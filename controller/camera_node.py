#!/usr/bin/env python3
import json
import os
import struct
import subprocess
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Image

CONDA_PYTHON = '/home/shamila/anaconda3/envs/drone_detect/bin/python3'
WORKER_SCRIPT = Path(__file__).parent / 'detect_worker.py'


class CameraNode(Node):
    def __init__(self) -> None:
        super().__init__('camera_stream')

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.create_subscription(Image, '/rover_camera/image', self._image_cb, qos)

        self._lock = threading.Lock()
        self._frame: bytes | None = None
        self._raw_arr: np.ndarray | None = None

        self._boxes_lock = threading.Lock()
        self._boxes: list = []

        # Frame diagnostics
        self._frame_count = 0
        self._got_first = False
        self._last_detected_count = -1
        self.create_timer(5.0, self._watchdog)

        # Detection mode: 'local' spawns the on-board YOLO subprocess; 'remote' waits
        # for a detector on another machine to feed boxes via the server's /ws/detect.
        self._mode = os.environ.get('DETECT_MODE', 'local')

        # Worker starts in background — __init__ returns immediately
        self._worker = None
        self._worker_ready = False
        if self._mode == 'local':
            threading.Thread(target=self._start_worker, daemon=True).start()

        self.get_logger().info(f'Camera stream node started (DETECT_MODE={self._mode})')

    # ── public API ───────────────────────────────────────────────────────────────

    def get_frame(self) -> bytes | None:
        with self._lock:
            return self._frame

    def get_frame_with_id(self):
        """Latest JPEG plus its frame counter (used by the remote detector to skip dupes)."""
        with self._lock:
            return self._frame, self._frame_count

    def get_boxes(self) -> list:
        with self._boxes_lock:
            return list(self._boxes)

    def set_boxes(self, boxes) -> None:
        """Called from the server's /ws/detect handler when boxes arrive from laptop 2."""
        with self._boxes_lock:
            self._boxes = list(boxes)

    def destroy_node(self):
        try:
            if self._worker:
                self._worker.terminate()
        except Exception:
            pass
        super().destroy_node()

    # ── internal ─────────────────────────────────────────────────────────────────

    def _start_worker(self) -> None:
        """Runs in a background thread — blocks until YOLO is loaded, then loops reading results."""
        self.get_logger().info(f'Starting detection worker: {CONDA_PYTHON}')
        self._worker = subprocess.Popen(
            [CONDA_PYTHON, str(WORKER_SCRIPT)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        self._worker.stdout.readline()   # blocks until worker writes 'ready\n'
        self._worker_ready = True
        self.get_logger().info('Detection worker ready — bbox overlay active')
        self._detect_loop()              # stays in this thread: send frame → read result, forever

    def _image_cb(self, msg: Image) -> None:
        try:
            arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
            if msg.encoding == 'rgb8':
                arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            elif msg.encoding == 'bgra8':
                arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
            _, buf = cv2.imencode('.jpg', arr, [cv2.IMWRITE_JPEG_QUALITY, 80])
            with self._lock:
                self._frame = buf.tobytes()
                self._raw_arr = arr
            self._frame_count += 1
            if not self._got_first:
                self._got_first = True
                self.get_logger().info(
                    f'FIRST camera frame received: {msg.width}x{msg.height} encoding={msg.encoding}')
            elif self._frame_count % 60 == 0:
                self.get_logger().info(
                    f'camera frame #{self._frame_count} ({msg.width}x{msg.height})')
        except Exception as e:
            self.get_logger().warn(f'Camera frame error: {e}')

    def _watchdog(self) -> None:
        if not self._got_first:
            self.get_logger().warn(
                'NO camera frames on /rover_camera/image yet — is the camera bridge running? '
                '(launch_baylands.sh step 4)')

    def _detect_loop(self) -> None:
        """Runs in the worker's background thread. Synchronous request→response so the
        blocking pipe I/O never touches the ROS executor (which would freeze the camera)."""
        detected = 0
        while True:
            # Grab the freshest frame; skip if unchanged since last inference.
            with self._lock:
                arr = self._raw_arr
                count = self._frame_count
            if arr is None or count == self._last_detected_count:
                time.sleep(0.02)
                continue
            self._last_detected_count = count
            try:
                _, buf = cv2.imencode('.jpg', arr, [cv2.IMWRITE_JPEG_QUALITY, 80])
                data = buf.tobytes()
                self._worker.stdin.write(struct.pack('>I', len(data)) + data)
                self._worker.stdin.flush()
                line = self._worker.stdout.readline()
                if not line:
                    self.get_logger().warn('Detection worker closed its output pipe')
                    break
                boxes = json.loads(line)
                with self._boxes_lock:
                    self._boxes = boxes
                detected += 1
                if detected == 1:
                    self.get_logger().info('FIRST detection result received — overlay live')
                elif detected % 30 == 0:
                    self.get_logger().info(f'detection #{detected}: {len(boxes)} box(es)')
                time.sleep(0.03)   # yield CPU to the video encoder between inferences
            except Exception as e:
                self.get_logger().warn(f'Detect loop error: {e}')
                break
