#!/usr/bin/env python3
import threading

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Image


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

        self.get_logger().info('Camera stream node started')

    def get_frame(self) -> bytes | None:
        with self._lock:
            return self._frame

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
        except Exception as e:
            self.get_logger().warn(f'Camera frame error: {e}')
