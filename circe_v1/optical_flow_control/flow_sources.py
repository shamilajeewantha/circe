"""
Pluggable frame sources for the optical-flow Flow Lab.

Every source hands back plain **BGR numpy frames** (what OpenCV and
FlowStabilizer expect) plus the wall-clock timestamp the frame became
available, behind one small interface so flow_processor.py doesn't care where
frames come from:

    src.start()
    frame, ts = src.read()   # frame is None if nothing new is ready yet
    src.stop()
    src.name                 # human label for the UI/logs
    src.fps                  # rolling estimate of delivered frames/sec

Three implementations:
  - DroneFrameSource  — the real path: decode the DM002HW's reconstructed
                        JPEGs coming off video_stream.VideoReceiver.
  - WebcamFrameSource — a laptop webcam, for developing/debugging the
                        algorithm + visualization with no drone in the air.
  - ReplayFrameSource — replay .jpg frames already saved under
                        video_debug/sessions/frames_* (or any folder of jpgs),
                        deterministically, as many times as you like — the
                        offline-debug path.

read() is non-blocking-ish: it returns (None, ts) when no new frame is
available rather than blocking the processor loop, EXCEPT the drone source
which blocks up to a short timeout on the receiver's Condition (that's the
efficient push path, no busy-polling).
"""

from __future__ import annotations

import glob
import os
import time
from typing import Optional

import cv2
import numpy as np


class _FpsMeter:
    """Rolling frames/sec estimate over the last ~2s of deliveries."""

    def __init__(self, window: float = 2.0):
        self._window = window
        self._stamps: list[float] = []

    def tick(self, ts: float):
        self._stamps.append(ts)
        cutoff = ts - self._window
        while self._stamps and self._stamps[0] < cutoff:
            self._stamps.pop(0)

    @property
    def fps(self) -> float:
        if len(self._stamps) < 2:
            return 0.0
        span = self._stamps[-1] - self._stamps[0]
        return (len(self._stamps) - 1) / span if span > 0 else 0.0


class FrameSource:
    """Base interface. Subclasses implement _read_frame()."""

    name = "base"

    def __init__(self):
        self._fps = _FpsMeter()
        self._started = False

    def start(self):
        self._started = True

    def stop(self):
        self._started = False

    @property
    def fps(self) -> float:
        return self._fps.fps

    def read(self) -> tuple[Optional[np.ndarray], float]:
        """Return (frame_bgr | None, timestamp). None means 'nothing new'."""
        frame, ts = self._read_frame()
        if frame is not None:
            self._fps.tick(ts)
        return frame, ts

    def _read_frame(self) -> tuple[Optional[np.ndarray], float]:
        raise NotImplementedError


class DroneFrameSource(FrameSource):
    """Decodes the reconstructed JPEG frames coming off a running
    video_stream.VideoReceiver. Uses the receiver's push Condition
    (wait_for_next_frame) so we sleep until a genuinely new frame exists
    instead of busy-polling — the same mechanism the MJPEG server uses."""

    name = "drone"

    def __init__(self, receiver, wait_timeout: float = 1.0):
        super().__init__()
        self.receiver = receiver
        self._wait_timeout = wait_timeout
        self._version = 0

    def _read_frame(self):
        jpeg, version = self.receiver.wait_for_next_frame(self._version,
                                                          timeout=self._wait_timeout)
        if jpeg is None or version == self._version:
            return None, time.time()  # timed out with no new frame
        self._version = version
        arr = np.frombuffer(jpeg, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            # Corrupt/incomplete JPEG slipped through — skip, don't crash.
            return None, time.time()
        return frame, time.time()


class WebcamFrameSource(FrameSource):
    """A local webcam via cv2.VideoCapture. For developing/debugging the
    algorithm and visualization without a drone."""

    name = "webcam"

    def __init__(self, index: int = 0, width: int = 640, height: int = 360):
        super().__init__()
        self.index = index
        self.width = width
        self.height = height
        self._cap: Optional[cv2.VideoCapture] = None
        self.open_error: Optional[str] = None

    def start(self):
        super().start()
        cap = cv2.VideoCapture(self.index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if not cap.isOpened():
            self.open_error = f"Could not open webcam index {self.index}"
            cap.release()
            self._cap = None
            return
        self._cap = cap

    def stop(self):
        super().stop()
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def _read_frame(self):
        if self._cap is None:
            return None, time.time()
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return None, time.time()
        return frame, time.time()


class ReplayFrameSource(FrameSource):
    """Replays .jpg frames from a folder (e.g. a video_debug/sessions/frames_*
    dir) at a fixed fps. Deterministic offline debugging — no hardware. Set
    loop=True to cycle forever; otherwise it stops after the last frame
    (read() returns (None, ...) once exhausted)."""

    name = "replay"

    def __init__(self, folder: str, fps: float = 7.0, loop: bool = True):
        super().__init__()
        self.folder = folder
        self.play_fps = max(0.1, fps)
        self.loop = loop
        self.files: list[str] = sorted(glob.glob(os.path.join(folder, "*.jpg")))
        self._idx = 0
        self._next_due = 0.0
        self.exhausted = False
        self.load_error: Optional[str] = None
        if not self.files:
            self.load_error = f"No .jpg frames found in {folder}"

    def start(self):
        super().start()
        self._idx = 0
        self._next_due = time.time()
        self.exhausted = False

    def _read_frame(self):
        if not self.files or self.exhausted:
            return None, time.time()
        now = time.time()
        if now < self._next_due:
            return None, now  # pace to play_fps; nothing new yet this tick
        path = self.files[self._idx]
        self._idx += 1
        if self._idx >= len(self.files):
            if self.loop:
                self._idx = 0
            else:
                self.exhausted = True
        self._next_due = now + 1.0 / self.play_fps
        frame = cv2.imread(path, cv2.IMREAD_COLOR)
        if frame is None:
            return None, now  # unreadable file — skip
        return frame, now
