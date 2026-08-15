"""Network camera backend for VGGT-SLAM.

VGGT-SLAM's real-time loop (main_realtime.py) pulls frames through the pluggable
``vggt_slam.cameras.Camera`` interface (``start`` / ``capture`` / ``stop``). The
stock backends are physical devices (RealSense). This backend instead receives
frames **over the network** from the circe rover — the Gazebo sim laptop in the
simulation, and the real RPi camera on the deployed robot. Because the SLAM side
only sees the ``Camera`` interface, it never knows or cares which source pushed
the frames: **the exact same server runs unchanged from sim to real hardware.**

``slam_server.py`` decodes incoming JPEGs and calls :meth:`NetworkCamera.push`;
the SLAM loop calls :meth:`capture` as if it were a local camera.

Register it before VGGT-SLAM parses its ``--camera`` choice::

    from vggt_slam.cameras import BACKENDS
    from network_camera import NetworkCamera
    BACKENDS["network"] = NetworkCamera
"""

from __future__ import annotations

import queue
import threading
from typing import Optional

import numpy as np

from vggt_slam.cameras import Camera


class NetworkCamera(Camera):
    """A ``Camera`` whose frames arrive over the wire instead of from a device.

    Frames are BGR uint8 ``(H, W, 3)`` — identical to what ``RealSenseCamera``
    yields — so no downstream VGGT-SLAM code changes.
    """

    #: The most-recently constructed instance, so the FastAPI server can find the
    #: live camera the SLAM loop is holding and push frames into it. VGGT-SLAM
    #: constructs the backend itself (``BACKENDS[name]()``), so we can't pass the
    #: server a reference at construction time — we grab it here instead.
    latest: "Optional[NetworkCamera]" = None

    def __init__(self, maxbuffer: int = 256):
        self._q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=maxbuffer)
        self._running = False
        self._lock = threading.Lock()
        self._received = 0
        self._dropped = 0
        self._latest_frame: Optional[np.ndarray] = None  # peek copy, not consumed from _q
        NetworkCamera.latest = self

    # --- Camera interface -------------------------------------------------
    def start(self) -> None:
        self._running = True

    def capture(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """Return the next queued frame, or ``None`` if none arrive in *timeout*.

        Returning ``None`` (rather than blocking forever) lets the SLAM loop stay
        responsive to shutdown while the rover is between frames — matching how
        the RealSense backend returns ``None`` on a dropped frame.
        """
        if not self._running:
            return None
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self) -> None:
        self._running = False

    # --- server-facing push ----------------------------------------------
    def push(self, frame_bgr: np.ndarray) -> None:
        """Enqueue a frame received over the network.

        The queue is bounded; if the SLAM thread falls behind we drop the
        **oldest** frame and keep the freshest, because SLAM is latency-tolerant
        (project.md §9) and a stale backlog only hurts.
        """
        with self._lock:
            self._received += 1
            self._latest_frame = frame_bgr
            try:
                self._q.put_nowait(frame_bgr)
            except queue.Full:
                try:
                    self._q.get_nowait()
                    self._q.put_nowait(frame_bgr)
                    self._dropped += 1
                except queue.Empty:
                    pass

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Peek at the most recently POSTed frame without consuming it from the
        SLAM queue — for a diagnostic viewer to show real proof frames arrived."""
        with self._lock:
            return self._latest_frame

    # --- introspection for /status ---------------------------------------
    def stats(self) -> dict:
        return {
            "running": self._running,
            "queued": self._q.qsize(),
            "received": self._received,
            "dropped": self._dropped,
        }
