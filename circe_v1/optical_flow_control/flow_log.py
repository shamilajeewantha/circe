"""
Per-session logging for the Flow Lab, modeled on video_stream.PacketLogger.

Same hard-won policy as the video debug logs (see PROTOCOL_NOTES.md and the
user's standing rule): every run gets its OWN uniquely-named file (timestamp +
pid), nothing is ever auto-deleted, and `latest_session.txt` points at the
most recent one. This is so a future debugging session can reconstruct exactly
what the algorithm saw and what it sent to the drone, entirely offline.

Two record streams into one JSONL:
  - one record per processed frame (dx/dy raw + smoothed, corrections,
    compute_ms, source, frame index, valid) — event "flow_frame"
  - one record per corrective command ACTUALLY SENT to the drone, with the
    exact roll/pitch/throttle/yaw bytes, engaged/pulsed state, and how fresh
    the driving video frame was at send time — event "correction_sent"

Plus a bounded set of saved annotated .jpg frames for eyeballing later.
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Optional

import cv2


class FlowLogger:
    def __init__(self, out_dir: Optional[str] = None, max_saved_frames: int = 120):
        self.pid = os.getpid()
        stamp = time.strftime("%Y%m%d_%H%M%S")
        session_id = f"{stamp}_pid{self.pid}"
        base = out_dir or os.path.join(os.path.dirname(__file__), "flow_debug")
        self.out_dir = base
        self.session_dir = os.path.join(base, "sessions")
        self.frames_dir = os.path.join(self.session_dir, f"frames_{session_id}")
        os.makedirs(self.frames_dir, exist_ok=True)
        self._jsonl_path = os.path.join(self.session_dir, f"flow_{session_id}.jsonl")
        self._lock = threading.Lock()
        self._saved = 0
        self._max_saved = max_saved_frames
        with open(os.path.join(base, "latest_session.txt"), "w") as f:
            f.write(self._jsonl_path + "\n")
        self.log({"event": "logger_init", "pid": self.pid, "session_id": session_id})

    @property
    def jsonl_path(self) -> str:
        return self._jsonl_path

    def log(self, record: dict):
        record.setdefault("t", time.time())
        record.setdefault("pid", self.pid)
        with self._lock:
            with open(self._jsonl_path, "a") as f:
                f.write(json.dumps(record) + "\n")

    def log_frame(self, corr, *, frame_idx: int, source_name: str, source_fps: float):
        self.log({
            "event": "flow_frame",
            "frame_idx": frame_idx,
            "source": source_name,
            "source_fps": round(source_fps, 2),
            "valid": corr.valid,
            "raw_dx": round(corr.raw_dx, 3),
            "raw_dy": round(corr.raw_dy, 3),
            "smooth_dx": round(corr.flow_dx, 3),
            "smooth_dy": round(corr.flow_dy, 3),
            "roll_delta": corr.roll_delta,
            "throttle_delta": corr.throttle_delta,
            "compute_ms": round(corr.compute_ms, 3),
        })

    def log_correction_sent(self, *, roll: int, pitch: int, throttle: int, yaw: int,
                            engaged: bool, pulsed: bool, frame_age: float,
                            reason: str = ""):
        """Log a control command actually pushed to the drone (or the neutral
        it was forced to). `frame_age` = seconds since the driving video frame."""
        self.log({
            "event": "correction_sent",
            "roll": roll, "pitch": pitch, "throttle": throttle, "yaw": yaw,
            "engaged": engaged, "pulsed": pulsed,
            "frame_age": round(frame_age, 3),
            "reason": reason,
        })

    def save_annotated(self, bgr_img, tag: str = "ann") -> Optional[str]:
        with self._lock:
            if self._saved >= self._max_saved:
                return None
            self._saved += 1
            n = self._saved
        path = os.path.join(self.frames_dir, f"frame_{n:04d}_{tag}.jpg")
        cv2.imwrite(path, bgr_img)
        return path
