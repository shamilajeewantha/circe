"""
Dense-optical-flow-based drift estimator, meant for "hold position" style
corrections (roll to cancel left/right drift, throttle to cancel climb/
descent) from a live camera feed.

Model choice (researched before writing this): OpenCV's DIS (Dense Inverse
Search) optical flow, ultrafast preset. Deep-learning models (RAFT,
NeuFlow v2, etc.) are more accurate but need a GPU/ONNX runtime and are
overkill here — DIS-ultrafast runs on CPU alone, is already in the
opencv-python we depend on elsewhere in this repo, and only needs "which
way is the dominant apparent motion", not per-pixel precision. Classical
Farneback/Lucas-Kanade were the other CPU-only options; DIS beats both on
speed and quality (see PROTOCOL_NOTES-style research notes in the PR/commit
that added this file).

IMPORTANT CAVEAT — forward-facing camera:
Every off-the-shelf optical-flow hover-hold (PX4, the PMW3901 sensor, toy
drones with "position hold", etc.) assumes a DOWNWARD-facing camera over a
roughly planar, constant-distance surface. Under that assumption, apparent
flow is close to pure translation and directly gives X/Y drift.

This module is fed the drone's forward-facing camera instead, which mixes:
  - translation (the drift we actually want to correct)
  - rotation (yaw/pitch/roll all add a uniform pan/tilt component across
    the whole frame, indistinguishable from translation by flow alone)
  - depth-dependent parallax (near objects flow faster than far ones, so
    there isn't really one true "the flow vector" even under pure
    translation)

There is no way to fully recover translational drift from a forward camera
alone without fusing IMU data to subtract out rotation (the MPU-9250 gyro
in the circe_v1/docs/mothership-scout.md design is earmarked for exactly
this, on the rover). Until then, this is a deliberate approximation:
robust (median) flow over a central ROI, treated as "dominant apparent
drift". Good enough for a HOVER-IN-PLACE feature that isn't deliberately
yawing; wrong if the drone is also intentionally turning while this runs.

Sign conventions (roll/throttle direction vs. flow direction) are a
best-effort guess and NOT yet live-verified against the real drone — flip
`gain` negative for either axis independently if a first live test shows
it correcting the wrong way.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np

NEUTRAL_STICK = 128


@dataclass
class FlowCorrection:
    roll_delta: int
    throttle_delta: int
    flow_dx: float          # smoothed dx that actually drove the correction
    flow_dy: float          # smoothed dy that actually drove the correction
    valid: bool             # False on the first frame of a session (no prior frame yet)
    # ── extras for visualization / logging (added for the Flow Lab) ──
    raw_dx: float = 0.0     # this-frame median dx BEFORE smoothing/deadband
    raw_dy: float = 0.0
    compute_ms: float = 0.0  # time spent inside update() for this frame
    # Raw dense flow field (in work_size coords) + the ROI rectangle that was
    # sampled, both in work_size pixel coordinates. `flow` is None on the first
    # (valid=False) frame. The annotator scales these up to the display frame.
    flow: "np.ndarray | None" = None
    roi: "tuple[int, int, int, int] | None" = None  # (x0, y0, w, h) in work_size coords
    work_size: "tuple[int, int]" = (0, 0)            # (w, h) the flow was computed at


class FlowStabilizer:
    """Feed consecutive BGR frames in; get back a small roll/throttle
    stick-delta meant to cancel the dominant apparent drift. Stateful
    (holds the previous frame + a smoothing filter) — call `reset()`
    when starting a new session or after a gap, so stale state from a
    previous run doesn't get treated as motion.
    """

    def __init__(self, work_width: int = 320, work_height: int = 180,
                 roi_fraction: float = 0.6, deadband_px: float = 0.8,
                 gain: float = 1.5, max_correction: int = 6,
                 smoothing_alpha: float = 0.4,
                 flip_roll: bool = False, flip_throttle: bool = False):
        self.work_size = (work_width, work_height)
        self.roi_fraction = roi_fraction
        self.deadband_px = deadband_px
        self.gain = gain
        self.max_correction = max_correction
        self.smoothing_alpha = smoothing_alpha
        # Correction sign is not yet live-verified against the real drone (see
        # module docstring). These let the UI flip either axis independently
        # if the drift arrow shows the correction fighting the wrong way.
        self.flip_roll = flip_roll
        self.flip_throttle = flip_throttle

        self._flow_engine = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
        self._prev_gray: np.ndarray | None = None
        self._smoothed_dx = 0.0
        self._smoothed_dy = 0.0

    def reset(self):
        self._prev_gray = None
        self._smoothed_dx = 0.0
        self._smoothed_dy = 0.0

    def params(self) -> dict:
        """Full snapshot of the tunables in effect, so every logged frame /
        correction can be reconstructed offline (why was the delta that big?)."""
        return {
            "gain": self.gain,
            "max_correction": self.max_correction,
            "deadband_px": self.deadband_px,
            "roi_fraction": self.roi_fraction,
            "smoothing_alpha": self.smoothing_alpha,
            "flip_roll": self.flip_roll,
            "flip_throttle": self.flip_throttle,
            "work_size": list(self.work_size),
        }

    def update(self, frame_bgr: np.ndarray) -> FlowCorrection:
        t0 = time.perf_counter()

        # Guard: accept already-grayscale frames too (a replay/webcam source
        # could hand us single-channel), so cvtColor never blows up.
        if frame_bgr.ndim == 2:
            gray = frame_bgr
        else:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, self.work_size, interpolation=cv2.INTER_AREA)

        w, h = self.work_size
        rw, rh = int(w * self.roi_fraction), int(h * self.roi_fraction)
        x0, y0 = (w - rw) // 2, (h - rh) // 2
        roi_rect = (x0, y0, rw, rh)

        if self._prev_gray is None:
            self._prev_gray = gray
            return FlowCorrection(0, 0, 0.0, 0.0, valid=False,
                                  compute_ms=(time.perf_counter() - t0) * 1000.0,
                                  roi=roi_rect, work_size=self.work_size)

        flow = self._flow_engine.calc(self._prev_gray, gray, None)
        self._prev_gray = gray

        roi = flow[y0:y0 + rh, x0:x0 + rw]

        # Median, not mean: robust against a noisy/outlier patch (e.g. a
        # moving object in frame) dominating the estimate.
        raw_dx = float(np.median(roi[..., 0]))
        raw_dy = float(np.median(roi[..., 1]))

        dx = 0.0 if abs(raw_dx) < self.deadband_px else raw_dx
        dy = 0.0 if abs(raw_dy) < self.deadband_px else raw_dy

        a = self.smoothing_alpha
        self._smoothed_dx = a * dx + (1 - a) * self._smoothed_dx
        self._smoothed_dy = a * dy + (1 - a) * self._smoothed_dy

        roll_sign = -1.0 if self.flip_roll else 1.0
        throttle_sign = -1.0 if self.flip_throttle else 1.0
        roll_delta = int(np.clip(roll_sign * self._smoothed_dx * self.gain,
                                  -self.max_correction, self.max_correction))
        throttle_delta = int(np.clip(throttle_sign * -self._smoothed_dy * self.gain,
                                      -self.max_correction, self.max_correction))

        return FlowCorrection(
            roll_delta, throttle_delta,
            self._smoothed_dx, self._smoothed_dy, valid=True,
            raw_dx=raw_dx, raw_dy=raw_dy,
            compute_ms=(time.perf_counter() - t0) * 1000.0,
            flow=flow, roi=roi_rect, work_size=self.work_size,
        )
