"""
FlowProcessor — the orchestrator that ties the Flow Lab together.

One background thread:
    source.read()  ->  FlowStabilizer.update()  ->  annotate (flow_viz)
      ->  publish two annotated JPEG streams (arrow view + HSV view)
      ->  log every frame + save bounded annotated frames
      ->  (opt-in) send corrective commands to the drone

The two annotated streams are exposed through the SAME `wait_for_next_frame`
push interface that video_stream.VideoReceiver uses, so they plug straight
into the existing video_stream.MjpegServer with zero changes — the browser
just points two <img> tags at two ports.

Actuation is off unless explicitly engaged, and even then:
  - a stale-frame watchdog forces the sticks back to neutral if the video
    feed freezes (never latch a stale correction onto a flying drone),
  - "pulsed" mode (default) nudges then releases, mimicking the proven-healthy
    button-press pattern; "continuous" holds armed for comparison,
  - every command actually sent is logged with its exact bytes.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Optional

import cv2

import flow_viz
from flow_log import FlowLogger
from flow_sources import FrameSource
from flow_stabilizer import FlowCorrection, FlowStabilizer
from video_stream import MjpegServer

NEUTRAL = 128


def _clamp_stick(v: int) -> int:
    return max(0, min(255, int(v)))


class _AnnotatedStream:
    """Minimal frame sink exposing exactly the `wait_for_next_frame` push
    contract that MjpegServer consumes — one per annotated view."""

    def __init__(self):
        self._cond = threading.Condition()
        self._jpeg: Optional[bytes] = None
        self._version = 0

    def publish(self, jpeg_bytes: bytes):
        with self._cond:
            self._jpeg = jpeg_bytes
            self._version += 1
            self._cond.notify_all()

    def wait_for_next_frame(self, after_version: int, timeout: float = 5.0):
        with self._cond:
            if self._version == after_version:
                self._cond.wait(timeout=timeout)
            return self._jpeg, self._version


class FlowProcessor:
    def __init__(self, source: FrameSource, stabilizer: FlowStabilizer,
                 drone=None, logger: Optional[FlowLogger] = None,
                 host: str = "127.0.0.1", arrow_port: int = 8091, hsv_port: int = 8092,
                 save_every: int = 8, plot_history: int = 400,
                 corrections_history: int = 200,
                 stale_after: float = 0.4):
        self.source = source
        self.stabilizer = stabilizer
        self.drone = drone
        self.logger = logger or FlowLogger()
        self.host = host
        self.arrow_port = arrow_port
        self.hsv_port = hsv_port
        self.save_every = save_every
        self.stale_after = stale_after

        self._arrow_stream = _AnnotatedStream()
        self._hsv_stream = _AnnotatedStream()
        self._arrow_server = MjpegServer(self._arrow_stream, host=host, port=arrow_port)
        self._hsv_server = MjpegServer(self._hsv_stream, host=host, port=hsv_port)

        # Guards start()/stop() against concurrent calls (e.g. a double-click
        # racing two Gradio callback invocations) so we never spin up two
        # processing threads or attempt to bind the MJPEG ports twice.
        self._lifecycle_lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._running = False

        # actuation state
        self.engaged = False
        self.pulsed = True
        self.pulse_on_s = 0.15
        self.pulse_period_s = 0.5
        self._pulse_next = 0.0
        self._pulse_off_at: Optional[float] = None
        self._last_sent_neutral = False

        # live state / stats
        self.last_corr: FlowCorrection = FlowCorrection(0, 0, 0.0, 0.0, valid=False)
        self.last_frame_ts: Optional[float] = None
        self.frames_processed = 0
        self.start_error: Optional[str] = None
        self._plot = deque(maxlen=plot_history)  # (t, dx, dy, roll, throttle)
        self._plot_lock = threading.Lock()
        self._corrections = deque(maxlen=corrections_history)  # dicts, newest last
        self._corrections_lock = threading.Lock()

    # ── lifecycle ───────────────────────────────────────────────────────

    def start(self):
        with self._lifecycle_lock:
            if self._running:
                return  # idempotent — already running, nothing to do
            self.stabilizer.reset()  # never carry stale motion state across runs
            self.source.start()
            try:
                self._arrow_server.start()
                self._hsv_server.start()
            except OSError as e:
                self.start_error = f"MJPEG server bind failed: {e}"
                self.source.stop()
                raise
            self._running = True
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
            self.logger.log({"event": "processor_start", "source": self.source.name,
                             "arrow_url": self.arrow_url, "hsv_url": self.hsv_url})

    def stop(self):
        with self._lifecycle_lock:
            if not self._running:
                return  # idempotent — already stopped, nothing to do
            self._running = False
            if self._thread:
                self._thread.join(timeout=2.0)
                self._thread = None
            # return the drone to the video-friendly idle state on the way out
            self._go_idle(reason="processor_stop")
            self._arrow_server.stop()
            self._hsv_server.stop()
            self.source.stop()
            self.logger.log({"event": "processor_stop", "frames_processed": self.frames_processed})

    @property
    def arrow_url(self) -> str:
        return self._arrow_server.url

    @property
    def hsv_url(self) -> str:
        return self._hsv_server.url

    # ── actuation control (called from the UI) ──────────────────────────

    def engage(self):
        self.engaged = True
        self._pulse_next = 0.0
        self._pulse_off_at = None
        self.logger.log({"event": "engage", "pulsed": self.pulsed})

    def disengage(self):
        self.engaged = False
        self._go_idle(reason="disengage")
        self.logger.log({"event": "disengage"})

    def _go_idle(self, reason: str):
        """Drop the drone back to the streaming-friendly idle/neutral state."""
        if self.drone is None:
            return
        try:
            self.drone.set_controls(roll=NEUTRAL, pitch=NEUTRAL, throttle=NEUTRAL, yaw=NEUTRAL)
            self.drone.set_idle_mode(True)
        except Exception as e:  # noqa: BLE001 — never let cleanup crash the loop
            self.logger.log({"event": "go_idle_error", "reason": reason, "error": str(e)})

    @property
    def is_running(self) -> bool:
        return self._running

    # ── plot / table access for the UI ──────────────────────────────────

    def plot_snapshot(self):
        with self._plot_lock:
            return list(self._plot)

    def corrections_snapshot(self):
        """Most-recent-last list of correction-sent dicts, for the UI table."""
        with self._corrections_lock:
            return list(self._corrections)

    def _record_correction(self, *, roll, pitch, throttle, yaw, pulsed, frame_age, reason):
        with self._corrections_lock:
            self._corrections.append({
                "time": time.strftime("%H:%M:%S"),
                "roll": roll, "pitch": pitch, "throttle": throttle, "yaw": yaw,
                "pulsed": pulsed, "frame_age_ms": round(frame_age * 1000, 1),
                "reason": reason,
            })

    # ── the processing loop ─────────────────────────────────────────────

    def _loop(self):
        while self._running:
            frame, ts = self.source.read()
            now = time.time()

            if frame is not None:
                self.last_frame_ts = now
                corr = self.stabilizer.update(frame)
                self.last_corr = corr
                self.frames_processed += 1
                self._publish_views(frame, corr)
                self.logger.log_frame(corr, frame_idx=self.frames_processed,
                                      source_name=self.source.name,
                                      source_fps=self.source.fps)
                with self._plot_lock:
                    self._plot.append((now, corr.flow_dx, corr.flow_dy,
                                       corr.roll_delta, corr.throttle_delta))

            frame_age = (now - self.last_frame_ts) if self.last_frame_ts else 999.0
            self._apply_actuation(self.last_corr, frame_age)

            if frame is None:
                time.sleep(0.005)  # avoid a busy spin when a source returns None fast

    def _publish_views(self, frame, corr: FlowCorrection):
        sent = self.engaged and corr.valid
        arrow = flow_viz.render_arrow_view(
            frame, corr, source_fps=self.source.fps, engaged=self.engaged,
            source_name=self.source.name, sent=sent)
        hsv = flow_viz.render_hsv_view(frame, corr)
        ok_a, buf_a = cv2.imencode(".jpg", arrow)
        ok_h, buf_h = cv2.imencode(".jpg", hsv)
        if ok_a:
            self._arrow_stream.publish(buf_a.tobytes())
        if ok_h:
            self._hsv_stream.publish(buf_h.tobytes())
        if ok_a and self.frames_processed % self.save_every == 0:
            self.logger.save_annotated(arrow, "arrow")

    def _apply_actuation(self, corr: FlowCorrection, frame_age: float):
        if not self.engaged or self.drone is None:
            return

        fresh = corr.valid and frame_age <= self.stale_after
        if not fresh:
            # Frozen/invalid feed — never keep pushing a stale correction.
            self._send_neutral(frame_age, reason="stale_or_invalid")
            return

        roll = _clamp_stick(NEUTRAL + corr.roll_delta)
        throttle = _clamp_stick(NEUTRAL + corr.throttle_delta)

        if self.pulsed:
            now = time.time()
            if now >= self._pulse_next:
                self._send_correction(roll, throttle, frame_age, pulsed=True, reason="pulse_on")
                self._pulse_off_at = now + self.pulse_on_s
                self._pulse_next = now + self.pulse_period_s
            elif self._pulse_off_at is not None and now >= self._pulse_off_at:
                self._go_idle(reason="pulse_off")
                self._pulse_off_at = None
                self.logger.log_correction_sent(
                    roll=NEUTRAL, pitch=NEUTRAL, throttle=NEUTRAL, yaw=NEUTRAL,
                    engaged=True, pulsed=True, frame_age=frame_age, reason="pulse_off")
                self._record_correction(roll=NEUTRAL, pitch=NEUTRAL, throttle=NEUTRAL,
                                        yaw=NEUTRAL, pulsed=True, frame_age=frame_age,
                                        reason="pulse_off")
        else:
            self._send_correction(roll, throttle, frame_age, pulsed=False, reason="continuous")

    def _send_correction(self, roll: int, throttle: int, frame_age: float,
                         pulsed: bool, reason: str):
        self.drone.set_idle_mode(False)
        self.drone.set_controls(roll=roll, pitch=NEUTRAL, throttle=throttle, yaw=NEUTRAL)
        self._last_sent_neutral = False
        self.logger.log_correction_sent(
            roll=roll, pitch=NEUTRAL, throttle=throttle, yaw=NEUTRAL,
            engaged=True, pulsed=pulsed, frame_age=frame_age, reason=reason)
        self._record_correction(roll=roll, pitch=NEUTRAL, throttle=throttle, yaw=NEUTRAL,
                                pulsed=pulsed, frame_age=frame_age, reason=reason)

    def _send_neutral(self, frame_age: float, reason: str):
        if self._last_sent_neutral:
            return  # already neutral; don't spam identical commands/logs
        self._go_idle(reason=reason)
        self._last_sent_neutral = True
        self.logger.log_correction_sent(
            roll=NEUTRAL, pitch=NEUTRAL, throttle=NEUTRAL, yaw=NEUTRAL,
            engaged=True, pulsed=self.pulsed, frame_age=frame_age, reason=reason)
        self._record_correction(roll=NEUTRAL, pitch=NEUTRAL, throttle=NEUTRAL, yaw=NEUTRAL,
                                pulsed=self.pulsed, frame_age=frame_age, reason=reason)
