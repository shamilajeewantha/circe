"""
DM002HW drone controller — one page, one job: fly it, and hold its position.

The DM002HW cannot hover on its own (no pitch/roll/yaw stability at all), so
the optical-flow control loop below IS the hover mechanism, not an optional
add-on — it lives on this same page as flight controls, not on a separate
tab you'd have to switch to mid-flight.

Layout (single Controller page):
  - status bar (connection / video / hold-position, always visible)
  - Connection & Video (toggles + raw feed)
  - Optical Flow — Hold Position (labeled arrow/HSV views + the one toggle
    that starts the control loop AND engages actuation together)
  - Manual flight controls (takeoff/land/calibrate/e-stop, gimbals, raw axis)
  - Diagnostics (collapsed by default): corrections table, live numbers,
    drift/correction plot, event log

All mutable app state (drone, video, MJPEG, optical-flow processors, event log)
is owned by a single DroneController instance — the single source of truth every
toggle handler AND the poll timer render from, so button label/color and the
video pane always reflect real backend state, never "what was last clicked."
This is what makes an automatic safety disengage (stale video / lost link) show
up without another click, clears the video pane on disconnect instead of
freezing the last frame, and disables flight buttons whenever we're not connected.

Actuation is Displacement Hold: while accumulated drift (cum_dx/cum_dy) stays
under a threshold, nothing is sent; once it crosses, one fixed-size nudge
fires and the drone goes fully neutral for a settle window before the next
check — never a continuous stream of corrections.

Run with:  python main.py
Then open the printed local URL in a browser.
"""

import atexit
import os
import signal
import threading
import time
import warnings

# Gradio 6 / Starlette combo emits this on every queued request (so once per
# gr.Timer tick, i.e. constantly) — cosmetic version-skew noise, not a bug.
warnings.filterwarnings("ignore", message=".*HTTP_422_UNPROCESSABLE_ENTITY.*")

import gradio as gr
import pandas as pd

from applog import get_logger, close_logging
from drone import Drone, NEUTRAL
from video_stream import VideoReceiver, MjpegServer
from flow_stabilizer import FlowStabilizer
from flow_sources import WebcamFrameSource, DroneFrameSource
from flow_processor import FlowProcessor
from flow_log import FlowLogger
from analog_control import AnalogInputServer

log = get_logger("main")

HOLD_ARROW_PORT, HOLD_HSV_PORT = 8091, 8092
DEMO_ARROW_PORT, DEMO_HSV_PORT = 8093, 8094
ANALOG_PORT = 8095

_BADGE_CSS = "display:inline-block; padding:0.35em 0.9em; border-radius:999px; font-weight:700; margin-right:0.5em;"
_EMPTY_PLOT = pd.DataFrame({"t": [], "value": [], "series": []})


def _badge(text: str, color: str) -> str:
    return f'<span style="{_BADGE_CSS} background:{color}; color:white;">{text}</span>'


# ════════════════════════════════════════════════════════════════════════════
# DroneController — owns ALL live state (connection, video, both optical-flow
# processors, event log) and is the single source of truth the UI renders from.
# Cascade teardown, lost-link auto-recovery, and one render/sync path shared by
# every handler AND the poll timer all live here.
# ════════════════════════════════════════════════════════════════════════════

class DroneController:
    def __init__(self):
        self.drone = Drone()
        self.video: VideoReceiver | None = None
        self.mjpeg: MjpegServer | None = None
        self.stabilizer = FlowStabilizer()
        self.processor: FlowProcessor | None = None       # Hold Position (drone) processor
        self.demo_processor: FlowProcessor | None = None  # Optical Flow tab (webcam) processor
        self.analog_server: AnalogInputServer | None = None  # smooth "gyro-style" analog control
        # Per-axis invert flags for analog mode — shared by reference with the
        # running server, so an invert checkbox takes effect without a restart.
        self.analog_invert = {"roll": False, "pitch": False, "throttle": False, "yaw": False}
        self._events: list[str] = []

        # Serializes all four toggle handlers (Connect / Start video / Hold
        # Position / webcam Demo) so a rapid double-click can't race two
        # connects/videos/demos and orphan a socket, thread, or bound port.
        # Also makes the Hold-Position/Demo mutual-exclusion check race-free.
        self.lock = threading.Lock()
        # Separate non-reentrant lock serializing ONLY the timed DIRECTIONAL
        # MOVES (up/down/fwd/back/left/right/yaw) — not toggles, not one-shot
        # commands (takeoff/land/hover), not absolute axis sets. Firing two
        # timed moves at once makes their set_controls()/sleep/hover sequences
        # interleave and clobber each other's axes (set_controls resets
        # unspecified axes to neutral) — a self-fighting trajectory a single
        # physical transmitter never produces. The 124-byte @ ~18Hz wire pattern
        # is unaffected (drone.py._loop), but the axis/cmd VALUES would deviate
        # from the real app. A move pressed while one runs is a "busy" no-op;
        # non-move commands and e-stop are never blocked by it.
        self._cmd_lock = threading.Lock()
        # Anti-churn: only re-emit video-feed HTML / button interactivity on a
        # real transition, so the 0.4s timer doesn't remount the MJPEG <img> or
        # spam identical updates every tick.
        self._video_feed_state: str | None = None
        self._gated_state: bool | None = None

    # ── state queries ───────────────────────────────────────────────────────
    @property
    def connected(self) -> bool:
        return self.drone._armed

    @property
    def video_running(self) -> bool:
        return self.video is not None

    @property
    def holding(self) -> bool:
        return self.processor is not None and self.processor.engaged

    @property
    def analog_engaged(self) -> bool:
        return self.analog_server is not None

    def _note(self, msg: str) -> str:
        log.info(msg)
        ts = time.strftime("%H:%M:%S")
        self._events.append(f"[{ts}] {msg}")
        # Fan the same action into whichever session logger(s) are live, so a
        # single JSONL shows button presses interleaved with packet/frame events
        # on one comparable timeline.
        for target in (self.video, self.processor, self.demo_processor):
            if target is not None:
                target.logger.log({"event": "ui_action", "msg": msg})
        return "\n".join(self._events[-16:])

    # ════════════════════════════════════════════════════════════════════════
    # Cascade teardown — each level tears down everything beneath it, in order,
    # so nothing is ever left dangling (a FlowProcessor reading a stopped
    # VideoReceiver, a receiver on a socket that's about to close, etc).
    # ════════════════════════════════════════════════════════════════════════
    def _release_hold_position(self, reason: str):
        if self.processor is not None:
            self.processor.disengage()
            self.processor.stop()
            self.processor = None
            log.info("Hold Position released (%s)", reason)

    def _release_analog(self, reason: str):
        if self.analog_server is not None:
            self.analog_server.stop()   # also centers the drone (hover) on the way out
            self.analog_server = None
            log.info("Analog control released (%s)", reason)

    def _teardown_video(self, reason: str):
        self._release_hold_position(reason)  # Hold Position can't outlive its video source
        if self.mjpeg is not None:
            self.mjpeg.stop()
            self.mjpeg = None
        if self.video is not None:
            self.video.stop()
            self.video = None
        self.drone.set_idle_mode(False)

    def _teardown_connection(self, reason: str):
        self._release_analog(reason)  # analog drives the drone directly — can't outlive the link
        self._teardown_video(reason)  # video can't outlive the connection
        if self.drone._armed:
            self.drone.disconnect()

    # ════════════════════════════════════════════════════════════════════════
    # UI state sync — every toggle handler AND the polling timer call this, so
    # button label/color/interactivity always reflect actual backend state.
    # ════════════════════════════════════════════════════════════════════════
    def _connect_button(self):
        # Always interactive — connect/disconnect is how you recover state.
        if self.connected:
            return gr.update(value="🔌 CONNECTED — click to disconnect")
        return gr.update(value="🔌 Connect")

    def _video_button(self):
        # Gated on connection: can't start/stop video without a link.
        if self.video is not None:
            return gr.update(value="🎥 VIDEO LIVE — click to stop", interactive=self.connected)
        return gr.update(value="🎥 Start video", interactive=self.connected)

    def _hold_button(self):
        # Gated on video being live (which implies connected).
        if self.holding:
            return gr.update(value="🎯 HOLDING — click to release", variant="stop",
                             interactive=self.video_running)
        return gr.update(value="🎯 HOLD POSITION", variant="secondary",
                         interactive=self.video_running)

    def _analog_button(self):
        # Gated on connection (analog drives the drone directly, no video needed).
        if self.analog_engaged:
            return gr.update(value="🎮 ANALOG ON — click to stop", variant="stop",
                             interactive=self.connected)
        return gr.update(value="🎮 ANALOG CONTROL (smooth)", variant="secondary",
                         interactive=self.connected)

    def status_badge_html(self) -> str:
        if self.drone._armed:
            conn = _badge(f"🟢 CONNECTED — {self.drone.ip}:{self.drone.port}", "#27ae60")
        elif self.drone.last_error:
            conn = _badge(f"🔴 LOST CONNECTION — {self.drone.last_error}", "#c0392b")
        else:
            conn = _badge("⚪ NOT CONNECTED", "#7f8c8d")

        if self.video is None:
            vid = _badge("⚪ VIDEO NOT STARTED", "#7f8c8d")
        elif self.video.last_decoded_ts is None:
            vid = _badge("🟡 VIDEO: waiting for first frame…", "#e67e22")
        else:
            age = time.time() - self.video.last_decoded_ts
            vid = (_badge(f"🟢 VIDEO LIVE ({self.video.frames_ok} frames)", "#27ae60") if age < 2.0
                   else _badge(f"🟡 VIDEO STALLED — last frame {age:.1f}s ago", "#e67e22"))

        if self.holding:
            p = self.processor
            age = (time.time() - p.last_frame_ts) if p.last_frame_ts else 999
            hold = (_badge("🟢 HOLDING POSITION", "#27ae60") if age < 0.6
                    else _badge(f"🟡 HOLDING (stale {age:.1f}s → neutral)", "#e67e22"))
        else:
            hold = _badge("⚪ NOT HOLDING", "#7f8c8d")

        analog = _badge("🎮 ANALOG ON", "#8e44ad") if self.analog_engaged else ""

        return f'<div style="font-size:1.05em;">{conn}{vid}{hold}{analog}</div>'

    def _ui_sync(self):
        return (self._connect_button(), self._video_button(), self._hold_button(),
                self._analog_button(), self.status_badge_html())

    def button_states(self, count: int) -> tuple:
        """interactive=connected for every flight/D-pad button, re-emitted only
        on an actual connect/disconnect transition (else no-op)."""
        en = self.connected
        if en == self._gated_state:
            return tuple(gr.update() for _ in range(count))
        self._gated_state = en
        return tuple(gr.update(interactive=en) for _ in range(count))

    def _check_lost_link(self):
        """If the drone control loop died (a send error auto-disarms it) while
        video or Hold Position is still live, cascade-teardown once — never keep
        a FlowProcessor/VideoReceiver running against a dead socket. Runs from
        the poll timer, so recovery is automatic without the user clicking."""
        if self.drone._armed or (self.video is None and self.processor is None
                                 and self.analog_server is None):
            return
        if not self.lock.acquire(blocking=False):
            return  # a toggle handler is mid-flight; it'll settle on the next tick
        try:
            if not self.drone._armed and (self.video is not None or self.processor is not None
                                          or self.analog_server is not None):
                self._release_analog("lost link — drone control loop stopped")
                self._teardown_video("lost link — drone control loop stopped")
                self._note("Lost connection to drone — video / Hold Position / analog torn down.")
        finally:
            self.lock.release()

    def poll_sync(self):
        """One timer tick: auto-recover from a lost link, then refresh every
        state-driven display (video feed, both flow views, all three toggle
        buttons, status badges) from real backend state."""
        self._check_lost_link()
        arrow_html, hsv_html = self._hold_views_html()
        return (self.render_video_feed(force=False), arrow_html, hsv_html,
                self._connect_button(), self._video_button(), self._hold_button(),
                self._analog_button(), self.status_badge_html())

    # ════════════════════════════════════════════════════════════════════════
    # Connection & Video toggles
    # ════════════════════════════════════════════════════════════════════════
    def _raw_video_html(self):
        if self.mjpeg is not None:
            return f'<img src="{self.mjpeg.url}" style="width:100%;border-radius:8px;">'
        return "<div id='video-box'>🎥 Video feed not connected yet</div>"

    def render_video_feed(self, force: bool = False):
        """Video-pane HTML, but only on a live↔placeholder transition (or
        force=True for click paths); steady-state timer ticks return gr.update()
        so the MJPEG <img> isn't remounted every 0.4s."""
        desired = "live" if self.mjpeg is not None else "placeholder"
        changed = desired != self._video_feed_state
        self._video_feed_state = desired
        if not changed and not force:
            return gr.update()
        return self._raw_video_html()

    def _start_video(self, width, height, color) -> str:
        """Start the video receiver + MJPEG server. Assumes the caller already
        holds self.lock and has confirmed we're connected and video isn't already
        running. Returns a status note (success or error)."""
        self.drone.set_idle_mode(True)
        self.video = VideoReceiver(self.drone.sock, self.drone.ip, self.drone.port,
                                   width=int(width), height=int(height),
                                   components=3 if color else 1)
        self.video.start()
        self.mjpeg = MjpegServer(self.video)
        try:
            self.mjpeg.start()
            return self._note(f"Video started ({int(width)}x{int(height)}). "
                              f"Log: {self.video.logger._jsonl_path}")
        except OSError as e:
            self.video.stop()
            self.video = None
            self.mjpeg = None
            return self._note(f"Video failed to bind: {e} (port may still be releasing)")

    def do_toggle_connect(self, width=640, height=360, color=True):
        with self.lock:  # serialize toggles — no double-connect race
            if self.connected:
                self._teardown_connection("user disconnect")
                note = self._note("Disconnected.")
            else:
                self.drone.last_error = None
                try:
                    self.drone.connect()
                    # Real app spends ~95% of a healthy video session disarmed/idle —
                    # engage idle mode immediately rather than waiting for Start
                    # video (a live test showed the stall locking in before a later
                    # switch took effect). Flight commands auto-wake the drone.
                    self.drone.set_idle_mode(True)
                    note = self._note(f"Connected — streaming controls to {self.drone.ip}:{self.drone.port}")
                    # Auto-start video on connect so it's one less manual step —
                    # the Start/Stop video button still works independently and
                    # stays in sync (recomputed from state on every click and tick).
                    note = self._start_video(width, height, color)
                except OSError as e:
                    note = self._note(f"Connect failed: {e}")
            return (note, self.render_video_feed(force=True)) + self._ui_sync()

    def do_toggle_video(self, width, height, color):
        with self.lock:  # serialize toggles — no double-start race
            if self.video is not None:
                self._teardown_video("user stop video")
                note = self._note("Video stopped, drone back to normal flight mode.")
            elif not self.connected:
                note = self._note("Ignored 'Start video' — not connected. Click Connect first.")
            else:
                note = self._start_video(width, height, color)
            return (note, self.render_video_feed(force=True)) + self._ui_sync()

    def poll_video_stats(self):
        if self.video is None:
            return "video: not running"
        stats = (f"pid={os.getpid()} packets={self.video.packets_seen} decoded_ok={self.video.frames_ok} "
                 f"decode_failed={self.video.frames_failed} unknown_pkts={self.video.unknown_packets} "
                 f"kicks_sent={self.video.kicks_sent} handshake_resends={self.video.handshake_resends}")
        if self.video.last_error:
            stats += f" | last_decode_error={self.video.last_error}"
        return stats

    # ════════════════════════════════════════════════════════════════════════
    # Hold Position — the optical-flow control loop IS the hover mechanism
    # ════════════════════════════════════════════════════════════════════════
    def _hold_views_html(self):
        if self.processor is not None:
            arrow = f'<img src="{self.processor.arrow_url}" style="width:100%;border-radius:8px;">'
            hsv = f'<img src="{self.processor.hsv_url}" style="width:100%;border-radius:8px;">'
            return arrow, hsv
        return (_hold_placeholder("🡒 Motion Arrows — waiting for Hold Position"),
                _hold_placeholder("🌈 Dense Field (HSV) — waiting for Hold Position"))

    def do_toggle_hold(self):
        with self.lock:  # serialize toggles — no double-start race
            if self.holding:
                self._release_hold_position("user released")
                note = self._note("Hold Position released. Drone back to idle/neutral.")
            elif self.video is None:
                note = self._note("Ignored 'HOLD POSITION' — video isn't live. Start video first "
                                  "(can't hold position with no frames).")
            elif self.analog_engaged:
                note = self._note("Ignored 'HOLD POSITION' — Analog control is engaged. Stop Analog "
                                  "first (both drive the drone; they can't run at once).")
            elif self.demo_processor is not None:
                note = self._note("Ignored 'HOLD POSITION' — webcam Demo is running. Stop the Demo "
                                  "first (shared stabilizer can't drive two live sources at once).")
            else:
                proc = FlowProcessor(DroneFrameSource(self.video), self.stabilizer, drone=self.drone,
                                     logger=FlowLogger(), arrow_port=HOLD_ARROW_PORT, hsv_port=HOLD_HSV_PORT)
                try:
                    proc.start()
                    proc.engage()
                    self.processor = proc
                    note = self._note(f"HOLD POSITION engaged (Displacement Hold). Log: {proc.logger.jsonl_path}")
                except OSError as e:
                    note = self._note(f"Hold Position failed to start: {e} (port may still be releasing)")
            arrow_html, hsv_html = self._hold_views_html()
            return (note, arrow_html, hsv_html) + self._ui_sync()

    # ════════════════════════════════════════════════════════════════════════
    # Analog "gyro-style" control — continuous fine axis values (like tilting
    # the phone), streamed from a browser joystick into set_controls at loop
    # rate. See analog_control.py / PROTOCOL_NOTES.md "flight-with-video".
    # ════════════════════════════════════════════════════════════════════════
    def _analog_log(self, event, **fields):
        """Fan analog events into whichever session logger(s) are live (no-op if
        analog is running without video/hold — JSONL for that case is deferred)."""
        rec = {"event": event, **fields}
        for target in (self.video, self.processor):
            if target is not None:
                target.logger.log(rec)

    def do_toggle_analog(self):
        with self.lock:  # serialize toggles — no double-start race
            if self.analog_engaged:
                self._release_analog("user disengaged")
                note = self._note("Analog control disengaged. Drone centered (hover).")
            elif not self.connected:
                note = self._note("Ignored 'ANALOG CONTROL' — not connected. Click Connect first.")
            elif self.holding:
                note = self._note("Ignored 'ANALOG CONTROL' — Hold Position is engaged. Release it "
                                  "first (both drive the drone; they can't run at once).")
            else:
                srv = AnalogInputServer(self.drone, logger=self._analog_log,
                                        port=ANALOG_PORT, invert=self.analog_invert)
                try:
                    srv.start()
                    self.analog_server = srv
                    note = self._note(f"Analog control engaged — joystick streaming to {srv.url}. "
                                      "Drag a pad to fly; release to recenter.")
                except OSError as e:
                    note = self._note(f"Analog control failed to start: {e} (port may still be releasing)")
            return (note,) + self._ui_sync()

    def analog_set_invert(self, axis, value):
        self.analog_invert[axis] = bool(value)
        self._analog_log("analog_invert", axis=axis, value=bool(value))
        return self._note(f"analog invert {axis}={bool(value)}")

    def _log_param(self, name, value):
        """Record a live tuning change into every active session log, so the JSONL
        fully explains later corrections — not just the transient UI note list."""
        for p in (self.processor, self.demo_processor):
            if p is not None:
                p.logger.log({"event": "param_change", "param": name, "value": value})

    def flow_set_roi(self, v):       self.stabilizer.roi_fraction = float(v);    self._log_param("roi_fraction", float(v));    return self._note(f"roi_fraction={v}")
    def flow_set_deadband(self, v):  self.stabilizer.deadband_px = float(v);     self._log_param("deadband_px", float(v));     return self._note(f"deadband_px={v}")
    def flow_set_flip_roll(self, v): self.stabilizer.flip_roll = bool(v);        self._log_param("flip_roll", bool(v));        return self._note(f"flip_roll={v}")
    def flow_set_flip_thr(self, v):  self.stabilizer.flip_throttle = bool(v);    self._log_param("flip_throttle", bool(v));    return self._note(f"flip_throttle={v}")

    def flow_set_roll_enabled(self, v):
        if self.processor is not None:
            self.processor.roll_enabled = bool(v)
        self._log_param("roll_enabled", bool(v))
        return self._note(f"roll_enabled={v}")

    def flow_set_throttle_enabled(self, v):
        if self.processor is not None:
            self.processor.throttle_enabled = bool(v)
        self._log_param("throttle_enabled", bool(v))
        return self._note(f"throttle_enabled={v}")

    def flow_set_hold_threshold(self, v):
        if self.processor is not None:
            self.processor.hold_threshold_px = float(v)
        self._log_param("hold_threshold_px", float(v))
        return self._note(f"hold_threshold_px={v}")

    def flow_set_hold_settle(self, v):
        if self.processor is not None:
            self.processor.hold_settle_s = float(v)
        self._log_param("hold_settle_s", float(v))
        return self._note(f"hold_settle_s={v}")

    # ── Diagnostics: live numbers, plot, corrections table ──────────────────
    def poll_hold_stats(self):
        if self.processor is None:
            return "Hold Position: not running."
        c = self.processor.last_corr
        return (f"frames={self.processor.frames_processed}  compute={c.compute_ms:.2f}ms  "
                f"src_fps={self.processor.source.fps:.1f}\n"
                f"raw  dx={c.raw_dx:+.2f} dy={c.raw_dy:+.2f}\n"
                f"flow dx={c.flow_dx:+.2f} dy={c.flow_dy:+.2f}  valid={c.valid}  coherent={c.coherent}\n"
                f"cumulative displacement  dx={c.cum_dx:+.1f}px  dy={c.cum_dy:+.1f}px  "
                f"(threshold={self.processor.hold_threshold_px:.0f}px)")

    def poll_hold_plot(self):
        if self.processor is None:
            return _EMPTY_PLOT
        pts = self.processor.plot_snapshot()
        if not pts:
            return _EMPTY_PLOT
        t0 = pts[0][0]
        rows = []
        for (t, dx, dy, cum_dx, cum_dy) in pts:
            rel = t - t0
            rows.append((rel, dx, "flow dx"))
            rows.append((rel, dy, "flow dy"))
            rows.append((rel, cum_dx, "cum dx"))
            rows.append((rel, cum_dy, "cum dy"))
        return pd.DataFrame(rows, columns=["t", "value", "series"])

    def poll_event_log(self):
        return "\n".join(self._events[-16:]) if self._events else "No events yet."

    # ════════════════════════════════════════════════════════════════════════
    # Manual flight controls (one-shot commands — serialized; e-stop bypasses)
    # ════════════════════════════════════════════════════════════════════════
    # serialize=True (the timed directional moves only) takes _cmd_lock so
    # overlapping moves can't clobber each other's axes; a second move while one
    # runs is a "busy" no-op. Everything else runs immediately.
    def _guarded(self, fn, label, *args, serialize=False, **kwargs):
        if not self.connected:
            reason = f" (lost connection: {self.drone.last_error})" if self.drone.last_error else ""
            return self._note(f"Ignored '{label}' — not connected{reason}. Click Connect first.")
        if serialize and not self._cmd_lock.acquire(blocking=False):
            return self._note(f"Busy — another move is running. Ignored '{label}'.")
        try:
            self.drone.set_idle_mode(False)  # any real flight command wakes the drone up
            fn(*args, **kwargs)
        finally:
            if serialize:
                self._cmd_lock.release()
        return self._note(label)

    def do_takeoff(self):    return self._guarded(self.drone.takeoff, "Takeoff")
    def do_land(self):       return self._guarded(self.drone.land, "Land")
    def do_hover(self):      return self._guarded(self.drone.hover, "Hover / center sticks")
    def do_calibrate(self):  return self._guarded(self.drone.calibrate, "Calibrate gyro (keep drone flat)")

    def do_stop(self):
        # EMERGENCY STOP must never be blocked by the command lock (instant
        # cmd=CMD_STOP set, no sleep) — bypass serialization.
        if not self.connected:
            return self._note("Ignored 'EMERGENCY STOP' — not connected.")
        self.drone.set_idle_mode(False)
        self.drone.stop()
        return self._note("EMERGENCY STOP")

    def do_up(self, duration):         return self._guarded(self.drone.up, f"Up ({duration}s)", duration=duration, serialize=True)
    def do_down(self, duration):       return self._guarded(self.drone.down, f"Down ({duration}s)", duration=duration, serialize=True)
    def do_forward(self, duration):    return self._guarded(self.drone.forward, f"Forward ({duration}s)", duration=duration, serialize=True)
    def do_backward(self, duration):   return self._guarded(self.drone.backward, f"Backward ({duration}s)", duration=duration, serialize=True)
    def do_move_left(self, duration):  return self._guarded(self.drone.move_left, f"Move left ({duration}s)", duration=duration, serialize=True)
    def do_move_right(self, duration): return self._guarded(self.drone.move_right, f"Move right ({duration}s)", duration=duration, serialize=True)
    def do_turn_left(self, duration):  return self._guarded(self.drone.turn_left, f"Turn left ({duration}s)", duration=duration, serialize=True)
    def do_turn_right(self, duration): return self._guarded(self.drone.turn_right, f"Turn right ({duration}s)", duration=duration, serialize=True)

    def do_set_axes(self, roll, pitch, throttle, yaw):
        # Explicit absolute stick position — latest-wins, always applied (no lock).
        if not self.connected:
            return self._note("Ignored manual axis update — not connected.")
        self.drone.set_idle_mode(False)
        self.drone.set_controls(roll=int(roll), pitch=int(pitch), throttle=int(throttle), yaw=int(yaw))
        return self._note(f"Manual axes -> roll={roll} pitch={pitch} throttle={throttle} yaw={yaw}")

    def do_center_axes(self):
        # Don't snap the sliders when disconnected — the hover no-ops, so the UI
        # would otherwise drift out of sync with the backend.
        if not self.connected:
            return gr.update(), gr.update(), gr.update(), gr.update(), self.do_hover()
        return NEUTRAL, NEUTRAL, NEUTRAL, NEUTRAL, self.do_hover()

    # ════════════════════════════════════════════════════════════════════════
    # Optical Flow tab — minimal webcam-only demo/debug sandbox (no drone)
    # ════════════════════════════════════════════════════════════════════════
    def _demo_views_html(self):
        if self.demo_processor is not None:
            arrow = f'<img src="{self.demo_processor.arrow_url}" style="width:100%;border-radius:8px;">'
            hsv = f'<img src="{self.demo_processor.hsv_url}" style="width:100%;border-radius:8px;">'
            return arrow, hsv
        return (_demo_placeholder("🡒 Motion Arrows — demo stopped"),
                _demo_placeholder("🌈 Dense Field (HSV) — demo stopped"))

    def _demo_button(self):
        if self.demo_processor is not None:
            return gr.update(value="⏹ STOP DEMO", variant="stop")
        return gr.update(value="▶ START DEMO (webcam)")

    def do_toggle_demo(self, webcam_idx):
        with self.lock:  # serialize toggles — no double-start race
            if self.demo_processor is not None:
                self.demo_processor.stop()
                self.demo_processor = None
                note = self._note("Webcam demo stopped.")
            elif self.holding:
                note = self._note("Ignored 'START DEMO' — Hold Position is engaged. Release "
                                  "Hold Position first (shared stabilizer can't drive two live "
                                  "sources at once).")
            else:
                src = WebcamFrameSource(index=int(webcam_idx), width=640, height=360)
                proc = FlowProcessor(src, self.stabilizer, drone=None, logger=FlowLogger(),
                                     arrow_port=DEMO_ARROW_PORT, hsv_port=DEMO_HSV_PORT)
                try:
                    proc.start()
                except OSError as e:
                    note = self._note(f"Demo failed to bind stream servers: {e}")
                    return (note,) + self._demo_views_html() + (self._demo_button(),)
                if getattr(src, "open_error", None):
                    proc.stop()
                    note = self._note(f"Webcam error: {src.open_error}")
                    return (note,) + self._demo_views_html() + (self._demo_button(),)
                self.demo_processor = proc
                note = self._note(f"Webcam demo started (index {int(webcam_idx)}).")
            return (note,) + self._demo_views_html() + (self._demo_button(),)

    def do_reset_origin(self):
        """Mark the current view as 'home' — zeroes the cumulative drift estimate
        without interrupting flow tracking. Shared stabilizer, so this affects
        whichever of Hold Position / Demo is currently running."""
        self.stabilizer.reset_origin()
        for p in (self.processor, self.demo_processor):
            if p is not None:
                p.logger.log({"event": "origin_reset"})
        return self._note("Origin reset — current view marked as home (cumulative drift zeroed).")

    def poll_demo_drift(self):
        if self.demo_processor is None:
            return "cumulative drift: demo not running"
        c = self.demo_processor.last_corr
        return (f"cumulative drift since origin:  dx={c.cum_dx:+.1f}px  dy={c.cum_dy:+.1f}px  "
                f"(|d|={(c.cum_dx**2 + c.cum_dy**2)**0.5:.1f}px)")

    def shutdown(self):
        """Tear the whole tree down so nothing survives process exit: Hold
        Position loop, video + every MJPEG server, the webcam demo, and the
        drone control thread — then release the log-file handles. Idempotent."""
        log.info("Shutting down — tearing down all threads/servers.")
        try:
            self._teardown_connection("app shutdown")  # cascades hold -> video -> drone
        except Exception as e:  # noqa: BLE001 — never let cleanup raise on the way out
            log.error("teardown_connection during shutdown failed: %s", e)
        if self.demo_processor is not None:
            try:
                self.demo_processor.stop()
            except Exception as e:  # noqa: BLE001
                log.error("demo_processor.stop during shutdown failed: %s", e)
            self.demo_processor = None


ctrl = DroneController()


# ── module-level placeholder helpers (used by both layout defaults + methods) ─
def _hold_placeholder(label: str) -> str:
    return f'<div class="flow-box">{label}</div>'


def _demo_placeholder(label: str) -> str:
    return f'<div class="flow-box">{label}</div>'


# ── handler wrappers: append flight/D-pad button interactivity to the toggle
# and poll-tick render tuples (keeps the controller methods' return shapes
# stable and independently testable).
def h_toggle_connect(width, height, color):
    return ctrl.do_toggle_connect(width, height, color) + ctrl.button_states(N_GATED)


def h_toggle_video(width, height, color):
    return ctrl.do_toggle_video(width, height, color) + ctrl.button_states(N_GATED)


def h_toggle_hold():
    return ctrl.do_toggle_hold() + ctrl.button_states(N_GATED)


def h_toggle_analog():
    return ctrl.do_toggle_analog() + ctrl.button_states(N_GATED)


def h_poll_sync():
    return ctrl.poll_sync() + ctrl.button_states(N_GATED)


# ════════════════════════════════════════════════════════════════════════════
# Styling — matches the original DM002HW_controller_tests/gradio_app.py: default
# light Gradio theme, a few id-scoped colored buttons, dashed placeholder boxes.
# ════════════════════════════════════════════════════════════════════════════

CSS = """
.dpad-btn {min-width: 3.2em !important;}
.dpad-spacer {visibility: hidden !important;}  /* keeps the D-pad cross aligned, no visible pill */
.big-btn button {font-size: 1.1em !important; font-weight: 600;}
/* This Gradio puts elem_id on the <button> itself; cover both DOM shapes. */
#stop-btn, #stop-btn button {background: #c0392b !important; color: white !important;}
#takeoff-btn, #takeoff-btn button {background: #27ae60 !important; color: white !important;}
#land-btn, #land-btn button {background: #e67e22 !important; color: white !important;}
#video-box, .flow-box {min-height: 260px; display:flex; align-items:center;
            justify-content:center; text-align:center; padding:0.5em;
            border: 2px dashed var(--border-color-primary); border-radius: 8px;}
#analog-btn, #analog-btn button {background:#8e44ad !important; color:white !important;}
.analog-pad {width: 170px; height: 170px; margin: 4px auto; border: 2px solid var(--border-color-primary);
            border-radius: 12px; display:flex; align-items:center; justify-content:center;
            text-align:center; user-select:none; touch-action:none; cursor: grab;
            background: var(--background-fill-secondary); font-size:0.9em;}
.analog-pad.analog-active {cursor: grabbing; border-color:#8e44ad; box-shadow:0 0 0 2px #8e44ad55 inset;}
"""

# Page-load JS: two virtual-joystick pads that stream held stick positions to the
# AnalogInputServer (analog_control.py) at ~20Hz. Runs entirely client-side; when
# analog mode is off the server isn't listening and fetch() fails silently. A
# <script> inside gr.HTML would NOT execute (Svelte {@html}), so this is injected
# via demo.load(js=...) and polls until the pads are in the DOM.
ANALOG_JS = """
() => {
  const PORT = 8095, NEUTRAL = 128;
  const state = { roll: NEUTRAL, pitch: NEUTRAL, throttle: NEUTRAL, yaw: NEUTRAL };
  const toByte = (n) => Math.max(0, Math.min(255, Math.round(NEUTRAL + n * 127)));
  function attachPad(el, xAxis, yAxis) {
    let active = false;
    const setFrom = (e) => {
      const r = el.getBoundingClientRect();
      let nx = Math.max(-1, Math.min(1, ((e.clientX - r.left) / r.width  - 0.5) * 2));
      let ny = Math.max(-1, Math.min(1, ((e.clientY - r.top)  / r.height - 0.5) * 2));
      state[xAxis] = toByte(nx);
      state[yAxis] = toByte(-ny);          // top of pad = positive (up / forward)
    };
    const recenter = () => { state[xAxis] = NEUTRAL; state[yAxis] = NEUTRAL; };
    el.addEventListener('pointerdown', (e) => { active = true; try { el.setPointerCapture(e.pointerId); } catch (_) {} setFrom(e); el.classList.add('analog-active'); });
    el.addEventListener('pointermove', (e) => { if (active) setFrom(e); });
    const end = () => { active = false; recenter(); el.classList.remove('analog-active'); };
    el.addEventListener('pointerup', end);
    el.addEventListener('pointercancel', end);
  }
  function init() {
    const left = document.getElementById('analog-left');
    const right = document.getElementById('analog-right');
    if (!left || !right) return false;
    if (left.dataset.analogInit) return true;
    left.dataset.analogInit = '1';
    attachPad(left,  'yaw',  'throttle');   // left stick: throttle (Y) / yaw (X)
    attachPad(right, 'roll', 'pitch');      // right stick: pitch (Y) / roll (X)
    setInterval(() => {
      const qs = 'roll=' + state.roll + '&pitch=' + state.pitch +
                 '&throttle=' + state.throttle + '&yaw=' + state.yaw;
      fetch('http://127.0.0.1:' + PORT + '/set?' + qs).catch(() => {});
    }, 50);
    return true;
  }
  const iv = setInterval(() => { if (init()) clearInterval(iv); }, 300);
}
"""

# ════════════════════════════════════════════════════════════════════════════
# Layout
# ════════════════════════════════════════════════════════════════════════════

with gr.Blocks(title="DM002HW — Controller") as demo:
    with gr.Tabs():
        with gr.Tab("🕹 Controller"):
            gr.Markdown(
                f"# DM002HW Drone Controller\n"
                f"`pid={os.getpid()}` — only one process should ever control the drone at a time."
            )
            status_badge = gr.HTML(ctrl.status_badge_html())

            with gr.Row():
                # ── video + optical-flow hold position ──────────────────────
                with gr.Column(scale=5):
                    video_feed = gr.HTML(ctrl._raw_video_html())
                    with gr.Row():
                        start_video_btn = gr.Button("🎥 Start video")
                    with gr.Row():
                        vid_w = gr.Number(value=640, label="Width", precision=0)
                        vid_h = gr.Number(value=360, label="Height", precision=0)
                        vid_color = gr.Checkbox(value=True, label="Color (uncheck if garbled)")
                    video_stats = gr.Textbox(label="Video stats", interactive=False)

                    gr.Markdown(
                        "### Hold Position — optical-flow hover\n"
                        "The DM002HW has no drift/heading stability of its own — this loop "
                        "**is** the hover mechanism. Start video first."
                    )
                    with gr.Row():
                        hold_arrow_view = gr.HTML(_hold_placeholder("🡒 Motion Arrows"), elem_id="hold-arrow")
                        hold_hsv_view = gr.HTML(_hold_placeholder("🌈 Dense Field (HSV)"), elem_id="hold-hsv")
                    hold_btn = gr.Button("🎯 HOLD POSITION", elem_id="hold-btn")
                    with gr.Accordion("Hold Position — tuning", open=False):
                        with gr.Row():
                            roi_s = gr.Slider(0.2, 1.0, value=ctrl.stabilizer.roi_fraction, step=0.05, label="ROI fraction")
                            dead_s = gr.Slider(0.0, 3.0, value=ctrl.stabilizer.deadband_px, step=0.1, label="Deadband px")
                        with gr.Row():
                            flip_roll_c = gr.Checkbox(value=False, label="Flip roll sign")
                            flip_thr_c = gr.Checkbox(value=False, label="Flip throttle sign")
                        gr.Markdown("**Displacement Hold** — fires one fixed nudge when accumulated "
                                   "drift crosses the threshold, then holds neutral for a settle "
                                   "window before checking again.")
                        with gr.Row():
                            roll_en_c = gr.Checkbox(value=True, label="Roll correction enabled")
                            thr_en_c = gr.Checkbox(value=False, label="Throttle correction enabled")
                        hold_threshold_s = gr.Slider(2, 40, value=10, step=1,
                                                     label="Displacement Hold: threshold (px)")
                        hold_settle_s_slider = gr.Slider(0.5, 10, value=2.0, step=0.5,
                                                         label="Displacement Hold: settle time (s, neutral between fires)")

                    status = gr.Textbox(label="Status log", lines=10, interactive=False)

                # ── manual control panel ────────────────────────────────────
                with gr.Column(scale=7):
                    with gr.Row():
                        connect_btn = gr.Button("🔌 Connect")
                        duration = gr.Slider(0.2, 3.0, value=1.0, step=0.1, label="Move duration (s)")

                    with gr.Row(elem_classes="big-btn"):
                        takeoff_btn = gr.Button("🛫 Takeoff", elem_id="takeoff-btn")
                        hover_btn = gr.Button("🖐 HOVER (center)")
                        land_btn = gr.Button("🛬 Land", elem_id="land-btn")
                        calibrate_btn = gr.Button("🧭 Calibrate gyro")
                        stop_btn = gr.Button("⛔ EMERGENCY STOP", elem_id="stop-btn")

                    gr.Markdown("### Gimbals")
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("**Left stick — Throttle / Yaw**")
                            with gr.Row():
                                gr.Button("", elem_classes=["dpad-btn", "dpad-spacer"])
                                up_btn = gr.Button("▲ Up", elem_classes="dpad-btn")
                                gr.Button("", elem_classes=["dpad-btn", "dpad-spacer"])
                            with gr.Row():
                                yaw_l_btn = gr.Button("↺ Yaw L", elem_classes="dpad-btn")
                                hover_btn2 = gr.Button("● Hover", elem_classes="dpad-btn")
                                yaw_r_btn = gr.Button("↻ Yaw R", elem_classes="dpad-btn")
                            with gr.Row():
                                gr.Button("", elem_classes=["dpad-btn", "dpad-spacer"])
                                down_btn = gr.Button("▼ Down", elem_classes="dpad-btn")
                                gr.Button("", elem_classes=["dpad-btn", "dpad-spacer"])

                        with gr.Column():
                            gr.Markdown("**Right stick — Pitch / Roll**")
                            with gr.Row():
                                gr.Button("", elem_classes=["dpad-btn", "dpad-spacer"])
                                fwd_btn = gr.Button("▲ Fwd", elem_classes="dpad-btn")
                                gr.Button("", elem_classes=["dpad-btn", "dpad-spacer"])
                            with gr.Row():
                                left_btn = gr.Button("◀ Left", elem_classes="dpad-btn")
                                hover_btn3 = gr.Button("● Hover", elem_classes="dpad-btn")
                                right_btn = gr.Button("▶ Right", elem_classes="dpad-btn")
                            with gr.Row():
                                gr.Button("", elem_classes=["dpad-btn", "dpad-spacer"])
                                back_btn = gr.Button("▼ Back", elem_classes="dpad-btn")
                                gr.Button("", elem_classes=["dpad-btn", "dpad-spacer"])

                    gr.Markdown("### 🎮 Analog control — smooth / gyro-style")
                    gr.Markdown(
                        "Drag a pad for continuous fine stick values — this is exactly what the "
                        "phone's *gyro mode* does (fine analog values streamed at ~20 Hz), just "
                        "with a joystick instead of tilt. Release a pad to recenter it. Engage "
                        "below; Hold Position must be off (both drive the drone)."
                    )
                    analog_btn = gr.Button("🎮 ANALOG CONTROL (smooth)", elem_id="analog-btn")
                    with gr.Row():
                        analog_inv_roll = gr.Checkbox(value=False, label="Invert roll")
                        analog_inv_pitch = gr.Checkbox(value=False, label="Invert pitch")
                        analog_inv_thr = gr.Checkbox(value=False, label="Invert throttle")
                        analog_inv_yaw = gr.Checkbox(value=False, label="Invert yaw")
                    with gr.Row():
                        gr.HTML('<div id="analog-left" class="analog-pad">Throttle ▲▼<br>Yaw ◀▶<br><small>(drag)</small></div>')
                        gr.HTML('<div id="analog-right" class="analog-pad">Pitch ▲▼<br>Roll ◀▶<br><small>(drag)</small></div>')

                    with gr.Accordion("Advanced — raw axis (manual trim)", open=False):
                        gr.Markdown(
                            "Set roll/pitch/throttle/yaw directly (0-255, neutral=128). "
                            "The background loop keeps re-sending whatever value you leave these at."
                        )
                        with gr.Row():
                            roll_s = gr.Slider(0, 255, value=NEUTRAL, step=1, label="Roll")
                            pitch_s = gr.Slider(0, 255, value=NEUTRAL, step=1, label="Pitch")
                            throttle_s = gr.Slider(0, 255, value=NEUTRAL, step=1, label="Throttle")
                            yaw_s = gr.Slider(0, 255, value=NEUTRAL, step=1, label="Yaw")
                        center_btn = gr.Button("Center all axes (hover)")

                    with gr.Accordion("Diagnostics", open=False):
                        hold_stats = gr.Textbox(label="Hold Position — live numbers", lines=4,
                                                interactive=False)
                        hold_plot = gr.LinePlot(_EMPTY_PLOT, x="t", y="value", color="series",
                                                title="drift & corrections over time", height=240,
                                                x_title="seconds")

        with gr.Tab("🌊 Optical Flow — Demo"):
            gr.Markdown(
                "### Webcam-only demo / debug sandbox\n"
                "Tune deadband / ROI / signs against your webcam before trusting them on the "
                "real drone — this shares the same tuning as Hold Position on the Controller tab. "
                "No drone or video code lives here.\n\n"
                "**Position estimate (red marker)** — the cyan arrow only shows *velocity* and "
                "vanishes the instant motion stops. The red cross+dot is the cumulative "
                "**position**: it stays wherever you've drifted to since the last 'Mark home' "
                "click, even while completely still. This is the estimator step — no corrections "
                "are sent from this tab."
            )
            demo_webcam_idx = gr.Number(value=0, label="Webcam index", precision=0)
            with gr.Row():
                demo_btn = gr.Button("▶ START DEMO (webcam)")
                demo_reset_btn = gr.Button("🏠 Mark home (reset origin)")
            with gr.Row():
                demo_arrow_view = gr.HTML(_demo_placeholder("🡒 Motion Arrows — demo stopped"))
                demo_hsv_view = gr.HTML(_demo_placeholder("🌈 Dense Field (HSV) — demo stopped"))
            demo_drift = gr.Textbox(label="Cumulative drift (position estimate)", interactive=False)
            demo_status = gr.Textbox(label="Demo status", lines=3, interactive=False)

    # Flight/D-pad buttons gated on connection (start-video & hold are gated via
    # their own value-updates in _video_button/_hold_button).
    FLIGHT_GATED_BUTTONS = [
        takeoff_btn, land_btn, calibrate_btn, stop_btn,
        hover_btn, hover_btn2, hover_btn3,
        up_btn, down_btn, fwd_btn, back_btn, left_btn, right_btn,
        yaw_l_btn, yaw_r_btn, center_btn,
    ]
    N_GATED = len(FLIGHT_GATED_BUTTONS)

    # ── wiring ────────────────────────────────────────────────────────────
    connect_btn.click(h_toggle_connect, inputs=[vid_w, vid_h, vid_color],
                      outputs=[status, video_feed, connect_btn, start_video_btn, hold_btn, analog_btn, status_badge] + FLIGHT_GATED_BUTTONS)
    start_video_btn.click(h_toggle_video, inputs=[vid_w, vid_h, vid_color],
                          outputs=[status, video_feed, connect_btn, start_video_btn, hold_btn, analog_btn, status_badge] + FLIGHT_GATED_BUTTONS)
    hold_btn.click(h_toggle_hold, outputs=[status, hold_arrow_view, hold_hsv_view,
                                           connect_btn, start_video_btn, hold_btn, analog_btn, status_badge] + FLIGHT_GATED_BUTTONS)
    analog_btn.click(h_toggle_analog, outputs=[status, connect_btn, start_video_btn, hold_btn,
                                               analog_btn, status_badge] + FLIGHT_GATED_BUTTONS)
    analog_inv_roll.change(lambda v: ctrl.analog_set_invert("roll", v), analog_inv_roll, status)
    analog_inv_pitch.change(lambda v: ctrl.analog_set_invert("pitch", v), analog_inv_pitch, status)
    analog_inv_thr.change(lambda v: ctrl.analog_set_invert("throttle", v), analog_inv_thr, status)
    analog_inv_yaw.change(lambda v: ctrl.analog_set_invert("yaw", v), analog_inv_yaw, status)

    takeoff_btn.click(ctrl.do_takeoff, outputs=status)
    land_btn.click(ctrl.do_land, outputs=status)
    calibrate_btn.click(ctrl.do_calibrate, outputs=status)
    stop_btn.click(ctrl.do_stop, outputs=status)
    for b in (hover_btn, hover_btn2, hover_btn3):
        b.click(ctrl.do_hover, outputs=status)

    up_btn.click(ctrl.do_up, inputs=duration, outputs=status)
    down_btn.click(ctrl.do_down, inputs=duration, outputs=status)
    fwd_btn.click(ctrl.do_forward, inputs=duration, outputs=status)
    back_btn.click(ctrl.do_backward, inputs=duration, outputs=status)
    left_btn.click(ctrl.do_move_left, inputs=duration, outputs=status)
    right_btn.click(ctrl.do_move_right, inputs=duration, outputs=status)
    yaw_l_btn.click(ctrl.do_turn_left, inputs=duration, outputs=status)
    yaw_r_btn.click(ctrl.do_turn_right, inputs=duration, outputs=status)

    for s in (roll_s, pitch_s, throttle_s, yaw_s):
        s.release(ctrl.do_set_axes, inputs=[roll_s, pitch_s, throttle_s, yaw_s], outputs=status)
    center_btn.click(ctrl.do_center_axes, outputs=[roll_s, pitch_s, throttle_s, yaw_s, status])

    roi_s.release(ctrl.flow_set_roi, roi_s, status)
    dead_s.release(ctrl.flow_set_deadband, dead_s, status)
    flip_roll_c.change(ctrl.flow_set_flip_roll, flip_roll_c, status)
    flip_thr_c.change(ctrl.flow_set_flip_thr, flip_thr_c, status)
    roll_en_c.change(ctrl.flow_set_roll_enabled, roll_en_c, status)
    thr_en_c.change(ctrl.flow_set_throttle_enabled, thr_en_c, status)
    hold_threshold_s.release(ctrl.flow_set_hold_threshold, hold_threshold_s, status)
    hold_settle_s_slider.release(ctrl.flow_set_hold_settle, hold_settle_s_slider, status)

    demo_btn.click(ctrl.do_toggle_demo, inputs=demo_webcam_idx,
                   outputs=[demo_status, demo_arrow_view, demo_hsv_view, demo_btn])
    demo_reset_btn.click(ctrl.do_reset_origin, outputs=demo_status)

    timer = gr.Timer(0.4)
    timer.tick(ctrl.poll_video_stats, outputs=video_stats)
    timer.tick(ctrl.poll_hold_stats, outputs=hold_stats)
    timer.tick(ctrl.poll_hold_plot, outputs=hold_plot)
    timer.tick(ctrl.poll_event_log, outputs=status)
    timer.tick(ctrl.poll_demo_drift, outputs=demo_drift)
    timer.tick(h_poll_sync,
               outputs=[video_feed, hold_arrow_view, hold_hsv_view,
                        connect_btn, start_video_btn, hold_btn, analog_btn, status_badge] + FLIGHT_GATED_BUTTONS)

    # Page-load: wire up the two analog joystick pads (streams to AnalogInputServer).
    demo.load(js=ANALOG_JS)

demo.queue()


def shutdown(*_args):
    """Module-level backstop that tears down the singleton controller and then
    releases the log-file handles (a held handle blocks deleting app_logs/ on
    Windows). Idempotent; safe to call more than once."""
    ctrl.shutdown()
    close_logging()


# Backstop: runs on any normal interpreter exit even if launch() returns oddly.
atexit.register(shutdown)


def _signal_shutdown(signum, _frame):
    shutdown()
    raise SystemExit(0)


if __name__ == "__main__":
    # SIGTERM isn't delivered as KeyboardInterrupt; handle it explicitly so a
    # kill (or IDE stop) still tears everything down. SIGINT (Ctrl-C) surfaces
    # as KeyboardInterrupt and is handled by the try/finally below.
    try:
        signal.signal(signal.SIGTERM, _signal_shutdown)
    except (ValueError, AttributeError, OSError):
        pass  # not on main thread / not supported on this platform
    try:
        demo.launch(css=CSS)
    finally:
        shutdown()
