"""
Gradio front-end for the DM002HW drone controller (drone.py).

Two "gimbal" D-pads (Throttle/Yaw on the left, Pitch/Roll on the right) mimic a
real transmitter, plus a raw-axis panel for direct manual control. The video
panel pulls the drone's UDP JPEG-fragment stream (port 1234) via
video_stream.VideoReceiver — reconstruction was verified offline against real
captured fragments in wireshark_1.pcapng (see PROTOCOL_NOTES.md) but has not
been flight-tested live; default resolution (640x360) is a best guess and
adjustable in the Video panel if frames come out skewed.

All mutable app state (drone connection, video receiver, MJPEG server, event
log) is owned by a single DroneController instance — the single source of truth
every handler AND the poll timer render from, so the UI never drifts out of sync
with the backend. Concretely, this is what makes Disconnect (and a lost-link
wifi drop) clear the video pane instead of leaving the last frame frozen, and
what disables flight buttons whenever we're not connected.

Run with:  python gradio_app.py
Then open the printed local URL in a browser.
"""

import os
import threading
import time
import warnings

# Gradio 6 / Starlette combo emits this on every queued request (so once per
# gr.Timer tick, i.e. constantly) — cosmetic version-skew noise, not a bug.
warnings.filterwarnings("ignore", message=".*HTTP_422_UNPROCESSABLE_ENTITY.*")

import gradio as gr

from applog import get_logger
from drone import Drone, NEUTRAL
from video_stream import VideoReceiver, MjpegServer
from analog_control import AnalogInputServer

log = get_logger("gradio_app")

ANALOG_PORT = 8095

# Video hard-stall auto-recovery (see DroneController._check_video_stall). The
# DM002HW can stop sending video entirely during aggressive flight; the in-place
# watchdog kicks don't recover that, so we restart the receiver to force a fresh
# handshake — but only after this many seconds with no decoded frame, and at most
# once per STALL_RESTART_EVERY so we never thrash while the drone refuses to stream.
STALL_RESTART_AFTER = 4.0
STALL_RESTART_EVERY = 6.0

_BADGE_CSS = "display:inline-block; padding:0.35em 0.9em; border-radius:999px; font-weight:700; margin-right:0.5em;"


def _badge(text: str, color: str) -> str:
    return f'<span style="{_BADGE_CSS} background:{color}; color:white;">{text}</span>'


# ════════════════════════════════════════════════════════════════════════════
# DroneController — owns ALL live state; the single source of truth the UI
# renders from. Every toggle handler and the poll timer call the same render
# helpers, so the displayed state can never diverge from the backend.
# ════════════════════════════════════════════════════════════════════════════

class DroneController:
    def __init__(self):
        self.drone = Drone()
        self.video: VideoReceiver | None = None
        self.mjpeg: MjpegServer | None = None
        self.analog_server: AnalogInputServer | None = None  # smooth "gyro-style" analog control
        # Per-axis invert flags for analog mode — shared by reference with the
        # running server, so an invert checkbox takes effect without a restart.
        self.analog_invert = {"roll": False, "pitch": False, "throttle": False, "yaw": False}
        self._events: list[str] = []
        # Serializes connect/disconnect/start-video/stop-video toggles so a
        # rapid double-click can't race two connects/videos and orphan a
        # socket, thread, or bound port.
        self.lock = threading.Lock()
        # Non-reentrant lock that serializes ONLY the timed DIRECTIONAL MOVES
        # (up/down/fwd/back/left/right/yaw). A verified concern: firing two of
        # those at once makes their set_controls()/sleep/hover sequences
        # interleave and clobber each other's axes (set_controls resets
        # unspecified axes to neutral), producing a self-fighting trajectory a
        # single physical transmitter never would. The 124-byte @ ~18Hz wire
        # pattern is unaffected (drone.py._loop), but the axis VALUES would
        # deviate from the real app — so overlapping moves get a "busy" no-op.
        # Stop/settle commands (Hover, Land, E-stop) and one-shot state-sets
        # deliberately do NOT serialize, so "make it stop/settle" is never
        # blockable and their UI never diverges from the drone.
        self._cmd_lock = threading.Lock()
        # Anti-churn bookkeeping so the 0.4s timer doesn't re-emit identical
        # HTML / interactive updates every tick (which would remount the MJPEG
        # <img> and spam the websocket). Only real transitions emit a value.
        self._video_feed_state: str | None = None  # None | "live" | "placeholder"
        self._gated_state: bool | None = None       # last-emitted button-enabled state
        # Auto-recovery for a hard-stalled video stream (see _check_video_stall):
        # remember the last-used video geometry to restart with, and throttle
        # restart attempts so we don't thrash while the drone refuses to stream.
        self._video_wh = (640, 360, True)
        self._last_video_restart_ts = 0.0
        # Bumped on each auto-restart so raw_video_html emits a *different* <img>
        # string, forcing the browser to drop the dead MJPEG connection and
        # reconnect to the fresh server (a browser won't auto-reconnect a
        # broken multipart <img> stream on its own).
        self._video_epoch = 0

    # ── state queries ───────────────────────────────────────────────────────
    @property
    def connected(self) -> bool:
        return self.drone._armed

    @property
    def video_running(self) -> bool:
        return self.video is not None

    @property
    def analog_engaged(self) -> bool:
        return self.analog_server is not None

    def _note(self, msg: str) -> str:
        log.info(msg)
        ts = time.strftime("%H:%M:%S")
        self._events.append(f"[{ts}] {msg}")
        return "\n".join(self._events[-12:])

    # ── cascade teardown — each level tears down everything beneath it ───────
    def _teardown_video(self, reason: str):
        if self.mjpeg is not None:
            self.mjpeg.stop()
            self.mjpeg = None
        if self.video is not None:
            self.video.stop()
            self.video = None
        self.drone.set_idle_mode(False)

    def _release_analog(self, reason: str):
        if self.analog_server is not None:
            self.analog_server.stop()   # also centers the drone (hover) on the way out
            self.analog_server = None
            log.info("Analog control released (%s)", reason)

    def _teardown_connection(self, reason: str):
        self._release_analog(reason)  # analog drives the drone directly — can't outlive the link
        self._teardown_video(reason)  # video can't outlive the connection
        if self.drone._armed:
            self.drone.disconnect()

    # ── rendering (the single source of truth) ──────────────────────────────
    def raw_video_html(self) -> str:
        if self.mjpeg is not None:
            # data-epoch makes the HTML string differ after an auto-restart, so
            # Gradio replaces the <img> element and the browser reconnects.
            return (f'<img data-epoch="{self._video_epoch}" src="{self.mjpeg.url}" '
                    f'style="width:100%; border-radius:8px;">')
        return "<div id='video-box'>🎥 Video feed not connected yet</div>"

    def render_video_feed(self, force: bool = False):
        """Return video-pane HTML, but only when the live↔placeholder state (or the
        restart epoch) actually changes — or force=True for click paths. Every
        steady-state timer tick returns gr.update() so the MJPEG <img> isn't
        remounted; a bumped epoch (auto-restart) forces a fresh <img> reconnect."""
        desired = ("live", self._video_epoch) if self.mjpeg is not None else ("placeholder", 0)
        changed = desired != self._video_feed_state
        self._video_feed_state = desired
        if not changed and not force:
            return gr.update()
        return self.raw_video_html()

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
            if age < 2.0:
                vid = _badge(f"🟢 VIDEO LIVE ({self.video.frames_ok} frames)", "#27ae60")
            else:
                vid = _badge(f"🟡 VIDEO STALLED — last frame {age:.1f}s ago", "#e67e22")

        analog = _badge("🎮 ANALOG ON", "#8e44ad") if self.analog_engaged else ""

        return f'<div style="font-size:1.05em;">{conn}{vid}{analog}</div>'

    def analog_button(self):
        # Single toggle, gated on connection; label reflects backend state.
        if self.analog_engaged:
            return gr.update(value="🎮 ANALOG ON — click to stop", variant="stop",
                             interactive=self.connected)
        return gr.update(value="🎮 ANALOG CONTROL (smooth)", variant="secondary",
                         interactive=self.connected)

    def button_states(self, count: int) -> tuple:
        """interactive=connected for every connection-gated button, but only
        re-emit on an actual connect/disconnect transition (else no-op)."""
        en = self.connected
        if en == self._gated_state:
            return tuple(gr.update() for _ in range(count))
        self._gated_state = en
        return tuple(gr.update(interactive=en) for _ in range(count))

    def video_stats(self) -> str:
        if self.video is None:
            return "Video not running."
        stats = (
            f"pid={os.getpid()} packets={self.video.packets_seen} decoded_ok={self.video.frames_ok} "
            f"decode_failed={self.video.frames_failed} unknown_pkts={self.video.unknown_packets} "
            f"kicks_sent={self.video.kicks_sent} handshake_resends={self.video.handshake_resends}"
        )
        if self.video.last_error:
            stats += f" | last_decode_error={self.video.last_error}"
        return stats

    # ── connection / video toggles ──────────────────────────────────────────
    def connect(self) -> str:
        with self.lock:
            if self.connected:
                return self._note("Already connected.")
            self.drone.last_error = None
            try:
                self.drone.connect()
                # Real app spends ~95% of a healthy video session disarmed/idle
                # and switches to it within ~0.5s of connecting (see
                # PROTOCOL_NOTES.md). Movement commands auto-wake the drone (see
                # _guarded), so flying still works normally.
                self.drone.set_idle_mode(True)
                return self._note(f"Connected — streaming controls to {self.drone.ip}:{self.drone.port}")
            except OSError as e:
                return self._note(f"Connect failed: {e}")

    def disconnect(self) -> str:
        with self.lock:
            if not self.connected:
                return self._note("Not connected.")
            self._teardown_connection("user disconnect")  # cascades video -> drone
            return self._note("Disconnected.")

    def start_video(self, width, height, color) -> str:
        with self.lock:
            if not self.connected:
                return self._note("Ignored 'Start video' — not connected. Click Connect first.")
            if self.video is not None:
                return self._note("Video already running.")
            return self._start_video_locked(width, height, color, set_idle=True)

    def _start_video_locked(self, width, height, color, set_idle=True) -> str:
        """Create the receiver + MJPEG server. Caller must hold self.lock and have
        confirmed we're connected and video isn't already running. `set_idle` is
        True for a user Start-video (matches the real app's grounded idle mode);
        the auto-restart path passes False so it NEVER forces idle mode mid-flight
        (that would send cmd=0 = disarm — dangerous while airborne)."""
        self._video_wh = (int(width), int(height), bool(color))
        if set_idle:
            # Real app spends ~95% of a healthy stream disarmed/idle (cmd=0, zero
            # axes, frozen counter) — see PROTOCOL_NOTES.md. Not required for video
            # (armed flight streams fine), so the restart path leaves it alone.
            self.drone.set_idle_mode(True)
        self.video = VideoReceiver(self.drone.sock, self.drone.ip, self.drone.port,
                                   width=int(width), height=int(height),
                                   components=3 if color else 1)
        self.video.start()
        # Push-based MJPEG stream, not client polling — see video_stream.py's
        # MjpegServer docstring. Frame updates the instant one decodes.
        self.mjpeg = MjpegServer(self.video)
        try:
            self.mjpeg.start()
        except OSError as e:
            self.video.stop()
            self.video = None
            self.mjpeg = None
            return self._note(f"Video started, but the MJPEG server failed to bind: {e}. "
                              f"Try again in a moment (port may still be releasing).")
        return self._note(
            f"Video receiver started ({int(width)}x{int(height)}). "
            f"MJPEG stream: {self.mjpeg.url}  Log: {self.video.logger._jsonl_path}"
        )

    def stop_video(self) -> str:
        with self.lock:
            if self.video is None:
                return self._note("Video not running.")
            self._teardown_video("user stop video")
            return self._note("Video receiver stopped, drone back to normal flight mode.")

    # ── lost-link auto-recovery (timer-only) ────────────────────────────────
    def check_lost_link(self):
        """If the drone control loop died (a send error auto-disarms it) while
        video is still live, cascade-teardown once — never keep a VideoReceiver
        running against a dead socket, and clear the frozen frame. Runs from the
        poll timer, so recovery is automatic without the user clicking."""
        if self.drone._armed or (self.video is None and self.analog_server is None):
            return
        if not self.lock.acquire(blocking=False):
            return  # a toggle handler is mid-flight; it'll settle next tick
        try:
            if not self.drone._armed and (self.video is not None or self.analog_server is not None):
                self._release_analog("lost link — drone control loop stopped")
                self._teardown_video("lost link — drone control loop stopped")
                self._note("Lost connection to drone — video / analog torn down.")
        finally:
            self.lock.release()

    def _check_video_stall(self):
        """The DM002HW can stop sending video entirely mid-flight; the receiver's
        own in-place kicks don't recover that. If we've gone STALL_RESTART_AFTER
        seconds with no decoded frame while still connected, restart the receiver
        to force a fresh handshake — throttled to once per STALL_RESTART_EVERY so
        we don't thrash. Runs from the poll timer (automatic, no click)."""
        if not self.connected or self.video is None:
            return
        ts = self.video.last_decoded_ts
        if ts is None:
            return  # never got a first frame — the start-time handshake owns that
        if time.time() - ts < STALL_RESTART_AFTER:
            return
        if time.time() - self._last_video_restart_ts < STALL_RESTART_EVERY:
            return  # already tried recently; give the drone time to come back
        if not self.lock.acquire(blocking=False):
            return  # a toggle handler is mid-flight; it'll settle next tick
        try:
            # Re-check under the lock (a manual stop/start may have just landed).
            if (self.connected and self.video is not None
                    and self.video.last_decoded_ts is not None
                    and time.time() - self.video.last_decoded_ts >= STALL_RESTART_AFTER):
                idle = time.time() - self.video.last_decoded_ts
                self._last_video_restart_ts = time.time()
                self._video_epoch += 1  # force the browser to reconnect to the fresh stream
                w, h, color = self._video_wh
                self._teardown_video("video hard stall — auto-restart")
                self._note(f"Video hard-stalled {idle:.0f}s with no frames — auto-restarting receiver.")
                self._start_video_locked(w, h, color, set_idle=False)  # never force idle mid-flight
        finally:
            self.lock.release()

    def poll_tick(self):
        self.check_lost_link()
        self._check_video_stall()

    # ── analog "gyro-style" control (continuous fine axis stream) ────────────
    def do_toggle_analog(self) -> str:
        with self.lock:  # serialize toggles — no double-start race
            if self.analog_engaged:
                self._release_analog("user disengaged")
                return self._note("Analog control disengaged. Drone centered (hover).")
            if not self.connected:
                return self._note("Ignored 'ANALOG CONTROL' — not connected. Click Connect first.")
            srv = AnalogInputServer(self.drone, port=ANALOG_PORT, invert=self.analog_invert)
            try:
                srv.start()
                self.analog_server = srv
                return self._note(f"Analog control engaged — joystick streaming to {srv.url}. "
                                  "Drag a pad to fly; release to recenter.")
            except OSError as e:
                return self._note(f"Analog control failed to start: {e} (port may still be releasing)")

    def analog_set_invert(self, axis, value) -> str:
        self.analog_invert[axis] = bool(value)
        return self._note(f"analog invert {axis}={bool(value)}")

    # ── guarded flight commands ─────────────────────────────────────────────
    # serialize=True (the timed directional moves only) takes _cmd_lock so
    # overlapping moves can't clobber each other's axes; a second move while one
    # runs is a "busy" no-op. Everything else runs immediately.
    def _guarded(self, fn, label, *args, serialize=False, **kwargs) -> str:
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

    def do_takeoff(self):   return self._guarded(self.drone.takeoff, "Takeoff")
    def do_land(self):      return self._guarded(self.drone.land, "Land")
    def do_hover(self):     return self._guarded(self.drone.hover, "Hover / center sticks")
    def do_calibrate(self): return self._guarded(self.drone.calibrate, "Calibrate gyro (keep drone flat)")

    def do_stop(self):
        # EMERGENCY STOP must never be blocked (instant cmd=CMD_STOP, no sleep).
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

    def do_set_axes(self, roll, pitch, throttle, yaw) -> str:
        # Explicit absolute stick position — latest-wins, always applied.
        if not self.connected:
            return self._note("Ignored manual axis update — not connected.")
        # Defensive coercion: a stale browser tab (a leftover tab from a previous
        # run of this app on the same port) can replay this event with junk/empty
        # values; never let that crash — just ignore an un-parseable payload.
        try:
            r, p, t, y = int(roll), int(pitch), int(throttle), int(yaw)
        except (TypeError, ValueError):
            return self._note("Ignored manual axis update — bad values (stale browser tab?).")
        self.drone.set_idle_mode(False)
        self.drone.set_controls(roll=r, pitch=p, throttle=t, yaw=y)
        return self._note(f"Manual axes -> roll={r} pitch={p} throttle={t} yaw={y}")

    def do_center_axes(self):
        # Don't snap the sliders when disconnected — the hover no-ops, so the UI
        # would otherwise drift out of sync with the backend.
        if not self.connected:
            return gr.update(), gr.update(), gr.update(), gr.update(), self.do_hover()
        return NEUTRAL, NEUTRAL, NEUTRAL, NEUTRAL, self.do_hover()


ctrl = DroneController()


# ── handler wrappers: assemble the full rendered output tuple ───────────────
# The four state-changing toggles and the poll tick return, in addition to the
# status log, the video-feed HTML + status badge + interactive-state of every
# connection-gated button — the single render path that keeps UI == backend.

def h_connect():
    note = ctrl.connect()
    return (note, ctrl.render_video_feed(force=True), ctrl.status_badge_html(), ctrl.analog_button()) + ctrl.button_states(N_GATED)


def h_disconnect():
    note = ctrl.disconnect()
    return (note, ctrl.render_video_feed(force=True), ctrl.status_badge_html(), ctrl.analog_button()) + ctrl.button_states(N_GATED)


def h_start_video(width, height, color):
    note = ctrl.start_video(width, height, color)
    return (note, ctrl.render_video_feed(force=True), ctrl.status_badge_html(), ctrl.analog_button()) + ctrl.button_states(N_GATED)


def h_stop_video():
    note = ctrl.stop_video()
    return (note, ctrl.render_video_feed(force=True), ctrl.status_badge_html(), ctrl.analog_button()) + ctrl.button_states(N_GATED)


def h_toggle_analog():
    note = ctrl.do_toggle_analog()
    return (note, ctrl.render_video_feed(force=False), ctrl.status_badge_html(), ctrl.analog_button()) + ctrl.button_states(N_GATED)


def h_poll_sync():
    ctrl.poll_tick()  # lost-link cascade teardown check — timer-only
    return (ctrl.render_video_feed(force=False), ctrl.status_badge_html(), ctrl.analog_button()) + ctrl.button_states(N_GATED)


CSS = """
.dpad-btn {min-width: 3.2em !important;}
.big-btn button {font-size: 1.1em !important; font-weight: 600;}
#stop-btn button {background: #c0392b !important; color: white !important;}
#takeoff-btn button {background: #27ae60 !important; color: white !important;}
#land-btn button {background: #e67e22 !important; color: white !important;}
#video-box {min-height: 360px; display:flex; align-items:center; justify-content:center;
            border: 2px dashed var(--border-color-primary); border-radius: 8px;}
#analog-btn button {background:#8e44ad !important; color:white !important;}
.analog-pad {width: 170px; height: 170px; margin: 4px auto; border: 2px solid var(--border-color-primary);
            border-radius: 12px; display:flex; align-items:center; justify-content:center;
            text-align:center; user-select:none; touch-action:none; cursor: grab;
            background: var(--background-fill-secondary); font-size:0.9em;}
.analog-pad.analog-active {cursor: grabbing; border-color:#8e44ad; box-shadow:0 0 0 2px #8e44ad55 inset;}
"""

# Page-load JS: two virtual-joystick pads streaming held stick positions to the
# AnalogInputServer (analog_control.py) at ~20Hz. Runs client-side; when analog
# mode is off the server isn't listening and fetch() fails silently. A <script>
# inside gr.HTML would NOT execute (Svelte {@html}), so this is injected via
# demo.load(js=...) and polls until the pads are in the DOM.
ANALOG_JS = """
() => {
  const PORT = 8095, NEUTRAL = 128;
  const state = { roll: NEUTRAL, pitch: NEUTRAL, throttle: NEUTRAL, yaw: NEUTRAL };
  const toByte = (n) => Math.max(0, Math.min(255, Math.round(NEUTRAL + n * 127)));
  const post = () => {
    const qs = 'roll=' + state.roll + '&pitch=' + state.pitch +
               '&throttle=' + state.throttle + '&yaw=' + state.yaw;
    fetch('http://127.0.0.1:' + PORT + '/set?' + qs).catch(() => {});
  };
  // Only stream while a pad is actively held — no always-on 20Hz fetch storm
  // (which spammed the console and competed with the MJPEG <img> stream).
  let held = 0, timer = null;
  const startPosting = () => { if (timer === null) timer = setInterval(post, 50); };
  const stopPosting  = () => { if (timer !== null) { clearInterval(timer); timer = null; } };
  function attachPad(el, xAxis, yAxis) {
    let down = false;
    const setFrom = (e) => {
      const r = el.getBoundingClientRect();
      let nx = Math.max(-1, Math.min(1, ((e.clientX - r.left) / r.width  - 0.5) * 2));
      let ny = Math.max(-1, Math.min(1, ((e.clientY - r.top)  / r.height - 0.5) * 2));
      state[xAxis] = toByte(nx);
      state[yAxis] = toByte(-ny);          // top of pad = positive (up / forward)
    };
    const recenter = () => { state[xAxis] = NEUTRAL; state[yAxis] = NEUTRAL; };
    el.addEventListener('pointerdown', (e) => {
      down = true; held++; try { el.setPointerCapture(e.pointerId); } catch (_) {}
      setFrom(e); el.classList.add('analog-active'); startPosting(); post();
    });
    el.addEventListener('pointermove', (e) => { if (down) setFrom(e); });
    const end = () => {
      if (!down) return;
      down = false; held = Math.max(0, held - 1);
      recenter(); el.classList.remove('analog-active');
      post();                              // send the recenter for this pad at once
      if (held === 0) stopPosting();       // idle -> zero traffic
    };
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
    return true;
  }
  const iv = setInterval(() => { if (init()) clearInterval(iv); }, 300);
}
"""

with gr.Blocks(title="DM002HW Drone Controller") as demo:
    gr.Markdown(
        f"# DM002HW Drone Controller\n"
        f"`process pid={os.getpid()}` — **use a single browser tab.** Only one PID "
        f"should ever send to the drone at a time, and a leftover tab from a previous "
        f"run keeps its own 0.4s timer alive (double commands) and replays stale "
        f"events (the `do_set_axes ... got 0` tracebacks) against the new server. "
        f"Close old terminals **and** old browser tabs before testing."
    )
    status_badge = gr.HTML(ctrl.status_badge_html())

    with gr.Row():
        # ── video panel ────────────────────────────────────────────────────
        with gr.Column(scale=5):
            video_feed = gr.HTML(ctrl.raw_video_html())
            with gr.Row():
                start_video_btn = gr.Button("🎥 Start video")
                stop_video_btn = gr.Button("Stop video")
            with gr.Row():
                vid_w = gr.Number(value=640, label="Width", precision=0)
                vid_h = gr.Number(value=360, label="Height", precision=0)
                vid_color = gr.Checkbox(value=True, label="Color (uncheck if garbled)")
            video_stats = gr.Textbox(label="Video stats", interactive=False)
            gr.Markdown(
                "Reconstruction verified offline against real captured packets, "
                "but never flight-tested live — if the picture looks skewed/wrong "
                "colors, try toggling Color or adjusting Width/Height. "
                "See `PROTOCOL_NOTES.md`. Raw packets + saved frames land in "
                "`video_debug/` for offline debugging."
            )
            status = gr.Textbox(label="Status log", lines=10, interactive=False)

        # ── controller panel ──────────────────────────────────────────────
        with gr.Column(scale=7):
            with gr.Row():
                connect_btn = gr.Button("🔌 Connect")
                disconnect_btn = gr.Button("Disconnect")
                duration = gr.Slider(0.2, 3.0, value=1.0, step=0.1, label="Move duration (s)")

            with gr.Row(elem_classes="big-btn"):
                takeoff_btn = gr.Button("🛫 Takeoff", elem_id="takeoff-btn")
                hover_btn = gr.Button("🖐 HOVER (center)")
                land_btn = gr.Button("🛬 Land", elem_id="land-btn")
                calibrate_btn = gr.Button("🧭 Calibrate gyro")
                stop_btn = gr.Button("⛔ EMERGENCY STOP", elem_id="stop-btn")

            gr.Markdown("### Gimbals")
            with gr.Row():
                # Left stick: throttle (up/down) + yaw (rotate left/right)
                with gr.Column():
                    gr.Markdown("**Left stick — Throttle / Yaw**")
                    with gr.Row():
                        gr.Button("", elem_classes="dpad-btn")
                        up_btn = gr.Button("▲ Up", elem_classes="dpad-btn")
                        gr.Button("", elem_classes="dpad-btn")
                    with gr.Row():
                        yaw_l_btn = gr.Button("↺ Yaw L", elem_classes="dpad-btn")
                        hover_btn2 = gr.Button("● Hover", elem_classes="dpad-btn")
                        yaw_r_btn = gr.Button("↻ Yaw R", elem_classes="dpad-btn")
                    with gr.Row():
                        gr.Button("", elem_classes="dpad-btn")
                        down_btn = gr.Button("▼ Down", elem_classes="dpad-btn")
                        gr.Button("", elem_classes="dpad-btn")

                # Right stick: pitch (forward/back) + roll (left/right)
                with gr.Column():
                    gr.Markdown("**Right stick — Pitch / Roll**")
                    with gr.Row():
                        gr.Button("", elem_classes="dpad-btn")
                        fwd_btn = gr.Button("▲ Fwd", elem_classes="dpad-btn")
                        gr.Button("", elem_classes="dpad-btn")
                    with gr.Row():
                        left_btn = gr.Button("◀ Left", elem_classes="dpad-btn")
                        hover_btn3 = gr.Button("● Hover", elem_classes="dpad-btn")
                        right_btn = gr.Button("▶ Right", elem_classes="dpad-btn")
                    with gr.Row():
                        gr.Button("", elem_classes="dpad-btn")
                        back_btn = gr.Button("▼ Back", elem_classes="dpad-btn")
                        gr.Button("", elem_classes="dpad-btn")

            gr.Markdown("### 🎮 Analog control — smooth / gyro-style")
            gr.Markdown(
                "Drag a pad for continuous fine stick values — this is exactly what the "
                "phone's *gyro mode* does (fine analog values streamed at ~20 Hz), just with a "
                "joystick instead of tilt. Release a pad to recenter it."
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
                    "The background loop keeps re-sending whatever value you leave these at, "
                    "so this behaves like holding a real stick in place."
                )
                with gr.Row():
                    roll_s = gr.Slider(0, 255, value=NEUTRAL, step=1, label="Roll")
                    pitch_s = gr.Slider(0, 255, value=NEUTRAL, step=1, label="Pitch")
                    throttle_s = gr.Slider(0, 255, value=NEUTRAL, step=1, label="Throttle")
                    yaw_s = gr.Slider(0, 255, value=NEUTRAL, step=1, label="Yaw")
                center_btn = gr.Button("Center all axes (hover)")

    # Connection-gated buttons: disabled whenever we're not connected (which
    # includes a lost link, since the poll tick recomputes this from real state).
    GATED_BUTTONS = [
        start_video_btn, stop_video_btn,
        takeoff_btn, hover_btn, land_btn, calibrate_btn, stop_btn,
        up_btn, down_btn, fwd_btn, back_btn, left_btn, right_btn,
        yaw_l_btn, yaw_r_btn, hover_btn2, hover_btn3, center_btn,
    ]
    N_GATED = len(GATED_BUTTONS)

    # ── wiring ────────────────────────────────────────────────────────────
    _sync_outputs = [status, video_feed, status_badge, analog_btn] + GATED_BUTTONS
    connect_btn.click(h_connect, outputs=_sync_outputs)
    disconnect_btn.click(h_disconnect, outputs=_sync_outputs)
    start_video_btn.click(h_start_video, inputs=[vid_w, vid_h, vid_color], outputs=_sync_outputs)
    stop_video_btn.click(h_stop_video, outputs=_sync_outputs)
    analog_btn.click(h_toggle_analog, outputs=_sync_outputs)
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

    video_timer = gr.Timer(0.4)
    video_timer.tick(ctrl.video_stats, outputs=video_stats)
    video_timer.tick(h_poll_sync, outputs=[video_feed, status_badge, analog_btn] + GATED_BUTTONS)

    # Page-load: wire up the two analog joystick pads (streams to AnalogInputServer).
    demo.load(js=ANALOG_JS)

demo.queue()

if __name__ == "__main__":
    demo.launch(css=CSS)
