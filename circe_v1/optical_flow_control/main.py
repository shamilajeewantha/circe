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

Every toggle (Connect, Start video, Hold Position) is a single button whose
label/color reflects real backend state — recomputed on every click AND every
poll tick, never just "what was last clicked" — so an automatic safety
disengage (stale video) shows up on the button without another click.

A second, minimal tab ("Optical Flow — Demo") is a webcam-only sandbox for
tuning deadband/ROI/signs before trusting them on the real drone. It shares
the same FlowStabilizer instance, so tuning there carries over to Hold
Position.

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

log = get_logger("main")

# ── shared backend state (single source of truth for everything below) ─────
drone = Drone()
video: VideoReceiver | None = None
mjpeg: MjpegServer | None = None
stabilizer = FlowStabilizer()
processor: FlowProcessor | None = None       # Hold Position (drone) processor
demo_processor: FlowProcessor | None = None  # Optical Flow tab (webcam) processor
_log: list[str] = []

# Serializes all four toggle handlers (Connect / Start video / Hold Position /
# webcam Demo) so a rapid double-click can't race two connects/videos/demos and
# orphan a socket, thread, or bound port. Also makes the Hold-Position/Demo
# mutual-exclusion check below race-free, since both read-check-and-write the
# processor/demo_processor globals only from inside this same lock.
# Mirrors FlowProcessor's own _lifecycle_lock.
_ui_lock = threading.Lock()

HOLD_ARROW_PORT, HOLD_HSV_PORT = 8091, 8092
DEMO_ARROW_PORT, DEMO_HSV_PORT = 8093, 8094

def _note(msg: str) -> str:
    log.info(msg)
    ts = time.strftime("%H:%M:%S")
    _log.append(f"[{ts}] {msg}")
    # Fan the same action into whichever session logger(s) are live, so a
    # single JSONL shows button presses interleaved with packet/frame events
    # on one comparable timeline — no more manually cross-referencing app.log
    # (human timestamps) against the video/flow JSONL (relative-float timestamps).
    for target in (video, processor, demo_processor):
        if target is not None:
            target.logger.log({"event": "ui_action", "msg": msg})
    return "\n".join(_log[-16:])


def _connected() -> bool:
    return drone._armed


def _holding() -> bool:
    return processor is not None and processor.engaged


# ════════════════════════════════════════════════════════════════════════════
# Cascade teardown — each level tears down everything beneath it, in order,
# so nothing is ever left dangling (a FlowProcessor reading a stopped
# VideoReceiver, a receiver on a socket that's about to close, etc).
# ════════════════════════════════════════════════════════════════════════════

def _release_hold_position(reason: str):
    global processor
    if processor is not None:
        processor.disengage()
        processor.stop()
        processor = None
        log.info("Hold Position released (%s)", reason)


def _teardown_video(reason: str):
    global video, mjpeg
    _release_hold_position(reason)  # Hold Position can't outlive its video source
    if mjpeg is not None:
        mjpeg.stop()
        mjpeg = None
    if video is not None:
        video.stop()
        video = None
    drone.set_idle_mode(False)


def _teardown_connection(reason: str):
    _teardown_video(reason)  # video can't outlive the connection
    if drone._armed:
        drone.disconnect()


# ════════════════════════════════════════════════════════════════════════════
# UI state sync — every toggle handler AND the polling timer call this, so
# button label/color always reflects actual backend state, never a client-side
# assumption. This is what makes the stale-frame auto-disengage visible
# without requiring the user to click anything.
# ════════════════════════════════════════════════════════════════════════════

def _connect_button():
    if _connected():
        return gr.update(value="🔌 CONNECTED — click to disconnect")
    return gr.update(value="🔌 Connect")


def _video_button():
    if video is not None:
        return gr.update(value="🎥 VIDEO LIVE — click to stop")
    return gr.update(value="🎥 Start video")


def _hold_button():
    if _holding():
        return gr.update(value="🎯 HOLDING — click to release", variant="stop")
    return gr.update(value="🎯 HOLD POSITION", variant="secondary")


_BADGE_CSS = "display:inline-block; padding:0.35em 0.9em; border-radius:999px; font-weight:700; margin-right:0.5em;"


def _badge(text: str, color: str) -> str:
    return f'<span style="{_BADGE_CSS} background:{color}; color:white;">{text}</span>'


def poll_status():
    if drone._armed:
        conn = _badge(f"🟢 CONNECTED — {drone.ip}:{drone.port}", "#27ae60")
    elif drone.last_error:
        conn = _badge(f"🔴 LOST CONNECTION — {drone.last_error}", "#c0392b")
    else:
        conn = _badge("⚪ NOT CONNECTED", "#7f8c8d")

    if video is None:
        vid = _badge("⚪ VIDEO NOT STARTED", "#7f8c8d")
    elif video.last_decoded_ts is None:
        vid = _badge("🟡 VIDEO: waiting for first frame…", "#e67e22")
    else:
        age = time.time() - video.last_decoded_ts
        vid = (_badge(f"🟢 VIDEO LIVE ({video.frames_ok} frames)", "#27ae60") if age < 2.0
               else _badge(f"🟡 VIDEO STALLED — last frame {age:.1f}s ago", "#e67e22"))

    if _holding():
        p = processor
        age = (time.time() - p.last_frame_ts) if p.last_frame_ts else 999
        hold = (_badge("🟢 HOLDING POSITION", "#27ae60") if age < 0.6
                else _badge(f"🟡 HOLDING (stale {age:.1f}s → neutral)", "#e67e22"))
    else:
        hold = _badge("⚪ NOT HOLDING", "#7f8c8d")

    return f'<div style="font-size:1.05em;">{conn}{vid}{hold}</div>'


def _ui_sync():
    return _connect_button(), _video_button(), _hold_button(), poll_status()


def _check_lost_link():
    """If the drone control loop died (a send error auto-disarms it) while video
    or Hold Position is still live, cascade-teardown once — never keep a
    FlowProcessor/VideoReceiver running against a dead socket. Runs from the poll
    timer, so recovery is automatic without the user clicking anything."""
    if drone._armed or (video is None and processor is None):
        return
    if not _ui_lock.acquire(blocking=False):
        return  # a toggle handler is mid-flight; it'll settle on the next tick
    try:
        if not drone._armed and (video is not None or processor is not None):
            _teardown_video("lost link — drone control loop stopped")
            _note("Lost connection to drone — video / Hold Position torn down.")
    finally:
        _ui_lock.release()


def _poll_sync():
    """One timer tick: auto-recover from a lost link, then refresh every
    state-driven display (video feed, both flow views, all three toggle
    buttons, status badges) from real backend state."""
    _check_lost_link()
    arrow_html, hsv_html = _hold_views_html()
    return (_raw_video_html(), arrow_html, hsv_html,
            _connect_button(), _video_button(), _hold_button(), poll_status())


# ════════════════════════════════════════════════════════════════════════════
# Connection & Video toggles
# ════════════════════════════════════════════════════════════════════════════

def _raw_video_html():
    if mjpeg is not None:
        return f'<img src="{mjpeg.url}" style="width:100%;border-radius:8px;">'
    return "<div id='video-box'>🎥 Video feed not connected yet</div>"


def _start_video(width, height, color) -> str:
    """Start the video receiver + MJPEG server. Assumes the caller already
    holds _ui_lock and has confirmed we're connected and video isn't already
    running. Returns a status note (success or error)."""
    global video, mjpeg
    drone.set_idle_mode(True)
    video = VideoReceiver(drone.sock, drone.ip, drone.port,
                          width=int(width), height=int(height),
                          components=3 if color else 1)
    video.start()
    mjpeg = MjpegServer(video)
    try:
        mjpeg.start()
        return _note(f"Video started ({int(width)}x{int(height)}). "
                     f"Log: {video.logger._jsonl_path}")
    except OSError as e:
        video.stop()
        video = None
        mjpeg = None
        return _note(f"Video failed to bind: {e} (port may still be releasing)")


def do_toggle_connect(width=640, height=360, color=True):
    with _ui_lock:  # serialize toggles — no double-connect race
        if _connected():
            _teardown_connection("user disconnect")
            note = _note("Disconnected.")
        else:
            drone.last_error = None
            try:
                drone.connect()
                # Real app spends ~95% of a healthy video session disarmed/idle —
                # engage idle mode immediately rather than waiting for Start
                # video (a live test showed the stall locking in before a later
                # switch took effect). Flight commands auto-wake the drone.
                drone.set_idle_mode(True)
                note = _note(f"Connected — streaming controls to {drone.ip}:{drone.port}")
                # Auto-start video on connect so it's one less manual step —
                # the Start/Stop video button still works independently and
                # stays in sync (it's recomputed from the `video` global on
                # every click and poll tick, same as every other toggle here).
                note = _start_video(width, height, color)
            except OSError as e:
                note = _note(f"Connect failed: {e}")
        return (note, _raw_video_html()) + _ui_sync()


def do_toggle_video(width, height, color):
    with _ui_lock:  # serialize toggles — no double-start race
        if video is not None:
            _teardown_video("user stop video")
            note = _note("Video stopped, drone back to normal flight mode.")
        elif not _connected():
            note = _note("Ignored 'Start video' — not connected. Click Connect first.")
        else:
            note = _start_video(width, height, color)
        return (note, _raw_video_html()) + _ui_sync()


def poll_video_stats():
    if video is None:
        return "video: not running"
    stats = (f"pid={os.getpid()} packets={video.packets_seen} decoded_ok={video.frames_ok} "
             f"decode_failed={video.frames_failed} unknown_pkts={video.unknown_packets} "
             f"kicks_sent={video.kicks_sent} handshake_resends={video.handshake_resends}")
    if video.last_error:
        stats += f" | last_decode_error={video.last_error}"
    return stats


# ════════════════════════════════════════════════════════════════════════════
# Hold Position — the optical-flow control loop IS the hover mechanism
# ════════════════════════════════════════════════════════════════════════════

def _hold_placeholder(label: str) -> str:
    return f'<div class="flow-box">{label}</div>'


def _hold_views_html():
    if processor is not None:
        arrow = f'<img src="{processor.arrow_url}" style="width:100%;border-radius:8px;">'
        hsv = f'<img src="{processor.hsv_url}" style="width:100%;border-radius:8px;">'
        return arrow, hsv
    return (_hold_placeholder("🡒 Motion Arrows — waiting for Hold Position"),
            _hold_placeholder("🌈 Dense Field (HSV) — waiting for Hold Position"))


def do_toggle_hold():
    global processor
    with _ui_lock:  # serialize toggles — no double-start race
        if _holding():
            _release_hold_position("user released")
            note = _note("Hold Position released. Drone back to idle/neutral.")
        elif video is None:
            note = _note("Ignored 'HOLD POSITION' — video isn't live. Start video first "
                         "(can't hold position with no frames).")
        elif demo_processor is not None:
            note = _note("Ignored 'HOLD POSITION' — webcam Demo is running. Stop the Demo "
                         "first (shared stabilizer can't drive two live sources at once).")
        else:
            proc = FlowProcessor(DroneFrameSource(video), stabilizer, drone=drone,
                                 logger=FlowLogger(), arrow_port=HOLD_ARROW_PORT, hsv_port=HOLD_HSV_PORT)
            try:
                proc.start()
                proc.engage()
                processor = proc
                note = _note(f"HOLD POSITION engaged (Displacement Hold). Log: {proc.logger.jsonl_path}")
            except OSError as e:
                note = _note(f"Hold Position failed to start: {e} (port may still be releasing)")
        arrow_html, hsv_html = _hold_views_html()
        return (note, arrow_html, hsv_html) + _ui_sync()


def _log_param(name, value):
    """Record a live tuning change into every active session log, so the JSONL
    fully explains later corrections — not just the transient UI note list."""
    for p in (processor, demo_processor):
        if p is not None:
            p.logger.log({"event": "param_change", "param": name, "value": value})


def flow_set_roi(v):       stabilizer.roi_fraction = float(v);    _log_param("roi_fraction", float(v));    return _note(f"roi_fraction={v}")
def flow_set_deadband(v):  stabilizer.deadband_px = float(v);     _log_param("deadband_px", float(v));     return _note(f"deadband_px={v}")
def flow_set_flip_roll(v): stabilizer.flip_roll = bool(v);        _log_param("flip_roll", bool(v));        return _note(f"flip_roll={v}")
def flow_set_flip_thr(v):  stabilizer.flip_throttle = bool(v);    _log_param("flip_throttle", bool(v));    return _note(f"flip_throttle={v}")


def flow_set_roll_enabled(v):
    if processor is not None:
        processor.roll_enabled = bool(v)
    _log_param("roll_enabled", bool(v))
    return _note(f"roll_enabled={v}")


def flow_set_throttle_enabled(v):
    if processor is not None:
        processor.throttle_enabled = bool(v)
    _log_param("throttle_enabled", bool(v))
    return _note(f"throttle_enabled={v}")


def flow_set_hold_threshold(v):
    if processor is not None:
        processor.hold_threshold_px = float(v)
    _log_param("hold_threshold_px", float(v))
    return _note(f"hold_threshold_px={v}")


def flow_set_hold_settle(v):
    if processor is not None:
        processor.hold_settle_s = float(v)
    _log_param("hold_settle_s", float(v))
    return _note(f"hold_settle_s={v}")


# ── Diagnostics: live numbers, plot, corrections table ──────────────────────

_EMPTY_PLOT = pd.DataFrame({"t": [], "value": [], "series": []})


def poll_hold_stats():
    if processor is None:
        return "Hold Position: not running."
    c = processor.last_corr
    return (f"frames={processor.frames_processed}  compute={c.compute_ms:.2f}ms  "
            f"src_fps={processor.source.fps:.1f}\n"
            f"raw  dx={c.raw_dx:+.2f} dy={c.raw_dy:+.2f}\n"
            f"flow dx={c.flow_dx:+.2f} dy={c.flow_dy:+.2f}  valid={c.valid}  coherent={c.coherent}\n"
            f"cumulative displacement  dx={c.cum_dx:+.1f}px  dy={c.cum_dy:+.1f}px  "
            f"(threshold={processor.hold_threshold_px:.0f}px)")


def poll_hold_plot():
    if processor is None:
        return _EMPTY_PLOT
    pts = processor.plot_snapshot()
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


def poll_event_log():
    return "\n".join(_log[-16:]) if _log else "No events yet."


# ════════════════════════════════════════════════════════════════════════════
# Manual flight controls (unchanged one-shot commands — not start/stop pairs)
# ════════════════════════════════════════════════════════════════════════════

def _guarded(fn, label, *args, **kwargs):
    if not _connected():
        reason = f" (lost connection: {drone.last_error})" if drone.last_error else ""
        return _note(f"Ignored '{label}' — not connected{reason}. Click Connect first.")
    drone.set_idle_mode(False)  # any real flight command wakes the drone up
    fn(*args, **kwargs)
    return _note(label)


def do_takeoff():    return _guarded(drone.takeoff, "Takeoff")
def do_land():        return _guarded(drone.land, "Land")
def do_hover():       return _guarded(drone.hover, "Hover / center sticks")
def do_calibrate():   return _guarded(drone.calibrate, "Calibrate gyro (keep drone flat)")
def do_stop():        return _guarded(drone.stop, "EMERGENCY STOP")

def do_up(duration):         return _guarded(drone.up, f"Up ({duration}s)", duration=duration)
def do_down(duration):       return _guarded(drone.down, f"Down ({duration}s)", duration=duration)
def do_forward(duration):    return _guarded(drone.forward, f"Forward ({duration}s)", duration=duration)
def do_backward(duration):   return _guarded(drone.backward, f"Backward ({duration}s)", duration=duration)
def do_move_left(duration):  return _guarded(drone.move_left, f"Move left ({duration}s)", duration=duration)
def do_move_right(duration): return _guarded(drone.move_right, f"Move right ({duration}s)", duration=duration)
def do_turn_left(duration):  return _guarded(drone.turn_left, f"Turn left ({duration}s)", duration=duration)
def do_turn_right(duration): return _guarded(drone.turn_right, f"Turn right ({duration}s)", duration=duration)


def do_set_axes(roll, pitch, throttle, yaw):
    if not _connected():
        return _note("Ignored manual axis update — not connected.")
    drone.set_idle_mode(False)
    drone.set_controls(roll=int(roll), pitch=int(pitch), throttle=int(throttle), yaw=int(yaw))
    return _note(f"Manual axes -> roll={roll} pitch={pitch} throttle={throttle} yaw={yaw}")


def do_center_axes():
    return NEUTRAL, NEUTRAL, NEUTRAL, NEUTRAL, do_hover()


# ════════════════════════════════════════════════════════════════════════════
# Optical Flow tab — minimal webcam-only demo/debug sandbox
# ════════════════════════════════════════════════════════════════════════════

def _demo_placeholder(label: str) -> str:
    return f'<div class="flow-box">{label}</div>'


def _demo_views_html():
    if demo_processor is not None:
        arrow = f'<img src="{demo_processor.arrow_url}" style="width:100%;border-radius:8px;">'
        hsv = f'<img src="{demo_processor.hsv_url}" style="width:100%;border-radius:8px;">'
        return arrow, hsv
    return (_demo_placeholder("🡒 Motion Arrows — demo stopped"),
            _demo_placeholder("🌈 Dense Field (HSV) — demo stopped"))


def _demo_button():
    if demo_processor is not None:
        return gr.update(value="⏹ STOP DEMO", variant="stop")
    return gr.update(value="▶ START DEMO (webcam)")


def do_toggle_demo(webcam_idx):
    global demo_processor
    with _ui_lock:  # serialize toggles — no double-start race
        if demo_processor is not None:
            demo_processor.stop()
            demo_processor = None
            note = _note("Webcam demo stopped.")
        elif _holding():
            note = _note("Ignored 'START DEMO' — Hold Position is engaged. Release "
                         "Hold Position first (shared stabilizer can't drive two live "
                         "sources at once).")
        else:
            src = WebcamFrameSource(index=int(webcam_idx), width=640, height=360)
            proc = FlowProcessor(src, stabilizer, drone=None, logger=FlowLogger(),
                                 arrow_port=DEMO_ARROW_PORT, hsv_port=DEMO_HSV_PORT)
            try:
                proc.start()
            except OSError as e:
                note = _note(f"Demo failed to bind stream servers: {e}")
                return (note,) + _demo_views_html() + (_demo_button(),)
            if getattr(src, "open_error", None):
                proc.stop()
                note = _note(f"Webcam error: {src.open_error}")
                return (note,) + _demo_views_html() + (_demo_button(),)
            demo_processor = proc
            note = _note(f"Webcam demo started (index {int(webcam_idx)}).")
        return (note,) + _demo_views_html() + (_demo_button(),)


def do_reset_origin():
    """Mark the current view as 'home' — zeroes the cumulative drift estimate
    without interrupting flow tracking. Shared stabilizer, so this affects
    whichever of Hold Position / Demo is currently running."""
    stabilizer.reset_origin()
    for p in (processor, demo_processor):
        if p is not None:
            p.logger.log({"event": "origin_reset"})
    return _note("Origin reset — current view marked as home (cumulative drift zeroed).")


def poll_demo_drift():
    if demo_processor is None:
        return "cumulative drift: demo not running"
    c = demo_processor.last_corr
    return (f"cumulative drift since origin:  dx={c.cum_dx:+.1f}px  dy={c.cum_dy:+.1f}px  "
            f"(|d|={(c.cum_dx**2 + c.cum_dy**2)**0.5:.1f}px)")


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
            status_badge = gr.HTML(poll_status())

            with gr.Row():
                # ── video + optical-flow hold position ──────────────────────
                with gr.Column(scale=5):
                    video_feed = gr.HTML(_raw_video_html())
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
                            roi_s = gr.Slider(0.2, 1.0, value=stabilizer.roi_fraction, step=0.05, label="ROI fraction")
                            dead_s = gr.Slider(0.0, 3.0, value=stabilizer.deadband_px, step=0.1, label="Deadband px")
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

    # ── wiring ────────────────────────────────────────────────────────────
    connect_btn.click(do_toggle_connect, inputs=[vid_w, vid_h, vid_color],
                      outputs=[status, video_feed, connect_btn, start_video_btn, hold_btn, status_badge])
    start_video_btn.click(do_toggle_video, inputs=[vid_w, vid_h, vid_color],
                          outputs=[status, video_feed, connect_btn, start_video_btn, hold_btn, status_badge])
    hold_btn.click(do_toggle_hold, outputs=[status, hold_arrow_view, hold_hsv_view,
                                            connect_btn, start_video_btn, hold_btn, status_badge])

    takeoff_btn.click(do_takeoff, outputs=status)
    land_btn.click(do_land, outputs=status)
    calibrate_btn.click(do_calibrate, outputs=status)
    stop_btn.click(do_stop, outputs=status)
    for b in (hover_btn, hover_btn2, hover_btn3):
        b.click(do_hover, outputs=status)

    up_btn.click(do_up, inputs=duration, outputs=status)
    down_btn.click(do_down, inputs=duration, outputs=status)
    fwd_btn.click(do_forward, inputs=duration, outputs=status)
    back_btn.click(do_backward, inputs=duration, outputs=status)
    left_btn.click(do_move_left, inputs=duration, outputs=status)
    right_btn.click(do_move_right, inputs=duration, outputs=status)
    yaw_l_btn.click(do_turn_left, inputs=duration, outputs=status)
    yaw_r_btn.click(do_turn_right, inputs=duration, outputs=status)

    for s in (roll_s, pitch_s, throttle_s, yaw_s):
        s.release(do_set_axes, inputs=[roll_s, pitch_s, throttle_s, yaw_s], outputs=status)
    center_btn.click(do_center_axes, outputs=[roll_s, pitch_s, throttle_s, yaw_s, status])

    roi_s.release(flow_set_roi, roi_s, status)
    dead_s.release(flow_set_deadband, dead_s, status)
    flip_roll_c.change(flow_set_flip_roll, flip_roll_c, status)
    flip_thr_c.change(flow_set_flip_thr, flip_thr_c, status)
    roll_en_c.change(flow_set_roll_enabled, roll_en_c, status)
    thr_en_c.change(flow_set_throttle_enabled, thr_en_c, status)
    hold_threshold_s.release(flow_set_hold_threshold, hold_threshold_s, status)
    hold_settle_s_slider.release(flow_set_hold_settle, hold_settle_s_slider, status)

    demo_btn.click(do_toggle_demo, inputs=demo_webcam_idx,
                   outputs=[demo_status, demo_arrow_view, demo_hsv_view, demo_btn])
    demo_reset_btn.click(do_reset_origin, outputs=demo_status)

    timer = gr.Timer(0.4)
    timer.tick(poll_video_stats, outputs=video_stats)
    timer.tick(poll_hold_stats, outputs=hold_stats)
    timer.tick(poll_hold_plot, outputs=hold_plot)
    timer.tick(poll_event_log, outputs=status)
    timer.tick(poll_demo_drift, outputs=demo_drift)
    timer.tick(_poll_sync,
               outputs=[video_feed, hold_arrow_view, hold_hsv_view,
                        connect_btn, start_video_btn, hold_btn, status_badge])

demo.queue()


def shutdown(*_args):
    """Tear the whole tree down so nothing survives process exit: Hold Position
    loop, video + every MJPEG server, the webcam demo, and the drone control
    thread — then release the log-file handles (a held handle is what blocks
    deleting app_logs/ on Windows). Idempotent; safe to call more than once."""
    global demo_processor
    log.info("Shutting down — tearing down all threads/servers.")
    try:
        _teardown_connection("app shutdown")  # cascades hold -> video -> drone
    except Exception as e:  # noqa: BLE001 — never let cleanup raise on the way out
        log.error("teardown_connection during shutdown failed: %s", e)
    if demo_processor is not None:
        try:
            demo_processor.stop()
        except Exception as e:  # noqa: BLE001
            log.error("demo_processor.stop during shutdown failed: %s", e)
        demo_processor = None
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
