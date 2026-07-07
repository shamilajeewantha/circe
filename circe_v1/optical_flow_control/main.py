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
tuning gain/deadband/signs before trusting them on the real drone. It shares
the same FlowStabilizer instance, so tuning there carries over to Hold
Position.

Run with:  python main.py
Then open the printed local URL in a browser.
"""

import os
import time
import warnings

# Gradio 6 / Starlette combo emits this on every queued request (so once per
# gr.Timer tick, i.e. constantly) — cosmetic version-skew noise, not a bug.
warnings.filterwarnings("ignore", message=".*HTTP_422_UNPROCESSABLE_ENTITY.*")

import gradio as gr
import pandas as pd

from applog import get_logger
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

HOLD_ARROW_PORT, HOLD_HSV_PORT = 8091, 8092
DEMO_ARROW_PORT, DEMO_HSV_PORT = 8093, 8094

_CORR_COLUMNS = ["time", "roll", "pitch", "throttle", "yaw", "pulsed", "frame_age_ms", "reason"]


def _note(msg: str) -> str:
    log.info(msg)
    ts = time.strftime("%H:%M:%S")
    _log.append(f"[{ts}] {msg}")
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


# ════════════════════════════════════════════════════════════════════════════
# Connection & Video toggles
# ════════════════════════════════════════════════════════════════════════════

def _raw_video_html():
    if mjpeg is not None:
        return f'<img src="{mjpeg.url}" style="width:100%;border-radius:8px;">'
    return "<div id='video-box'>🎥 Video feed not connected yet</div>"


def do_toggle_connect():
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
        except OSError as e:
            note = _note(f"Connect failed: {e}")
    return (note, _raw_video_html()) + _ui_sync()


def do_toggle_video(width, height, color):
    global video, mjpeg
    if video is not None:
        _teardown_video("user stop video")
        note = _note("Video stopped, drone back to normal flight mode.")
    elif not _connected():
        note = _note("Ignored 'Start video' — not connected. Click Connect first.")
    else:
        drone.set_idle_mode(True)
        video = VideoReceiver(drone.sock, drone.ip, drone.port,
                              width=int(width), height=int(height),
                              components=3 if color else 1)
        video.start()
        mjpeg = MjpegServer(video)
        try:
            mjpeg.start()
            note = _note(f"Video started ({int(width)}x{int(height)}). "
                         f"Log: {video.logger._jsonl_path}")
        except OSError as e:
            video.stop()
            video = None
            mjpeg = None
            note = _note(f"Video failed to bind: {e} (port may still be releasing)")
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
    if _holding():
        _release_hold_position("user released")
        note = _note("Hold Position released. Drone back to idle/neutral.")
    elif video is None:
        note = _note("Ignored 'HOLD POSITION' — video isn't live. Start video first "
                     "(can't hold position with no frames).")
    else:
        proc = FlowProcessor(DroneFrameSource(video), stabilizer, drone=drone,
                             logger=FlowLogger(), arrow_port=HOLD_ARROW_PORT, hsv_port=HOLD_HSV_PORT)
        try:
            proc.start()
            proc.engage()
            processor = proc
            note = _note(f"HOLD POSITION engaged (pulsed). Log: {proc.logger.jsonl_path}")
        except OSError as e:
            note = _note(f"Hold Position failed to start: {e} (port may still be releasing)")
    arrow_html, hsv_html = _hold_views_html()
    return (note, arrow_html, hsv_html) + _ui_sync()


def flow_set_roi(v):       stabilizer.roi_fraction = float(v);    return _note(f"roi_fraction={v}")
def flow_set_deadband(v):  stabilizer.deadband_px = float(v);     return _note(f"deadband_px={v}")
def flow_set_gain(v):      stabilizer.gain = float(v);            return _note(f"gain={v}")
def flow_set_maxcorr(v):   stabilizer.max_correction = int(v);    return _note(f"max_correction={v}")
def flow_set_smooth(v):    stabilizer.smoothing_alpha = float(v); return _note(f"smoothing_alpha={v}")
def flow_set_flip_roll(v): stabilizer.flip_roll = bool(v);        return _note(f"flip_roll={v}")
def flow_set_flip_thr(v):  stabilizer.flip_throttle = bool(v);    return _note(f"flip_throttle={v}")


def flow_set_mode(mode):
    if processor is not None:
        processor.pulsed = (mode == "Pulsed (recommended)")
    return _note(f"actuation mode = {mode}")


# ── Diagnostics: live numbers, plot, corrections table ──────────────────────

_EMPTY_PLOT = pd.DataFrame({"t": [], "value": [], "series": []})
_EMPTY_CORR = pd.DataFrame(columns=_CORR_COLUMNS)


def poll_hold_stats():
    if processor is None:
        return "Hold Position: not running."
    c = processor.last_corr
    return (f"frames={processor.frames_processed}  compute={c.compute_ms:.2f}ms  "
            f"src_fps={processor.source.fps:.1f}\n"
            f"raw    dx={c.raw_dx:+.2f} dy={c.raw_dy:+.2f}\n"
            f"smooth dx={c.flow_dx:+.2f} dy={c.flow_dy:+.2f}  valid={c.valid}\n"
            f"corr   roll={c.roll_delta:+d}  throttle={c.throttle_delta:+d}")


def poll_hold_plot():
    if processor is None:
        return _EMPTY_PLOT
    pts = processor.plot_snapshot()
    if not pts:
        return _EMPTY_PLOT
    t0 = pts[0][0]
    rows = []
    for (t, dx, dy, roll, thr) in pts:
        rel = t - t0
        rows.append((rel, dx, "drift dx"))
        rows.append((rel, dy, "drift dy"))
        rows.append((rel, roll, "roll corr"))
        rows.append((rel, thr, "throttle corr"))
    return pd.DataFrame(rows, columns=["t", "value", "series"])


def poll_corrections_table():
    if processor is None:
        return _EMPTY_CORR
    rows = processor.corrections_snapshot()
    if not rows:
        return _EMPTY_CORR
    df = pd.DataFrame(rows, columns=_CORR_COLUMNS)
    return df.iloc[::-1].head(50)  # newest first, capped


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
    if demo_processor is not None:
        demo_processor.stop()
        demo_processor = None
        note = _note("Webcam demo stopped.")
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
                            gain_s = gr.Slider(0.1, 10.0, value=stabilizer.gain, step=0.1, label="Gain")
                            max_s = gr.Slider(1, 40, value=stabilizer.max_correction, step=1, label="Max correction")
                            smooth_s = gr.Slider(0.05, 1.0, value=stabilizer.smoothing_alpha, step=0.05, label="Smoothing α")
                        with gr.Row():
                            flip_roll_c = gr.Checkbox(value=False, label="Flip roll sign")
                            flip_thr_c = gr.Checkbox(value=False, label="Flip throttle sign")
                            mode_r = gr.Radio(["Pulsed (recommended)", "Continuous"],
                                              value="Pulsed (recommended)", label="Actuation mode")

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
                "Tune gain / deadband / signs against your webcam before trusting them on the "
                "real drone — this shares the same tuning as Hold Position on the Controller tab. "
                "No drone or video code lives here."
            )
            demo_webcam_idx = gr.Number(value=0, label="Webcam index", precision=0)
            demo_btn = gr.Button("▶ START DEMO (webcam)")
            with gr.Row():
                demo_arrow_view = gr.HTML(_demo_placeholder("🡒 Motion Arrows — demo stopped"))
                demo_hsv_view = gr.HTML(_demo_placeholder("🌈 Dense Field (HSV) — demo stopped"))
            demo_status = gr.Textbox(label="Demo status", lines=3, interactive=False)

    # ── wiring ────────────────────────────────────────────────────────────
    connect_btn.click(do_toggle_connect, outputs=[status, video_feed, connect_btn, start_video_btn, hold_btn, status_badge])
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
    gain_s.release(flow_set_gain, gain_s, status)
    max_s.release(flow_set_maxcorr, max_s, status)
    smooth_s.release(flow_set_smooth, smooth_s, status)
    flip_roll_c.change(flow_set_flip_roll, flip_roll_c, status)
    flip_thr_c.change(flow_set_flip_thr, flip_thr_c, status)
    mode_r.change(flow_set_mode, mode_r, status)

    demo_btn.click(do_toggle_demo, inputs=demo_webcam_idx,
                   outputs=[demo_status, demo_arrow_view, demo_hsv_view, demo_btn])

    timer = gr.Timer(0.4)
    timer.tick(poll_video_stats, outputs=video_stats)
    timer.tick(poll_hold_stats, outputs=hold_stats)
    timer.tick(poll_hold_plot, outputs=hold_plot)
    timer.tick(poll_event_log, outputs=status)
    timer.tick(lambda: (_connect_button(), _video_button(), _hold_button(), poll_status()),
              outputs=[connect_btn, start_video_btn, hold_btn, status_badge])

demo.queue()

if __name__ == "__main__":
    demo.launch(css=CSS)
