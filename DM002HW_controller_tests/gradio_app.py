"""
Gradio front-end for the DM002HW drone controller (drone.py).

Two "gimbal" D-pads (Throttle/Yaw on the left, Pitch/Roll on the right) mimic a
real transmitter, plus a raw-axis panel for direct manual control. The video
panel pulls the drone's UDP JPEG-fragment stream (port 1234) via
video_stream.VideoReceiver — reconstruction was verified offline against real
captured fragments in wireshark_1.pcapng (see PROTOCOL_NOTES.md) but has not
been flight-tested live; default resolution (640x360) is a best guess and
adjustable in the Video panel if frames come out skewed.

Run with:  python gradio_app.py
Then open the printed local URL in a browser.
"""

import io
import os
import time
import warnings

# Gradio 6 / Starlette combo emits this on every queued request (so once per
# gr.Timer tick, i.e. constantly) — cosmetic version-skew noise, not a bug.
warnings.filterwarnings("ignore", message=".*HTTP_422_UNPROCESSABLE_ENTITY.*")

import gradio as gr
from PIL import Image

from drone import Drone, NEUTRAL
from video_stream import VideoReceiver

drone = Drone()
video: VideoReceiver | None = None
_log: list[str] = []


def _note(msg: str) -> str:
    ts = time.strftime("%H:%M:%S")
    _log.append(f"[{ts}] {msg}")
    return "\n".join(_log[-12:])


def _connected() -> bool:
    return drone._armed


# ── connection ────────────────────────────────────────────────────────────

def do_connect():
    if _connected():
        return _note("Already connected.")
    drone.last_error = None
    try:
        drone.connect()
        return _note(f"Connected — streaming controls to {drone.ip}:{drone.port}")
    except OSError as e:
        return _note(f"Connect failed: {e}")


def do_disconnect():
    if not _connected():
        return _note("Not connected.")
    if video is not None:
        do_stop_video()
    drone.disconnect()
    return _note("Disconnected.")


# ── video ─────────────────────────────────────────────────────────────────

def do_start_video(width, height, color):
    global video
    if not _connected():
        return _note("Ignored 'Start video' — not connected. Click Connect first.")
    if video is not None:
        return _note("Video already running.")
    video = VideoReceiver(drone.sock, drone.ip, drone.port,
                           width=int(width), height=int(height),
                           components=3 if color else 1)
    video.start()
    return _note(
        f"Video receiver started ({int(width)}x{int(height)}). "
        f"Log: {video.logger._jsonl_path}"
    )


def do_stop_video():
    global video
    if video is None:
        return _note("Video not running.")
    video.stop()
    video = None
    return _note("Video receiver stopped.")


def poll_video():
    if video is None:
        return None, "Video not running."
    jpeg = video.latest_jpeg()
    img = Image.open(io.BytesIO(jpeg)).convert("RGB") if jpeg else None
    stats = (
        f"pid={os.getpid()} packets={video.packets_seen} decoded_ok={video.frames_ok} "
        f"decode_failed={video.frames_failed} unknown_pkts={video.unknown_packets} "
        f"kicks_sent={video.kicks_sent} handshake_resends={video.handshake_resends}"
    )
    if video.last_error:
        stats += f" | last_decode_error={video.last_error}"
    return img, stats


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
        if age < 2.0:
            vid = _badge(f"🟢 VIDEO LIVE ({video.frames_ok} frames)", "#27ae60")
        else:
            vid = _badge(f"🟡 VIDEO STALLED — last frame {age:.1f}s ago", "#e67e22")

    return f'<div style="font-size:1.05em;">{conn}{vid}</div>'


# ── guarded command wrapper ──────────────────────────────────────────────

def _guarded(fn, label, *args, **kwargs):
    if not _connected():
        reason = f" (lost connection: {drone.last_error})" if drone.last_error else ""
        return _note(f"Ignored '{label}' — not connected{reason}. Click Connect first.")
    fn(*args, **kwargs)
    return _note(label)


# ── flight commands ──────────────────────────────────────────────────────

def do_takeoff():
    return _guarded(drone.takeoff, "Takeoff")


def do_land():
    return _guarded(drone.land, "Land")


def do_hover():
    return _guarded(drone.hover, "Hover / center sticks")


def do_calibrate():
    return _guarded(drone.calibrate, "Calibrate gyro (keep drone flat)")


def do_stop():
    return _guarded(drone.stop, "EMERGENCY STOP")


# ── timed D-pad moves ─────────────────────────────────────────────────────

def do_up(duration):
    return _guarded(drone.up, f"Up ({duration}s)", duration=duration)


def do_down(duration):
    return _guarded(drone.down, f"Down ({duration}s)", duration=duration)


def do_forward(duration):
    return _guarded(drone.forward, f"Forward ({duration}s)", duration=duration)


def do_backward(duration):
    return _guarded(drone.backward, f"Backward ({duration}s)", duration=duration)


def do_move_left(duration):
    return _guarded(drone.move_left, f"Move left ({duration}s)", duration=duration)


def do_move_right(duration):
    return _guarded(drone.move_right, f"Move right ({duration}s)", duration=duration)


def do_turn_left(duration):
    return _guarded(drone.turn_left, f"Turn left ({duration}s)", duration=duration)


def do_turn_right(duration):
    return _guarded(drone.turn_right, f"Turn right ({duration}s)", duration=duration)


# ── raw manual axis panel ────────────────────────────────────────────────

def do_set_axes(roll, pitch, throttle, yaw):
    if not _connected():
        return _note("Ignored manual axis update — not connected.")
    drone.set_controls(roll=int(roll), pitch=int(pitch), throttle=int(throttle), yaw=int(yaw))
    return _note(f"Manual axes -> roll={roll} pitch={pitch} throttle={throttle} yaw={yaw}")


def do_center_axes():
    return NEUTRAL, NEUTRAL, NEUTRAL, NEUTRAL, do_hover()


CSS = """
.dpad-btn {min-width: 3.2em !important;}
.big-btn button {font-size: 1.1em !important; font-weight: 600;}
#stop-btn button {background: #c0392b !important; color: white !important;}
#takeoff-btn button {background: #27ae60 !important; color: white !important;}
#land-btn button {background: #e67e22 !important; color: white !important;}
#video-box {min-height: 360px; display:flex; align-items:center; justify-content:center;
            border: 2px dashed var(--border-color-primary); border-radius: 8px;}
"""

with gr.Blocks(title="DM002HW Drone Controller") as demo:
    gr.Markdown(
        f"# DM002HW Drone Controller\n"
        f"`process pid={os.getpid()}` — if you have more than one terminal "
        f"running this app, only one PID should ever be sending to the drone "
        f"at a time. Close any others before testing video."
    )
    status_badge = gr.HTML(poll_status())

    with gr.Row():
        # ── video panel ────────────────────────────────────────────────────
        with gr.Column(scale=5):
            video_feed = gr.Image(label="Video feed (experimental)", height=360,
                                   show_label=True, elem_id="video-box")
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

    # ── wiring ────────────────────────────────────────────────────────────
    connect_btn.click(do_connect, outputs=status)
    disconnect_btn.click(do_disconnect, outputs=status)

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

    start_video_btn.click(do_start_video, inputs=[vid_w, vid_h, vid_color], outputs=status)
    stop_video_btn.click(do_stop_video, outputs=status)
    video_timer = gr.Timer(0.4)
    video_timer.tick(poll_video, outputs=[video_feed, video_stats])
    video_timer.tick(poll_status, outputs=status_badge)

demo.queue()

if __name__ == "__main__":
    demo.launch(css=CSS)
