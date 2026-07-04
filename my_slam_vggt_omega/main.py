import sys
import os

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import argparse
import gc
import glob
import json
import logging
import re
import shutil
import time
from datetime import datetime

import cv2
import torch
import gradio as gr

from vggt_omega.models import VGGTOmega
import pipeline as pipeline_module

_LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),        # console, as before
        logging.FileHandler(_LOG_FILE), # also persisted to disk, survives terminal scrollback
    ],
)
log = logging.getLogger("vggt_omega.main")


def _log_uncaught_exceptions(exc_type, exc_value, exc_traceback):
    """
    Without this, an uncaught crash only ever prints to the terminal and is
    lost the moment the buffer scrolls — nothing on disk captures it. Route
    it through logging (-> app.log) too, with the full traceback.
    """
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    log.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))


sys.excepthook = _log_uncaught_exceptions

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

_model = None
_checkpoint_path = None


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(checkpoint_path=None):
    global _model
    if _model is not None:
        return _model

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    t0 = time.perf_counter()
    log.info("Loading VGGT-Omega model...")

    # Meta-device skips weight init (~8 GB RAM saved); assign=True avoids a copy spike.
    # Weights stay float32 — converting to bfloat16 breaks the heads which run under
    # autocast(enabled=False) and expect float32 parameters.
    with torch.device("meta"):
        model = VGGTOmega().eval()

    ckpt = checkpoint_path or _checkpoint_path
    if ckpt and os.path.isfile(ckpt):
        ckpt_path = ckpt
    else:
        from huggingface_hub import hf_hub_download
        log.info("Downloading checkpoint from HuggingFace...")
        ckpt_path = hf_hub_download(
            repo_id="facebook/VGGT-Omega", filename="vggt_omega_1b_512.pt"
        )

    log.info(f"Loading weights: {ckpt_path}")
    t_weights = time.perf_counter()
    state_dict = torch.load(ckpt_path, map_location="cpu", mmap=True)
    model.load_state_dict(state_dict, assign=True)
    del state_dict
    model = model.to("cuda")
    log.info(f"Weights loaded to GPU in {time.perf_counter() - t_weights:.2f}s")

    params_b = sum(p.numel() for p in model.parameters()) / 1e9
    vram_gb = torch.cuda.memory_allocated() / 1e9
    log.info(f"Ready — {params_b:.2f}B params, VRAM: {vram_gb:.2f} GB, total load time {time.perf_counter() - t0:.2f}s")

    _model = model
    return _model


def _load_model_ui():
    """
    Load Model button handler — explicit and required. Viewing past runs never
    pays this cost, and Start stays disabled until this succeeds.

    Generator: the BUTTON ITSELF carries the state (label + interactive),
    not a separate status line nobody looks at — "Load Model" -> "Loading
    model..." (disabled) -> "Model Loaded ✓" or "Load Failed — Retry".
    The Markdown line still carries the detailed params/VRAM/timing info.
    Yields (load_model_btn_update, model_status_text, start_btn_update).
    """
    yield (
        gr.update(value="⏳ Loading model...", interactive=False),
        "First load can take a while.",
        gr.update(interactive=False),
    )

    t0 = time.perf_counter()
    try:
        model = load_model()
    except Exception as e:
        yield (
            gr.update(value="❌ Load Failed — Retry", interactive=True),
            f"Model load failed: {e}",
            gr.update(interactive=False),
        )
        return

    params_b = sum(p.numel() for p in model.parameters()) / 1e9
    vram_gb = torch.cuda.memory_allocated() / 1e9
    elapsed = time.perf_counter() - t0
    yield (
        gr.update(value="✅ Model Loaded", interactive=True),
        f"{params_b:.2f}B params, {vram_gb:.2f} GB VRAM, loaded in {elapsed:.2f}s",
        gr.update(interactive=True),
    )


# ---------------------------------------------------------------------------
# Upload handling
# ---------------------------------------------------------------------------

def _file_path(file_data) -> str:
    if isinstance(file_data, dict):
        return file_data.get("name") or file_data.get("path") or str(file_data)
    if hasattr(file_data, "name"):
        return file_data.name
    return str(file_data)


def _handle_uploads(input_video, input_images, video_sample_fps=1.0):
    gc.collect()
    torch.cuda.empty_cache()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    image_dir = os.path.join(base, f"upload_{ts}", "images")
    os.makedirs(image_dir, exist_ok=True)

    image_paths = []

    if input_images:
        for item in input_images:
            src = _file_path(item)
            dst = os.path.join(image_dir, os.path.basename(src))
            shutil.copy(src, dst)
            image_paths.append(dst)

    if input_video:
        video_path = _file_path(input_video)
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 1.0
        video_sample_fps = max(float(video_sample_fps), 0.1)
        interval = max(int(round(fps / video_sample_fps)), 1)
        f_idx = saved = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if f_idx % interval == 0:
                out_path = os.path.join(image_dir, f"{saved:06d}.png")
                cv2.imwrite(out_path, frame)
                image_paths.append(out_path)
                saved += 1
            f_idx += 1
        cap.release()

    image_paths = sorted(image_paths)
    return image_dir, image_paths


# ---------------------------------------------------------------------------
# Gradio processing generator
# ---------------------------------------------------------------------------

_OUTPUTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")


def _make_run_dir() -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    d = os.path.join(_OUTPUTS_DIR, f"run_{ts}")
    os.makedirs(d, exist_ok=True)
    return d


def _stats_text(total_images: int, passes_done: int) -> str:
    return f"**Images loaded:** {total_images}   |   **Passes:** {passes_done}/{total_images}"


def _pack(submap_path, global_path) -> str:
    """Encode a (submap glb, global-map-snapshot glb) pair as one radio value."""
    return f"{submap_path or ''}||{global_path or ''}"


def _unpack(value):
    """Decode a packed radio value back to (submap_path_or_None, global_path_or_None)."""
    if not value:
        return None, None
    parts = value.split("||", 1)
    submap = parts[0] or None
    global_ = parts[1] if len(parts) > 1 and parts[1] else None
    return submap, global_


def _process(
    input_video,
    input_images,
    video_sample_fps,
    resolution,
    conf_thres,
    max_points_k,
):
    """
    Streaming generator.
    Yields: (log_text, frame_img_path, submap_viewer_value, global_map_viewer_value,
             glb_radio_update, stats_text)
    """
    if not input_video and not input_images:
        yield "Upload images or a video first.", None, None, None, gr.update(), ""
        return

    try:
        image_dir, image_paths = _handle_uploads(input_video, input_images, video_sample_fps)
    except Exception as e:
        yield f"Upload failed: {e}", None, None, None, gr.update(), ""
        return

    if not image_paths:
        yield "No images found after upload.", None, None, None, gr.update(), ""
        return

    image_res = int(resolution)
    run_dir = _make_run_dir()
    total = len(image_paths)

    log_lines = [
        f"Uploaded {total} images",
        f"Run dir : {run_dir}",
        f"Resolution: {image_res}",
        "─" * 50,
    ]

    # Start is only clickable once "Load Model" has succeeded (see
    # start_btn's interactive gating in build_ui), so _model must already be
    # loaded here. No lazy-loading fallback — if this ever fires with no
    # model loaded, fail loudly rather than silently paying the load cost.
    if _model is None:
        yield "Model not loaded — click '⚙ Load Model' first, then Start.", None, None, None, gr.update(), ""
        return
    model = _model

    yield "\n".join(log_lines), None, None, None, gr.update(), _stats_text(total, 0)

    all_glbs: list[tuple[str, str]] = []  # (label, packed_value)
    last_value = None
    submap_view = None
    global_view = None
    passes_done = 0

    for result in pipeline_module.run(
        image_dir, model, run_dir,
        image_resolution=image_res,
        conf_thres=float(conf_thres),
        max_points=int(max_points_k) * 1000,
    ):
        frame_idx = result["frame_idx"]
        is_global_map = result.get("is_global_map", False)
        if not is_global_map:
            passes_done += 1

        log_lines.append(result["log_msg"])
        log_text = "\n".join(log_lines[-80:])

        frame_img = result["current_frame_img"]  # None for the final global map

        submap_path = result.get("glb_path")
        global_path = result.get("global_snapshot_path")

        if submap_path or global_path:
            if is_global_map:
                label = f"FINAL GLOBAL MAP  ({os.path.basename(global_path)})"
            else:
                label = f"pass_{result['pass_number']:04d}  frame {frame_idx:04d}"
            value = _pack(submap_path, global_path)
            if not any(v == value for _, v in all_glbs):
                all_glbs.append((label, value))
            last_value = value

        submap_view = submap_path if submap_path and os.path.isfile(submap_path) else None
        global_view = global_path if global_path and os.path.isfile(global_path) else None

        radio_update = gr.update(choices=all_glbs, value=last_value)
        yield log_text, frame_img, submap_view, global_view, radio_update, _stats_text(total, passes_done)

    yield (
        "\n".join(log_lines[-80:]) + "\n[INFO] Processing complete.",
        None,
        submap_view,
        global_view,
        gr.update(choices=all_glbs, value=last_value),
        _stats_text(total, passes_done),
    )


_ROLE_CAPTIONS = {
    "new": "New frame",
    "recent": "Context — recent",
    "revisit": "Context — revisit (cube)",
    "map": "Keyframe",
}


def _read_image_entries(glb_path) -> list:
    """
    Return [(image_path, caption), ...] for the gallery.
    Prefers the role-tagged _images.json sidecar; falls back to the legacy
    plain _images.txt (no captions) for runs made before this format existed.
    """
    if not glb_path:
        return []

    json_path = glb_path.replace(".glb", "_images.json")
    if os.path.isfile(json_path):
        with open(json_path) as f:
            entries = json.load(f)
        result = []
        for e in entries:
            if not os.path.isfile(e["path"]):
                continue
            caption = _ROLE_CAPTIONS.get(e.get("role"), "")
            if "frame_idx" in e:
                caption = f"{caption} (frame {e['frame_idx']})" if caption else f"frame {e['frame_idx']}"
            result.append((e["path"], caption))
        return result

    txt_path = glb_path.replace(".glb", "_images.txt")
    if os.path.isfile(txt_path):
        with open(txt_path) as f:
            return [(l.strip(), "") for l in f if l.strip() and os.path.isfile(l.strip())]

    return []


def _load_glb_and_images(value):
    """Selecting a row: push the pair into both 3D viewers AND populate the gallery."""
    submap_path, global_path = _unpack(value)
    submap_view = submap_path if submap_path and os.path.isfile(submap_path) else None
    global_view = global_path if global_path and os.path.isfile(global_path) else None
    gallery = _read_image_entries(submap_path or global_path)
    return submap_view, global_view, gallery


# ---------------------------------------------------------------------------
# Past runs
# ---------------------------------------------------------------------------

def _scan_past_runs() -> list:
    """[(basename, fullpath), ...] for outputs/run_* dirs, newest first."""
    dirs = [d for d in glob.glob(os.path.join(_OUTPUTS_DIR, "run_*")) if os.path.isdir(d)]
    dirs.sort(reverse=True)
    return [(os.path.basename(d), d) for d in dirs]


def _refresh_past_runs():
    return gr.update(choices=_scan_past_runs())


def _load_run(run_dir):
    """
    Load a previously completed run directory as if it just finished processing.
    Returns the same 6-tuple shape _process yields:
        (log_text, frame_img, submap_viewer_value, global_map_viewer_value,
         glb_radio_update, stats_text)
    """
    if not run_dir or not os.path.isdir(run_dir):
        return "No run selected.", None, None, None, gr.update(), ""

    log_path = os.path.join(run_dir, "run_log.jsonl")
    frame_idx_by_glb = {}
    snapshot_by_glb = {}
    log_lines = []
    total_images = 0
    if os.path.isfile(log_path):
        with open(log_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "display" in entry:
                    log_lines.append(entry["display"])
                    m = re.search(r"Found (\d+) images", entry["display"])
                    if m:
                        total_images = int(m.group(1))
                if entry.get("glb_path"):
                    frame_idx_by_glb[entry["glb_path"]] = entry.get("frame_idx")
                    snapshot_by_glb[entry["glb_path"]] = entry.get("global_snapshot_path")

    pass_glbs = sorted(glob.glob(os.path.join(run_dir, "pass_*.glb")))
    global_map = os.path.join(run_dir, "global_map.glb")

    all_glbs: list[tuple[str, str]] = []
    for i, p in enumerate(pass_glbs):
        frame_idx = frame_idx_by_glb.get(p, "?")
        snap = snapshot_by_glb.get(p)
        all_glbs.append((f"pass_{i:04d}  frame {frame_idx}", _pack(p, snap)))

    last_value = None
    if os.path.isfile(global_map):
        all_glbs.append(
            (f"FINAL GLOBAL MAP  ({os.path.basename(global_map)})", _pack(None, global_map))
        )
        last_value = all_glbs[-1][1]
    elif all_glbs:
        last_value = all_glbs[-1][1]

    submap_path, global_path = _unpack(last_value)
    submap_view = submap_path if submap_path and os.path.isfile(submap_path) else None
    global_view = global_path if global_path and os.path.isfile(global_path) else None

    log_text = "\n".join(log_lines[-200:]) if log_lines else f"(no run_log.jsonl found in {run_dir})"
    stats_text = _stats_text(total_images or len(pass_glbs), len(pass_glbs))

    return (
        log_text, None, submap_view, global_view,
        gr.update(choices=all_glbs, value=last_value), stats_text,
    )


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

_MONO = "ui-monospace, 'Cascadia Mono', Consolas, 'Courier New', monospace"

_CSS = f"""
/* Full-bleed instrument panel */
.gradio-container {{ max-width: 100% !important; padding: 6px 14px !important; }}

/* Slim header strip */
#hdr {{ display: flex; align-items: baseline; gap: 18px; flex-wrap: wrap;
    padding: 2px 2px 6px 2px; border-bottom: 1px solid rgba(128,128,128,.25);
    margin-bottom: 6px; }}
#hdr .t {{ font-size: 17px; font-weight: 700; letter-spacing: .04em; }}
#hdr .chip {{ font-family: {_MONO}; font-size: 12px; opacity: .85; }}
#hdr .dot {{ display: inline-block; width: 9px; height: 9px; border-radius: 2px;
    margin-right: 5px; vertical-align: baseline; }}

/* Sidebar column stays narrow so the twin 3D viewers get the width */
#sidebar {{ max-width: 300px; }}

/* Processing log lives under the submap ledger in the main stage */
#log-box textarea {{ font-family: {_MONO}; font-size: 11.5px; line-height: 1.5; }}

/* Full-width image strip above the twin viewers, bordered like the ledger rows */
#submap-gallery {{ border: 1px solid rgba(128,128,128,.22); border-radius: 6px; padding: 4px; }}

/* True fullscreen for either 3D viewer via the Fullscreen API */
#model3d_viewer:fullscreen, #global_map_viewer:fullscreen {{
    width: 100vw !important; height: 100vh !important; background: #000; }}

/* Submap ledger: vertical, capped height, scrolls in place.
   Left rail reuses the 3D scene color language:
   blue = context/overlap, yellow = query sphere (global map). */
.glb-list .wrap {{ display: flex !important; flex-direction: column !important;
    flex-wrap: nowrap !important; gap: 3px !important;
    max-height: 260px; overflow-y: auto; padding-right: 6px; }}
.glb-list .wrap label {{ width: 100% !important; border: 1px solid rgba(128,128,128,.22) !important;
    border-left: 4px solid rgba(128,128,128,.35) !important;
    border-radius: 5px !important; padding: 5px 12px !important;
    font-family: {_MONO}; font-size: 12.5px; }}
.glb-list .wrap label:hover {{ border-left-color: #4c8dff !important; }}
.glb-list .wrap label.selected,
.glb-list .wrap label:has(input:checked) {{
    border-left-color: #4c8dff !important;
    background: rgba(76, 141, 255, .12) !important; }}
/* The FINAL global map row has no paired submap — its packed value starts "||" */
.glb-list .wrap label:has(input[value^="||"]) {{
    border-left-color: #ffc800 !important; font-weight: 700; }}
"""


def build_ui() -> gr.Blocks:
    # NOTE: the installed gradio==5.50.0 does NOT accept theme=/css= on
    # .launch() (that only works in Gradio 6.0+, despite what the deprecation
    # warning implies) — confirmed by testing: TypeError, unexpected keyword.
    # Keeping them on the Blocks() constructor; the warning is harmless until
    # gradio is actually upgraded past 6.0.
    with gr.Blocks(
        title="Streaming VGGT-Omega (V2)",
        theme=gr.themes.Soft(),
        css=_CSS,
    ) as demo:
        gr.HTML(
            "<div id='hdr'>"
            "<span class='t'>VGGT-OMEGA · STREAMING RECONSTRUCTION</span>"
            "<span class='chip'><span class='dot' style='background:#ff9600'></span>revisit (cube candidate)</span>"
            "<span class='chip'><span class='dot' style='background:#0064ff'></span>recent (temporal overlap)</span>"
            "<span class='chip'><span class='dot' style='background:#ff3232'></span>new frame</span>"
            "<span class='chip'><span class='dot' style='background:#ffc800'></span>query cube</span>"
            "</div>"
        )

        stats_box = gr.Markdown("**Images loaded:** —   |   **Passes:** —")

        # ── Past runs — load a completed session as if it just finished ─────
        with gr.Row():
            past_run_dropdown = gr.Dropdown(
                label="Past Runs", choices=_scan_past_runs(), value=None, scale=3,
            )
            refresh_runs_btn = gr.Button("↻ Refresh", scale=0)
            load_run_btn = gr.Button("Load Run", variant="secondary", scale=0)

        with gr.Row():
            # ── Sidebar: uploads, settings ───────────────────────────────────
            with gr.Column(scale=0, min_width=320, elem_id="sidebar"):
                input_video = gr.Video(label="Upload Video", interactive=True, height=160)
                input_images = gr.File(
                    file_count="multiple",
                    label="Upload Images (drag & drop multiple)",
                    interactive=True,
                    height=120,
                )
                video_sample_fps = gr.Slider(
                    minimum=0.5, maximum=2.0, value=1.0, step=0.1,
                    label="Video Sampling FPS",
                )
                resolution_dropdown = gr.Dropdown(
                    label="Resolution", choices=["256", "512"], value="256",
                )
                conf_thres = gr.Slider(
                    minimum=2, maximum=100, value=20, step=0.1,
                    label="Confidence Threshold (%)",
                )
                max_points_k = gr.Slider(
                    minimum=100, maximum=10000, value=300, step=100,
                    label="Max Points (K points)",
                )

                load_model_btn = gr.Button("⚙ Load Model", variant="secondary")
                model_status = gr.Markdown("Model not loaded yet.")

                # Single-slot toggle: only one of these is ever visible at a
                # time, so it reads as one button that swaps Start <-> Stop.
                start_btn = gr.Button(
                    "▶  Start", variant="primary", interactive=False, visible=True,
                )
                stop_btn = gr.Button("⏹  Stop", variant="stop", visible=False)

                frame_preview = gr.Image(
                    label="Current Frame", type="filepath", height=140,
                )

            # ── Main stage: gallery strip, twin viewers, submap ledger, log ──
            with gr.Column(scale=1):
                submap_gallery = gr.Gallery(
                    label="Images in Selected Row (New / Context-recent / Context-revisit / Keyframe)",
                    columns=8,
                    rows=1,
                    height=240,
                    object_fit="contain",
                    allow_preview=True,
                    elem_id="submap-gallery",
                )
                gallery_height_slider = gr.Slider(
                    minimum=120, maximum=600, value=240, step=20,
                    label="Gallery Height (px)",
                )

                with gr.Row(equal_height=True):
                    model3d = gr.Model3D(
                        label="Submap (this step)", height=620,
                        elem_id="model3d_viewer",
                    )
                    global_map_viewer = gr.Model3D(
                        label="Global Map (growth this step — lime = added/refreshed)", height=620,
                        elem_id="global_map_viewer",
                    )

                with gr.Row():
                    fullscreen_btn = gr.Button("⛶ Fullscreen Submap", scale=0)
                    fullscreen_global_btn = gr.Button("⛶ Fullscreen Global Map", scale=0)
                    gr.Markdown(
                        "**Saved Submaps** — click a row (or use ↑/↓ arrow keys once focused) "
                        "to instantly load both viewers + images"
                    )

                glb_radio = gr.Radio(
                    label="",
                    choices=[],
                    value=None,
                    interactive=True,
                    elem_classes=["glb-list"],
                )

                log_box = gr.Textbox(
                    label="Processing Log", interactive=False,
                    lines=12, max_lines=12,
                    elem_id="log-box",
                )

        # ── Event wiring ────────────────────────────────────────────────────
        # Flip to "Stop" the instant Start is clicked (fast, independent of
        # the long-running _process call below).
        start_btn.click(
            fn=lambda: (gr.update(visible=False), gr.update(visible=True)),
            outputs=[start_btn, stop_btn],
        )
        run_event = start_btn.click(
            fn=_process,
            inputs=[
                input_video, input_images, video_sample_fps,
                resolution_dropdown, conf_thres, max_points_k,
            ],
            outputs=[log_box, frame_preview, model3d, global_map_viewer, glb_radio, stats_box],
            show_progress="hidden",
        )
        # Flip back to "Start" once processing finishes on its own (only fires
        # on natural completion, not when cancelled by Stop below).
        run_event.then(
            fn=lambda: (gr.update(visible=True), gr.update(visible=False)),
            outputs=[start_btn, stop_btn],
        )

        stop_btn.click(fn=None, cancels=[run_event])
        stop_btn.click(
            fn=lambda: (gr.update(visible=True), gr.update(visible=False)),
            outputs=[start_btn, stop_btn],
        )

        load_model_btn.click(
            fn=_load_model_ui,
            outputs=[load_model_btn, model_status, start_btn],
        )

        # Selecting a row — by click OR arrow keys (native radio-group behavior
        # fires .change() on every arrow-key move once a radio is focused) —
        # instantly loads both 3D viewers + the gallery. No separate Load step,
        # so comparing across timestamps is just click/arrow, click/arrow.
        glb_radio.change(
            fn=_load_glb_and_images,
            inputs=[glb_radio],
            outputs=[model3d, global_map_viewer, submap_gallery],
            show_progress="hidden",
        )

        # Gallery height slider — resizes the strip live, no server round-trip needed
        gallery_height_slider.change(
            fn=lambda h: gr.update(height=int(h)),
            inputs=[gallery_height_slider],
            outputs=[submap_gallery],
        )

        # Fullscreen — pure client-side, no server round-trip
        fullscreen_btn.click(
            fn=None,
            js="""
            () => {
                const el = document.getElementById('model3d_viewer');
                if (el) (el.requestFullscreen || el.webkitRequestFullscreen)?.call(el);
            }
            """,
        )
        fullscreen_global_btn.click(
            fn=None,
            js="""
            () => {
                const el = document.getElementById('global_map_viewer');
                if (el) (el.requestFullscreen || el.webkitRequestFullscreen)?.call(el);
            }
            """,
        )

        # Past runs
        refresh_runs_btn.click(fn=_refresh_past_runs, outputs=[past_run_dropdown])
        load_run_btn.click(
            fn=_load_run,
            inputs=[past_run_dropdown],
            outputs=[log_box, frame_preview, model3d, global_map_viewer, glb_radio, stats_box],
        )

        # Arrow-key navigation for the submap ledger. The radio <input>s never
        # actually receive keyboard focus in this layout, so native browser
        # radio-group arrow behavior doesn't fire — instead, while the mouse
        # is over the list, ArrowUp/ArrowDown move the selection directly and
        # .click() the target input so Gradio's own change handler still fires.
        demo.load(
            fn=None,
            js="""
            () => {
                const list = document.querySelector('.glb-list');
                if (!list) return;
                let hovering = false;
                list.addEventListener('mouseenter', () => { hovering = true; });
                list.addEventListener('mouseleave', () => { hovering = false; });
                document.addEventListener('keydown', (e) => {
                    if (!hovering) return;
                    if (e.key !== 'ArrowUp' && e.key !== 'ArrowDown') return;
                    e.preventDefault();
                    const inputs = Array.from(list.querySelectorAll('input[type=radio]'));
                    if (inputs.length === 0) return;
                    let idx = inputs.findIndex((i) => i.checked);
                    if (idx === -1) {
                        idx = 0;
                    } else if (e.key === 'ArrowDown') {
                        idx = Math.min(idx + 1, inputs.length - 1);
                    } else {
                        idx = Math.max(idx - 1, 0);
                    }
                    inputs[idx].click();
                    inputs[idx].scrollIntoView({ block: 'nearest' });
                });
            }
            """,
        )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    global _checkpoint_path

    parser = argparse.ArgumentParser(description="Streaming VGGT-Omega Demo V2")
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to .pt checkpoint. If omitted, downloads from HuggingFace.",
    )
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    _checkpoint_path = args.checkpoint
    # Model is NOT loaded here — loading it costs GPU time/VRAM you don't want
    # to pay just to browse past runs. It loads lazily: either explicitly via
    # the "Load Model" button, or automatically the first time Start is clicked.

    t0 = time.perf_counter()
    demo = build_ui()
    log.info(f"UI built in {time.perf_counter() - t0:.3f}s")

    # Allow the submap list / gallery / load button to respond immediately
    # even while the long-running _process generator is still streaming —
    # otherwise they'd queue behind it and appear to freeze.
    demo.queue(default_concurrency_limit=10)

    log.info(f"Launching server on port {args.port} ({time.perf_counter() - t0:.3f}s since UI build started)...")
    demo.launch(server_name="0.0.0.0", server_port=args.port)


if __name__ == "__main__":
    main()
