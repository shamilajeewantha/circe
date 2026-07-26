"""Gradio UI wrapping the dataset_annotation pipeline - one tab per stage (01/03/04/06/05), each
with its own real CLI options exposed as controls, a live-streamed log panel (subprocess stdout/
stderr captured line-by-line - explicit instruction: "all the logs that get printed on terminal
must be visible on ui as well"), and an on-demand output gallery over that stage's own qc/ folder.

Runs natively in the WSL yolo_det_py312 conda env (explicit instruction - launched there directly
by the user), so every stage script is reachable as a same-environment subprocess
(sys.executable <script>.py <flags>) - no Windows<->WSL bridging needed. Stage 2
(02_propose_regions_dumb.py, SAM2) is deliberately NOT wired in here - explicit instruction: it's
no longer part of the active pipeline. Left untouched on disk, just not a tab here.

Follows the Gradio conventions established in circe_v1/optical_flow_control/main.py (the repo's
actively-maintained Gradio reference): single gr.Blocks, gr.Tabs() per feature area, default theme
(only a couple of elem_id color overrides), a SINGLETON controller object owning all mutable
state (every click handler is a thin wrapper around it, holds no state of its own), gr.Timer-based
polling driven by one shared render function, atexit/SIGTERM cleanup, applog.py for the app's own
log (separate from each stage script's own circe_datasets/<dataset>/run_log.txt).

A "Dataset" dropdown at the top selects which dataset (auto-discovered from datasets/<name>/)
every tab below operates on - --src/--out are computed per-dataset and passed explicitly on every
run, not left to each script's own hardcoded default. Stage 01 is the one exception that isn't a
single fixed script: which "01"-equivalent working-copy builder runs depends on the dataset (see
DATASET_STAGE01_SCRIPT) since each raw download has a different layout.

Usage
-----
    conda activate yolo_det_py312
    cd dataset_annotation
    python gradio_app.py
    # open the printed URL - server_name="0.0.0.0" so WSL2's localhost-forwarding reaches it from
    # a browser on the Windows host regardless of WSL's internal IP.
"""
import atexit
import importlib.util
import json
import signal
import subprocess
import sys
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gradio as gr
from PIL import Image

from annotate_common import BoltAnnotation, BoltBox, CLASSES, draw_visualization
from applog import close_logging, get_logger

log = get_logger("gradio_app")

HERE = Path(__file__).parent
PYTHON = sys.executable
IMG_SUFFIXES = {".jpg", ".jpeg", ".png"}
THUMB_DIR = HERE / "ui_logs" / "thumbs"
THUMB_MAX_DIM = 640  # gallery grid only needs to be big enough to spot defects at a glance - QC
                      # originals are often multi-MB full-res photos; sending 60 of those to the
                      # browser untouched is exactly what was making the gallery "fucking slow" /
                      # "super laggy" - downsizing here is the actual fix, not just a nice-to-have

MAX_LOG_LINES = 2000    # cap per-stage buffer so a very long run doesn't grow unbounded in memory
MAX_GALLERY_IMAGES = 60  # loading hundreds of full images into the browser at once is slow and
                          # rarely more useful for QC-by-eye than a representative sample

# Models actually tried/considered against this pipeline (see 04_annotate_with_gemini.py's own
# "Models tried/considered this session" comment block for the tested/untested notes behind each).
# allow_custom_value=True on the dropdown below means any other --model string still works too -
# this list is a convenience shortlist, not a hard restriction.
GEMINI_MODEL_CHOICES = [
    "gemini-robotics-er-1.6-preview",
    "gemini-robotics-er-1.5-preview",
    "gemini-3.5-flash",
    "gemini-3.1-flash-lite",
]

# Pipeline order: 01 (one-time) -> 03 (SAM3 regions) -> 04 (Gemini, costs quota) -> 06 (vote
# consensus, only relevant after 04 --passes > 1) -> 05 (SAM3 tightening + final dataset/).
# 03/04/05/06 are already dataset-agnostic (take --src/--out, no dataset-specific logic) - every
# handler below passes those explicitly per the selected dataset. Stage 01 is NOT generic: each
# dataset's raw download has a different layout (npu_bolt = flat + CAD-* filter, bolt-defects__ext-
# sdnet2025 = 3 nested COCO subfolders to flatten, no filter needed) - see
# DATASET_STAGE01_SCRIPT below for which "01" script runs per dataset.
STAGES = {
    "orig": {"label": "Original Labels", "script": None},  # resolved per-dataset, see below
    "01": {"label": "01 · Dataset Cleanup", "script": None},  # resolved per-dataset, see below
    "03": {"label": "03 · SAM3 Regions", "script": "03_propose_regions_concept.py"},
    "04": {"label": "04 · Gemini Annotate", "script": "04_annotate_with_gemini.py"},
    "06": {"label": "06 · Vote Consensus", "script": "06_vote_consensus.py"},
    "05": {"label": "05 · Box Tightening", "script": "05_tighten_boxes.py"},
}

# Which "01"-equivalent working-copy builder runs for each dataset - add an entry here whenever a
# new dataset is dropped into datasets/. No fallback/guess on purpose: an unmapped dataset means
# nobody has written its builder yet, and guessing wrong would silently corrupt a working copy.
DATASET_STAGE01_SCRIPT = {
    "npu_bolt": "01_remove_cad.py",
    "bolt-defects__ext-sdnet2025": "01_build_working_copy_bolt_defects.py",
}

# Which script draws a dataset's OWN original ground-truth annotations, as shipped (no remapping
# to this pipeline's bolt_ok/bolt_defective/bolt_corroded classes) - different per dataset because
# the raw annotation FORMAT differs (npu_bolt = one Pascal-VOC XML per image, bolt-defects__ext-
# sdnet2025 = two COCO JSON files covering many images each). Both scripts share the same
# --datasets-dir/--out/--limit flag names, so the caller below can build args identically either
# way. No fallback, same reasoning as DATASET_STAGE01_SCRIPT above.
DATASET_ORIGINAL_LABELS_SCRIPT = {
    "npu_bolt": "_visualize_original_annotations.py",
    "bolt-defects__ext-sdnet2025": "_visualize_original_annotations_bolt_defects.py",
}


def list_available_datasets() -> List[str]:
    """Auto-discovered from datasets/<name>/ (the read-only source root) - NOT circe_datasets/,
    since a brand-new dataset only gets a circe_datasets/ entry after stage 01 has already run once
    for it, which would make it invisible in the selector right when you need to pick it to run
    stage 01 in the first place."""
    root = HERE / "datasets"
    if not root.is_dir():
        return ["npu_bolt"]
    names = sorted(p.name for p in root.iterdir() if p.is_dir())
    return names or ["npu_bolt"]


def dataset_out_root(dataset: str) -> Path:
    return HERE / "circe_datasets" / dataset


def dataset_working_images(dataset: str) -> Path:
    return dataset_out_root(dataset) / "working_images"


class PipelineController:
    """Singleton - owns ALL mutable state (running subprocesses, per-stage log buffers). Every
    Gradio event handler below is a thin wrapper calling a method here and returning
    gr.update(...) - handlers hold no state of their own, matching
    circe_v1/optical_flow_control/main.py's DroneController convention."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.processes: Dict[str, subprocess.Popen] = {}
        self.logs: Dict[str, List[str]] = {key: [] for key in STAGES}
        self.running: Dict[str, bool] = {key: False for key in STAGES}

    def is_running(self, stage_key: str) -> bool:
        with self.lock:
            return self.running.get(stage_key, False)

    def get_log_text(self, stage_key: str) -> str:
        with self.lock:
            return "\n".join(self.logs.get(stage_key, []))

    def run_stage(self, stage_key: str, args: List[str], script: Optional[str] = None) -> None:
        """Starts the stage's script as a subprocess in a background thread and returns
        immediately - never blocks the calling Gradio handler (matches the "long-running work
        happens in background threads, started by a handler which returns immediately" convention).
        A second click while already running is a no-op (logged), not a duplicate launch.

        script: overrides STAGES[stage_key]["script"] - only stage "01" needs this (which builder
        runs depends on which dataset is selected, see DATASET_STAGE01_SCRIPT); every other stage
        always uses its one fixed script regardless of dataset."""
        with self.lock:
            if self.running.get(stage_key):
                log.warning("Stage %s already running - ignoring duplicate start request.", stage_key)
                return
            self.running[stage_key] = True
            self.logs[stage_key] = []

        def _worker() -> None:
            script_name = script or STAGES[stage_key]["script"]
            cmd = [PYTHON, str(HERE / script_name), *args]
            log.info("Starting stage %s: %s", stage_key, " ".join(cmd))
            self._append_log(stage_key, f"$ {' '.join(cmd)}")
            try:
                proc = subprocess.Popen(
                    cmd, cwd=str(HERE), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1, encoding="utf-8", errors="replace",
                )
                with self.lock:
                    self.processes[stage_key] = proc
                # Every stage script already logs to both its own run_log.txt AND stdout
                # (logging.basicConfig with a StreamHandler) - capturing stdout here naturally
                # surfaces "everything printed on terminal" with zero changes to those scripts.
                for line in proc.stdout:
                    self._append_log(stage_key, line.rstrip("\n"))
                proc.wait()
                self._append_log(stage_key, f"[process exited with code {proc.returncode}]")
                log.info("Stage %s finished, exit code %s", stage_key, proc.returncode)
            except Exception as e:  # noqa: BLE001 - a launch failure must not crash the whole UI
                self._append_log(stage_key, f"[FAILED TO START: {e!r}]")
                log.error("Stage %s failed to start: %r", stage_key, e)
            finally:
                with self.lock:
                    self.running[stage_key] = False
                    self.processes.pop(stage_key, None)

        threading.Thread(target=_worker, daemon=True).start()

    def stop_stage(self, stage_key: str) -> None:
        with self.lock:
            proc = self.processes.get(stage_key)
        if proc is not None and proc.poll() is None:
            log.info("Stopping stage %s (pid=%s) by user request", stage_key, proc.pid)
            proc.terminate()
            self._append_log(stage_key, "[stopped by user]")

    def _append_log(self, stage_key: str, line: str) -> None:
        with self.lock:
            buf = self.logs.setdefault(stage_key, [])
            buf.append(line)
            if len(buf) > MAX_LOG_LINES:
                del buf[: len(buf) - MAX_LOG_LINES]

    def shutdown(self) -> None:
        with self.lock:
            procs = list(self.processes.items())
        for stage_key, proc in procs:
            try:
                if proc.poll() is None:
                    log.info("Shutdown: terminating still-running stage %s (pid=%s)",
                              stage_key, proc.pid)
                    proc.terminate()
            except Exception:  # noqa: BLE001 - cleanup must never raise on the way out
                pass


ctrl = PipelineController()


def _thumbnail_path(src: Path, index: int, total: int) -> str:
    """Downsized, cached copy of src for gallery display. Cached under ui_logs/thumbs/, mirroring
    src's path relative to this script (so same-named files from different qc/ subfolders never
    collide), keyed by src's mtime so a re-run/edit invalidates the cached thumbnail. Falls back to
    the original path if thumbnailing fails for any reason - a slow/broken thumbnail must never
    hide an image from QC."""
    rel = src.resolve().relative_to(HERE.resolve())
    dst = (THUMB_DIR / rel).with_suffix(".jpg")
    if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
        return str(dst)
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(src) as im:
            im = im.convert("RGB")
            im.thumbnail((THUMB_MAX_DIM, THUMB_MAX_DIM), Image.LANCZOS)
            im.save(dst, "JPEG", quality=82)
        log.info("[%d/%d] thumbnailed %s -> %s", index, total, src.name, dst)
        return str(dst)
    except Exception as e:  # noqa: BLE001 - a thumbnail failure must not hide the image from QC
        log.error("Thumbnailing failed for %s: %r - falling back to original", src, e)
        return str(src)


def _load_gallery(qc_dir: Path) -> List[str]:
    if not qc_dir.exists():
        return []
    files = sorted(p for p in qc_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_SUFFIXES)
    files = files[:MAX_GALLERY_IMAGES]
    return [_thumbnail_path(p, i + 1, len(files)) for i, p in enumerate(files)]


def _available_pass_choices(dataset: str) -> List[str]:
    """Stage 04's gallery needs to pick between qc/gemini_raw/ (single-pass) and
    qc/gemini_raw_pass{N}/ (multi-pass, one folder per pass, N auto-detected from whatever
    actually exists on disk - never hardcoded)."""
    choices = ["single-pass"]
    qc_root = dataset_out_root(dataset) / "qc"
    if qc_root.exists():
        for d in sorted(qc_root.glob("gemini_raw_pass*"),
                         key=lambda p: int(p.name.replace("gemini_raw_pass", ""))):
            choices.append(d.name.replace("gemini_raw_pass", ""))
    return choices


def _load_gallery_for_pass(dataset: str, pass_value: str) -> List[str]:
    qc_root = dataset_out_root(dataset) / "qc"
    if pass_value == "single-pass":
        return _load_gallery(qc_root / "gemini_raw")
    return _load_gallery(qc_root / f"gemini_raw_pass{pass_value}")


def _load_gemini_prompts() -> Tuple[str, str]:
    """Loads SYSTEM_PROMPT + BATCH_USER_PROMPT straight from 04_annotate_with_gemini.py so the UI
    always shows whatever is ACTUALLY active (SYSTEM_PROMPT is built via chained .replace() calls
    over SYSTEM_PROMPT_V1/V2/V3 - not a static string anywhere in the file - so this has to import
    the module and read the real resulting value, not regex the source). Re-imports fresh on every
    call (module filename starts with a digit, can't just `import`) so a "Refresh" click reflects
    an on-disk edit without restarting the whole UI. Runs in the same env/interpreter that already
    runs 04_annotate_with_gemini.py as a subprocess, so its imports (google-genai, dotenv) are
    already satisfied here too."""
    try:
        spec = importlib.util.spec_from_file_location(
            "gemini_annotate_prompts_view", HERE / "04_annotate_with_gemini.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.SYSTEM_PROMPT, mod.BATCH_USER_PROMPT
    except Exception as e:  # noqa: BLE001 - this is a read-only viewer, must never crash the UI
        log.error("Failed to load prompts from 04_annotate_with_gemini.py: %r", e)
        err = f"[Failed to load prompt from 04_annotate_with_gemini.py: {e!r}]"
        return err, err


# --- Manual annotation correction (no API calls - pure local file editing) ---------------------
# For two real gaps found this session doing QC by eye: (1) Gemini misses a real fastener
# entirely - fixed by picking a SAM candidate REGION (already proposed by stage 03, zero new
# inference) and hand-picking its label, rather than hand-drawing a box pixel-by-pixel; (2) Gemini
# assigns the wrong label to a box it DID find - fixed by editing that box's label directly.
# Reads/writes cache/raw_gemini/<stem>.json (the SAME file 05_tighten_boxes.py reads as its source
# of truth) - re-run stage 05 after saving to regenerate the final dataset/+qc/visualized/ with
# these corrections folded in; 05 has no per-image skip logic, it always reprocesses everything
# in cache/raw_gemini/ from scratch, so nothing extra is needed to make it pick up an edit.

CORRECTION_PREVIEW_DIR = HERE / "ui_logs" / "manual_correction_preview"


def _correction_raw_json_path(dataset: str, stem: str) -> Path:
    return dataset_out_root(dataset) / "cache" / "raw_gemini" / f"{stem}.json"


def _correction_sam_json_path(dataset: str, stem: str) -> Path:
    return dataset_out_root(dataset) / "cache" / "sam_regions_concept" / f"{stem}.json"


def _find_image_file(folder: Path, stem: str) -> Optional[Path]:
    for ext in IMG_SUFFIXES:
        p = folder / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def _list_correction_stems(dataset: str) -> List[str]:
    """Only images Gemini has already annotated at least once are correctable here - this tool
    fixes/extends an existing annotation, it doesn't create one from nothing (that's stage 04's
    job)."""
    raw_dir = dataset_out_root(dataset) / "cache" / "raw_gemini"
    if not raw_dir.is_dir():
        return []
    return sorted(p.stem for p in raw_dir.glob("*.json"))


def _load_boxes_rows(dataset: str, stem: str) -> List[List]:
    """Rows for the gr.Dataframe: [label, ymin, xmin, ymax, xmax]."""
    path = _correction_raw_json_path(dataset, stem)
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return [[b["label"], *b["box_2d"]] for b in data.get("boxes", [])]


def _rows_to_boxes(rows) -> List[BoltBox]:
    """Converts the dataframe's current value (however Gradio hands it back - list of lists or a
    pandas DataFrame depending on version) into validated BoltBox objects. Raises ValueError with
    a human-readable message on anything invalid - the caller surfaces that straight into the log
    panel rather than silently dropping or guess-fixing a bad row."""
    if hasattr(rows, "values"):  # pandas DataFrame
        rows = rows.values.tolist()
    boxes = []
    for i, row in enumerate(rows or []):
        label, ymin, xmin, ymax, xmax = row
        label = str(label).strip()
        if label not in CLASSES:
            raise ValueError(f"Row {i + 1}: '{label}' is not a valid class - must be one of {CLASSES}")
        box_2d = [int(round(float(v))) for v in (ymin, xmin, ymax, xmax)]
        boxes.append(BoltBox(box_2d=box_2d, label=label))
    return boxes


def _render_correction_preview(dataset: str, stem: str, rows) -> Optional[str]:
    """Redraws a preview image from whatever's CURRENTLY in the dataframe (including unsaved
    staged edits/additions) - so the preview always reflects what "Save" would actually write, not
    stale disk content. Written to ui_logs/ (scratch, gitignored), never qc/gemini_raw/ itself -
    that folder is stage 04's own output, this preview must never be mistaken for it."""
    image_path = _find_image_file(dataset_working_images(dataset), stem)
    if image_path is None:
        return None
    try:
        boxes = _rows_to_boxes(rows)
    except ValueError:
        boxes = []  # invalid in-progress edit - preview just skips drawing rather than crashing
    ann = BoltAnnotation(boxes=boxes)
    out_dir = CORRECTION_PREVIEW_DIR / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{stem}.jpg"
    draw_visualization(image_path, ann, out_path)
    return str(out_path)


def _polygon_area(polygon: List[List[float]]) -> float:
    n = len(polygon)
    area = 0.0
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2


def _point_in_polygon(x: float, y: float, polygon: List[List[float]]) -> bool:
    """Standard ray-casting test. polygon and (x, y) must already be in the same coordinate space
    (both 0-1000 normalized here, see caller)."""
    inside = False
    n = len(polygon)
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def _polygon_to_box_2d(polygon: List[List[float]]) -> List[int]:
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    return [round(min(ys)), round(min(xs)), round(max(ys)), round(max(xs))]  # ymin,xmin,ymax,xmax


CSS = """
#run-btn { background: #2f6f2f !important; color: white !important; }
#stop-btn { background: #7a2020 !important; color: white !important; }
.resizable-gallery { resize: vertical; overflow: auto; min-height: 400px; }
.resizable-gallery .grid-wrap { height: 100% !important; }
"""

with gr.Blocks(title="Dataset Annotation Pipeline") as demo:
    gr.Markdown("# `dataset_annotation` pipeline")
    dataset_dd = gr.Dropdown(
        label="Dataset (drives every tab below - --src/--out are computed from this, not each "
              "script's own default)",
        choices=list_available_datasets(),
        value=list_available_datasets()[0],
    )
    log_boxes: Dict[str, gr.Textbox] = {}

    with gr.Tabs():
        with gr.Tab(STAGES["orig"]["label"]):
            orig_script_info = gr.Markdown()

            def _describe_orig(dataset: str) -> str:
                script = DATASET_ORIGINAL_LABELS_SCRIPT.get(dataset)
                if script is None:
                    return (f"**No original-labels viewer registered for `{dataset}`** - add an "
                            f"entry to `DATASET_ORIGINAL_LABELS_SCRIPT` in `gradio_app.py`. Run "
                            f"disabled.")
                return (f"Draws `datasets/{dataset}/`'s OWN original ground-truth annotations, "
                        f"exactly as shipped - NOT remapped to this pipeline's "
                        f"bolt_ok/bolt_defective/bolt_corroded classes, purely for visual "
                        f"reference. Read-only source, never written to. Runs `{script}`, writing "
                        f"to `circe_datasets/{dataset}/qc/original_labels/`.")

            with gr.Row():
                limit_orig = gr.Number(label="--limit (0 = all)", value=0, precision=0)
                run_orig_btn = gr.Button("Run", elem_id="run-btn")
            log_orig = gr.Textbox(label="Log", lines=16, max_lines=16, interactive=False, autoscroll=True)
            log_boxes["orig"] = log_orig
            refresh_orig_btn = gr.Button("Refresh gallery")
            gallery_orig = gr.Gallery(label="qc/original_labels/", columns=4, height=800,
                                       elem_classes=["resizable-gallery"], preview=True,
                                       object_fit="contain")
            demo.load(_describe_orig, inputs=[dataset_dd], outputs=[orig_script_info])
            dataset_dd.change(_describe_orig, inputs=[dataset_dd], outputs=[orig_script_info])

            def _run_orig(dataset: str, limit: float):
                script = DATASET_ORIGINAL_LABELS_SCRIPT.get(dataset)
                if script is None:
                    ctrl._append_log("orig", f"[No original-labels viewer registered for "
                                              f"'{dataset}' - add it to "
                                              f"DATASET_ORIGINAL_LABELS_SCRIPT in gradio_app.py. "
                                              f"Not running.]")
                    return
                args = ["--datasets-dir", str(HERE / "datasets" / dataset),
                        "--out", str(dataset_out_root(dataset) / "qc" / "original_labels")]
                if limit:
                    args += ["--limit", str(int(limit))]
                ctrl.run_stage("orig", args, script=script)

            run_orig_btn.click(_run_orig, inputs=[dataset_dd, limit_orig], outputs=[])
            refresh_orig_btn.click(
                lambda dataset: _load_gallery(dataset_out_root(dataset) / "qc" / "original_labels"),
                inputs=[dataset_dd], outputs=[gallery_orig],
            )

        with gr.Tab(STAGES["01"]["label"]):
            stage01_script_info = gr.Markdown()

            def _describe_stage01(dataset: str) -> str:
                script = DATASET_STAGE01_SCRIPT.get(dataset)
                if script is None:
                    return (f"**No dataset-cleanup script registered for `{dataset}`** - add an "
                            f"entry to `DATASET_STAGE01_SCRIPT` in `gradio_app.py`. Run disabled.")
                return (f"Builds the working copy every downstream stage reads from - the "
                        f"read-only `datasets/{dataset}/` source is never written to. Runs "
                        f"`{script}`, writing to `circe_datasets/{dataset}/working_images/`. Pure "
                        f"Python, no SAM/Gemini deps - safe to re-run any time (skips files "
                        f"already copied).")

            run01_btn = gr.Button("Run", elem_id="run-btn")
            log01 = gr.Textbox(label="Log", lines=16, max_lines=16, interactive=False, autoscroll=True)
            log_boxes["01"] = log01
            demo.load(_describe_stage01, inputs=[dataset_dd], outputs=[stage01_script_info])
            dataset_dd.change(_describe_stage01, inputs=[dataset_dd], outputs=[stage01_script_info])

            def _run01(dataset: str):
                script = DATASET_STAGE01_SCRIPT.get(dataset)
                if script is None:
                    ctrl._append_log("01", f"[No dataset-cleanup script registered for "
                                            f"'{dataset}' - add it to DATASET_STAGE01_SCRIPT in "
                                            f"gradio_app.py. Not running.]")
                    return
                ctrl.run_stage("01", [], script=script)

            run01_btn.click(_run01, inputs=[dataset_dd], outputs=[])

        with gr.Tab(STAGES["03"]["label"]):
            gr.Markdown("Local/free - needs SAM3 (this WSL env). Writes "
                        "`qc/sam_proposals_concept/` + the hint cache stage 04 reads.")
            with gr.Row():
                concepts03 = gr.Textbox(label="--concepts", value="bolt,screw,nut,fastener,rivet")
                conf03 = gr.Slider(label="--confidence-threshold", minimum=0.0, maximum=1.0,
                                    value=0.3, step=0.05)
            with gr.Row():
                limit03 = gr.Number(label="--limit (0 = all)", value=0, precision=0)
                force03 = gr.Checkbox(label="--force (recompute even if already cached)")
            with gr.Row():
                run03_btn = gr.Button("Run", elem_id="run-btn")
                stop03_btn = gr.Button("Stop", elem_id="stop-btn")
            log03 = gr.Textbox(label="Log", lines=16, max_lines=16, interactive=False, autoscroll=True)
            log_boxes["03"] = log03
            refresh03_btn = gr.Button("Refresh gallery")
            gallery03 = gr.Gallery(label="qc/sam_proposals_concept/", columns=4, height=800,
                                    elem_classes=["resizable-gallery"], preview=True,
                                    object_fit="contain")

            def _run03(dataset: str, concepts: str, conf: float, limit: float, force: bool):
                args = ["--src", str(dataset_working_images(dataset)),
                        "--out", str(dataset_out_root(dataset)),
                        "--concepts", concepts, "--confidence-threshold", str(conf)]
                if limit:
                    args += ["--limit", str(int(limit))]
                if force:
                    args.append("--force")
                ctrl.run_stage("03", args)

            run03_btn.click(_run03, inputs=[dataset_dd, concepts03, conf03, limit03, force03], outputs=[])
            stop03_btn.click(lambda: ctrl.stop_stage("03"), outputs=[])
            refresh03_btn.click(
                lambda dataset: _load_gallery(dataset_out_root(dataset) / "qc" / "sam_proposals_concept"),
                inputs=[dataset_dd], outputs=[gallery03],
            )

        with gr.Tab(STAGES["04"]["label"]):
            gr.Markdown("**Costs real Gemini API quota.** Auto-chunked internally to stay under "
                        "the real input-token cap - pick any `--limit`, chunking happens "
                        "automatically. `--passes > 1` runs independent passes for later majority "
                        "voting (stage 06) - each pass is stored separately for QC, never "
                        "overwritten. A per-run cost estimate (chunks x passes) is logged before "
                        "any API call is made - watch the log panel below.")
            with gr.Row():
                model04 = gr.Dropdown(label="--model", choices=GEMINI_MODEL_CHOICES,
                                       value="gemini-robotics-er-1.6-preview",
                                       allow_custom_value=True)
                limit04 = gr.Number(label="--limit (0 = all)", value=0, precision=0)
                passes04 = gr.Number(label="--passes", value=1, precision=0, minimum=1)
            with gr.Row():
                retries04 = gr.Number(label="--retries", value=3, precision=0)
                backoff04 = gr.Number(label="--backoff", value=3.0)
            with gr.Row():
                skip_hints04 = gr.Checkbox(label="--skip-region-hints")
                clean04 = gr.Checkbox(label="--clean (DANGER - forces full re-annotation)")
            with gr.Accordion("View active Gemini prompts (SYSTEM_PROMPT + BATCH_USER_PROMPT)",
                               open=False):
                refresh_prompts04_btn = gr.Button("Reload from 04_annotate_with_gemini.py")
                system_prompt04 = gr.Textbox(label="SYSTEM_PROMPT (sent as system_instruction)",
                                              lines=18, max_lines=40, interactive=False)
                batch_user_prompt04 = gr.Textbox(label="BATCH_USER_PROMPT (sent as the request text, "
                                                        "same for every call)",
                                                  lines=8, max_lines=20, interactive=False)
            with gr.Row():
                run04_btn = gr.Button("Run", elem_id="run-btn")
                stop04_btn = gr.Button("Stop", elem_id="stop-btn")
            log04 = gr.Textbox(label="Log (cost estimate + real per-chunk token/MB usage appear here)",
                                lines=20, max_lines=20, interactive=False, autoscroll=True)
            log_boxes["04"] = log04
            with gr.Row():
                pass_select04 = gr.Dropdown(label="View pass", choices=["single-pass"],
                                             value="single-pass")
                refresh04_btn = gr.Button("Refresh gallery")
            gallery04 = gr.Gallery(label="qc/gemini_raw[_pass{N}]/", columns=4, height=800,
                                    elem_classes=["resizable-gallery"], preview=True,
                                    object_fit="contain")

            def _run04(dataset: str, model: str, limit: float, passes: float, retries: float,
                       backoff: float, skip_hints: bool, clean: bool):
                args = ["--src", str(dataset_working_images(dataset)),
                        "--out", str(dataset_out_root(dataset)),
                        "--model", model, "--retries", str(int(retries)), "--backoff", str(backoff),
                        "--passes", str(int(passes))]
                if limit:
                    args += ["--limit", str(int(limit))]
                if skip_hints:
                    args.append("--skip-region-hints")
                if clean:
                    args.append("--clean")
                ctrl.run_stage("04", args)

            run04_btn.click(
                _run04,
                inputs=[dataset_dd, model04, limit04, passes04, retries04, backoff04, skip_hints04, clean04],
                outputs=[],
            )
            stop04_btn.click(lambda: ctrl.stop_stage("04"), outputs=[])
            refresh_prompts04_btn.click(_load_gemini_prompts,
                                         outputs=[system_prompt04, batch_user_prompt04])
            demo.load(_load_gemini_prompts, outputs=[system_prompt04, batch_user_prompt04])
            refresh04_btn.click(
                lambda dataset: gr.update(choices=_available_pass_choices(dataset)),
                inputs=[dataset_dd], outputs=[pass_select04],
            ).then(_load_gallery_for_pass, inputs=[dataset_dd, pass_select04], outputs=[gallery04])
            pass_select04.change(_load_gallery_for_pass, inputs=[dataset_dd, pass_select04], outputs=[gallery04])

        with gr.Tab(STAGES["06"]["label"]):
            gr.Markdown("Local/free - only relevant after stage 04 was run with `--passes > 1`. "
                        "Merges every `cache/raw_gemini_pass*/` into `cache/raw_gemini/` via "
                        "IoU-matched majority vote across passes; writes vote tallies (e.g. "
                        "\"4/5 detected - corroded:3 defective:0 ok:1\") onto each QC box.")
            with gr.Row():
                iou06 = gr.Slider(label="--iou-threshold", minimum=0.0, maximum=1.0, value=0.5,
                                   step=0.05)
                vote06 = gr.Number(label="--vote-threshold (0 = auto: majority)", value=0, precision=0)
            with gr.Row():
                limit06 = gr.Number(label="--limit (0 = all)", value=0, precision=0)
                clean06 = gr.Checkbox(label="--clean")
            with gr.Row():
                run06_btn = gr.Button("Run", elem_id="run-btn")
                stop06_btn = gr.Button("Stop", elem_id="stop-btn")
            log06 = gr.Textbox(label="Log", lines=16, max_lines=16, interactive=False, autoscroll=True)
            log_boxes["06"] = log06
            refresh06_btn = gr.Button("Refresh gallery")
            gallery06 = gr.Gallery(label="qc/gemini_consensus/ (vote tallies drawn on each box)",
                                    columns=4, height=800, elem_classes=["resizable-gallery"],
                                    preview=True, object_fit="contain")

            def _run06(dataset: str, iou: float, vote: float, limit: float, clean: bool):
                args = ["--src", str(dataset_working_images(dataset)),
                        "--out", str(dataset_out_root(dataset)),
                        "--iou-threshold", str(iou)]
                if vote:
                    args += ["--vote-threshold", str(int(vote))]
                if limit:
                    args += ["--limit", str(int(limit))]
                if clean:
                    args.append("--clean")
                ctrl.run_stage("06", args)

            run06_btn.click(_run06, inputs=[dataset_dd, iou06, vote06, limit06, clean06], outputs=[])
            stop06_btn.click(lambda: ctrl.stop_stage("06"), outputs=[])
            refresh06_btn.click(
                lambda dataset: _load_gallery(dataset_out_root(dataset) / "qc" / "gemini_consensus"),
                inputs=[dataset_dd], outputs=[gallery06],
            )

        with gr.Tab(STAGES["05"]["label"]):
            gr.Markdown("Local/free unless SAM3 tightening runs. Reads `cache/raw_gemini/` - "
                        "works identically whether that came from single-pass stage 04 directly "
                        "or from stage 06's post-vote consensus (same JSON schema either way, "
                        "zero changes needed here for multi-pass mode).")
            with gr.Row():
                skip_tighten05 = gr.Checkbox(label="--skip-tightening")
                limit05 = gr.Number(label="--limit (0 = all)", value=0, precision=0)
                clean05 = gr.Checkbox(label="--clean")
            with gr.Row():
                run05_btn = gr.Button("Run", elem_id="run-btn")
                stop05_btn = gr.Button("Stop", elem_id="stop-btn")
            log05 = gr.Textbox(label="Log", lines=16, max_lines=16, interactive=False, autoscroll=True)
            log_boxes["05"] = log05
            refresh05_btn = gr.Button("Refresh gallery")
            gallery05 = gr.Gallery(label="qc/visualized/ (FINAL - what dataset/labels/ contains)",
                                    columns=4, height=800, elem_classes=["resizable-gallery"],
                                    preview=True, object_fit="contain")

            def _run05(dataset: str, skip_tighten: bool, limit: float, clean: bool):
                args = ["--src", str(dataset_working_images(dataset)),
                        "--out", str(dataset_out_root(dataset))]
                if skip_tighten:
                    args.append("--skip-tightening")
                if limit:
                    args += ["--limit", str(int(limit))]
                if clean:
                    args.append("--clean")
                ctrl.run_stage("05", args)

            run05_btn.click(_run05, inputs=[dataset_dd, skip_tighten05, limit05, clean05], outputs=[])
            stop05_btn.click(lambda: ctrl.stop_stage("05"), outputs=[])
            refresh05_btn.click(
                lambda dataset: _load_gallery(dataset_out_root(dataset) / "qc" / "visualized"),
                inputs=[dataset_dd], outputs=[gallery05],
            )

        with gr.Tab("Manual Correction"):
            gr.Markdown("**No API calls** - pure local editing of `cache/raw_gemini/<stem>.json` "
                        "(the SAME file `05 · Box Tightening` reads). After saving, re-run "
                        "`05 · Box Tightening` above to fold corrections into the final "
                        "`dataset/` + `qc/visualized/` - it always reprocesses everything in "
                        "`cache/raw_gemini/` fresh, no extra flag needed to pick up an edit.\n\n"
                        "**New box Gemini missed**: click a highlighted region on the left image "
                        "(stage 03's SAM proposals) to auto-derive its bounding box from that "
                        "region's real outline, pick a label, click **Add box**.\n\n"
                        "**Wrong label on an existing box**: edit the `label` cell directly in the "
                        "table (must be exactly `bolt_ok` / `bolt_defective` / `bolt_corroded`) - "
                        "or delete the row if the whole box is a false positive (trash icon per "
                        "row). Nothing is written to disk until **Save corrections**.")
            with gr.Row():
                stem_dd = gr.Dropdown(label="Image (only images already annotated by stage 04 "
                                             "appear here)", choices=[])
                refresh_stems_btn = gr.Button("Refresh image list")
            with gr.Row():
                sam_overlay_img = gr.Image(
                    label="Click a SAM candidate region here (qc/sam_proposals_concept/)",
                    interactive=False, height=500,
                )
                preview_img = gr.Image(
                    label="Current annotation preview (includes unsaved staged edits)",
                    interactive=False, height=500,
                )
            pending_box_state = gr.State(None)  # a staged [ymin,xmin,ymax,xmax] awaiting Add, or None
            pending_info = gr.Markdown("No region selected yet.")
            with gr.Row():
                new_label_dd = gr.Dropdown(label="Label for new box", choices=CLASSES, value=CLASSES[0])
                add_box_btn = gr.Button("Add box")
            boxes_df = gr.Dataframe(
                headers=["label", "ymin", "xmin", "ymax", "xmax"],
                datatype=["str", "number", "number", "number", "number"],
                column_count=5, interactive=True,
            )
            with gr.Row():
                save_btn = gr.Button("Save corrections", elem_id="run-btn")
                correction_status = gr.Textbox(label="Status", interactive=False)

            def _refresh_stems(dataset: str):
                stems = _list_correction_stems(dataset)
                return gr.update(choices=stems, value=stems[0] if stems else None)

            def _on_stem_change(dataset: str, stem: str):
                if not stem:
                    return None, None, [], "No image selected."
                rows = _load_boxes_rows(dataset, stem)
                overlay_path = _find_image_file(
                    dataset_out_root(dataset) / "qc" / "sam_proposals_concept", stem
                )
                preview = _render_correction_preview(dataset, stem, rows)
                status = f"Loaded {len(rows)} box(es) for {stem}."
                if overlay_path is None:
                    status += (" No SAM regions cached for this image - run 03 . SAM3 Regions "
                                "first if you want click-to-add-box for it.")
                return (str(overlay_path) if overlay_path else None), preview, rows, status

            refresh_stems_btn.click(_refresh_stems, inputs=[dataset_dd], outputs=[stem_dd])
            dataset_dd.change(_refresh_stems, inputs=[dataset_dd], outputs=[stem_dd])
            stem_dd.change(_on_stem_change, inputs=[dataset_dd, stem_dd],
                            outputs=[sam_overlay_img, preview_img, boxes_df, correction_status])

            def _on_region_click(dataset: str, stem: str, evt: gr.SelectData):
                if not stem:
                    return None, "Pick an image first."
                sam_json = _correction_sam_json_path(dataset, stem)
                if not sam_json.exists():
                    return None, (f"No SAM regions cached for {stem} - run 03 . SAM3 Regions "
                                  f"first to get candidate regions for this image.")
                overlay_path = _find_image_file(
                    dataset_out_root(dataset) / "qc" / "sam_proposals_concept", stem
                )
                if overlay_path is None:
                    return None, f"SAM overlay image for {stem} not found."
                # evt.index is documented Gradio behavior to be in the ORIGINAL image's pixel
                # space, not the browser-rendered/CSS-scaled size - not independently re-verified
                # from source this session (frontend coordinate rescaling is compiled JS). Use the
                # SAME file being displayed/clicked (the overlay) for its true pixel dimensions, so
                # this is correct regardless of any resize gr.Image applies for display only.
                with Image.open(overlay_path) as im:
                    width, height = im.size
                px, py = evt.index
                nx, ny = px / width * 1000, py / height * 1000
                polygons = json.loads(sam_json.read_text(encoding="utf-8")).get("polygons", [])
                matches = [p for p in polygons if _point_in_polygon(nx, ny, p)]
                if not matches:
                    return None, "No SAM candidate region contains that point - click inside a highlighted region."
                best = min(matches, key=_polygon_area)
                box_2d = _polygon_to_box_2d(best)
                return box_2d, (f"Staged box {box_2d} from a SAM region ({len(matches)} region(s) "
                                f"contained that point, picked the smallest) - pick a label and "
                                f"click Add box. If this looks wrong, check the preview after "
                                f"adding before saving.")

            sam_overlay_img.select(_on_region_click, inputs=[dataset_dd, stem_dd],
                                    outputs=[pending_box_state, pending_info])

            def _add_box(dataset: str, stem: str, pending_box, label: str, rows):
                if pending_box is None:
                    return rows, None, "No staged box - click a SAM region first.", gr.update()
                new_rows = (rows or []) + [[label, *pending_box]]
                preview = _render_correction_preview(dataset, stem, new_rows)
                return new_rows, None, "Box added (not yet saved - click Save corrections).", preview

            add_box_btn.click(
                _add_box, inputs=[dataset_dd, stem_dd, pending_box_state, new_label_dd, boxes_df],
                outputs=[boxes_df, pending_box_state, pending_info, preview_img],
            )

            def _save_corrections(dataset: str, stem: str, rows):
                if not stem:
                    return "No image selected.", None
                try:
                    boxes = _rows_to_boxes(rows)
                except ValueError as e:
                    return f"NOT SAVED - {e}", None
                path = _correction_raw_json_path(dataset, stem)
                existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
                existing["boxes"] = [b.model_dump() for b in boxes]
                existing["manually_corrected"] = True
                path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
                preview = _render_correction_preview(dataset, stem, rows)
                log.info("Manual correction saved: %s (%d boxes)", path, len(boxes))
                return (f"Saved {len(boxes)} box(es) to {path.name}. Re-run 05 . Box Tightening "
                        f"to regenerate the final dataset/ with this correction."), preview

            save_btn.click(_save_corrections, inputs=[dataset_dd, stem_dd, boxes_df],
                            outputs=[correction_status, preview_img])

    # Shared render function used by the timer only here (no click handler needs the full log
    # text immediately - a click just starts a background thread) - matches
    # circe_v1/optical_flow_control/main.py's "recompute UI state from backend truth" pattern.
    def _poll_logs():
        return tuple(gr.update(value=ctrl.get_log_text(key)) for key in STAGES)

    timer = gr.Timer(1.0)
    timer.tick(_poll_logs, outputs=[log_boxes[key] for key in STAGES])


def _shutdown(*_args) -> None:
    log.info("Shutting down - terminating any still-running stage subprocess(es).")
    ctrl.shutdown()
    close_logging()


atexit.register(_shutdown)
if hasattr(signal, "SIGTERM"):
    # SIGTERM isn't delivered as KeyboardInterrupt - needs its own handler to actually run cleanup
    # (same reasoning as circe_v1/optical_flow_control/main.py's shutdown wiring).
    signal.signal(signal.SIGTERM, lambda *a: _shutdown())


if __name__ == "__main__":
    log.info("Starting Gradio UI...")
    try:
        demo.launch(server_name="0.0.0.0", css=CSS)
    finally:
        _shutdown()
