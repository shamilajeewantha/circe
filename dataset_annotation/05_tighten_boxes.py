"""Stage 4 of 4: SAM3 box-tightening pass + final dataset/qc output. Local and free (no Gemini
quota spent) - reads cache/raw_gemini/<stem>.json (written by 04_annotate_with_gemini.py) for every
image that has one, re-prompts SAM3 with each box (the actual Grounded-SAM pattern: use a rough
detection to prompt a segmentation model for a pixel-precise boundary), and writes the final
dataset/ (trainable YOLO output) + qc/visualized/ + qc/sam_regions_debug/.

Uses the OFFICIAL facebookresearch/sam3 package (Sam3Processor.add_geometric_prompt), not
ultralytics - same processor/checkpoint as 03_propose_regions_concept.py, different prompt type
(box instead of text).

Deliberately does NOT know about --model/PROMPT_VERSION staleness at all - that's
04_annotate_with_gemini.py's concern. This stage just reads whatever is currently cached for each
image (current or stale, migrate_boxes() is applied unconditionally and is a safe no-op on
already-current labels) and regenerates output from it. That means re-running this stage after a
code change to tightening/QC-drawing logic never re-calls Gemini or re-runs either SAM proposal
pass - only 04_annotate_with_gemini.py spends API quota, only 02/03 re-run SAM proposals.

Setup
-----
Run in the WSL yolo_det_py312 conda env - needs the official `sam3` package installed
(git clone https://github.com/facebookresearch/sam3 && cd sam3 && pip install -e .), unless
--skip-tightening is passed. Same gated-checkpoint access requirement as
03_propose_regions_concept.py (hf auth login with granted access to facebook/sam3).

Usage
-----
    python 05_tighten_boxes.py --limit 10   # smoke test first
    python 05_tighten_boxes.py              # full pass over everything cached in cache/raw_gemini/
    python 05_tighten_boxes.py --clean      # wipe + regenerate dataset/+qc/ from cache, no API calls

Output - dataset/ (the actual trainable YOLO output, standard Ultralytics/YOLO26 layout - same
shape as model_training/merged, nothing else in this folder so a future importer can point straight
at it) and qc/visualized/ + qc/sam_regions_debug/ (human-facing diagnostics, never consumed by
training code):
  <out>/dataset/data.yaml                 - nc=3, names, train/val both point at images/ (single
                                             unsplit source; real train/valid/test split happens
                                             later in model_training/pipeline)
  <out>/dataset/images/<stem>.<ext>       - copy of the original image (npu_bolt/ untouched)
  <out>/dataset/labels/<stem>.txt         - YOLO label: "class_id cx cy w h" normalized 0-1, one
                                             line per box (empty file if no fasteners found - valid
                                             YOLO background-image convention)
  <out>/qc/visualized/<stem>.<ext>        - STAGE 4 (FINAL): after SAM tightening - this is what
                                             dataset/labels/ actually contains
  <out>/qc/sam_regions_debug/<stem>.json  - per-box before/after SAM tightening record (numbers)
  <out>/qc/sam_regions_debug/<stem>.<ext> - same, drawn on one image: gray = before tightening,
                                             class color = after
"""
import argparse
import json
import logging
import shutil
from pathlib import Path

from annotate_common import (
    BoltAnnotation, BoltBox, IMG_EXTS, draw_tightening_debug, draw_visualization,
    filter_valid_boxes, migrate_boxes, tighten_boxes_for_image, write_data_yaml, write_yolo_label,
)

log = logging.getLogger("tighten_boxes")

# If sam3.pt already sits next to this script (the common case), default straight to it - only
# falls back to None (auto-download via huggingface_hub) if no local file is present.
_LOCAL_SAM3_CHECKPOINT = Path(__file__).parent / "sam3.pt"
DEFAULT_SAM3_CHECKPOINT = str(_LOCAL_SAM3_CHECKPOINT) if _LOCAL_SAM3_CHECKPOINT.exists() else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path, default=Path(__file__).parent / "npu_bolt")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "annotations")
    parser.add_argument("--sam3-checkpoint", type=str, default=DEFAULT_SAM3_CHECKPOINT,
                         help=f"Path to a local sam3.pt (default {DEFAULT_SAM3_CHECKPOINT!r} - "
                              f"auto-detected next to this script if present). If not given and no "
                              f"local file is found, auto-downloads via huggingface_hub, which "
                              f"requires `hf auth login`. Ignored if --skip-tightening is set.")
    parser.add_argument("--skip-tightening", action="store_true",
                         help="Skip the local SAM box-prompted pass that would otherwise tighten "
                              "each cached box to the real object edges (Grounded-SAM pattern). "
                              "Use this if the official sam3 package isn't available or you want "
                              "to isolate its effect - dataset/+qc/visualized/ are still produced, "
                              "just with Gemini's boxes exactly as cached.")
    parser.add_argument("--limit", type=int, default=None,
                         help="Only consider the first N images from --src this run.")
    parser.add_argument("--clean", action="store_true",
                         help="Wipe dataset/ and qc/visualized/ + qc/sam_regions_debug/ before "
                              "running, then regenerate them from cache/raw_gemini/ - FREE, no API "
                              "calls, since the expensive part (the Gemini cache) is kept. Use this "
                              "for a clean output layout, e.g. after a code change to tightening or "
                              "QC-drawing logic.")
    args = parser.parse_args()

    src = args.src.resolve()
    out = args.out.resolve()
    raw_dir = out / "cache" / "raw_gemini"
    dataset_dir = out / "dataset"
    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"
    viz_dir = out / "qc" / "visualized"
    sam_debug_dir = out / "qc" / "sam_regions_debug"

    if args.clean:
        shutil.rmtree(dataset_dir, ignore_errors=True)
        shutil.rmtree(viz_dir, ignore_errors=True)
        shutil.rmtree(sam_debug_dir, ignore_errors=True)

    for d in (images_dir, labels_dir, viz_dir, sam_debug_dir):
        d.mkdir(parents=True, exist_ok=True)

    # FileHandler flushes every record as it's emitted, so the log survives a crash/Ctrl-C
    # partway through - appends to the same run_log.txt every stage writes to.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(out / "run_log.txt", encoding="utf-8"), logging.StreamHandler()],
        force=True,  # a package imported before this point (sam3/torch/etc.) may already have
                     # attached its own root-logger handler, which makes a plain basicConfig() a
                     # silent no-op - force=True always (re)configures regardless (see
                     # 03_propose_regions_concept.py for the real crash that surfaced this).
    )
    log.info("Run started. Source: %s", src)
    if args.clean:
        log.info("--clean: wiped dataset/ and qc/visualized/+qc/sam_regions_debug/ (regenerated "
                  "from cache/raw_gemini/, free - no API calls)")

    images = sorted(f for f in src.iterdir() if f.is_file() and f.suffix.lower() in IMG_EXTS)
    if args.limit:
        images = images[: args.limit]
    total = len(images)
    log.info("Images found: %d", total)

    processed, missing, total_boxes = 0, 0, 0

    for i, img_path in enumerate(images, start=1):
        raw_path = raw_dir / f"{img_path.stem}.json"
        if not raw_path.exists():
            missing += 1
            log.warning("[%d/%d] %s - no cached Gemini annotation found; run "
                        "04_annotate_with_gemini.py first. Skipping.", i, total, img_path.name)
            continue

        cached_data = json.loads(raw_path.read_text(encoding="utf-8"))
        result = BoltAnnotation(boxes=migrate_boxes(cached_data["boxes"], img_path.name))
        result = filter_valid_boxes(result, img_path.name)

        if not args.skip_tightening:
            tight_boxes_2d = tighten_boxes_for_image(
                img_path, [box.box_2d for box in result.boxes], args.sam3_checkpoint
            )
            tightened_boxes = [
                BoltBox(box_2d=tight_box_2d, label=box.label)
                for box, tight_box_2d in zip(result.boxes, tight_boxes_2d)
            ]
            debug_records = [
                {"label": box.label, "before": box.box_2d, "after": tight_box_2d}
                for box, tight_box_2d in zip(result.boxes, tight_boxes_2d)
            ]
            result = BoltAnnotation(boxes=tightened_boxes)
            (sam_debug_dir / f"{img_path.stem}.json").write_text(
                json.dumps(debug_records, indent=2), encoding="utf-8"
            )
            draw_tightening_debug(img_path, debug_records, sam_debug_dir / img_path.name)

        write_yolo_label(result, labels_dir / f"{img_path.stem}.txt")
        shutil.copy2(img_path, images_dir / img_path.name)
        draw_visualization(img_path, result, viz_dir / img_path.name)
        total_boxes += len(result.boxes)
        processed += 1
        log.info("[%d/%d] %s -> %d box(es)%s", i, total, img_path.name, len(result.boxes),
                  "" if args.skip_tightening else " (tightened)")

    write_data_yaml(dataset_dir)

    log.info("Done. %d image(s) processed, %d missing (no cached Gemini annotation - run "
              "04_annotate_with_gemini.py first), %d total boxes.", processed, missing, total_boxes)
    log.info("YOLO dataset ready at %s (data.yaml + images/ + labels/); QC at %s", dataset_dir, viz_dir)


if __name__ == "__main__":
    main()
