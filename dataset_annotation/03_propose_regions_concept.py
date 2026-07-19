"""Stage 2 of 4: SAM3 concept-targeted region-proposal pass. Local and free (no Gemini quota
spent) - searches every image for fastener-like concepts (bolt, screw, nut, fastener, rivet by
default) and writes candidate regions to cache/sam_regions_concept/<stem>.json + a QC visualization
(real mask overlay, not a simplified outline) to qc/sam_proposals_concept/<stem>.<ext>. Downstream
stages (04_annotate_with_gemini.py, 05_tighten_boxes.py) read this stage's cache/ output - nothing
here calls Gemini or loads the box-tightening machinery beyond what SAM3 itself provides, so this
can be run, re-run, and tuned (e.g. --concepts) independently.

Uses the OFFICIAL facebookresearch/sam3 package (Sam3Processor), not ultralytics - switched this
session after real problems with the ultralytics path (a cascading CUDA OOM on a 100-image run,
never fully explained; repeated friction with ultralytics' own argument whitelist). See
annotate_common.py's module docstring for the full writeup.

Setup
-----
Run in the WSL yolo_det_py312 conda env - needs the official `sam3` package installed
(git clone https://github.com/facebookresearch/sam3 && cd sam3 && pip install -e .), Python >= 3.12,
torch >= 2.10. SAM3 (facebook/sam3 on Hugging Face) IS gated: request access, then `hf auth login`
- build_sam3_image_model() auto-downloads from there once access is granted, no manual checkpoint
placement needed.

Usage
-----
    python 03_propose_regions_concept.py --limit 5      # smoke test a handful first
    python 03_propose_regions_concept.py                # full pass over everything in npu_bolt/
    python 03_propose_regions_concept.py --concepts "bolt,screw,nut,fastener,rivet,washer" --force
                                                          # broaden concepts and recompute everything

Then inspect qc/sam_proposals_concept/ before running 04_annotate_with_gemini.py - this is the
"dry run" from an earlier single-script design; there's no separate --dry-run flag needed any more,
since running this stage alone already stops before any Gemini call exists to make.
"""
import argparse
import json
import logging
from pathlib import Path

from annotate_common import (
    DEFAULT_CONCEPTS, DEFAULT_SAM3_CONFIDENCE_THRESHOLD, IMG_EXTS, SAM_CONCEPT_COLOR,
    draw_mask_overlay, propose_regions_concept,
)

log = logging.getLogger("propose_regions_concept")

# If sam3.pt already sits next to this script (the common case - manually downloaded once, no
# reason to re-download or need hf auth login every run), default straight to it. Only falls back
# to None (auto-download via huggingface_hub) if no local file is present.
_LOCAL_SAM3_CHECKPOINT = Path(__file__).parent / "sam3.pt"
DEFAULT_SAM3_CHECKPOINT = str(_LOCAL_SAM3_CHECKPOINT) if _LOCAL_SAM3_CHECKPOINT.exists() else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path, default=Path(__file__).parent / "npu_bolt")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "annotations")
    parser.add_argument("--concepts", type=str, default=",".join(DEFAULT_CONCEPTS),
                         help=f"Comma-separated text concepts SAM3 searches for (default "
                              f"{','.join(DEFAULT_CONCEPTS)!r}).")
    parser.add_argument("--sam3-checkpoint", type=str, default=DEFAULT_SAM3_CHECKPOINT,
                         help=f"Path to a local sam3.pt (default {DEFAULT_SAM3_CHECKPOINT!r} - "
                              f"auto-detected next to this script if present). If not given and no "
                              f"local file is found, build_sam3_image_model() auto-downloads via "
                              f"huggingface_hub, which requires `hf auth login` with access already "
                              f"granted to the gated facebook/sam3 repo.")
    parser.add_argument("--confidence-threshold", type=float, default=DEFAULT_SAM3_CONFIDENCE_THRESHOLD,
                         help=f"SAM3's own confidence filter for a concept match (default "
                              f"{DEFAULT_SAM3_CONFIDENCE_THRESHOLD}, raised above Sam3Processor's "
                              f"own 0.5 default - real evidence this session that 0.5 let degenerate, "
                              f"perfectly-rectangular 'detections' through on images with no real "
                              f"matching object, identical across unrelated concepts). Raise further "
                              f"if junk candidates still show up in qc/sam_proposals_concept/; lower "
                              f"if real fasteners are being missed.")
    parser.add_argument("--limit", type=int, default=None,
                         help="Only consider the first N images from --src this run.")
    parser.add_argument("--force", action="store_true",
                         help="Recompute even for images whose cache already matches the current "
                              "--concepts (by default those are skipped for free). Use this after "
                              "changing --concepts and wanting a full redo, or if you suspect a "
                              "cached result is bad.")
    args = parser.parse_args()

    src = args.src.resolve()
    out = args.out.resolve()
    cache_dir = out / "cache" / "sam_regions_concept"
    qc_dir = out / "qc" / "sam_proposals_concept"
    cache_dir.mkdir(parents=True, exist_ok=True)
    qc_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(out / "run_log.txt", encoding="utf-8"), logging.StreamHandler()],
    )
    log.info("Run started. Source: %s", src)

    concepts_list = [c.strip() for c in args.concepts.split(",") if c.strip()]

    images = sorted(f for f in src.iterdir() if f.is_file() and f.suffix.lower() in IMG_EXTS)
    if args.limit:
        images = images[: args.limit]
    total = len(images)
    log.info("Images found: %d", total)

    proposed, skipped = 0, 0
    for i, img_path in enumerate(images, start=1):
        cache_path = cache_dir / f"{img_path.stem}.json"
        if cache_path.exists() and not args.force:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if (cached.get("concepts") == concepts_list
                    and cached.get("confidence_threshold") == args.confidence_threshold):
                skipped += 1
                log.info("[%d/%d] %s - SKIP: already proposed with these concepts+threshold. Pass "
                          "--force to redo.", i, total, img_path.name)
                continue
            log.info("[%d/%d] %s - RE-PROPOSE: cached concepts/confidence-threshold differ from "
                      "current run", i, total, img_path.name)

        polygons, masks = propose_regions_concept(
            img_path, concepts_list, args.sam3_checkpoint, args.confidence_threshold
        )
        cache_path.write_text(
            json.dumps({"concepts": concepts_list, "confidence_threshold": args.confidence_threshold,
                        "polygons": polygons}, indent=2),
            encoding="utf-8",
        )
        draw_mask_overlay(img_path, masks, SAM_CONCEPT_COLOR, qc_dir / img_path.name)
        log.info("[%d/%d] %s - %d concept-targeted candidate region(s) proposed", i, total,
                  img_path.name, len(polygons))
        proposed += 1

    log.info("Done. %d newly proposed, %d already cached (skipped, free). Inspect %s for quality, "
              "then run 04_annotate_with_gemini.py.", proposed, skipped, qc_dir)


if __name__ == "__main__":
    main()
