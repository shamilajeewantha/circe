"""Stage 1 of 4: SAM2 promptless/automatic ("segment everything") region-proposal pass. Local and
free (no Gemini quota spent) - no text prompt, no concept understanding at all, just flags every
visually-distinct region SAM2 finds by low-level structure. This is a genuinely different failure
mode from the concept-targeted pass (03_propose_regions_concept.py): that pass and Gemini both have
to understand what the WORD "bolt" visually means, so a fastener ambiguous enough to fool one has a
real chance of fooling both. This pass has zero notion of "bolt" at all, so it can catch a fastener
that's visually distinct-but-unusual even when both language-grounded passes miss it. Writes
candidate regions to cache/sam_regions_dumb/<stem>.json + a QC visualization (real mask overlay,
not a simplified outline) to qc/sam_proposals_dumb/<stem>.<ext>.

Uses the OFFICIAL facebookresearch/sam2 package (SAM2AutomaticMaskGenerator), not ultralytics - its
constructor takes points_per_side/crop_n_layers as genuine, directly-reachable arguments (confirmed
via source read), unlike the ultralytics wrapper's CLI-style argument whitelist that made grid-
density tuning unreachable when this pass was first attempted via SAM3/ultralytics earlier this
session. SAM2 checkpoints are NOT gated (unlike SAM3's) - no Hugging Face access request needed.

Setup
-----
Run in the WSL yolo_det_py312 conda env - needs the official `sam2` package installed
(git clone https://github.com/facebookresearch/sam2 && cd sam2 && pip install -e .), Python >= 3.10,
torch >= 2.5.1 (both satisfied by the same env set up for 03_propose_regions_concept.py's SAM3
requirement, no separate env needed).

Usage
-----
    python 02_propose_regions_dumb.py --limit 1     # speed is UNVERIFIED - test one image first
    python 02_propose_regions_dumb.py --limit 5
    python 02_propose_regions_dumb.py                # full pass over everything in npu_bolt/
    python 02_propose_regions_dumb.py --points-per-side 16 --force
                                                       # sparser grid (faster, coarser) - real,
                                                       # directly-reachable tuning knob here

Then inspect qc/sam_proposals_dumb/ before running 04_annotate_with_gemini.py.
"""
import argparse
import json
import logging
from pathlib import Path

from annotate_common import (
    DEFAULT_POINTS_PER_SIDE, DEFAULT_SAM2_MODEL_ID, IMG_EXTS, SAM_DUMB_COLOR,
    draw_mask_overlay, propose_regions_dumb,
)

log = logging.getLogger("propose_regions_dumb")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path, default=Path(__file__).parent / "npu_bolt")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "annotations")
    parser.add_argument("--sam2-model-id", type=str, default=DEFAULT_SAM2_MODEL_ID,
                         help=f"Hugging Face repo ID for the SAM2 checkpoint (default "
                              f"{DEFAULT_SAM2_MODEL_ID!r}, smallest/fastest SAM2.1 variant). Not "
                              "gated - auto-downloads via huggingface_hub, no access request "
                              "needed.")
    parser.add_argument("--points-per-side", type=int, default=DEFAULT_POINTS_PER_SIDE,
                         help=f"Grid density for automatic mode (default {DEFAULT_POINTS_PER_SIDE} "
                              "- SAM2's own default). Total sampled points = this squared. Lower "
                              "for a faster/coarser pass if the default is too slow on your "
                              "hardware - a real, directly-reachable tuning knob here.")
    parser.add_argument("--limit", type=int, default=None,
                         help="Only consider the first N images from --src this run.")
    parser.add_argument("--force", action="store_true",
                         help="Recompute even for images whose cache already matches the current "
                              "--sam2-model-id and --points-per-side (by default those are skipped "
                              "for free).")
    args = parser.parse_args()

    src = args.src.resolve()
    out = args.out.resolve()
    cache_dir = out / "cache" / "sam_regions_dumb"
    qc_dir = out / "qc" / "sam_proposals_dumb"
    cache_dir.mkdir(parents=True, exist_ok=True)
    qc_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(out / "run_log.txt", encoding="utf-8"), logging.StreamHandler()],
        force=True,  # a package imported before this point (sam2/torch/etc.) may already have
                     # attached its own root-logger handler, which makes a plain basicConfig() a
                     # silent no-op - force=True always (re)configures regardless (see
                     # 03_propose_regions_concept.py for the real crash that surfaced this).
    )
    log.info("Run started. Source: %s", src)

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
            if (cached.get("sam2_model_id") == args.sam2_model_id
                    and cached.get("points_per_side") == args.points_per_side):
                skipped += 1
                log.info("[%d/%d] %s - SKIP: already proposed with this model+points_per_side. "
                          "Pass --force to redo.", i, total, img_path.name)
                continue
            log.info("[%d/%d] %s - RE-PROPOSE: cached model/points_per_side differ from current run",
                      i, total, img_path.name)

        polygons, masks = propose_regions_dumb(img_path, args.sam2_model_id, args.points_per_side)
        cache_path.write_text(
            json.dumps({"sam2_model_id": args.sam2_model_id, "points_per_side": args.points_per_side,
                        "polygons": polygons}, indent=2),
            encoding="utf-8",
        )
        draw_mask_overlay(img_path, masks, SAM_DUMB_COLOR, qc_dir / img_path.name)
        log.info("[%d/%d] %s - %d geometric candidate region(s) proposed", i, total, img_path.name,
                  len(polygons))
        proposed += 1

    log.info("Done. %d newly proposed, %d already cached (skipped, free). Inspect %s for quality, "
              "then run 04_annotate_with_gemini.py.", proposed, skipped, qc_dir)


if __name__ == "__main__":
    main()
