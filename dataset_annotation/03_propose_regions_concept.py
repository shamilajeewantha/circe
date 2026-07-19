"""Stage 2 of 4: SAM3 concept-targeted region-proposal pass. Local and free (no Gemini quota
spent) - searches every image for fastener-like concepts (bolt, screw, nut, fastener, rivet by
default) and writes candidate regions to cache/sam_regions_concept/<stem>.json + a QC visualization
(real mask overlay, not a simplified outline) to qc/sam_proposals_concept/<stem>.<ext>.

Deliberately as simple as sam3_minimal.py's proven-working call pattern: build the model once,
loop over images, set_image/set_text_prompt per image/concept. One thing this fixes vs. an earlier
version of this script that routed images through annotate_common._load_rgb() first: that function
returns a numpy HWC array, and Sam3Processor.set_image()'s numpy/tensor branch reads
`height, width = image.shape[-2:]` - a CHW assumption. On an HWC array that reads the channel count
(3) as the width, corrupting every returned mask's scale factor and silently destroying real
detections (confirmed: that path returned 0 candidates on images with confirmed real bolts, while
this script's PIL-direct call finds them correctly - same image, same threshold, same checkpoint).
Passing a PIL Image straight to set_image(), like the official README does, hits the correct
isinstance(image, PIL.Image.Image) branch instead.

Confidence threshold default is 0.3, not Sam3Processor's own 0.5: real evidence this session (raw,
unfiltered scores on npu_bolt/AAAA.jpg, visually confirmed via mask overlay to all be genuine bolt
hardware) showed real detections scoring as low as 0.256, so a higher cutoff was silently dropping
correct hits.
"""
import argparse
import json
import logging
from pathlib import Path

import torch
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

from annotate_common import DEFAULT_CONCEPTS, IMG_EXTS, MAX_REGION_HINTS, SAM_CONCEPT_COLOR, _mask_to_polygon, draw_mask_overlay

log = logging.getLogger("propose_regions_concept")

DEFAULT_CONFIDENCE_THRESHOLD = 0.3

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
    parser.add_argument("--confidence-threshold", type=float, default=DEFAULT_CONFIDENCE_THRESHOLD,
                         help=f"Sam3Processor's own confidence filter for a concept match (default "
                              f"{DEFAULT_CONFIDENCE_THRESHOLD}, tuned down from Sam3Processor's own "
                              f"0.5 default - real evidence this session that 0.5+ silently dropped "
                              f"genuine detections scoring as low as 0.256).")
    parser.add_argument("--limit", type=int, default=None,
                         help="Only consider the first N images from --src this run.")
    parser.add_argument("--force", action="store_true",
                         help="Recompute even for images whose cache already matches the current "
                              "--concepts/--confidence-threshold (by default those are skipped for "
                              "free).")
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
        force=True,  # real bug found this session: importing sam3 at module load time attaches its
                     # own root-logger StreamHandler first, which makes a plain basicConfig() a
                     # silent no-op (Python's documented behavior when the root logger already has
                     # handlers) - every log.info() call was being dropped with no error, no
                     # run_log.txt entry, exit code 0. force=True always (re)configures regardless.
    )
    log.info("Run started. Source: %s", src)

    concepts_list = [c.strip() for c in args.concepts.split(",") if c.strip()]

    images = sorted(f for f in src.iterdir() if f.is_file() and f.suffix.lower() in IMG_EXTS)
    if args.limit:
        images = images[: args.limit]
    total = len(images)
    log.info("Images found: %d. Concepts: %s. Confidence threshold: %s", total, concepts_list,
              args.confidence_threshold)

    # Load once, loop over images - matches the official README's own usage pattern.
    model = build_sam3_image_model(checkpoint_path=args.sam3_checkpoint)
    processor = Sam3Processor(model, confidence_threshold=args.confidence_threshold)

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

        image = Image.open(img_path).convert("RGB")
        target_shape = (image.height, image.width)

        polygons: list = []
        masks: list = []
        seen_polygons: set = set()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            state = processor.set_image(image)
            for concept in concepts_list:
                processor.reset_all_prompts(state)
                output = processor.set_text_prompt(state=state, prompt=concept)
                concept_masks = output["masks"].cpu().numpy().squeeze(1)  # (N,H,W) bool
                for mask_arr in concept_masks:
                    polygon = _mask_to_polygon(mask_arr, target_shape)
                    if polygon is None:
                        continue
                    poly_key = tuple(tuple(pt) for pt in polygon)
                    if poly_key in seen_polygons:
                        continue
                    seen_polygons.add(poly_key)
                    polygons.append(polygon)
                    masks.append(mask_arr)

        polygons = polygons[:MAX_REGION_HINTS]
        masks = masks[:MAX_REGION_HINTS]

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
