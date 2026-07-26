"""Build a CAD-excluded WORKING COPY of the NPU-BOLT images for the rest of this pipeline to use.

NPU-BOLT ships 3 image groups distinguishable by filename prefix (verified against
the actual files in datasets/npu_bolt/images/ this session: 204 AUT-*, 116 WEB-*, 17 CAD-*):
    AUT-*  field-captured photographs (real)
    WEB-*  photographs sourced from the internet (real)
    CAD-*  synthetic CAD renders (NOT real photographs)

Only AUT-*/WEB-* should ever reach the Gemini annotation step.

datasets/npu_bolt/ is READ-ONLY (same invariant as model_training/datasets/ at the repo root) -
this script never writes there. It COPIES the real (non-CAD) images into
circe_datasets/npu_bolt/working_images/, which is what every downstream stage (03/04/05/06)
reads from. This used to instead MOVE CAD-* files out of a mutable npu_bolt/ folder - that
approach is no longer possible now that the source is read-only, so the direction flipped: copy
the good ones IN, rather than move the bad ones OUT. Idempotent and safe to re-run any time -
skips a destination file that's already present and at least as new as its source.

Usage:
    python 01_remove_cad.py
    python 01_remove_cad.py --src datasets/npu_bolt/images --dest circe_datasets/npu_bolt/working_images
"""
import argparse
import logging
import shutil
from pathlib import Path

log = logging.getLogger("remove_cad")

CAD_PREFIX = "cad"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path, default=Path(__file__).parent / "datasets" / "npu_bolt" / "images",
                         help="Read-only source folder of NPU-BOLT images (default: ./datasets/npu_bolt/images)")
    parser.add_argument("--dest", type=Path, default=Path(__file__).parent / "circe_datasets" / "npu_bolt" / "working_images",
                         help="Working copy destination, CAD-* excluded (default: ./circe_datasets/npu_bolt/working_images)")
    args = parser.parse_args()

    src = args.src.resolve()
    dest = args.dest.resolve()
    dest.mkdir(parents=True, exist_ok=True)

    # FileHandler flushes every record as it's emitted, so the log survives a crash partway
    # through, unlike accumulate-then-write-once-at-the-end.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(dest.parent / "cleanup_cad_log.txt", encoding="utf-8"), logging.StreamHandler()],
        force=True,  # defensive consistency with the other numbered scripts - see
                     # 03_propose_regions_concept.py for the real bug (a package attaching its own
                     # root-logger handler before this point silently makes basicConfig() a no-op)
                     # that motivated adding this everywhere, even though this script doesn't
                     # import torch/sam and is lower-risk.
    )

    if not src.is_dir():
        raise SystemExit(f"Source folder not found: {src}")

    real_images = sorted(
        f for f in src.iterdir()
        if f.is_file() and f.suffix.lower() in IMG_EXTS and not f.name.lower().startswith(CAD_PREFIX)
    )
    cad_count = sum(
        1 for f in src.iterdir()
        if f.is_file() and f.suffix.lower() in IMG_EXTS and f.name.lower().startswith(CAD_PREFIX)
    )
    log.info("Found %d real (AUT-*/WEB-*) image(s) and %d CAD-* image(s) in %s", len(real_images), cad_count, src)

    total = len(real_images)
    copied = 0
    skipped = 0
    for i, f in enumerate(real_images, start=1):
        dst = dest / f.name
        if dst.exists() and dst.stat().st_mtime >= f.stat().st_mtime:
            skipped += 1
            log.info("[%d/%d] skip (already up to date) %s", i, total, f.name)
            continue
        shutil.copy2(str(f), str(dst))
        copied += 1
        log.info("[%d/%d] copied %s -> %s", i, total, f.name, dst)

    log.info("Done. %d copied, %d already up to date, %d CAD-* excluded. Working copy at %s",
              copied, skipped, cad_count, dest)


if __name__ == "__main__":
    main()
