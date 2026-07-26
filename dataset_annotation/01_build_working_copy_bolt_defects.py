"""Build a WORKING COPY of the bolt-defects__ext-sdnet2025 dataset for the rest of this pipeline
to use - the "01" equivalent of 01_remove_cad.py, but for this dataset instead of npu_bolt.

Unlike npu_bolt, this dataset needs no CAD-style prefix filtering (no synthetic renders) - every
real image is worth annotating. What it does need is FLATTENING: the raw download spreads its 826
real images across 3 separate nested subfolders with spaces in their names (verified this session):
    Defected/Annotated Loosen bolt & nuts/Resized images 640-640/   324 images, "Loosen-*.jpg"
    Defected/Annotated Missing bolt & nuts/Resized- 640-640/        200 images, "Missing-*.jpg"
    Fixed/640-640/                                                  302 images, "RAW (*).jpg"
                                                                     --------------------------
                                                                     826 total (matches
                                                                     model_training/reports/stats.json's
                                                                     orig_split_counts.coco: 826)
Filename prefixes are distinct across all 3 groups (checked this session) - no collision risk
flattening them into one folder.

datasets/bolt-defects__ext-sdnet2025/ is READ-ONLY (same invariant as datasets/npu_bolt/ and
model_training/datasets/) - this script never writes there. It COPIES the 826 real images into
circe_datasets/bolt-defects__ext-sdnet2025/working_images/, which is what every downstream stage
(03/04/05/06) reads from. Idempotent and safe to re-run any time - skips a destination file that's
already present and at least as new as its source.

This dataset's own original annotations (Loosen/Missing COCO JSON bbox labels - localization/type,
not defect-severity classes) are NOT read by this script - same situation as npu_bolt's original
bolt_a/b/c/vague XML labels: informational only, unrelated to the bolt_ok/bolt_defective/
bolt_corroded taxonomy this pipeline produces via Gemini.

Usage:
    python 01_build_working_copy_bolt_defects.py
    python 01_build_working_copy_bolt_defects.py --src datasets/bolt-defects__ext-sdnet2025 --dest circe_datasets/bolt-defects__ext-sdnet2025/working_images
"""
import argparse
import logging
import shutil
from pathlib import Path

log = logging.getLogger("build_working_copy_bolt_defects")

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Relative to --src. Order doesn't matter - filenames are prefix-distinct across all 3.
SOURCE_SUBFOLDERS = [
    "Defected/Annotated Loosen bolt & nuts/Resized images 640-640",
    "Defected/Annotated Missing bolt & nuts/Resized- 640-640",
    "Fixed/640-640",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path, default=Path(__file__).parent / "datasets" / "bolt-defects__ext-sdnet2025",
                         help="Read-only source root (default: ./datasets/bolt-defects__ext-sdnet2025)")
    parser.add_argument("--dest", type=Path,
                         default=Path(__file__).parent / "circe_datasets" / "bolt-defects__ext-sdnet2025" / "working_images",
                         help="Working copy destination, flattened (default: ./circe_datasets/bolt-defects__ext-sdnet2025/working_images)")
    args = parser.parse_args()

    src = args.src.resolve()
    dest = args.dest.resolve()
    dest.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(dest.parent / "cleanup_log.txt", encoding="utf-8"), logging.StreamHandler()],
        force=True,
    )

    if not src.is_dir():
        raise SystemExit(f"Source folder not found: {src}")

    all_images = []
    for sub in SOURCE_SUBFOLDERS:
        subdir = src / sub
        if not subdir.is_dir():
            log.warning("Expected subfolder not found, skipping: %s", subdir)
            continue
        found = sorted(f for f in subdir.iterdir() if f.is_file() and f.suffix.lower() in IMG_EXTS)
        log.info("Found %d image(s) in %s", len(found), sub)
        all_images.extend(found)

    seen_names = set()
    collisions = [f for f in all_images if f.name in seen_names or seen_names.add(f.name)]
    if collisions:
        log.warning("%d filename collision(s) across source subfolders - later copy wins, "
                    "first is shadowed: %s", len(collisions), [f.name for f in collisions])

    total = len(all_images)
    log.info("Total real images to copy: %d", total)

    copied = 0
    skipped = 0
    for i, f in enumerate(all_images, start=1):
        dst = dest / f.name
        if dst.exists() and dst.stat().st_mtime >= f.stat().st_mtime:
            skipped += 1
            log.info("[%d/%d] skip (already up to date) %s", i, total, f.name)
            continue
        shutil.copy2(str(f), str(dst))  # copy2 preserves mtime - required for the idempotency
                                         # check above to work correctly on the NEXT run
        copied += 1
        log.info("[%d/%d] copied %s -> %s", i, total, f.name, dst)

    log.info("Done. %d copied, %d already up to date. Working copy at %s", copied, skipped, dest)


if __name__ == "__main__":
    main()
