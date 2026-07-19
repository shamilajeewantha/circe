"""Remove synthetic CAD renders from a copy of the NPU-BOLT dataset.

NPU-BOLT ships 3 image groups distinguishable by filename prefix (verified against
the actual files in npu_bolt/ this session: 204 AUT-*, 116 WEB-*, 17 CAD-*):
    AUT-*  field-captured photographs (real)
    WEB-*  photographs sourced from the internet (real)
    CAD-*  synthetic CAD renders (NOT real photographs)

Only AUT-*/WEB-* should ever reach the Gemini annotation step. This script moves
(does not delete) CAD-* images to a sibling folder so the operation is reversible.

Usage:
    python 01_remove_cad.py
    python 01_remove_cad.py --src npu_bolt --dest npu_bolt_cad_excluded
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
    parser.add_argument("--src", type=Path, default=Path(__file__).parent / "npu_bolt",
                         help="Folder containing the NPU-BOLT images (default: ./npu_bolt)")
    parser.add_argument("--dest", type=Path, default=None,
                         help="Where to move CAD images (default: sibling folder <src>_cad_excluded)")
    args = parser.parse_args()

    src = args.src.resolve()
    dest = (args.dest or src.parent / f"{src.name}_cad_excluded").resolve()

    # FileHandler flushes every record as it's emitted, so the log survives a crash partway
    # through, unlike accumulate-then-write-once-at-the-end.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(src.parent / "cleanup_cad_log.txt", encoding="utf-8"), logging.StreamHandler()],
        force=True,  # defensive consistency with the other numbered scripts - see
                     # 03_propose_regions_concept.py for the real bug (a package attaching its own
                     # root-logger handler before this point silently makes basicConfig() a no-op)
                     # that motivated adding this everywhere, even though this script doesn't
                     # import torch/sam and is lower-risk.
    )

    if not src.is_dir():
        raise SystemExit(f"Source folder not found: {src}")

    to_move = sorted(
        f for f in src.iterdir()
        if f.is_file() and f.suffix.lower() in IMG_EXTS and f.name.lower().startswith(CAD_PREFIX)
    )

    if not to_move:
        log.info("No CAD-* images found in %s - nothing to do.", src)
        return

    dest.mkdir(parents=True, exist_ok=True)
    log.info("Moving %d CAD image(s) from %s to %s", len(to_move), src, dest)
    for f in to_move:
        shutil.move(str(f), str(dest / f.name))
        log.info("  moved %s", f.name)

    remaining = sorted(f for f in src.iterdir() if f.is_file() and f.suffix.lower() in IMG_EXTS)

    log.info("Moved %d CAD image(s) to %s", len(to_move), dest)
    log.info("%d real (AUT-*/WEB-*) images remain in %s", len(remaining), src)


if __name__ == "__main__":
    main()
