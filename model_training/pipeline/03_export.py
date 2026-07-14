"""Step 3 — export the curated dataset to a training-ready YOLO set in merged/.

- Drops `duplicate`-tagged samples (already logged by 02).
- Regenerates a STRATIFIED 70/20/10 split (by each image's primary class) so every
  class gets val/test coverage and the old welding-defects train-only issue is gone.
- Caps background (no-box) images at config.BACKGROUND_MAX_SHARE of each split;
  any background dropped by the cap is written to the drop manifest with a reason.
- COPIES pixels out of datasets/ into merged/ (source folder never modified).
- Filenames are prefixed with the source folder to avoid collisions.

Run in `drone_detect`:  python pipeline/03_export.py
"""
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import fiftyone as fo

from config import (BACKGROUND_MAX_SHARE, FO_DATASET_NAME, MERGED_DIR, NAME2ID,
                    SPLIT_RATIOS, SPLIT_SEED, TAXONOMY)
from _util import DropLog, fo_to_yolo


def collect():
    """Return list of records for all NON-duplicate samples."""
    dataset = fo.load_dataset(FO_DATASET_NAME)
    keep = dataset.match_tags("duplicate", bool=False)
    recs = []
    for s in keep:
        dets = [(d.label, *d.bounding_box) for d in s.ground_truth.detections]
        primary = Counter(l for l, *_ in dets).most_common(1)[0][0] if dets else "background"
        recs.append({"filepath": s.filepath, "source": s.source,
                     "dets": dets, "primary": primary, "bg": not dets})
    return recs


def stratified_split(recs):
    """Assign each record a split, stratified by primary label, with a fixed seed."""
    rng = random.Random(SPLIT_SEED)
    by_label = defaultdict(list)
    for r in recs:
        by_label[r["primary"]].append(r)
    for label, items in by_label.items():
        rng.shuffle(items)
        n = len(items)
        n_tr = int(n * SPLIT_RATIOS["train"])
        n_va = int(n * SPLIT_RATIOS["valid"])
        for i, r in enumerate(items):
            r["split"] = "train" if i < n_tr else "valid" if i < n_tr + n_va else "test"
    return recs


def apply_background_cap(recs, drops):
    """Keep foreground; cap backgrounds per split; log dropped backgrounds."""
    fg = [r for r in recs if not r["bg"]]
    bg = [r for r in recs if r["bg"]]
    fg_per_split = Counter(r["split"] for r in fg)
    kept, per_split_bg = [], Counter()
    rng = random.Random(SPLIT_SEED + 1)
    rng.shuffle(bg)
    for r in bg:
        cap = int(fg_per_split[r["split"]] * BACKGROUND_MAX_SHARE)
        if per_split_bg[r["split"]] < cap:
            per_split_bg[r["split"]] += 1
            kept.append(r)
        else:
            drops.add(r["filepath"], r["source"], "export:background-cap",
                      f"background image beyond {int(BACKGROUND_MAX_SHARE*100)}% cap for split '{r['split']}'")
    return fg + kept


def write(recs):
    if MERGED_DIR.exists():
        shutil.rmtree(MERGED_DIR)
    for split in SPLIT_RATIOS:
        (MERGED_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (MERGED_DIR / split / "labels").mkdir(parents=True, exist_ok=True)

    counts = defaultdict(Counter)   # split -> Counter(label -> boxes)
    img_counts = Counter()
    for r in recs:
        src_img = Path(r["filepath"])
        stem = f"{r['source']}__{src_img.stem}"
        dst_img = MERGED_DIR / r["split"] / "images" / (stem + src_img.suffix.lower())
        shutil.copy2(src_img, dst_img)
        lines = []
        for label, x, y, w, h in r["dets"]:
            xc, yc, bw, bh = fo_to_yolo(x, y, w, h)
            lines.append(f"{NAME2ID[label]} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
            counts[r["split"]][label] += 1
        (MERGED_DIR / r["split"] / "labels" / (stem + ".txt")).write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        img_counts[r["split"]] += 1

    names_block = "\n".join(f"  {i}: {n}" for i, n in enumerate(TAXONOMY))
    (MERGED_DIR / "data.yaml").write_text(
        f"path: {MERGED_DIR.as_posix()}\n"
        f"train: train/images\nval: valid/images\ntest: test/images\n\n"
        f"nc: {len(TAXONOMY)}\nnames:\n{names_block}\n", encoding="utf-8")
    return img_counts, counts


def main():
    drops = DropLog(reset=False)
    recs = apply_background_cap(stratified_split(collect()), drops)
    drops.close()
    img_counts, counts = write(recs)

    print(f"merged/ written: {sum(img_counts.values())} images")
    print(f"  {'class':<16}" + "".join(f"{s:>8}" for s in SPLIT_RATIOS) + f"{'total':>8}")
    for label in TAXONOMY:
        row = [counts[s][label] for s in SPLIT_RATIOS]
        print(f"  {label:<16}" + "".join(f"{v:>8}" for v in row) + f"{sum(row):>8}")
    print(f"  {'images':<16}" + "".join(f"{img_counts[s]:>8}" for s in SPLIT_RATIOS)
          + f"{sum(img_counts.values()):>8}")
    # sanity: every class present in train
    missing = [l for l in TAXONOMY if counts["train"][l] == 0]
    if missing:
        print(f"  WARNING: no TRAIN boxes for: {missing}")
    print(f"\ndata.yaml: {MERGED_DIR/'data.yaml'}")


if __name__ == "__main__":
    main()
