"""Shared helpers for the pipeline: source label reading, bbox conversion, and the
drop-manifest logger (every dropped IMAGE is recorded with a reason)."""
import csv
from pathlib import Path

import yaml

from config import DATASETS_DIR, IMG_EXTS, REPORTS_DIR

DROP_CSV = REPORTS_DIR / "dropped_images.csv"
DROP_FIELDS = ["filepath", "source", "stage", "reason"]


def source_names(folder: str) -> list:
    """Return the ordered class names from a source folder's data.yaml."""
    d = yaml.safe_load((DATASETS_DIR / folder / "data.yaml").read_text(encoding="utf-8"))
    names = d["names"]
    if isinstance(names, dict):                      # some exports use {0: name, ...}
        names = [names[i] for i in range(len(names))]
    return names


def iter_split_images(folder: str, split: str):
    """Yield (image_path, label_path) for one split of a source folder."""
    img_dir = DATASETS_DIR / folder / split / "images"
    lbl_dir = DATASETS_DIR / folder / split / "labels"
    if not img_dir.is_dir():
        return
    for img in sorted(img_dir.iterdir()):
        if img.suffix.lower() in IMG_EXTS and img.is_file():
            yield img, lbl_dir / (img.stem + ".txt")


def read_yolo(label_path: Path):
    """Parse a YOLO label file -> list of (cls_id, xc, yc, w, h) floats."""
    out = []
    if not label_path.exists():
        return out
    for line in label_path.read_text(encoding="utf-8").splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        try:
            out.append((int(float(p[0])), *(float(v) for v in p[1:5])))
        except ValueError:
            continue
    return out


def yolo_to_fo(xc, yc, w, h):
    """YOLO (norm center xc,yc,w,h) -> FiftyOne bbox [top-left-x, y, w, h], clipped to [0,1]."""
    x = xc - w / 2.0
    y = yc - h / 2.0
    x = min(max(x, 0.0), 1.0)
    y = min(max(y, 0.0), 1.0)
    w = min(w, 1.0 - x)
    h = min(h, 1.0 - y)
    return [x, y, w, h]


def fo_to_yolo(x, y, w, h):
    """FiftyOne bbox [top-left-x, y, w, h] -> YOLO (xc, yc, w, h)."""
    return x + w / 2.0, y + h / 2.0, w, h


class DropLog:
    """Append-only manifest of every dropped image + reason. Reset once per run by 01_import."""

    def __init__(self, reset=False):
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        self._new = reset or not DROP_CSV.exists()
        self._f = open(DROP_CSV, "w" if reset else "a", newline="", encoding="utf-8")
        self._w = csv.DictWriter(self._f, fieldnames=DROP_FIELDS)
        if self._new:
            self._w.writeheader()

    def add(self, filepath, source, stage, reason):
        self._w.writerow({"filepath": str(filepath), "source": source,
                           "stage": stage, "reason": reason})

    def close(self):
        self._f.close()
