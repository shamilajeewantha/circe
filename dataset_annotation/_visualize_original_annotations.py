"""One-off script: draw the NPU-BOLT dataset's OWN original Pascal-VOC XML annotations
(datasets/npu_bolt/annotations/*.xml) onto their source images (datasets/npu_bolt/images/*.jpg),
completely as-shipped - no relabeling, no remapping to this pipeline's bolt_ok/bolt_defective/
bolt_corroded classes (those are a different, incompatible label set - the original data uses
bolt_a/bolt_b/bolt_c/vague, see class counts below). Purpose is purely visual inspection of what
the dataset's original ground truth actually looked like.

datasets/npu_bolt/ is READ-ONLY (same invariant as model_training/datasets/ at the repo root) -
this script only ever reads from it. All output goes to circe_datasets/npu_bolt/qc/original_labels/
- the same regenerable working/output tree the numbered pipeline (01/03/04/05/06) uses - never into
datasets/ itself.

Original class counts across all 337 XML files (checked this session, informational only - this
script does not interpret or validate against these):
    bolt_b: 768   bolt_a: 314   vague: 132   bolt_c: 61

Not part of the numbered 01-06 pipeline - a manual inspection aid only, run directly (plain cv2/
Pillow deps already in the env, no torch/sam/genai needed).

Usage:
    python _visualize_original_annotations.py
    python _visualize_original_annotations.py --limit 20
    python _visualize_original_annotations.py --datasets-dir datasets/npu_bolt --out circe_datasets/npu_bolt/qc/original_labels
"""
import argparse
import logging
from pathlib import Path
from typing import List, Tuple

import cv2
from defusedxml import ElementTree as ET

log = logging.getLogger("visualize_original_annotations")

HERE = Path(__file__).parent
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Original NPU-BOLT classes - fixed palette so colors stay consistent run over run. Distinct from
# BOX_COLORS in annotate_common.py (that one keys off this pipeline's OWN bolt_ok/bolt_defective/
# bolt_corroded labels, a different label set entirely).
CLASS_COLORS_BGR = {
    "bolt_a": (40, 180, 40),
    "bolt_b": (20, 20, 220),
    "bolt_c": (20, 90, 150),
    "vague": (180, 180, 40),
}
FALLBACK_COLOR_BGR = (255, 255, 255)


def _rects_overlap(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> bool:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return ax0 < bx1 and bx0 < ax1 and ay0 < by1 and by0 < ay1


def parse_voc_annotation(xml_path: Path) -> List[Tuple[str, int, int, int, int]]:
    """Returns [(class_name, xmin, ymin, xmax, ymax), ...] straight from the XML, no filtering."""
    root = ET.parse(xml_path).getroot()
    boxes = []
    for obj in root.findall("object"):
        name = obj.find("name").text
        bnd = obj.find("bndbox")
        xmin = round(float(bnd.find("xmin").text))
        ymin = round(float(bnd.find("ymin").text))
        xmax = round(float(bnd.find("xmax").text))
        ymax = round(float(bnd.find("ymax").text))
        boxes.append((name, xmin, ymin, xmax, ymax))
    return boxes


def draw_original_annotation(image_path: Path, boxes: List[Tuple[str, int, int, int, int]], out_path: Path) -> None:
    img = cv2.imread(str(image_path))
    if img is None:
        raise RuntimeError(f"cv2 failed to read {image_path}")
    height, width = img.shape[:2]
    line_width = max(4, round(min(width, height) / 250))
    font_scale = max(1.0, min(width, height) / 900)
    thickness = max(2, round(font_scale))
    font = cv2.FONT_HERSHEY_SIMPLEX

    placed_labels: List[Tuple[int, int, int, int]] = []

    for name, x0, y0, x1, y1 in boxes:
        color = CLASS_COLORS_BGR.get(name, FALLBACK_COLOR_BGR)
        cv2.rectangle(img, (x0, y0), (x1, y1), color, line_width)

        (text_w, text_h), baseline = cv2.getTextSize(name, font, font_scale, thickness)
        label_h = text_h + baseline + 6
        label_top = max(0, y0 - label_h)
        label_rect = (x0, label_top, x0 + text_w + 8, label_top + label_h)
        while any(_rects_overlap(label_rect, other) for other in placed_labels):
            label_top += label_h + 2
            label_rect = (x0, label_top, x0 + text_w + 8, label_top + label_h)
        placed_labels.append(label_rect)

        text_pos = (x0 + 4, label_rect[1] + text_h)
        cv2.putText(img, name, text_pos, font, font_scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
        cv2.putText(img, name, text_pos, font, font_scale, color, thickness, cv2.LINE_AA)

    cv2.imwrite(str(out_path), img)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets-dir", type=Path, default=HERE / "datasets" / "npu_bolt",
                         help="Read-only source root, containing images/ and annotations/ (default: ./datasets/npu_bolt)")
    parser.add_argument("--out", type=Path, default=HERE / "circe_datasets" / "npu_bolt" / "qc" / "original_labels",
                         help="Dedicated output folder - never datasets/ itself "
                              "(default: ./circe_datasets/npu_bolt/qc/original_labels)")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N annotation files (debugging)")
    args = parser.parse_args()

    src = args.datasets_dir.resolve()
    images_dir = src / "images"
    annotations_dir = src / "annotations"
    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(out_dir / "run_log.txt", encoding="utf-8"), logging.StreamHandler()],
        force=True,
    )

    if not images_dir.is_dir() or not annotations_dir.is_dir():
        raise SystemExit(f"Expected {images_dir} and {annotations_dir} to both exist (read-only source layout)")

    xml_files = sorted(annotations_dir.glob("*.xml"))
    if args.limit:
        xml_files = xml_files[: args.limit]
    log.info("Found %d annotation XML file(s) in %s", len(xml_files), annotations_dir)

    done = 0
    skipped = 0
    total = len(xml_files)
    for i, xml_path in enumerate(xml_files, start=1):
        stem = xml_path.stem
        matches = sorted(images_dir.glob(f"{stem}.*"))
        if not matches:
            log.warning("[%d/%d] SKIP %s: no matching image found in %s", i, total, stem, images_dir)
            skipped += 1
            continue
        image_path = matches[0]
        boxes = parse_voc_annotation(xml_path)
        out_path = out_dir / image_path.name
        draw_original_annotation(image_path, boxes, out_path)
        log.info("[%d/%d] %s: %d box(es) -> %s", i, total, stem, len(boxes), out_path.name)
        done += 1

    log.info("Done. %d image(s) visualized, %d skipped (no matching image). Output: %s", done, skipped, out_dir)


if __name__ == "__main__":
    main()
