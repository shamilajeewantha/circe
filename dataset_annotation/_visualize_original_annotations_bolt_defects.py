"""One-off script: draw bolt-defects__ext-sdnet2025's OWN original COCO-format annotations onto
their source images, completely as-shipped - no relabeling, no remapping to this pipeline's
bolt_ok/bolt_defective/bolt_corroded classes. This is the COCO equivalent of
_visualize_original_annotations.py (which handles npu_bolt's Pascal-VOC XML format instead) - this
dataset ships two SEPARATE COCO JSON files, one per defect group, not one XML per image, so it
needs its own script rather than sharing that one.

Authoritative label files (confirmed against model_training/pipeline/config.py's SDNET entry,
the single source of truth for this dataset's category map, this session):
    Defected/Annotated Loosen bolt & nuts/Annotated File/_annotations.coco.json   category "Loosen"
        (id 0 "LOOSEN" is an unused Roboflow root category, 0 real annotations - ignored)
    Defected/Annotated Missing bolt & nuts/Annotated File/annotations.coco.json   category "Missing"
        (+ 2 stray "Loosen" annotations mixed into this file - drawn too, not dropped, since this
        script draws originals as-is; config.py's own import pipeline is what remaps/drops them)
    Fixed/ has NO annotation file at all - these are the no-defect images, drawn with zero boxes
        (that absence IS the original label: "nothing wrong here").

datasets/bolt-defects__ext-sdnet2025/ is READ-ONLY (same invariant as datasets/npu_bolt/) - this
script only ever reads from it. Output goes to
circe_datasets/bolt-defects__ext-sdnet2025/qc/original_labels/, same tree shape as npu_bolt's
equivalent output, never into datasets/ itself.

Not part of the numbered 01-06 pipeline - a manual inspection aid only, run directly (plain cv2
dep already in the env, no torch/sam/genai needed).

Usage:
    python _visualize_original_annotations_bolt_defects.py
    python _visualize_original_annotations_bolt_defects.py --limit 20
"""
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import cv2

log = logging.getLogger("visualize_original_annotations_bolt_defects")

HERE = Path(__file__).parent

# (group label, images subdir relative to --datasets-dir, COCO json path relative to --datasets-dir)
DEFECT_GROUPS = [
    ("Loosen", "Defected/Annotated Loosen bolt & nuts/Resized images 640-640",
     "Defected/Annotated Loosen bolt & nuts/Annotated File/_annotations.coco.json"),
    ("Missing", "Defected/Annotated Missing bolt & nuts/Resized- 640-640",
     "Defected/Annotated Missing bolt & nuts/Annotated File/annotations.coco.json"),
]
FIXED_IMAGES_SUBDIR = "Fixed/640-640"

CLASS_COLORS_BGR = {
    "Loosen": (20, 20, 220),
    "Missing": (20, 90, 150),
}
FALLBACK_COLOR_BGR = (255, 255, 255)


def _rects_overlap(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> bool:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return ax0 < bx1 and bx0 < ax1 and ay0 < by1 and by0 < ay1


def load_coco_boxes_by_filename(json_path: Path) -> Dict[str, List[Tuple[str, int, int, int, int]]]:
    """Returns {file_name: [(category_name, x, y, w, h), ...]} - only categories that actually have
    at least one real annotation (a "supercategory: none" root with 0 annotations, e.g. Roboflow's
    auto-added "LOOSEN"/"Missing-dPh5", is never a real label - excluding it here isn't relabeling,
    it just never had any boxes to draw in the first place)."""
    data = json.loads(json_path.read_text(encoding="utf-8"))
    id_to_name = {c["id"]: c["name"] for c in data["categories"]}
    image_id_to_filename = {img["id"]: img["file_name"] for img in data["images"]}
    by_filename: Dict[str, List[Tuple[str, int, int, int, int]]] = {}
    for ann in data["annotations"]:
        filename = image_id_to_filename[ann["image_id"]]
        name = id_to_name[ann["category_id"]]
        x, y, w, h = (round(v) for v in ann["bbox"])
        by_filename.setdefault(filename, []).append((name, x, y, w, h))
    return by_filename


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

    for name, x, y, w, h in boxes:
        x0, y0, x1, y1 = x, y, x + w, y + h
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
    parser.add_argument("--datasets-dir", type=Path,
                         default=HERE / "datasets" / "bolt-defects__ext-sdnet2025",
                         help="Read-only source root (default: ./datasets/bolt-defects__ext-sdnet2025)")
    parser.add_argument("--out", type=Path,
                         default=HERE / "circe_datasets" / "bolt-defects__ext-sdnet2025" / "qc" / "original_labels",
                         help="Dedicated output folder - never datasets/ itself")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N images total (debugging)")
    args = parser.parse_args()

    src = args.datasets_dir.resolve()
    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(out_dir / "run_log.txt", encoding="utf-8"), logging.StreamHandler()],
        force=True,
    )

    jobs: List[Tuple[Path, List[Tuple[str, int, int, int, int]]]] = []

    for group_label, images_subdir, json_rel in DEFECT_GROUPS:
        json_path = src / json_rel
        images_dir = src / images_subdir
        if not json_path.is_file() or not images_dir.is_dir():
            log.warning("Skipping group %s - missing %s or %s", group_label, json_path, images_dir)
            continue
        by_filename = load_coco_boxes_by_filename(json_path)
        log.info("%s: %d image(s) with annotations, from %s", group_label, len(by_filename), json_path)
        for filename, boxes in by_filename.items():
            image_path = images_dir / filename
            if not image_path.is_file():
                log.warning("  %s: annotated in JSON but file not found at %s - skipping",
                            filename, image_path)
                continue
            jobs.append((image_path, boxes))

    fixed_dir = src / FIXED_IMAGES_SUBDIR
    if fixed_dir.is_dir():
        fixed_images = sorted(p for p in fixed_dir.iterdir() if p.is_file())
        log.info("Fixed (no-defect): %d image(s), zero boxes by definition", len(fixed_images))
        jobs.extend((p, []) for p in fixed_images)
    else:
        log.warning("Fixed images dir not found: %s", fixed_dir)

    if args.limit:
        jobs = jobs[: args.limit]
    total = len(jobs)
    log.info("Total images to visualize: %d", total)

    done = 0
    for i, (image_path, boxes) in enumerate(jobs, start=1):
        out_path = out_dir / image_path.name
        draw_original_annotation(image_path, boxes, out_path)
        log.info("[%d/%d] %s: %d box(es) -> %s", i, total, image_path.name, len(boxes), out_path.name)
        done += 1

    log.info("Done. %d image(s) visualized. Output: %s", done, out_dir)


if __name__ == "__main__":
    main()
