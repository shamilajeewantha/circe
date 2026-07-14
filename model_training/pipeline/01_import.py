"""Step 1 — import all source YOLO folders into ONE FiftyOne dataset.

- Reads every INCLUDED folder in datasets/ (by reference; source files untouched).
- Remaps each source's raw classes into the unified 9-class `ground_truth` field
  using config.CLASS_MAP; boxes mapped to None are dropped (box-level).
- Records per-sample `source` and `orig_split`.
- Images whose only boxes were dropped are kept as background/negative samples.
- EXCLUDED folders (config.EXCLUDE_FOLDERS) are not imported; every one of their
  images is written to the drop manifest with a reason.

Run in the `drone_detect` env:  python pipeline/01_import.py
"""
import json
from pathlib import Path

import fiftyone as fo

from config import (BACKGROUND_DIRS, CLASS_MAP, COCO_SOURCES, DATASETS_DIR, EXCLUDE_FOLDERS,
                    FO_DATASET_NAME, IMG_EXTS, SPLITS_ON_DISK)
from _util import (DropLog, iter_split_images, read_yolo, source_names, yolo_to_fo)


def import_coco(source, spec, samples, per_source, drops):
    """Manually parse one COCO json; remap category names -> taxonomy; drop unmapped boxes.
    Images referenced by the json but missing on disk are skipped and logged."""
    data_dir = DATASETS_DIR / spec["data_path"]
    coco = json.loads((DATASETS_DIR / spec["labels_path"]).read_text(encoding="utf-8"))
    cats = {c["id"]: c["name"] for c in coco.get("categories", [])}
    imgs = {im["id"]: im for im in coco.get("images", [])}
    anns_by_img = {}
    for a in coco.get("annotations", []):
        anns_by_img.setdefault(a["image_id"], []).append(a)
    cmap = spec["category_map"]

    stats = {"images": 0, "boxes": 0, "background": 0}
    for iid, im in imgs.items():
        path = data_dir / im["file_name"]
        if not path.exists():
            drops.add(path, source, "import:coco-missing-file",
                      "image referenced by COCO json not found on disk")
            continue
        W, H = im.get("width"), im.get("height")
        dets = []
        for a in anns_by_img.get(iid, []):
            tgt = cmap.get(cats.get(a["category_id"]))
            if tgt is None:
                continue
            x, y, w, h = a["bbox"]                       # COCO: absolute [x,y,w,h]
            bb = [max(x / W, 0.0), max(y / H, 0.0), min(w / W, 1.0), min(h / H, 1.0)]
            dets.append(fo.Detection(label=tgt, bounding_box=bb))
        ns = fo.Sample(filepath=str(path))
        ns["source"] = source
        ns["orig_split"] = "coco"
        ns["ground_truth"] = fo.Detections(detections=dets)
        ns.tags.append(source)
        if not dets:
            ns.tags.append("background")
            stats["background"] += 1
        stats["images"] += 1
        stats["boxes"] += len(dets)
        samples.append(ns)
    per_source[source] = stats


def import_backgrounds(source, rel_dir, samples, per_source):
    """Import a folder of no-defect images as background/negative samples."""
    d = DATASETS_DIR / rel_dir
    n = 0
    for img in sorted(d.iterdir()) if d.is_dir() else []:
        if img.suffix.lower() not in IMG_EXTS:
            continue
        ns = fo.Sample(filepath=str(img))
        ns["source"] = source
        ns["orig_split"] = "coco"
        ns["ground_truth"] = fo.Detections(detections=[])
        ns.tags.extend([source, "background"])
        samples.append(ns)
        n += 1
    per_source[source] = {"images": n, "boxes": 0, "background": n}


def main():
    drops = DropLog(reset=True)   # fresh manifest for the whole run

    if FO_DATASET_NAME in fo.list_datasets():
        fo.delete_dataset(FO_DATASET_NAME)
    dataset = fo.Dataset(FO_DATASET_NAME)
    dataset.persistent = True

    samples = []
    per_source = {}     # folder -> {"images": n, "boxes": n, "background": n}

    for folder, class_map in CLASS_MAP.items():
        names = source_names(folder)
        stats = {"images": 0, "boxes": 0, "background": 0}
        for split in SPLITS_ON_DISK:
            for img_path, lbl_path in iter_split_images(folder, split):
                dets = []
                for cid, xc, yc, w, h in read_yolo(lbl_path):
                    target = class_map.get(cid)          # None or missing => drop this box
                    if target is None:
                        continue
                    dets.append(fo.Detection(label=target, bounding_box=yolo_to_fo(xc, yc, w, h)))
                s = fo.Sample(filepath=str(img_path))
                s["source"] = folder
                s["orig_split"] = split
                s["ground_truth"] = fo.Detections(detections=dets)
                s.tags.append(folder)
                if not dets:
                    s.tags.append("background")
                    stats["background"] += 1
                stats["images"] += 1
                stats["boxes"] += len(dets)
                samples.append(s)
        per_source[folder] = stats

    # COCO sources (SDNET bolt defects) + no-defect backgrounds
    for source, spec in COCO_SOURCES.items():
        import_coco(source, spec, samples, per_source, drops)
    for source, rel_dir in BACKGROUND_DIRS.items():
        import_backgrounds(source, rel_dir, samples, per_source)

    dataset.add_samples(samples)

    # log excluded folders (image-level drops, with reason)
    for folder, reason in EXCLUDE_FOLDERS.items():
        n = 0
        for split in SPLITS_ON_DISK:
            for img_path, _ in iter_split_images(folder, split):
                drops.add(img_path, folder, "import:excluded-folder", reason)
                n += 1
        per_source[folder] = {"excluded": reason, "images": n}
    drops.close()

    # ---- report to console ----
    print(f"FiftyOne dataset '{FO_DATASET_NAME}': {len(dataset)} samples")
    for folder, st in per_source.items():
        if "excluded" in st:
            print(f"  EXCLUDED {folder:32s} {st['images']:6d} imgs  ({st['excluded']})")
        else:
            print(f"  {folder:32s} imgs={st['images']:6d} boxes={st['boxes']:6d} bg={st['background']}")

    present = set(dataset.distinct("ground_truth.detections.label"))
    from config import TAXONOMY
    print("\nlabels present:", sorted(present))
    unexpected = present - set(TAXONOMY)
    assert not unexpected, f"labels outside taxonomy: {unexpected}"
    print(f"OK: all labels within the {len(TAXONOMY)}-class taxonomy.")


if __name__ == "__main__":
    main()
