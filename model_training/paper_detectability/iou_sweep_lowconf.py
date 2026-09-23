"""Experiment 1 — is diffuse-defect failure a RECOGNITION or a LOCALISATION problem?

Motivation: per-class mAP50 does not track object size. efflorescence (median footprint
660 px) scores 0.209 while missing_bolt (31 px) scores 0.455. So the failing classes are not
resolution-limited. The leading alternative explanation is BOUNDARY AMBIGUITY: texture-defined
defects have no crisp edge, so the model finds them but cannot delineate them, and predictions
are discarded by the IoU criterion rather than never made.

Test: per-class recall as a function of the IoU matching threshold.
  * If a class has high recall at IoU 0.1 that collapses by IoU 0.5 -> the model IS finding it;
    the failure is localisation/extent, not recognition.
  * If recall is already low at IoU 0.1 -> the model genuinely does not detect it (recognition).

Frozen detector, held-out test split, no training.
"""
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
WEIGHTS = ROOT / "models" / "circe_yolo26n_13cls_100ep" / "best.pt"
IMAGES = ROOT / "merged_v2" / "test" / "images"
LABELS = ROOT / "merged_v2" / "test" / "labels"

NAMES = ["concrete_crack", "corrosion", "fluid_patch", "fire", "smoke", "gauge_face",
         "efflorescence", "exposed_rebar", "spalling", "missing_bolt", "bolt_ok",
         "bolt_defective", "bolt_corroded"]
IOUS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
CONF = 0.001


def xywhn_to_xyxy(b, W, H):
    xc, yc, w, h = b
    return np.array([(xc - w / 2) * W, (yc - h / 2) * H, (xc + w / 2) * W, (yc + h / 2) * H])


def iou_mat(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ax1, ay1, ax2, ay2 = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    iw = np.maximum(0, np.minimum(ax2, bx2) - np.maximum(ax1, bx1))
    ih = np.maximum(0, np.minimum(ay2, by2) - np.maximum(ay1, by1))
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return np.where(ua > 0, inter / np.maximum(ua, 1e-9), 0.0)


def main():
    model = YOLO(str(WEIGHTS))
    files = sorted(p for p in IMAGES.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    print(f"test images: {len(files)}")

    # matched[cls][iou] = count of GT matched ; total[cls] = GT count
    matched = defaultdict(lambda: defaultdict(int))
    total = defaultdict(int)

    B = 8
    for i in range(0, len(files), B):
        batch = files[i:i + B]
        results = model.predict(batch, conf=CONF, verbose=False, device=0)
        for p, res in zip(batch, results):
            lf = LABELS / f"{p.stem}.txt"
            if not lf.exists():
                continue
            H, W = res.orig_shape
            gt = defaultdict(list)
            for line in lf.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                c, *box = line.split()
                gt[int(c)].append(xywhn_to_xyxy([float(v) for v in box], W, H))

            pr = defaultdict(list)
            if res.boxes is not None and len(res.boxes):
                cls = res.boxes.cls.cpu().numpy().astype(int)
                xyxy = res.boxes.xyxy.cpu().numpy()
                for c, bb in zip(cls, xyxy):
                    pr[int(c)].append(bb)

            for c, boxes in gt.items():
                g = np.array(boxes)
                total[c] += len(g)
                q = np.array(pr.get(c, []))
                if len(q) == 0:
                    continue
                M = iou_mat(g, q)          # (n_gt, n_pred)
                best = M.max(axis=1)       # best IoU per GT, same class
                for t in IOUS:
                    matched[c][t] += int((best >= t).sum())

        if (i // B) % 20 == 0:
            print(f"  {i}/{len(files)}", flush=True)

    hdr = "class".ljust(16) + "n".rjust(6) + "".join(f"{t:>8.1f}" for t in IOUS)
    print("\nPER-CLASS RECALL vs IoU MATCHING THRESHOLD")
    print(hdr)
    print("-" * len(hdr))
    rows = {}
    for ci, nm in enumerate(NAMES):
        n = total.get(ci, 0)
        if n == 0:
            continue
        r = [matched[ci][t] / n for t in IOUS]
        rows[nm] = r
        print(nm.ljust(16) + str(n).rjust(6) + "".join(f"{v:>8.3f}" for v in r))

    print("\nCOLLAPSE RATIO  recall@0.5 / recall@0.1   (low = found but poorly delineated)")
    for nm, r in sorted(rows.items(), key=lambda kv: (kv[1][4] / kv[1][0]) if kv[1][0] > 0 else 9):
        ratio = (r[4] / r[0]) if r[0] > 0 else float("nan")
        print(f"  {nm:<16} r@0.1={r[0]:.3f}  r@0.5={r[4]:.3f}  ratio={ratio:.3f}")


if __name__ == "__main__":
    main()
