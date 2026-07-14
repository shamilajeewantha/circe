"""Evaluate a trained YOLO26 model on the held-out TEST split — official Ultralytics layout.

Training auto-validates on the `val` split every epoch, but the `test` split is NEVER touched during
training. Run this AFTER training for an honest, final generalization number on data the model never saw:
    python val.py
Docs: https://docs.ultralytics.com/modes/val/

Reports overall mAP50 / mAP50-95 plus PER-CLASS mAP (so you can see how the sparse classes — missing_bolt,
loose_bolt, efflorescence — actually do). Results + plots go to runs/detect/<RUN_NAME>/ and, with
SAVE_JSON, a COCO-style predictions.json for later analysis.
"""
from pathlib import Path

from ultralytics import YOLO

# ---- settings (edit these) --------------------------------------
WEIGHTS  = "runs/detect/train/weights/best.pt"   # model to evaluate (best val checkpoint).
# ^ set to the exact best.pt path train.py printed — on re-runs the dir auto-increments (train2, train3, ...)
DATA     = "/home/shamila/datasets/circe_merged/data.yaml"
SPLIT    = "test"          # "test" = held-out split; "val" reproduces the training-time validation
IMGSZ    = 640
BATCH    = 16
DEVICE   = "0"
RUN_NAME = "val"           # Ultralytics' OFFICIAL DEFAULT name -> runs/detect/val/ (auto-increments val2, ...)
SAVE_JSON = True           # write COCO-format predictions.json for offline analysis
# -----------------------------------------------------------------

HERE = Path(__file__).parent
PROJECT = HERE / "runs" / "detect"


def main():
    weights = HERE / WEIGHTS
    if not weights.exists():
        raise SystemExit(f"weights not found: {weights}\nTrain first (python train.py).")

    model = YOLO(str(weights))
    m = model.val(
        data=DATA, split=SPLIT, imgsz=IMGSZ, batch=BATCH, device=DEVICE,
        project=str(PROJECT), name=RUN_NAME, exist_ok=False,   # official default: auto-increment, never overwrite
        save_json=SAVE_JSON, plots=True,
    )

    print(f"\n=== {SPLIT} split ===")
    print(f"mAP50-95: {m.box.map:.4f}")
    print(f"mAP50   : {m.box.map50:.4f}")
    print(f"mAP75   : {m.box.map75:.4f}")
    print("\nper-class mAP50-95:")
    # ap_class_index only contains classes that HAVE ground-truth instances in this split, so iterate over
    # ALL taxonomy classes and mark the ones with no test data explicitly — otherwise sparse classes
    # (missing_bolt, loose_bolt, efflorescence) would silently vanish from the report.
    present = {int(i): float(ap) for i, ap in zip(m.box.ap_class_index, m.box.maps[m.box.ap_class_index])}
    for cid in range(len(model.names)):
        name = model.names[cid]
        if cid in present:
            print(f"  {name:16s} {present[cid]:.4f}")
        else:
            print(f"  {name:16s}   (no {SPLIT} instances)")
    print(f"\nresults saved to: {m.save_dir}")


if __name__ == "__main__":
    main()
