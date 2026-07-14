"""Run the trained YOLO26 model on images / a folder / a video — official Ultralytics layout.

Edit the constants below and run:  python predict.py
Annotated results are saved to the official runs/detect/predict/ tree.
Docs: https://docs.ultralytics.com/modes/predict/
"""
from pathlib import Path

from ultralytics import YOLO

# ---- settings (edit these) --------------------------------------
WEIGHTS = "runs/detect/train/weights/best.pt"   # trained model (best.pt = highest val mAP).
# ^ set to the exact best.pt path train.py printed — on re-runs the dir auto-increments (train2, train3, ...)
# SOURCE  = "sample_downloaded_nuts_clean"        # folder / single image / video / glob
SOURCE  = "sample_images"        # folder / single image / video / glob

CONF    = 0.25                                  # min confidence to keep a detection
IOU     = 0.70                                  # NMS IoU threshold
IMGSZ   = 640
DEVICE  = "0"                                   # "0" = GPU, "cpu" = CPU
RUN_NAME = "predict"                            # -> runs/detect/predict/ (auto-increments predict2, ...)
SAVE     = True                                 # write annotated images/video
SAVE_TXT = False                                # also write YOLO-format label .txt per image
# -----------------------------------------------------------------

HERE = Path(__file__).parent
PROJECT = HERE / "runs" / "detect"


def main():
    weights = HERE / WEIGHTS
    if not weights.exists():
        raise SystemExit(f"weights not found: {weights}\nTrain first (python train.py) or fix WEIGHTS.")

    src = HERE / SOURCE
    model = YOLO(str(weights))
    results = model.predict(
        source=str(src if src.exists() else SOURCE),   # allow absolute/glob sources too
        conf=CONF, iou=IOU, imgsz=IMGSZ, device=DEVICE,
        project=str(PROJECT), name=RUN_NAME, exist_ok=False,   # official default: auto-increment, never overwrite
        save=SAVE, save_txt=SAVE_TXT,
    )

    # per-image detection summary
    for r in results:
        n = len(r.boxes)
        labels = [r.names[int(c)] for c in r.boxes.cls] if n else []
        print(f"{Path(r.path).name}: {n} detections {labels}")

    # actual dir (auto-increments predict2, ... on re-runs), not the static PROJECT/RUN_NAME
    print(f"\nannotated results saved to: {model.predictor.save_dir}")


if __name__ == "__main__":
    main()
