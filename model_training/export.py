"""Export a trained YOLO26 model to a deployment format — official Ultralytics layout.

    python export.py
Docs: https://docs.ultralytics.com/modes/export/

ONNX is the portable default (runs on CPU/GPU via onnxruntime, easy to ship to the rover). For max GPU
throughput on an NVIDIA target, export FORMAT="engine" (TensorRT) ON THE DEPLOYMENT GPU. The exported
file is written next to the source weights (runs/detect/train/weights/best.onnx).
"""
from pathlib import Path

from ultralytics import YOLO

# ---- settings (edit these) --------------------------------------
WEIGHTS = "runs/detect/train/weights/best.pt"   # set to the best.pt path train.py printed
# ^ on re-runs the run dir auto-increments (train2, train3, ...); match whatever train.py reported
FORMAT  = "onnx"     # onnx | engine (TensorRT) | openvino | torchscript | tflite | ...
IMGSZ   = 640
HALF    = False      # FP16 export (GPU formats only, e.g. engine)
DEVICE  = "0"
# -----------------------------------------------------------------

HERE = Path(__file__).parent


def main():
    weights = HERE / WEIGHTS
    if not weights.exists():
        raise SystemExit(f"weights not found: {weights}\nTrain first (python train.py).")

    model = YOLO(str(weights))
    out = model.export(format=FORMAT, imgsz=IMGSZ, half=HALF, device=DEVICE)
    print(f"\nexported {FORMAT} -> {out}")


if __name__ == "__main__":
    main()
