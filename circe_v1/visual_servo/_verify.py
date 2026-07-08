"""Isolated smoke test of the same calls main.py makes — does NOT import
main.py (that would spin up a second Engine + open the webcam a second time
while the real app is already running)."""
import os
import time

import cv2
from ultralytics import YOLO

_BEST_PT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "drone_detection", "best.pt"))

out = []

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)
out.append(f"camera opened: {cap.isOpened()}")

frame = None
if cap.isOpened():
    for _ in range(5):
        ok, frame = cap.read()
        if ok and frame is not None:
            break
        time.sleep(0.2)
out.append(f"frame captured: {frame is not None}  shape={getattr(frame, 'shape', None)}")
cap.release()

model = YOLO(_BEST_PT)
out.append(f"model loaded: {model is not None}")

if frame is not None:
    t0 = time.time()
    results = model(frame, imgsz=640, conf=0.3, device="cuda:0", verbose=False)
    dt = time.time() - t0
    annotated = results[0].plot()
    out.append(f"inference ok: {results is not None}  took {dt*1000:.1f}ms  annotated shape={annotated.shape}")
    out.append(f"detections: {len(results[0].boxes)}")

with open(os.path.join(os.path.dirname(__file__), "_verify_out.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(out) + "\n")
