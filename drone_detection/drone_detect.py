from pathlib import Path
from ultralytics import YOLO

images = list(Path("images").glob("*.jpg")) + list(Path("images").glob("*.png"))

model = YOLO("best.pt")
results = model.predict(source=images, conf=0.3, iou=0.5, save=True)
print("Done — check runs/detect/predict/")