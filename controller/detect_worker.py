#!/usr/bin/env python3
"""
Runs under /home/shamila/anaconda3/envs/drone_detect/bin/python3
Pipe protocol:
  stdin:  4-byte big-endian length + JPEG bytes (one frame at a time)
  stdout: one JSON line per frame — list of {x1,y1,x2,y2,conf} dicts
          first line is 'ready' (sent once model finishes loading)
"""
import json
import struct
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

MODEL_PATH = Path(__file__).parent.parent / 'drone_detection' / 'best.pt'
model = YOLO(str(MODEL_PATH))

sys.stdout.write('ready\n')
sys.stdout.flush()

while True:
    header = sys.stdin.buffer.read(4)
    if len(header) < 4:
        break
    n = struct.unpack('>I', header)[0]
    data = sys.stdin.buffer.read(n)
    if len(data) < n:
        break

    arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    boxes = []
    for r in model(arr, verbose=False, conf=0.3, imgsz=1280):
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            boxes.append({
                'x1': int(x1), 'y1': int(y1),
                'x2': int(x2), 'y2': int(y2),
                'conf': round(float(box.conf[0]), 3),
            })

    sys.stdout.write(json.dumps(boxes) + '\n')
    sys.stdout.flush()
