#!/usr/bin/env python3
"""Remote YOLO detector — runs on a second machine (ideally with a GPU).

Connects to the controller server running on laptop 1, receives JPEG frames over
the /ws/detect WebSocket, runs YOLO, and sends back JSON boxes. Same model and
inference params as the on-board worker (detect_worker.py) — only the transport
differs (network WebSocket instead of a local stdin/stdout pipe).

Run in a conda env that has ultralytics + websockets + opencv:
    python3 detect_client.py --server ws://<laptop1-ip>:8080
"""
import argparse
import asyncio
import json
import time
from pathlib import Path

import cv2
import numpy as np
import websockets
from ultralytics import YOLO

MODEL_PATH = Path(__file__).parent.parent / 'drone_detection' / 'best.pt'


async def run(url: str) -> None:
    print(f'[detector] loading model: {MODEL_PATH}')
    model = YOLO(str(MODEL_PATH))
    print(f'[detector] model loaded on device={model.device}')

    endpoint = url.rstrip('/') + '/ws/detect'
    while True:
        try:
            async with websockets.connect(endpoint, max_size=None, ping_interval=None) as ws:
                print(f'[detector] connected to {endpoint}')
                n, t0 = 0, time.time()
                async for message in ws:                   # binary JPEG, one at a time
                    arr = cv2.imdecode(np.frombuffer(message, np.uint8), cv2.IMREAD_COLOR)
                    boxes = []
                    for r in model(arr, verbose=False, conf=0.5, imgsz=1280):
                        for b in r.boxes:
                            x1, y1, x2, y2 = b.xyxy[0].tolist()
                            boxes.append({
                                'x1': int(x1), 'y1': int(y1),
                                'x2': int(x2), 'y2': int(y2),
                                'conf': round(float(b.conf[0]), 3),
                            })
                    await ws.send(json.dumps(boxes))
                    n += 1
                    if n % 30 == 0:
                        fps = n / (time.time() - t0)
                        print(f'[detector] {n} frames, {fps:.1f} fps, last={len(boxes)} boxes')
        except Exception as e:
            print(f'[detector] connection lost ({e}); retry in 2s')
            await asyncio.sleep(2)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--server', required=True, help='ws://<laptop1-ip>:8080')
    asyncio.run(run(ap.parse_args().server))
