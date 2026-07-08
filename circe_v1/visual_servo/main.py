"""
Webcam YOLO detector bench — Gradio app.

Live webcam -> inference -> annotated feed, with dropdowns to swap
model/imgsz/conf/device and see FPS + model size change in real time.

No drone control here — this is just for picking detector settings before
wiring detection into anything. Run with the `drone_detect` conda env
(has torch+CUDA+ultralytics already):

    "D:\\PROGRAM_FILES\\anaconda\\envs\\drone_detect\\python" main.py
"""

import os
import threading
import time

import cv2
import gradio as gr
import torch
from ultralytics import YOLO

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_BEST_PT = os.path.normpath(os.path.join(_THIS_DIR, "..", "..", "drone_detection", "best.pt"))

MODELS = {
    "best.pt (drone detector)": _BEST_PT,
    "yolov8n.pt (COCO, nano)": "yolov8n.pt",
    "yolov8s.pt (COCO, small)": "yolov8s.pt",
    "yolov8m.pt (COCO, medium)": "yolov8m.pt",
    "yolov8l.pt (COCO, large)": "yolov8l.pt",
    "yolov8x.pt (COCO, xlarge)": "yolov8x.pt",
}
IMGSZ_OPTIONS = [320, 480, 640, 960, 1280]
DEVICES = ["cuda:0", "cpu"] if torch.cuda.is_available() else ["cpu"]


class FpsMeter:
    """Rolling frames/sec over the last ~2s of ticks."""

    def __init__(self, window: float = 2.0):
        self._window = window
        self._stamps: list[float] = []

    def tick(self):
        now = time.time()
        self._stamps.append(now)
        cutoff = now - self._window
        while self._stamps and self._stamps[0] < cutoff:
            self._stamps.pop(0)

    @property
    def fps(self) -> float:
        if len(self._stamps) < 2:
            return 0.0
        span = self._stamps[-1] - self._stamps[0]
        return (len(self._stamps) - 1) / span if span > 0 else 0.0


class Engine:
    """Owns the webcam, the current model, and the two background loops
    (capture, inference) that run decoupled so inference speed never
    blocks/slows frame capture."""

    def __init__(self):
        self.lock = threading.Lock()
        self.cap = None
        self.model = None
        self.model_name = None
        self.model_info = ""
        self.imgsz = 640
        self.conf = 0.3
        self.device = DEVICES[0]
        self.latest_frame = None
        self.latest_annotated = None
        self.det_count = 0
        self.top_conf = 0.0
        self.status = "starting..."
        self.cap_fps = FpsMeter()
        self.inf_fps = FpsMeter()
        self.running = True

        self._open_camera(0)
        self._load_model("best.pt (drone detector)")
        threading.Thread(target=self._capture_loop, daemon=True).start()
        threading.Thread(target=self._infer_loop, daemon=True).start()

    # ── camera ──────────────────────────────────────────────────────────
    def _open_camera(self, index: int):
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(index)
        with self.lock:
            if self.cap is not None:
                self.cap.release()
            self.cap = cap if cap.isOpened() else None
            if self.cap is None:
                self.status = f"Could not open webcam index {index}"

    def set_camera(self, index):
        threading.Thread(target=self._open_camera, args=(int(index),), daemon=True).start()

    # ── model ───────────────────────────────────────────────────────────
    def _load_model(self, label: str):
        path = MODELS[label]
        with self.lock:
            self.status = f"Loading {label}..."
        try:
            model = YOLO(path)
            info = []
            try:
                nparams = sum(p.numel() for p in model.model.parameters()) / 1e6
                info.append(f"{nparams:.1f}M params")
            except Exception:
                pass
            for candidate in (path, getattr(model, "ckpt_path", None), getattr(model, "pt_path", None)):
                if candidate and os.path.exists(candidate):
                    info.append(f"{os.path.getsize(candidate) / 1e6:.0f} MB")
                    break
            with self.lock:
                self.model = model
                self.model_name = label
                self.model_info = " · ".join(info)
                self.status = f"Loaded {label}"
        except Exception as e:
            with self.lock:
                self.status = f"Failed to load {label}: {e}"

    def set_model(self, label):
        threading.Thread(target=self._load_model, args=(label,), daemon=True).start()

    def set_imgsz(self, v):
        with self.lock:
            self.imgsz = int(v)

    def set_conf(self, v):
        with self.lock:
            self.conf = float(v)

    def set_device(self, v):
        with self.lock:
            self.device = v

    # ── background loops ───────────────────────────────────────────────
    def _capture_loop(self):
        while self.running:
            with self.lock:
                cap = self.cap
            if cap is None:
                time.sleep(0.2)
                continue
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.05)
                continue
            with self.lock:
                self.latest_frame = frame
                self.cap_fps.tick()

    def _infer_loop(self):
        while self.running:
            with self.lock:
                frame = self.latest_frame
                model = self.model
                imgsz, conf, device = self.imgsz, self.conf, self.device
            if frame is None or model is None:
                time.sleep(0.02)
                continue
            try:
                results = model(frame, imgsz=imgsz, conf=conf, device=device, verbose=False)
                annotated = results[0].plot()
                boxes = results[0].boxes
                det_count = len(boxes)
                top_conf = float(boxes.conf.max()) if det_count else 0.0
            except Exception as e:
                with self.lock:
                    self.status = f"Inference error: {e}"
                time.sleep(0.2)
                continue
            with self.lock:
                self.latest_annotated = annotated
                self.det_count = det_count
                self.top_conf = top_conf
                self.inf_fps.tick()

    def snapshot(self):
        with self.lock:
            frame = self.latest_annotated if self.latest_annotated is not None else self.latest_frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if frame is not None else None
            stats = (f"capture: {self.cap_fps.fps:.1f} fps   |   inference: {self.inf_fps.fps:.1f} fps   |   "
                     f"detections: {self.det_count} (top conf {self.top_conf:.2f})   |   "
                     f"{self.model_name} — {self.model_info}   |   device: {self.device}   |   "
                     f"{self.status}")
        return frame_rgb, stats


engine = Engine()

with gr.Blocks(title="Visual Servo — Detector Bench") as demo:
    gr.Markdown("# Detector Bench\nLive webcam inference — pick a model/imgsz/conf/device and watch FPS. No drone control.")
    with gr.Row():
        with gr.Column(scale=3):
            img = gr.Image(label="Live")
            stats = gr.Textbox(label="Stats", interactive=False)
        with gr.Column(scale=1):
            model_dd = gr.Dropdown(list(MODELS.keys()), value="best.pt (drone detector)", label="Model")
            imgsz_dd = gr.Dropdown(IMGSZ_OPTIONS, value=640, label="Image size")
            conf_sl = gr.Slider(0.05, 0.9, value=0.3, step=0.05, label="Confidence threshold")
            device_dd = gr.Dropdown(DEVICES, value=DEVICES[0], label="Device")
            cam_num = gr.Number(value=0, precision=0, label="Camera index")

    model_dd.change(engine.set_model, inputs=model_dd)
    imgsz_dd.change(engine.set_imgsz, inputs=imgsz_dd)
    conf_sl.change(engine.set_conf, inputs=conf_sl)
    device_dd.change(engine.set_device, inputs=device_dd)
    cam_num.change(engine.set_camera, inputs=cam_num)

    timer = gr.Timer(0.1)
    timer.tick(engine.snapshot, outputs=[img, stats])

if __name__ == "__main__":
    demo.launch()
