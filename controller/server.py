#!/usr/bin/env python3
"""
FastAPI controller server.
Starts ROS2 nodes in a background thread, serves the web UI and REST endpoints.

Usage:
    source /opt/ros/jazzy/setup.bash
    source ~/ws_px4/install/local_setup.bash
    python3 server.py

Open http://localhost:8080
"""

import asyncio
import json
import logging
import os
import signal
import sys
import threading

import rclpy
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel
from rclpy.executors import MultiThreadedExecutor

from camera_node import CameraNode
from drone_node import DroneNode
from rover_node import RoverNode

# ── Logging ──────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S',
    stream=sys.stdout,
)
log = logging.getLogger('controller')

# ── Shared shutdown event ────────────────────────────────────────────────────────
_shutdown = threading.Event()

# ── Node instances ───────────────────────────────────────────────────────────────
drone_node: DroneNode | None = None
rover_node: RoverNode | None = None
camera_node: CameraNode | None = None


def _ros_thread() -> None:
    global drone_node, rover_node, camera_node
    rclpy.init()
    drone_node = DroneNode()
    rover_node = RoverNode()
    camera_node = CameraNode()
    executor = MultiThreadedExecutor()
    executor.add_node(drone_node)
    executor.add_node(rover_node)
    executor.add_node(camera_node)
    try:
        while not _shutdown.is_set():
            try:
                executor.spin_once(timeout_sec=0.1)
            except Exception:
                if _shutdown.is_set():
                    break
                log.exception('[ros] spin_once error')
    finally:
        log.info('[ros] executor shutting down')
        executor.shutdown()
        if rclpy.ok():
            rclpy.shutdown()


# ── FastAPI app ─────────────────────────────────────────────────────────────────
app = FastAPI(title='Drone & Rover Controller')

STATIC_DIR = os.path.join(os.path.dirname(__file__), 'static')


@app.get('/')
async def index():
    return FileResponse(os.path.join(STATIC_DIR, 'index.html'),
                        headers={'Cache-Control': 'no-store'})


# ── Camera stream (WebSocket) ────────────────────────────────────────────────────

@app.websocket('/ws/camera')
async def camera_ws(websocket: WebSocket):
    client = websocket.client
    await websocket.accept()
    log.info(f'[WS] client connected: {client}')
    sent = 0
    none_count = 0
    try:
        while not _shutdown.is_set():
            frame = camera_node.get_frame() if camera_node else None
            if frame:
                await websocket.send_bytes(frame)
                sent += 1
                if sent == 1 or sent % 60 == 0:
                    log.info(f'[WS] sent {sent} frames (last={len(frame)} bytes) to {client}')
            else:
                none_count += 1
                if none_count == 1 or none_count % 30 == 0:
                    state = 'set' if camera_node else 'None'
                    log.warning(f'[WS] no frame available yet (camera_node={state}), count={none_count}')
            await asyncio.sleep(1 / 15)
    except WebSocketDisconnect:
        log.info(f'[WS] client disconnected: {client} after {sent} frames')
    except Exception:
        log.exception(f'[WS] handler error for {client} after {sent} frames')


# ── Remote detector link (laptop 2 connects here) ────────────────────────────────

@app.websocket('/ws/detect')
async def detect_ws(websocket: WebSocket):
    """A YOLO detector on another machine connects here: we push the latest JPEG,
    it pushes back JSON boxes. Lockstep request→response, skips unchanged frames."""
    client = websocket.client
    await websocket.accept()
    log.info(f'[detect] remote detector connected: {client}')
    last_id, n = -1, 0
    try:
        while not _shutdown.is_set():
            if camera_node is None:
                await asyncio.sleep(0.1)
                continue
            frame, fid = camera_node.get_frame_with_id()
            if frame is None or fid == last_id:
                await asyncio.sleep(0.02)
                continue
            last_id = fid
            await websocket.send_bytes(frame)
            boxes_json = await websocket.receive_text()
            camera_node.set_boxes(json.loads(boxes_json))
            n += 1
            if n == 1 or n % 30 == 0:
                log.info(f'[detect] result #{n} from {client}')
    except WebSocketDisconnect:
        log.info(f'[detect] remote detector disconnected: {client} after {n} results')
    except Exception:
        log.exception(f'[detect] handler error for {client}')


@app.get('/camera/snapshot')
async def camera_snapshot():
    frame = camera_node.get_frame() if camera_node else None
    if not frame:
        log.warning('[snapshot] requested but no frame available')
        return Response(status_code=503, content=b'no frame')
    return Response(content=frame, media_type='image/jpeg')


# ── Frontend log sink ────────────────────────────────────────────────────────────

class FrontendLog(BaseModel):
    level: str = 'info'
    msg: str


@app.post('/log')
async def frontend_log(entry: FrontendLog):
    lvl = getattr(logging, entry.level.upper(), logging.INFO)
    log.log(lvl, f'[FRONTEND] {entry.msg}')
    return {'ok': True}


@app.get('/detection/boxes')
async def detection_boxes():
    boxes = camera_node.get_boxes() if camera_node else []
    return {'boxes': boxes}


# ── Drone endpoints ──────────────────────────────────────────────────────────────

class DroneMoveRequest(BaseModel):
    dir: str
    step: float = 0.20


@app.post('/drone/move')
async def drone_move(req: DroneMoveRequest):
    if drone_node is None:
        return {'ok': False, 'error': 'node not ready'}
    drone_node.apply_increment(req.dir, req.step)
    return {'ok': True}


@app.get('/drone/status')
async def drone_status():
    if drone_node is None:
        return {'armed': False, 'preflight_ok': False, 'z_ned': 0.0, 'nav_state': -1}
    return drone_node.get_status()


# ── Rover endpoints ──────────────────────────────────────────────────────────────

class RoverMoveRequest(BaseModel):
    linear: float = 0.4
    angular: float = 0.0
    duration: float = 0.5


@app.post('/rover/move')
async def rover_move(req: RoverMoveRequest):
    if rover_node is None:
        return {'ok': False, 'error': 'node not ready'}
    rover_node.move_step(req.linear, req.angular, req.duration)
    return {'ok': True}


@app.post('/rover/brake')
async def rover_brake():
    if rover_node is None:
        return {'ok': False, 'error': 'node not ready'}
    rover_node.brake()
    return {'ok': True}


@app.post('/rover/release')
async def rover_release():
    if rover_node is None:
        return {'ok': False, 'error': 'node not ready'}
    rover_node.release()
    return {'ok': True}


@app.get('/rover/status')
async def rover_status():
    if rover_node is None:
        return {'braked': True}
    return rover_node.get_status()


# ── Entry point ──────────────────────────────────────────────────────────────────

def _handle_signal(sig, frame):
    print(f'\n[controller] Signal {sig} received, shutting down...')
    _shutdown.set()
    sys.exit(0)


if __name__ == '__main__':
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    ros_t = threading.Thread(target=_ros_thread, daemon=True)
    ros_t.start()

    print('Controller server starting on http://0.0.0.0:8080')
    uvicorn.run(
        app,
        host='0.0.0.0',
        port=8080,
        log_level='info',
        # Disable uvicorn's own signal handling so our handler runs first
        workers=1,
    )
    _shutdown.set()
