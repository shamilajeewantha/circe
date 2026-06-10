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
import os
import signal
import sys
import threading

import rclpy
import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from rclpy.executors import MultiThreadedExecutor

from camera_node import CameraNode
from drone_node import DroneNode
from rover_node import RoverNode

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
            executor.spin_once(timeout_sec=0.1)
    finally:
        executor.shutdown()
        if rclpy.ok():
            rclpy.shutdown()


# ── FastAPI app ─────────────────────────────────────────────────────────────────
app = FastAPI(title='Drone & Rover Controller')

STATIC_DIR = os.path.join(os.path.dirname(__file__), 'static')


@app.get('/')
async def index():
    return FileResponse(os.path.join(STATIC_DIR, 'index.html'))


# ── Camera stream ────────────────────────────────────────────────────────────────

@app.get('/camera/stream')
async def camera_stream():
    async def generate():
        try:
            while not _shutdown.is_set():
                frame = camera_node.get_frame() if camera_node else None
                if frame:
                    yield (
                        b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
                    )
                await asyncio.sleep(1 / 15)
        except (asyncio.CancelledError, GeneratorExit):
            pass

    return StreamingResponse(
        generate(),
        media_type='multipart/x-mixed-replace; boundary=frame',
    )


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
        log_level='warning',
        # Disable uvicorn's own signal handling so our handler runs first
        workers=1,
    )
    _shutdown.set()
