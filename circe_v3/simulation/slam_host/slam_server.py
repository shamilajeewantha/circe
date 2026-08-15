"""VGGT-SLAM host server — runs the SLAM on THIS laptop, exposes a small HTTP API.

Topology (see circe_v3/simulation/BUILD_GUIDE.md and the plan): the rover's brain
runs elsewhere (the Gazebo sim laptop, or the real robot) and talks to this
off-board SLAM over the network — exactly the project.md §9 split. This server:

  * builds one VGGT + Solver (same sequence as VGGT-SLAM's main_realtime.py),
  * feeds it frames through a NetworkCamera backend (frames arrive over HTTP),
  * runs the incremental submap loop in a background thread,
  * serves poses + point cloud back out.

Because VGGT-SLAM ingests frames only through its Camera interface, the SAME
server runs unchanged whether frames come from Gazebo or a real RPi camera.

Endpoints
---------
  POST /session   {intrinsics, width, height, submap_size?}  -> config ack
  POST /frames    multipart JPEG file(s)                     -> {received, queued}
  GET  /pose/latest                                          -> latest camera pose (rel scale)
  GET  /map?after_submap=&known_loops=                       -> poses + cloud (delta or full_refresh)
  GET  /status                                               -> solver + camera counters

Run (WSL `vggt` env, GPU):
  python slam_server.py --port 8000 --submap_size 8 --vis_map
The Solver also opens VGGT-SLAM's own viser raw-map viewer (default :8080).

This file imports the installed `vggt_slam` and does NOT edit vendored source, so
`git pull`s of VGGT-SLAM stay clean.
"""

from __future__ import annotations

import argparse
import base64
import io
import logging
import os
import threading
import time
from datetime import datetime, timezone
from typing import List, Optional

import cv2
import numpy as np
import torch

log = logging.getLogger("slam_server")


def _setup_logging(log_dir: str) -> None:
    """Console + a UTF-8 log file per run (repo convention: never print-only for
    a long-running loop, always leave a file a session can be reconstructed from)."""
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = os.path.join(log_dir, f"slam_server_{ts}.log")
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh = logging.FileHandler(path, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(fh)
    root.addHandler(sh)
    log.info("Logging to %s", path)

# Register the network backend BEFORE anything reads BACKENDS.
from vggt_slam.cameras import BACKENDS
from network_camera import NetworkCamera

BACKENDS["network"] = NetworkCamera

from vggt_slam.solver import Solver          # noqa: E402
from vggt.models.vggt import VGGT            # noqa: E402

from fastapi import FastAPI, UploadFile, File, Query   # noqa: E402
from fastapi.responses import JSONResponse             # noqa: E402
import uvicorn                                          # noqa: E402


# --------------------------------------------------------------------------- #
# Shared SLAM state (one Solver for the process lifetime; viser binds a port
# once, so we never rebuild it — same constraint the repo's gradio_demo.py notes)
# --------------------------------------------------------------------------- #
solver_lock = threading.Lock()   # only one submap inference in flight
data_lock = threading.Lock()     # protects solver map/graph reads & writes

_model: Optional[VGGT] = None
_solver: Optional[Solver] = None
_camera: Optional[NetworkCamera] = None
_cfg = argparse.Namespace()
_worker_started = threading.Event()
_stop = threading.Event()


def load_model(device: str) -> VGGT:
    """Load VGGT-1B — from the torch.hub checkpoint if present, else the HF URL
    (mirrors main.py's cached path with main_realtime.py's URL as fallback)."""
    model = VGGT()
    cached = os.path.join(torch.hub.get_dir(), "checkpoints/model.pt")
    if os.path.exists(cached):
        # VGGT-1B is a pure state_dict → weights_only=True (safe unpickler).
        state = torch.load(cached, map_location="cpu", mmap=True, weights_only=True)
        model.load_state_dict(state)
        del state
    else:
        url = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
        model.load_state_dict(torch.hub.load_state_dict_from_url(url))
    model.eval()
    model = model.to(torch.bfloat16).to(device)
    return model


def _process_submap(frames: List[str]) -> None:
    """Run VGGT + graph optimisation for one submap (background thread)."""
    try:
        preds = _solver.run_predictions(frames, _model, _cfg.max_loops, None, None)
        with data_lock:
            _solver.add_points(preds)
            _solver.graph.optimize()
            if _cfg.vis_map:
                if len(preds.get("detected_loops", [])) > 0:
                    _solver.update_all_submap_vis()
                else:
                    _solver.update_latest_submap_vis()
        log.info("submap done (submaps=%d loops=%d)",
                 _solver.map.get_num_submaps(), _solver.graph.get_num_loops())
    except Exception:  # keep the loop alive on a bad submap
        log.exception("submap processing failed")
    finally:
        if solver_lock.locked():
            solver_lock.release()


def slam_worker() -> None:
    """The capture -> keyframe-gate -> submap loop (mirrors main_realtime.py)."""
    _camera.start()
    subset: List[str] = []
    frame_count = 0
    target = _cfg.submap_size + _cfg.overlapping_window_size
    os.makedirs(_cfg.keyframe_folder, exist_ok=True)
    _worker_started.set()
    log.info("worker started; waiting for frames...")

    while not _stop.is_set():
        img = _camera.capture(timeout=1.0)
        if img is None:
            continue
        frame_count += 1
        if _solver.flow_tracker.compute_disparity(img, _cfg.min_disparity, False):
            path = os.path.join(_cfg.keyframe_folder, f"frame_{frame_count:06d}.png")
            cv2.imwrite(path, img)
            subset.append(path)

        if frame_count % 25 == 0:      # mandatory progress signal for a loop with no fixed N
            log.info("[frame %d] keyframes_pending=%d/%d camera=%s",
                      frame_count, len(subset), target, _camera.stats())

        if len(subset) >= target:
            if solver_lock.acquire(blocking=False):
                t = threading.Thread(target=_process_submap, args=(list(subset),), daemon=True)
                t.start()
                subset = subset[-_cfg.overlapping_window_size:]
            elif len(subset) > target * 2:      # SLAM busy; bound the backlog
                subset = subset[-target:]


# --------------------------------------------------------------------------- #
# Egress helpers — read the map through the same public getters main.py /
# gradio_demo.py use (confirmed present at commit 35327ac).
# --------------------------------------------------------------------------- #
def _b64(arr: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode("ascii")


def _collect(submaps, voxel: float, max_points: int):
    poses_out, xyz_all, rgb_all = [], [], []
    for sm in submaps:
        poses = sm.get_all_poses_world(_solver.graph)          # (F,4,4) rel-scale
        poses_out.append({"submap_id": int(sm.get_id()),
                          "loop_closure": bool(sm.get_lc_status()),
                          "poses": poses.astype(np.float32).tolist()})
        pts = sm.get_points_in_world_frame(_solver.graph)
        cols = sm.get_points_colors()
        if pts is not None and len(pts) > 0:
            xyz_all.append(np.asarray(pts).reshape(-1, 3))
            rgb_all.append(np.asarray(cols).reshape(-1, 3))

    if xyz_all:
        xyz = np.concatenate(xyz_all, 0).astype(np.float32)
        rgb = np.concatenate(rgb_all, 0).astype(np.uint8)
        if voxel > 0:                                          # cheap grid dedupe
            keys = np.floor(xyz / voxel).astype(np.int64)
            _, idx = np.unique(keys, axis=0, return_index=True)
            xyz, rgb = xyz[idx], rgb[idx]
        if max_points and len(xyz) > max_points:
            sel = np.linspace(0, len(xyz) - 1, max_points).astype(int)
            xyz, rgb = xyz[sel], rgb[sel]
        cloud = {"n": int(len(xyz)), "xyz_f32_b64": _b64(xyz), "rgb_u8_b64": _b64(rgb)}
    else:
        cloud = {"n": 0, "xyz_f32_b64": "", "rgb_u8_b64": ""}
    return poses_out, cloud


# --------------------------------------------------------------------------- #
# HTTP API
# --------------------------------------------------------------------------- #
app = FastAPI(title="circe VGGT-SLAM host")


@app.post("/session")
def session(payload: dict):
    """Record camera intrinsics / config for the current run. Note: a *fresh*
    map needs a server restart (viser binds its port once) — this endpoint
    configures, it does not rebuild the Solver."""
    _cfg.intrinsics = payload.get("intrinsics")
    _cfg.width = payload.get("width")
    _cfg.height = payload.get("height")
    if payload.get("submap_size"):
        _cfg.submap_size = int(payload["submap_size"])
    return {"ok": True, "submap_size": _cfg.submap_size,
            "note": "relative scale; restart server for a fresh map"}


@app.post("/frames")
async def frames(files: List[UploadFile] = File(...)):
    """Receive JPEG keyframes from the rover and push them to the SLAM loop."""
    n = 0
    for f in files:
        buf = np.frombuffer(await f.read(), np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)   # BGR uint8
        if img is not None:
            _camera.push(img)
            n += 1
    return {"received": n, "queued": _camera.stats()["queued"]}


@app.get("/pose/latest")
def pose_latest():
    with data_lock:
        if _solver.map.get_num_submaps() == 0:
            return JSONResponse({"available": False})
        last = list(_solver.map.ordered_submaps_by_key())[-1]
        poses = last.get_all_poses_world(_solver.graph)
        return {"available": True, "submap_id": int(last.get_id()),
                "T_cam_world": poses[-1].astype(np.float32).tolist(),
                "num_submaps": int(_solver.map.get_num_submaps()),
                "num_loops": int(_solver.graph.get_num_loops())}


@app.get("/map")
def get_map(after_submap: int = Query(-1), known_loops: int = Query(0),
            voxel: float = Query(0.0), max_points: int = Query(400000)):
    """Return poses + cloud. A loop closure re-optimises *all* poses, so if the
    loop count grew we flag ``full_refresh`` and return the whole map for the
    client to rebuild (and re-run its motion self-calibration); otherwise we
    return only submaps newer than ``after_submap``."""
    with data_lock:
        num_loops = int(_solver.graph.get_num_loops())
        num_submaps = int(_solver.map.get_num_submaps())
        all_submaps = list(_solver.map.ordered_submaps_by_key())
        full_refresh = num_loops > known_loops
        chosen = all_submaps if full_refresh else \
            [sm for sm in all_submaps if int(sm.get_id()) > after_submap]
        poses_out, cloud = _collect(chosen, voxel, max_points)
    return {"full_refresh": full_refresh, "num_submaps": num_submaps,
            "num_loops": num_loops, "submaps": poses_out, "cloud": cloud}


@app.get("/status")
def status():
    with data_lock:
        ns = int(_solver.map.get_num_submaps()) if _solver else 0
        nl = int(_solver.graph.get_num_loops()) if _solver else 0
    return {"worker_started": _worker_started.is_set(),
            "num_submaps": ns, "num_loops": nl,
            "camera": _camera.stats() if _camera else None}


def main() -> None:
    global _model, _solver, _camera
    p = argparse.ArgumentParser(description="circe VGGT-SLAM host server")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--submap_size", type=int, default=8)
    p.add_argument("--overlapping_window_size", type=int, default=1)
    p.add_argument("--max_loops", type=int, default=1)
    p.add_argument("--min_disparity", type=float, default=50.0)
    p.add_argument("--conf_threshold", type=float, default=25.0)
    p.add_argument("--lc_thres", type=float, default=0.95)
    p.add_argument("--vis_map", action="store_true", help="open VGGT-SLAM's viser raw-map viewer")
    p.add_argument("--vis_voxel_size", type=float, default=None)
    p.add_argument("--keyframe_folder", default="slam_host_keyframes")
    p.add_argument("--log_dir", default="slam_server_logs")
    args = p.parse_args()
    _cfg.__dict__.update(vars(args))

    _setup_logging(args.log_dir)   # also captures uvicorn's own POST/GET access logs to file

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("device=%s; loading VGGT...", device)
    _model = load_model(device)
    _solver = Solver(init_conf_threshold=args.conf_threshold, lc_thres=args.lc_thres,
                     vis_voxel_size=args.vis_voxel_size)
    _camera = NetworkCamera()
    log.info("model + solver ready.")

    threading.Thread(target=slam_worker, daemon=True).start()
    _worker_started.wait(timeout=10)
    log.info("serving on %s:%d", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
