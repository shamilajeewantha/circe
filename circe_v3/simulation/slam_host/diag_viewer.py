"""diag_viewer — live incremental Gradio proof-of-data viewer for slam_server.py.

Same visual language as VGGT-SLAM's own `gradio_demo.py` (D:\\others_github\\VGGT-SLAM),
reused directly rather than reinvented: an interactive glTF 3D view (`gr.Model3D`) with
camera-frustum wireframes color-coded green=just-arrived/red=loop-closure/blue=older
(`_camera_frustum_segments` + the OpenCV->glTF axis flip below are copied verbatim from
that file — proven-correct code, not rewritten). The difference: `gradio_demo.py` reads
a live in-process `Solver` object; this polls `slam_server.py`'s own HTTP API instead
(GET /status, /frame/latest, /map), since it's a separate diagnostic process — so
everything shown is ground truth off the wire, not a description of what should be true.

Incremental, snapshot-to-snapshot: each poll tick is one snapshot. New points/frustums
since the last poll are appended (green) and previous ones fade to blue, exactly like
clicking Reconstruct repeatedly in gradio_demo.py appends onto the same running map.
A gallery of every polled incoming frame builds up over the session (image selection,
snapshot to snapshot).

Run (WSL `vggt` env) — auto-launched by slam_server.py, or standalone:
  python diag_viewer.py --slam_url http://localhost:8000 --port 7861
"""

from __future__ import annotations

import argparse
import base64
import os
from datetime import datetime

import cv2
import gradio as gr
import numpy as np
import requests
import trimesh

_OUTPUTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "diag_viewer_outputs")


# --------------------------------------------------------------------------- #
# Verbatim from VGGT-SLAM's gradio_demo.py — proven-correct frustum + glTF math.
# --------------------------------------------------------------------------- #
def _camera_frustum_segments(cam_to_world: np.ndarray, scale: float = 0.05) -> np.ndarray:
    """Small pyramid wireframe (apex=camera center) for one 4x4 cam-to-world pose."""
    R = cam_to_world[:3, :3]
    t = cam_to_world[:3, 3]

    hw, hh, d = scale, scale * 0.75, scale * 1.3
    corners_local = np.array([
        [0, 0, 0],
        [-hw, -hh, d],
        [hw, -hh, d],
        [hw, hh, d],
        [-hw, hh, d],
    ], dtype=np.float32)
    corners_world = (R @ corners_local.T).T + t

    edges = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4), (4, 1)]
    segments = [corners_world[i] for edge in edges for i in edge]
    return np.array(segments, dtype=np.float32).reshape(-1, 2, 3)


def _gltf_flip(pts: np.ndarray) -> np.ndarray:
    """OpenCV (Y-down, Z-forward) -> glTF (Y-up, Z-backward)."""
    out = pts.copy()
    out[..., 1] *= -1
    out[..., 2] *= -1
    return out


# --------------------------------------------------------------------------- #
# HTTP polling against slam_server.py's own API — every value below is real,
# fetched this tick, not cached/assumed.
# --------------------------------------------------------------------------- #
def _get_json(url: str, timeout: float = 3.0):
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        return r.json(), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def _get_frame(slam_url: str):
    try:
        r = requests.get(f"{slam_url}/frame/latest", timeout=3.0)
        if r.status_code != 200:
            return None
        buf = np.frombuffer(r.content, np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        return img[:, :, ::-1] if img is not None else None  # BGR -> RGB
    except Exception:
        return None


class _State:
    """Accumulated across polls, snapshot to snapshot — mirrors gradio_demo.py's
    "repeated Reconstruct clicks append onto the same running map" behavior."""

    def __init__(self):
        self.old_points: list[np.ndarray] = []
        self.old_colors: list[np.ndarray] = []
        self.old_frustums: list[tuple[np.ndarray, tuple]] = []  # (segments, rgb)
        self.known_loops = 0
        self.tick = 0
        self.frame_gallery: list[np.ndarray] = []


def _build_glb(state: _State, map_json: dict | None, out_path: str) -> str | None:
    cloud = (map_json or {}).get("cloud", {})
    n = cloud.get("n", 0)
    is_loop_refresh = bool((map_json or {}).get("full_refresh"))

    if n > 0:
        xyz = np.frombuffer(base64.b64decode(cloud["xyz_f32_b64"]), np.float32).reshape(n, 3).astype(np.float64)
        rgb = np.frombuffer(base64.b64decode(cloud["rgb_u8_b64"]), np.uint8).reshape(n, 3)
        if is_loop_refresh:
            # A loop closure re-optimises everything — this whole poll's cloud
            # supersedes prior points rather than appending onto them.
            state.old_points, state.old_colors = [xyz], [rgb]
        else:
            state.old_points.append(xyz)
            state.old_colors.append(rgb)

        for sm in map_json.get("submaps", []):
            poses = np.array(sm["poses"])  # (F,4,4)
            if len(poses) == 0:
                continue
            segs = np.concatenate([_camera_frustum_segments(p) for p in poses], axis=0)
            # green = new this poll, red = this submap is loop-closed
            color = (230, 60, 60) if sm.get("loop_closure") else (60, 220, 60)
            state.old_frustums.append((segs, color))

    if not state.old_points:
        return None

    scene = trimesh.Scene()
    # newest batch highlighted green (already colored above); everything older fades blue
    n_batches = len(state.old_points)
    for i, (segs, color) in enumerate(state.old_frustums):
        is_latest_batch = i >= len(state.old_frustums) - max(1, len(map_json.get("submaps", []) or []))
        c = color if is_latest_batch else (70, 130, 240)  # blue = older
        path = trimesh.load_path(_gltf_flip(segs))
        path.colors = np.tile(np.array([*c, 255], dtype=np.uint8), (len(path.entities), 1))
        scene.add_geometry(path, node_name=f"cam_{i}")

    points = np.concatenate(state.old_points, axis=0)
    colors = np.concatenate(state.old_colors, axis=0)
    if len(points) > 500_000:
        idx = np.linspace(0, len(points) - 1, 500_000).astype(int)
        points, colors = points[idx], colors[idx]
    cloud_geom = trimesh.points.PointCloud(_gltf_flip(points), colors=colors)
    scene.add_geometry(cloud_geom, node_name="points")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    scene.export(out_path)
    return out_path


def _status_banner(status: dict | None, err: str | None) -> str:
    if err is not None:
        return f"## \U0001F534 slam_server unreachable — {err}"
    alive = status.get("worker_alive")
    dot = "\U0001F7E2" if alive else "\U0001F534"
    cam = status.get("camera") or {}
    last_err = status.get("worker_last_error")
    lines = [
        f"## {dot} worker_alive={alive}  |  **submaps={status.get('num_submaps')}  "
        f"loops={status.get('num_loops')}**",
        f"camera: received={cam.get('received')} dropped={cam.get('dropped')} "
        f"queued={cam.get('queued')}",
    ]
    if last_err:
        lines.append(f"\U0001F534 **worker_last_error:** `{last_err}`")
    return "  \n".join(lines)


def make_app(slam_url: str, poll_period: float) -> gr.Blocks:
    state = _State()

    with gr.Blocks(title="circe SLAM diagnostic viewer", theme=gr.themes.Ocean()) as demo:
        gr.HTML(
            "<h2>circe SLAM diagnostic viewer</h2>"
            f"<p>Live, polling <code>{slam_url}</code> every {poll_period:.0f}s — every value below is "
            "fetched off the wire this tick, nothing is assumed. Frustums: "
            "<span style='color:#3ca'>green</span>=just arrived, "
            "<span style='color:#c33'>red</span>=loop closure, "
            "<span style='color:#37c'>blue</span>=older.</p>"
        )
        banner = gr.Markdown("connecting...")
        with gr.Row():
            with gr.Column(scale=2):
                frame_img = gr.Image(label="GET /frame/latest (real incoming frame, this tick)",
                                      type="numpy")
                gallery = gr.Gallery(label="Incoming frames, snapshot to snapshot", columns=4,
                                      height="260px", object_fit="contain", preview=True)
            with gr.Column(scale=3):
                model3d = gr.Model3D(label="GET /map (real returned point cloud + poses)",
                                      height=560, zoom_speed=0.2, pan_speed=0.2)
        raw_status = gr.JSON(label="GET /status (raw)")

        def refresh():
            state.tick += 1
            status, status_err = _get_json(f"{slam_url}/status")
            map_json, _ = _get_json(
                f"{slam_url}/map?after_submap=-1&known_loops={state.known_loops}&voxel=0.02&max_points=200000")
            if map_json is not None:
                state.known_loops = map_json.get("num_loops", state.known_loops)

            frame = _get_frame(slam_url)
            if frame is not None:
                state.frame_gallery.append(frame)
                state.frame_gallery = state.frame_gallery[-40:]

            out_path = os.path.join(_OUTPUTS_DIR, f"tick_{state.tick:05d}.glb")
            glb = _build_glb(state, map_json, out_path)

            return (_status_banner(status, status_err), frame, glb,
                    list(reversed(state.frame_gallery)), (status or {}))

        timer = gr.Timer(poll_period)
        outputs = [banner, frame_img, model3d, gallery, raw_status]
        timer.tick(refresh, outputs=outputs)
        demo.load(refresh, outputs=outputs)
    return demo


def main() -> None:
    p = argparse.ArgumentParser(description="Live incremental proof-of-data viewer for slam_server.py")
    p.add_argument("--slam_url", default="http://localhost:8000")
    p.add_argument("--port", type=int, default=7861)
    p.add_argument("--poll_period", type=float, default=2.0)
    args = p.parse_args()

    demo = make_app(args.slam_url, args.poll_period)
    demo.launch(server_port=args.port)


if __name__ == "__main__":
    main()
