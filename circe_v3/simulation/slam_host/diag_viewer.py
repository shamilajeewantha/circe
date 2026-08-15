"""diag_viewer — live Gradio proof-of-data viewer for slam_server.py.

Standalone (no ROS, no Gazebo) — polls slam_server.py's own HTTP API on this
laptop and renders what it actually returns: the real latest incoming frame
(GET /frame/latest), real worker/queue counters (GET /status), and the real
returned point cloud + poses (GET /map). Nothing here is a claim or a summary
of a log file — every value on screen came directly off the wire this poll
cycle, so this is the ground-truth check for "did we really get a submap."

Run (WSL `vggt` env, alongside a running slam_server.py):
  python diag_viewer.py --slam_url http://localhost:8000 --port 7861
"""

from __future__ import annotations

import argparse

import cv2
import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import requests


def _get_json(url: str, timeout: float = 3.0):
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        return r.json(), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def _status_banner(status: dict | None, err: str | None) -> str:
    if err is not None:
        return f"## \U0001F534 slam_server unreachable — {err}"
    alive = status.get("worker_alive")
    started = status.get("worker_started")
    last_err = status.get("worker_last_error")
    cam = status.get("camera") or {}
    dot = "\U0001F7E2" if alive else "\U0001F534"
    lines = [
        f"## {dot} worker_alive={alive}  (worker_started={started})",
        f"**num_submaps={status.get('num_submaps')}  num_loops={status.get('num_loops')}**",
        f"camera: received={cam.get('received')} dropped={cam.get('dropped')} "
        f"queued={cam.get('queued')} running={cam.get('running')}",
    ]
    if last_err:
        lines.append(f"\U0001F534 **worker_last_error:** `{last_err}`")
    return "\n\n".join(lines)


def _frame_image(slam_url: str):
    try:
        r = requests.get(f"{slam_url}/frame/latest", timeout=3.0)
        if r.status_code != 200:
            return None
        buf = np.frombuffer(r.content, np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            return None
        return img[:, :, ::-1]  # BGR -> RGB for gr.Image
    except Exception:
        return None


def _map_plot(map_json: dict | None):
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection="3d")
    if not map_json or map_json.get("cloud", {}).get("n", 0) == 0:
        ax.set_title("no points yet")
        return fig

    import base64
    cloud = map_json["cloud"]
    n = cloud["n"]
    xyz = np.frombuffer(base64.b64decode(cloud["xyz_f32_b64"]), np.float32).reshape(n, 3)
    rgb = np.frombuffer(base64.b64decode(cloud["rgb_u8_b64"]), np.uint8).reshape(n, 3) / 255.0

    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=rgb, s=1)
    for sm in map_json.get("submaps", []):
        poses = np.array(sm["poses"])  # (F,4,4)
        if len(poses):
            centers = poses[:, :3, 3]
            ax.plot(centers[:, 0], centers[:, 1], centers[:, 2], "k-", linewidth=1)
    ax.set_title(f"{n} points, {len(map_json.get('submaps', []))} submap(s) in this delta")
    return fig


def make_app(slam_url: str, poll_period: float) -> gr.Blocks:
    with gr.Blocks(title="circe SLAM diagnostic viewer") as demo:
        gr.Markdown(f"# circe SLAM diagnostic viewer — polling `{slam_url}`")
        banner = gr.Markdown("connecting...")
        with gr.Row():
            frame_img = gr.Image(label="GET /frame/latest (real incoming frame)", type="numpy")
            map_plot = gr.Plot(label="GET /map (real returned point cloud)")
        raw_status = gr.JSON(label="GET /status (raw)")

        def refresh():
            status, status_err = _get_json(f"{slam_url}/status")
            map_json, _ = _get_json(f"{slam_url}/map?after_submap=-1&voxel=0.05&max_points=20000")
            frame = _frame_image(slam_url)
            fig = _map_plot(map_json)
            return _status_banner(status, status_err), frame, fig, (status or {})

        timer = gr.Timer(poll_period)
        timer.tick(refresh, outputs=[banner, frame_img, map_plot, raw_status])
        demo.load(refresh, outputs=[banner, frame_img, map_plot, raw_status])
    return demo


def main() -> None:
    p = argparse.ArgumentParser(description="Live proof-of-data viewer for slam_server.py")
    p.add_argument("--slam_url", default="http://localhost:8000")
    p.add_argument("--port", type=int, default=7861)
    p.add_argument("--poll_period", type=float, default=2.0)
    args = p.parse_args()

    demo = make_app(args.slam_url, args.poll_period)
    demo.launch(server_port=args.port)


if __name__ == "__main__":
    main()
