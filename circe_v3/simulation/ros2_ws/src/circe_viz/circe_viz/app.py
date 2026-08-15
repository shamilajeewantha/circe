"""circe_viz — real-time 3D viewer for what the rover sees AND decides.

Runs on the SIM laptop. A ROS 2 (rclpy) node subscribes to the live circe
topics; a Gradio page renders the growing map + the rover's coverage/planning
decisions and refreshes on a timer. It improves on the repo's gradio_demo.py by
being **live from ROS** (not upload-and-click) and by overlaying the two coverage
layers + the robot pose/goal, not just the raw VGGT point cloud.

Subscribes (all optional — the view degrades gracefully if a topic is silent):
  /vggt/cloud          sensor_msgs/PointCloud2   the map (relative scale)
  /circe/pose_fused    nav_msgs/Odometry         robot pose (frustum + trajectory)
  /circe/surfels       sensor_msgs/PointCloud2   inspection surfels (rgb encodes covered/q_best)
  /circe/frontiers     sensor_msgs/PointCloud2   fog frontier cells
  /circe/detection_gaps sensor_msgs/PointCloud2  uncovered surfel clusters
  /circe/goal_station  nav_msgs/Odometry         next station marker

Launch:  python -m circe_viz.app --port 7860     (or via circe_bringup)

Reuses gradio_demo.py's GLB conventions: camera-frustum wireframes, the
OpenCV(Y-down,Z-forward) → glTF(Y-up,Z-back) axis flip, voxel downsample and a
max-points cap.
"""

from __future__ import annotations

import argparse
import struct
import threading
from collections import deque
from typing import Optional

import numpy as np
import trimesh
import gradio as gr

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry


# --------------------------------------------------------------------------- #
# PointCloud2 → numpy (xyz + optional packed rgb). Minimal, no ros2 numpy dep.
# --------------------------------------------------------------------------- #
def _pc2_to_xyzrgb(msg: PointCloud2):
    names = {f.name: f for f in msg.fields}
    if not {"x", "y", "z"} <= set(names):
        return np.empty((0, 3), np.float32), None
    step = msg.point_step
    n = msg.width * msg.height
    buf = bytes(msg.data)
    ox, oy, oz = names["x"].offset, names["y"].offset, names["z"].offset
    xyz = np.empty((n, 3), np.float32)
    for i, off in enumerate((ox, oy, oz)):
        xyz[:, i] = _strided(buf, off, step, n)
    rgb = None
    if "rgb" in names:
        packed = _strided_u32(buf, names["rgb"].offset, step, n)
        rgb = np.empty((n, 3), np.uint8)
        rgb[:, 0] = (packed >> 16) & 0xFF
        rgb[:, 1] = (packed >> 8) & 0xFF
        rgb[:, 2] = packed & 0xFF
    return xyz, rgb


def _strided(buf: bytes, off: int, step: int, n: int) -> np.ndarray:
    a = np.frombuffer(buf, np.uint8).reshape(-1)[: n * step].reshape(n, step)
    return a[:, off:off + 4].copy().view(np.float32).reshape(n)


def _strided_u32(buf: bytes, off: int, step: int, n: int) -> np.ndarray:
    a = np.frombuffer(buf, np.uint8).reshape(-1)[: n * step].reshape(n, step)
    return a[:, off:off + 4].copy().view(np.uint32).reshape(n)


def _frustum_segments(center, R, scale=0.06):
    hw, hh, d = scale, scale * 0.75, scale * 1.3
    local = np.array([[0, 0, 0], [-hw, -hh, d], [hw, -hh, d], [hw, hh, d], [-hw, hh, d]], np.float32)
    world = (R @ local.T).T + center
    edges = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4), (4, 1)]
    return np.array([world[i] for e in edges for i in e], np.float32).reshape(-1, 2, 3)


def _quat_to_R(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], np.float32)


# --------------------------------------------------------------------------- #
# ROS node: keep the latest of each topic in memory for the render thread.
# --------------------------------------------------------------------------- #
class VizState(Node):
    def __init__(self):
        super().__init__("circe_viz")
        self.lock = threading.Lock()
        self.cloud = (np.empty((0, 3), np.float32), None)
        self.surfels = (np.empty((0, 3), np.float32), None)
        self.frontiers = np.empty((0, 3), np.float32)
        self.gaps = np.empty((0, 3), np.float32)
        self.pose = None            # (center, R)
        self.goal = None            # center
        self.traj = deque(maxlen=2000)
        self.create_subscription(PointCloud2, "/vggt/cloud", self._cloud, 1)
        self.create_subscription(PointCloud2, "/circe/surfels", self._surfels, 1)
        self.create_subscription(PointCloud2, "/circe/frontiers", self._frontiers, 1)
        self.create_subscription(PointCloud2, "/circe/detection_gaps", self._gaps, 1)
        self.create_subscription(Odometry, "/circe/pose_fused", self._pose, 10)
        self.create_subscription(Odometry, "/circe/goal_station", self._goal, 10)

    def _cloud(self, m):
        with self.lock: self.cloud = _pc2_to_xyzrgb(m)
    def _surfels(self, m):
        with self.lock: self.surfels = _pc2_to_xyzrgb(m)
    def _frontiers(self, m):
        with self.lock: self.frontiers = _pc2_to_xyzrgb(m)[0]
    def _gaps(self, m):
        with self.lock: self.gaps = _pc2_to_xyzrgb(m)[0]
    def _pose(self, m):
        p, o = m.pose.pose.position, m.pose.pose.orientation
        c = np.array([p.x, p.y, p.z], np.float32)
        with self.lock:
            self.pose = (c, _quat_to_R((o.w, o.x, o.y, o.z)))
            self.traj.append(c)
    def _goal(self, m):
        p = m.pose.pose.position
        with self.lock: self.goal = np.array([p.x, p.y, p.z], np.float32)

    def snapshot(self):
        with self.lock:
            return dict(cloud=self.cloud, surfels=self.surfels,
                        frontiers=self.frontiers.copy(), gaps=self.gaps.copy(),
                        pose=self.pose, goal=self.goal,
                        traj=np.array(self.traj, np.float32) if self.traj else np.empty((0, 3), np.float32))


_STATE: Optional[VizState] = None


def _flip(p: np.ndarray) -> np.ndarray:
    """OpenCV → glTF viewer convention (same as gradio_demo.py)."""
    q = p.copy()
    q[..., 1] *= -1
    q[..., 2] *= -1
    return q


def build_glb(show_map, show_surfels, show_frontier, show_gaps, voxel, max_pts) -> Optional[str]:
    if _STATE is None:
        return None
    s = _STATE.snapshot()
    scene = trimesh.Scene()

    def _add_points(xyz, rgb, default_rgb, name):
        if xyz is None or len(xyz) == 0:
            return
        pts = xyz.astype(np.float64)
        cols = rgb if rgb is not None else np.tile(np.array(default_rgb, np.uint8), (len(pts), 1))
        if voxel and voxel > 0:
            keys = np.floor(pts / voxel).astype(np.int64)
            _, idx = np.unique(keys, axis=0, return_index=True)
            pts, cols = pts[idx], cols[idx]
        if max_pts and len(pts) > max_pts:
            sel = np.linspace(0, len(pts) - 1, int(max_pts)).astype(int)
            pts, cols = pts[sel], cols[sel]
        scene.add_geometry(trimesh.points.PointCloud(_flip(pts), colors=cols), node_name=name)

    if show_map:
        _add_points(s["cloud"][0], s["cloud"][1], (170, 170, 170), "map")
    if show_surfels:
        _add_points(s["surfels"][0], s["surfels"][1], (60, 200, 60), "surfels")
    if show_frontier:
        _add_points(s["frontiers"], None, (60, 120, 240), "frontier")
    if show_gaps:
        _add_points(s["gaps"], None, (230, 60, 60), "gaps")

    if s["pose"] is not None:
        c, R = s["pose"]
        seg = _flip(_frustum_segments(c, R))
        path = trimesh.load_path(seg)
        path.colors = np.tile(np.array([255, 200, 0, 255], np.uint8), (len(path.entities), 1))
        scene.add_geometry(path, node_name="robot")
    if len(s["traj"]) > 1:
        tf = _flip(s["traj"])
        segs = np.stack([tf[:-1], tf[1:]], axis=1)          # (N-1, 2, 3) line segments
        tp = trimesh.load_path(segs)
        tp.colors = np.tile(np.array([255, 255, 0, 255], np.uint8), (len(tp.entities), 1))
        scene.add_geometry(tp, node_name="trajectory")
    if s["goal"] is not None:
        m = trimesh.creation.uv_sphere(radius=0.08)
        m.apply_translation(_flip(s["goal"]))
        m.visual.vertex_colors = [255, 0, 255, 255]
        scene.add_geometry(m, node_name="goal")

    if len(scene.geometry) == 0:
        return None
    out = "/tmp/circe_viz_map.glb"
    scene.export(out)
    return out


def stats_md() -> str:
    if _STATE is None:
        return "waiting for ROS..."
    s = _STATE.snapshot()
    nmap = len(s["cloud"][0]); nsurf = len(s["surfels"][0])
    nfr = len(s["frontiers"]); ng = len(s["gaps"])
    state = "DONE" if (nfr == 0 and ng == 0 and nsurf > 0) else "EXPLORING"
    return (f"**map pts:** {nmap}  |  **surfels:** {nsurf}  |  **frontier:** {nfr}  |  "
            f"**detection-gaps:** {ng}  |  **state:** {state}")


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="circe_viz", theme=gr.themes.Ocean()) as demo:
        gr.HTML("<h2>circe_viz — live map &amp; decisions</h2>"
                "<p>Live from ROS: the VGGT map, inspection surfels (green=covered, red=gap), fog frontier, "
                "robot pose/trajectory (yellow), and next station (magenta).</p>")
        with gr.Row():
            with gr.Column(scale=4):
                view = gr.Model3D(height=620, zoom_speed=0.2, pan_speed=0.2, label="live map")
                stats = gr.Markdown("waiting for ROS...")
            with gr.Column(scale=1):
                show_map = gr.Checkbox(True, label="map cloud")
                show_surf = gr.Checkbox(True, label="surfels (covered/gap)")
                show_fr = gr.Checkbox(True, label="fog frontier")
                show_gap = gr.Checkbox(True, label="detection gaps")
                voxel = gr.Slider(0.0, 0.2, 0.02, step=0.005, label="viz voxel size")
                max_pts = gr.Slider(50000, 800000, 300000, step=50000, label="max points")
                rate = gr.Slider(0.5, 5.0, 1.0, step=0.5, label="refresh (s)")
        timer = gr.Timer(1.0)
        rate.change(lambda r: gr.Timer(r), rate, timer)
        timer.tick(build_glb, [show_map, show_surf, show_fr, show_gap, voxel, max_pts], view)
        timer.tick(stats_md, None, stats)
    return demo


def _ros_spin():
    rclpy.spin(_STATE)


def main():
    global _STATE
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()

    rclpy.init()
    _STATE = VizState()
    threading.Thread(target=_ros_spin, daemon=True).start()

    demo = build_ui()
    demo.queue(default_concurrency_limit=1)
    try:
        demo.launch(server_name="0.0.0.0", server_port=args.port)
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
