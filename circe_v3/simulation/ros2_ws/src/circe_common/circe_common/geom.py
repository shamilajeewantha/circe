"""Shared geometry + PointCloud2 helpers for the circe rover nodes.

Kept dependency-light (numpy + ROS message types only). All maps are RELATIVE
scale (VGGT-SLAM property, project.md §3/§5) — nothing here assumes metric units.
"""
from __future__ import annotations

import numpy as np
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header


# --- rotations -------------------------------------------------------------
def quat_to_R(w, x, y, z) -> np.ndarray:
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], np.float64)


def R_to_quat(R: np.ndarray):
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        return ((0.25 * s), (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s)
    i = int(np.argmax(np.diag(R)))
    if i == 0:
        s = np.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        return ((R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s)
    if i == 1:
        s = np.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        return ((R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s)
    s = np.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
    return ((R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s)


def T_from_odom(odom) -> np.ndarray:
    p, o = odom.pose.pose.position, odom.pose.pose.orientation
    T = np.eye(4)
    T[:3, :3] = quat_to_R(o.w, o.x, o.y, o.z)
    T[:3, 3] = [p.x, p.y, p.z]
    return T


# --- PointCloud2 <-> numpy -------------------------------------------------
def _field(name, offset, dtype=PointField.FLOAT32):
    return PointField(name=name, offset=offset, datatype=dtype, count=1)


def read_points(msg: PointCloud2, fields=("x", "y", "z")) -> np.ndarray:
    names = {f.name: f for f in msg.fields}
    if not set(fields) <= set(names):
        return np.empty((0, len(fields)), np.float32)
    n = msg.width * msg.height
    a = np.frombuffer(bytes(msg.data), np.uint8)[: n * msg.point_step].reshape(n, msg.point_step)
    out = np.empty((n, len(fields)), np.float32)
    for i, f in enumerate(fields):
        off = names[f].offset
        out[:, i] = a[:, off:off + 4].copy().view(np.float32).reshape(n)
    return out


def make_cloud(points: np.ndarray, frame: str, stamp, extra: dict | None = None) -> PointCloud2:
    """points (N, k). Names default x,y,z; pass extra={name: (N,) array} for more fields."""
    n = points.shape[0]
    cols = {"x": points[:, 0], "y": points[:, 1], "z": points[:, 2]}
    if extra:
        cols.update(extra)
    names = list(cols.keys())
    data = np.zeros(n, dtype=[(nm, np.float32) for nm in names])
    for nm in names:
        data[nm] = np.asarray(cols[nm], np.float32)
    msg = PointCloud2()
    msg.header = Header(stamp=stamp, frame_id=frame)
    msg.height, msg.width = 1, n
    msg.is_dense, msg.is_bigendian = False, False
    msg.fields = [_field(nm, 4 * i) for i, nm in enumerate(names)]
    msg.point_step = 4 * len(names)
    msg.row_step = msg.point_step * n
    msg.data = data.tobytes()
    return msg


def pack_rgb(rgb_u8: np.ndarray) -> np.ndarray:
    """(N,3) uint8 → (N,) float32 packed rgb (rviz/gradio convention)."""
    p = (rgb_u8[:, 0].astype(np.uint32) << 16 |
         rgb_u8[:, 1].astype(np.uint32) << 8 |
         rgb_u8[:, 2].astype(np.uint32))
    return p.view(np.float32)


def voxel_downsample(xyz: np.ndarray, voxel: float):
    """Return (kept_xyz, kept_idx) by first-point-per-voxel."""
    if voxel <= 0 or len(xyz) == 0:
        return xyz, np.arange(len(xyz))
    keys = np.floor(xyz / voxel).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return xyz[idx], idx
