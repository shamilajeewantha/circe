import sys
import os

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import trimesh
from visual_util import (
    predictions_to_glb,
    integrate_camera_into_scene,
    get_opengl_conversion_matrix,
)

# Camera role colors (RGB)
_COLOR_REVISIT = (255, 150, 0)   # orange — cube-candidate revisit frames
_COLOR_RECENT = (0, 100, 255)    # blue   — recent-overlap (temporal) frames
_COLOR_NEW = (255, 50, 50)       # red    — newly processed frame


def build_glb(
    predictions_np: dict,
    cube_center: np.ndarray,
    cube_half_size: float,
    n_revisit: int,
    n_recent: int,
    conf_thres: float = 20.0,
    max_points: int = 300000,
) -> trimesh.Scene:
    """
    Build one GLB snapshot for the current pass.

    predictions_np contains globally-aligned data:
      - world_points_from_depth : (N, H, W, 3)
      - extrinsic               : (N, 3, 4)
      - depth_conf              : (N, H, W)
      - images                  : (N, 3, H, W)
      - depth                   : (N, H, W, 1)

    Batch layout (always old_covisible + recent_overlap + new, per pipeline.py):
      frames [0, n_revisit)                = revisit  (orange)
      frames [n_revisit, n_revisit+n_recent) = recent  (blue)
      frame  -1                             = new      (red)
    """
    # Point cloud (show_cam=False — cameras added manually below)
    scene = predictions_to_glb(
        predictions_np,
        conf_thres=conf_thres,
        show_cam=False,
        max_points=max_points,
    )

    # Scene scale for camera frustum sizing
    pts = predictions_np["world_points_from_depth"].reshape(-1, 3)
    finite = np.isfinite(pts).all(axis=1)
    if finite.sum() > 10:
        p5 = np.percentile(pts[finite], 5, axis=0)
        p95 = np.percentile(pts[finite], 95, axis=0)
        scene_scale = float(np.linalg.norm(p95 - p5))
    else:
        scene_scale = 1.0
    scene_scale = max(scene_scale, 1e-6)

    # Alignment transform T = inv(extrinsics[0]) @ opengl_conv
    # (same transform apply_scene_alignment applied to the point cloud)
    extrinsics = predictions_np["extrinsic"]   # (N, 3, 4)
    N = len(extrinsics)
    ext4 = np.eye(4, dtype=np.float64)[None].repeat(N, axis=0)
    ext4[:, :3, :4] = extrinsics

    opengl_conv = get_opengl_conversion_matrix()
    T_align = np.linalg.inv(ext4[0]) @ opengl_conv

    # Role-colored camera frustums
    for i in range(N):
        cam_to_world_pre = np.linalg.inv(ext4[i])
        cam_to_world_aligned = T_align @ cam_to_world_pre
        if i == N - 1:
            color = _COLOR_NEW
        elif i < n_revisit:
            color = _COLOR_REVISIT
        else:
            color = _COLOR_RECENT
        integrate_camera_into_scene(scene, cam_to_world_aligned, color, scene_scale)

    # Wireframe cube showing the next-pass query region (matches the actual
    # cube used for covisibility candidate selection — see context_selector.py)
    if cube_center is not None and cube_half_size > 0:
        center_h = np.append(cube_center.astype(np.float64), 1.0)
        center_aligned = (T_align @ center_h)[:3]

        cube_mesh = trimesh.creation.box(extents=[2 * cube_half_size] * 3)
        edge_verts = cube_mesh.vertices[cube_mesh.edges_unique]   # (E, 2, 3)
        path = trimesh.load_path(edge_verts + center_aligned[None, None, :])
        for entity in path.entities:
            entity.color = np.array([255, 200, 0, 120], dtype=np.uint8)
        scene.add_geometry(path, geom_name="query_cube")

    return scene
