import sys
import os

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import gc
import glob
import json
import logging
import time

import numpy as np
import torch
import trimesh

from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera
from aligner import (
    align_extrinsics,
    apply_scale_to_extrinsics,
    compute_delta_averaged,
    estimate_scale,
)
from context_selector import (
    build_global_point_cloud,
    compute_cube,
    filtered_keyframe_points,
    select_covisible_keyframes,
)
from glb_builder import build_glb
from logger import RunLogger
from visual_util import depth_edge, get_opengl_conversion_matrix

log = logging.getLogger("vggt_omega.pipeline")

# N_OVERLAP is a FLOOR (guaranteed minimum for temporal continuity), not a
# fixed count — cube-candidate selection (context_selector.py) now adapts its
# own count to hit a coverage target, so recent-overlap no longer needs to
# reserve a large fixed share of the batch.
N_OVERLAP = 2
CONF_THRES_PCT = 20.0
_MAX_ACTIVE = N_OVERLAP * 2      # cap active_window memory footprint
_IMAGE_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp")


# ---------------------------------------------------------------------------
# Depth unprojection (local coords)
# ---------------------------------------------------------------------------

def _unproject(
    depth_map: np.ndarray,   # (N, H, W, 1)
    extrinsic: np.ndarray,   # (N, 3, 4)
    intrinsic: np.ndarray,   # (N, 3, 3)
) -> np.ndarray:
    """Returns (N, H, W, 3) world-space XYZ in the model's local coordinate system."""
    depth = depth_map[..., 0]  # (N, H, W)
    N, H, W = depth.shape

    fx = intrinsic[:, 0, 0][:, None, None]
    fy = intrinsic[:, 1, 1][:, None, None]
    cx = intrinsic[:, 0, 2][:, None, None]
    cy = intrinsic[:, 1, 2][:, None, None]

    u = np.arange(W)[None, None, :]
    v = np.arange(H)[None, :, None]

    x_c = (u - cx) / fx * depth
    y_c = (v - cy) / fy * depth
    pts_cam = np.stack([x_c, y_c, depth], axis=-1)   # (N, H, W, 3)

    R = extrinsic[:, :3, :3]
    t = extrinsic[:, :3, 3]

    pts_flat = pts_cam.reshape(N, -1, 3)
    pts_world = np.einsum(
        "sij,snj->sni", R.transpose(0, 2, 1), pts_flat - t[:, None, :]
    )
    return pts_world.reshape(N, H, W, 3)


# ---------------------------------------------------------------------------
# Global map builder
# ---------------------------------------------------------------------------

_GROWTH_HIGHLIGHT_COLOR = (0, 255, 60)  # lime green


def _build_global_map(
    keyframe_db: list,
    run_dir: str,
    max_points: int = 600000,
    out_path: str | None = None,
    highlight_frame_indices: set | None = None,
) -> str | None:
    """
    Merge all globally-aligned keyframe point clouds into a single GLB.
    Points are coloured with the original image RGB; camera trajectory is red.

    If highlight_frame_indices is given, keyframes whose frame_idx is in that
    set are flat-colored lime green instead of their real RGB — used for
    per-step snapshots to show what this pass added/refreshed. None (the
    default, used for the final map) means no highlighting — every keyframe
    keeps its real color.
    """
    if not keyframe_db:
        return None

    all_pts_list = []
    all_colors_list = []
    cam_positions = []

    for kf in keyframe_db:
        pts = kf["world_points"].reshape(-1, 3)
        conf = kf["conf"].reshape(-1)
        mask = np.isfinite(pts).all(axis=1) & np.isfinite(conf) & (conf > 1e-5)

        # Same 20th-percentile confidence filter the submaps use
        if mask.any():
            thr = np.percentile(conf[mask], CONF_THRES_PCT)
            mask &= conf >= thr

        # Depth-edge rejection (drops the smeared points at depth discontinuities)
        edges = depth_edge(kf["depth"], rtol=0.03).reshape(-1)
        mask &= ~edges

        if not mask.any():
            cam_positions.append(kf["camera_center"])
            continue
        all_pts_list.append(pts[mask])

        is_highlighted = (
            highlight_frame_indices is not None
            and kf["frame_idx"] in highlight_frame_indices
        )
        if is_highlighted:
            cols = np.tile(_GROWTH_HIGHLIGHT_COLOR, (int(mask.sum()), 1)).astype(np.uint8)
        elif "colors" in kf:
            cols = kf["colors"].reshape(-1, 3)[mask]
        else:
            cols = np.full((int(mask.sum()), 3), 180, dtype=np.uint8)
        all_colors_list.append(cols)

        cam_positions.append(kf["camera_center"])

    if not all_pts_list:
        return None

    all_pts = np.concatenate(all_pts_list, axis=0).astype(np.float32)
    all_colors = np.concatenate(all_colors_list, axis=0)

    # Uniform subsample to keep file size manageable
    if len(all_pts) > max_points:
        idx = np.linspace(0, len(all_pts) - 1, max_points, dtype=int)
        all_pts = all_pts[idx]
        all_colors = all_colors[idx]

    scene = trimesh.Scene()
    scene.add_geometry(trimesh.PointCloud(vertices=all_pts, colors=all_colors))

    # Camera trajectory — red point cloud
    if cam_positions:
        cam_pts = np.array(cam_positions, dtype=np.float32)
        red = np.tile([255, 50, 50], (len(cam_pts), 1)).astype(np.uint8)
        scene.add_geometry(trimesh.PointCloud(vertices=cam_pts, colors=red))

    # Viewer-convention transform — same as apply_scene_alignment does for the
    # submaps: inv(first frame's extrinsic) @ opengl_conv. Without this the
    # global map renders upside-down/back-to-front vs. every submap (GLB viewers
    # are +Y-up; the camera convention is +Y-down).
    E4_first = np.eye(4, dtype=np.float64)
    E4_first[:3, :4] = keyframe_db[0]["extrinsic_global"]
    scene.apply_transform(np.linalg.inv(E4_first) @ get_opengl_conversion_matrix())

    if out_path is None:
        out_path = os.path.join(run_dir, "global_map.glb")
    try:
        scene.export(out_path)
        json_path = out_path.replace(".glb", "_images.json")
        entries = [
            {"path": kf["image_path"], "role": "map", "frame_idx": kf["frame_idx"]}
            for kf in keyframe_db
        ]
        with open(json_path, "w") as f:
            json.dump(entries, f)
        return out_path
    except Exception as e:
        log.info(f"Global map export failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Main streaming generator
# ---------------------------------------------------------------------------

def run(
    image_dir: str,
    model,
    run_dir: str,
    image_resolution: int = 256,
    conf_thres: float = CONF_THRES_PCT,
    max_points: int = 300000,
    stop_flag=None,
):
    """
    Streaming generator — yields one dict per frame:
        pass_number, frame_idx, glb_path, log_msg,
        current_frame_img, total_frames, inference_time,
        context_size, gpu_mem_GB
    """
    logger = RunLogger(run_dir)

    paths = []
    for ext in _IMAGE_EXTS:
        paths += glob.glob(os.path.join(image_dir, ext))
    paths.sort()
    total = len(paths)

    if total == 0:
        raise ValueError(f"No images found in: {image_dir}")

    logger.info(f"Found {total} images  run_dir={run_dir}")

    # --- State ---
    keyframe_db: list = []
    active_window: list = []
    pass_number = 0
    current_camera_center = None
    # Cube (center, half-size) computed from the LATEST already-processed
    # frame's own points at the end of the previous iteration — used both to
    # draw that pass's "next-query-region" wireframe AND to select this
    # pass's cube candidates. None/0 until at least one frame has been
    # processed (natural bootstrap: zero cube candidates on pass 0).
    cube_center = None
    cube_half_size = 0.0

    for img_idx, image_path in enumerate(paths):
        if stop_flag is not None and stop_flag():
            logger.info("Stop requested.")
            break

        # --- Cube-based candidate query (real Open3D box query against the
        # full global point cloud, not a brute-force per-keyframe scan) ---
        recent_overlap = active_window[-N_OVERLAP:]
        recent_idxs = {e["frame_idx"] for e in recent_overlap}

        # Snapshot the query cube used THIS pass — cube_center/cube_half_size
        # get reassigned later in this same iteration (for the NEXT pass), so
        # this copy is what gets logged below as "what was actually queried."
        query_cube_center = cube_center
        query_cube_half_size = cube_half_size

        old_covisible: list = []
        universe_size = 0
        uncovered_count = 0
        cube_coverage_fraction = 0.0
        candidate_details: list = []
        if keyframe_db and query_cube_center is not None and query_cube_half_size > 0:
            global_pcd = build_global_point_cloud(keyframe_db)
            keyframe_db_by_idx = {kf["frame_idx"]: kf for kf in keyframe_db}
            old_covisible, universe_size, uncovered_count, cube_coverage_fraction, candidate_details = select_covisible_keyframes(
                global_pcd, query_cube_center, query_cube_half_size,
                keyframe_db_by_idx, recent_idxs,
            )

        # --- Batch construction: cube candidates + recent overlap + new ---
        batch_entries = old_covisible + recent_overlap
        batch_paths = [e["image_path"] for e in batch_entries] + [image_path]
        n_context = len(batch_entries)

        # --- Run VGGT-Omega ---
        torch.cuda.reset_peak_memory_stats()
        gpu_mem_before = torch.cuda.memory_allocated() / 1e9

        try:
            images_tensor = load_and_preprocess_images(
                batch_paths, image_resolution=image_resolution
            ).to("cuda")
        except Exception as e:
            logger.info(f"Image preprocessing failed frame {img_idx}: {e}")
            continue

        t0 = time.perf_counter()
        with torch.inference_mode():
            predictions = model(images_tensor)
        inference_time = time.perf_counter() - t0

        gpu_mem_after = torch.cuda.memory_allocated() / 1e9
        gpu_mem_peak = torch.cuda.max_memory_allocated() / 1e9

        extrinsic_t, intrinsic_t = encoding_to_camera(
            predictions["pose_enc"].float(),
            predictions["images"].shape[-2:],
        )
        predictions["extrinsic"] = extrinsic_t
        predictions["intrinsic"] = intrinsic_t

        # Convert all tensors to float32 numpy (squeeze batch dim)
        predictions_np: dict = {}
        for key, val in predictions.items():
            if isinstance(val, torch.Tensor):
                arr = val.detach().float().cpu().numpy()
                if arr.ndim > 0 and arr.shape[0] == 1:
                    arr = arr[0]
                predictions_np[key] = arr

        extrinsics_local = predictions_np["extrinsic"]   # (N_batch, 3, 4)
        intrinsics_np = predictions_np["intrinsic"]       # (N_batch, 3, 3)

        # depth : ensure (N_batch, H, W, 1)
        depth_np = predictions_np["depth"]
        if depth_np.ndim == 3:
            depth_np = depth_np[..., None]

        # depth_conf : ensure (N_batch, H, W)
        depth_conf_np = predictions_np["depth_conf"]
        if depth_conf_np.ndim == 4:
            depth_conf_np = depth_conf_np[..., 0]

        # --- Sim(3) alignment, part 1: scale ---
        # Median depth ratio of known batch frames vs their stored global-scale depth.
        scale = estimate_scale(batch_entries, depth_np, depth_conf_np, n_context)

        # Apply scale to the GEOMETRY before alignment (arXiv:2511.16282: the
        # block scale is applied to all depths AND extrinsics). Without this,
        # every submap lands in the global frame at its own arbitrary scale.
        depth_scaled = depth_np * scale                                     # (N, H, W, 1)
        extrinsics_local_scaled = apply_scale_to_extrinsics(extrinsics_local, scale)

        # --- Sim(3) alignment, part 2: SE(3) Delta averaged over all known frames ---
        Delta, delta_residual_deg = compute_delta_averaged(
            extrinsics_local_scaled, batch_entries
        )
        delta_norm = float(np.linalg.norm(Delta - np.eye(4)))

        extrinsics_global = align_extrinsics(extrinsics_local_scaled, Delta)  # (N, 3, 4)

        # Derive world_points from the global extrinsics + scaled depth directly.
        # This guarantees the point cloud and camera positions are always in
        # the same coordinate system by construction (no separate point transform).
        world_points_global = _unproject(depth_scaled, extrinsics_global, intrinsics_np)

        # --- New frame (last in batch) ---
        E_new = extrinsics_global[-1]                      # (3, 4)
        camera_center_global = -(E_new[:3, :3].T @ E_new[:3, 3])
        depth_new_global = depth_scaled[-1, ..., 0]        # (H, W) already global scale
        conf_new = depth_conf_np[-1]                        # (H, W)

        current_camera_center = camera_center_global.copy()

        # --- Refresh-on-participation: every batch frame that already exists in
        # keyframe_db gets its stored geometry OVERWRITTEN with this pass's fresh
        # reconstruction. old_covisible / active_window entries are the SAME dict
        # objects stored in keyframe_db, so mutating them in place updates the
        # database directly — no separate merge step needed. This is what makes
        # a cube-revisit act as loop closure: the old and new views of the same
        # place come out of one joint inference, so they can't disagree.
        for i, kf in enumerate(batch_entries):
            img_chw_i = predictions_np["images"][i]              # (3, H, W)
            img_hwc_i = np.transpose(img_chw_i, (1, 2, 0))       # (H, W, 3)
            colors_i = (np.clip(img_hwc_i, 0, 1) * 255).astype(np.uint8)

            kf["world_points"] = world_points_global[i].copy()
            kf["colors"] = colors_i.copy()
            kf["depth"] = depth_scaled[i, ..., 0].copy()
            kf["conf"] = depth_conf_np[i].copy()
            kf["extrinsic_global"] = extrinsics_global[i].copy()
            E_i = extrinsics_global[i]
            kf["camera_center"] = -(E_i[:3, :3].T @ E_i[:3, 3])

        # --- Store the brand-new frame (last in batch) ---
        img_chw = predictions_np["images"][-1]                 # (3, H, W)
        img_hwc = np.transpose(img_chw, (1, 2, 0))            # (H, W, 3)
        colors_new = (np.clip(img_hwc, 0, 1) * 255).astype(np.uint8)

        new_entry = {
            "frame_idx": img_idx,
            "image_path": image_path,
            "camera_center": current_camera_center.copy(),
            "world_points": world_points_global[-1].copy(),   # (H, W, 3)
            "colors": colors_new.copy(),                       # (H, W, 3) uint8
            "depth": depth_new_global.copy(),                  # (H, W)
            "conf": conf_new.copy(),
            "extrinsic_global": E_new.copy(),                  # (3, 4)
        }
        keyframe_db.append(new_entry)
        active_window.append(new_entry)
        if len(active_window) > _MAX_ACTIVE:
            active_window.pop(0)

        # --- Compute the cube for THIS frame's own points — used to draw
        # THIS pass's "next-query-region" wireframe below, and (carried
        # forward via the loop variable) to select the NEXT pass's cube
        # candidates. Replaces the old R_SPHERE (locked once at pass 0 and
        # never revisited) — this is recomputed fresh every single pass from
        # real, already-computed geometry, so it always tracks wherever the
        # robot actually is right now. ---
        new_frame_pts = filtered_keyframe_points(new_entry)
        if len(new_frame_pts) > 0:
            cube_center, cube_half_size = compute_cube(new_frame_pts)
        else:
            cube_center, cube_half_size = None, 0.0

        # --- Build globally-aligned predictions dict for GLB ---
        predictions_np_global = {
            "world_points_from_depth": world_points_global,
            "extrinsic": extrinsics_global,
            "depth_conf": depth_conf_np,
            "images": predictions_np["images"],
            "depth": depth_scaled,
        }

        # --- Save GLB ---
        glb_path = os.path.join(run_dir, f"pass_{pass_number:06d}.glb")
        try:
            scene = build_glb(
                predictions_np=predictions_np_global,
                cube_center=cube_center,
                cube_half_size=cube_half_size,
                n_revisit=len(old_covisible),
                n_recent=len(recent_overlap),
                conf_thres=conf_thres,
                max_points=max_points,
            )
            scene.export(glb_path)
            # Sidecar: images used to build this GLB, tagged by batch role —
            # same order as batch_paths (old_covisible + recent_overlap + new).
            json_path = glb_path.replace(".glb", "_images.json")
            entries = (
                [{"path": e["image_path"], "role": "revisit", "frame_idx": e["frame_idx"]} for e in old_covisible]
                + [{"path": e["image_path"], "role": "recent", "frame_idx": e["frame_idx"]} for e in recent_overlap]
                + [{"path": image_path, "role": "new", "frame_idx": img_idx}]
            )
            with open(json_path, "w") as f:
                json.dump(entries, f)
        except Exception as e:
            logger.info(f"GLB export failed: {e}")
            glb_path = None

        # --- Per-step global map snapshot: everything this pass touched
        # (new frame + refreshed overlap/revisit frames) highlighted lime green,
        # rest of the map in real color. Lets you see how the map grew at
        # every step instead of only at the very end. ---
        highlight_idxs = {e["frame_idx"] for e in batch_entries} | {img_idx}
        snapshot_path = os.path.join(run_dir, f"global_map_step_{pass_number:06d}.glb")
        snapshot_path = _build_global_map(
            keyframe_db, run_dir, max_points=max_points * 2,
            out_path=snapshot_path, highlight_frame_indices=highlight_idxs,
        )

        # --- Log ---
        log_entry = {
            "pass": pass_number,
            "frame_idx": img_idx,
            "batch_size": len(batch_paths),
            "n_context": n_context,
            "delta_norm": round(delta_norm, 6),
            "delta_residual_deg": round(delta_residual_deg, 3),
            "scale_estimated": round(scale, 6),
            "scale_frame_count": n_context,
            "recent_overlap_indices": [e["frame_idx"] for e in recent_overlap],
            "old_covisible_indices_used": [e["frame_idx"] for e in old_covisible],
            "new_frame_idx": img_idx,
            "context_size": len(batch_paths),
            "covisible_query_count": len(old_covisible),
            "active_window_count": len(active_window),
            "keyframes_refreshed": n_context,
            "inference_time_s": round(inference_time, 4),
            "gpu_mem_before_GB": round(gpu_mem_before, 3),
            "gpu_mem_after_GB": round(gpu_mem_after, 3),
            "gpu_mem_peak_GB": round(gpu_mem_peak, 3),
            "query_cube_center": query_cube_center.tolist() if query_cube_center is not None else None,
            "query_cube_half_size": round(query_cube_half_size, 6),
            "current_camera_center": current_camera_center.tolist(),
            "keyframe_db_size": len(keyframe_db),
            "universe_size": universe_size,
            "uncovered_count": uncovered_count,
            "cube_coverage_pct": round(cube_coverage_fraction * 100, 2),
            "candidate_details": candidate_details,
            "glb_path": glb_path,
            "global_snapshot_path": snapshot_path,
        }
        log_msg = logger.log_pass(log_entry)

        del predictions, images_tensor
        torch.cuda.empty_cache()
        gc.collect()

        pass_number += 1

        yield {
            "pass_number": pass_number - 1,
            "frame_idx": img_idx,
            "glb_path": glb_path,
            "global_snapshot_path": snapshot_path,
            "log_msg": log_msg,
            "current_frame_img": image_path,
            "total_frames": total,
            "inference_time": inference_time,
            "context_size": len(batch_paths),
            "gpu_mem_GB": gpu_mem_after,
        }

    logger.info(f"Finished. {pass_number} inference passes over {total} frames.")

    # --- Final global map: merge all keyframe point clouds, no highlighting ---
    global_map_path = _build_global_map(keyframe_db, run_dir, max_points=max_points * 2)
    if global_map_path:
        log_msg = logger.info(f"Global map saved: {global_map_path}")
        yield {
            "pass_number": pass_number,
            "frame_idx": -1,
            "glb_path": None,
            "global_snapshot_path": global_map_path,
            "log_msg": log_msg,
            "current_frame_img": None,
            "total_frames": total,
            "inference_time": 0.0,
            "context_size": 0,
            "gpu_mem_GB": torch.cuda.memory_allocated() / 1e9,
            "is_global_map": True,
        }
