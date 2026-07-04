import numpy as np


def pad_extrinsic(E: np.ndarray) -> np.ndarray:
    """(3,4) or (4,4) -> (4,4) float64."""
    E = np.asarray(E, dtype=np.float64)
    if E.shape == (4, 4):
        return E
    E4 = np.eye(4, dtype=np.float64)
    E4[:3, :4] = E
    return E4


def compute_delta(E_local_0: np.ndarray, E_global_0: np.ndarray) -> np.ndarray:
    """
    Delta = inv(E_local_0) @ E_global_0
    Maps current-pass local coordinates → global coordinates.
    Inputs may be (3,4) or (4,4).  Returns (4,4).
    """
    L = pad_extrinsic(E_local_0)
    G = pad_extrinsic(E_global_0)
    return np.linalg.inv(L) @ G


def apply_scale_to_extrinsics(extrinsics_np: np.ndarray, s: float) -> np.ndarray:
    """
    Scale the translation of every extrinsic: [R | t] -> [R | s·t].

    Per arXiv:2511.16282, the per-block scale must be applied to all VGGT
    depths AND extrinsics BEFORE computing the alignment Delta — otherwise
    each submap lands in the global frame at its own arbitrary scale.

    extrinsics_np : (N, 3, 4)
    returns        : (N, 3, 4) copy with scaled translations
    """
    out = extrinsics_np.copy()
    out[:, :, 3] *= float(s)
    return out


def compute_delta_averaged(
    extrinsics_local_scaled: np.ndarray,
    batch_entries: list,
) -> tuple[np.ndarray, float]:
    """
    Delta averaged over ALL known batch frames instead of a single anchor.

    Per known frame i:  Δ_i = inv(E_local_scaled[i]) @ E_global_stored[i]
    Rotation average : SVD projection of the summed rotation blocks onto SO(3)
    Translation      : per-axis median of the Δ_i translations

    Returns (Delta 4x4, residual_deg) where residual_deg is the maximum
    geodesic angle between any Δ_i rotation and the average — a free drift
    diagnostic (large values mean the batch frames disagree about the
    local→global transform).
    """
    n = len(batch_entries)
    if n == 0:
        return np.eye(4, dtype=np.float64), 0.0

    deltas = []
    for i in range(n):
        L = pad_extrinsic(extrinsics_local_scaled[i])
        G = pad_extrinsic(batch_entries[i]["extrinsic_global"])
        deltas.append(np.linalg.inv(L) @ G)

    if n == 1:
        return deltas[0], 0.0

    R_sum = np.zeros((3, 3), dtype=np.float64)
    for d in deltas:
        R_sum += d[:3, :3]
    U, _, Vt = np.linalg.svd(R_sum)
    det_sign = np.sign(np.linalg.det(U @ Vt))
    R_avg = U @ np.diag([1.0, 1.0, det_sign]) @ Vt

    t_med = np.median(np.stack([d[:3, 3] for d in deltas]), axis=0)

    Delta = np.eye(4, dtype=np.float64)
    Delta[:3, :3] = R_avg
    Delta[:3, 3] = t_med

    residual_deg = 0.0
    for d in deltas:
        R_rel = R_avg.T @ d[:3, :3]
        cos_ang = np.clip((np.trace(R_rel) - 1.0) / 2.0, -1.0, 1.0)
        residual_deg = max(residual_deg, float(np.degrees(np.arccos(cos_ang))))

    return Delta, residual_deg


def align_extrinsics(extrinsics_np: np.ndarray, Delta: np.ndarray) -> np.ndarray:
    """
    E_global[i] = E_local[i] @ Delta   (RIGHT-multiply)
    where Delta = inv(E_local_0) @ E_global_0  (global→local change-of-basis).

    Proof: E_local[i] @ Delta = E_local[i] @ inv(E_local_0) @ E_global_0
    For i=0: E_local_0 @ inv(E_local_0) @ E_global_0 = E_global_0  ✓
    World points are then re-unprojected using extrinsics_global, guaranteeing
    the point cloud and cameras are always in the same coordinate system.

    extrinsics_np : (N, 3, 4)  world→camera in local coords
    returns        : (N, 3, 4)  world→camera in global coords
    """
    N = len(extrinsics_np)
    out = np.empty_like(extrinsics_np)
    for i in range(N):
        out[i] = (pad_extrinsic(extrinsics_np[i]) @ Delta)[:3, :]
    return out


def align_world_points(
    world_points_np: np.ndarray,
    Delta: np.ndarray,
    scale: float = 1.0,
) -> np.ndarray:
    """
    Transform local-coord world points to global coords.
    world_points_np : (N, H, W, 3)
    returns          : (N, H, W, 3)
    """
    R = Delta[:3, :3]
    t = Delta[:3, 3]
    N, H, W, _ = world_points_np.shape
    pts_flat = world_points_np.reshape(-1, 3)
    pts_global = (pts_flat @ R.T + t) * scale
    return pts_global.reshape(N, H, W, 3)


def estimate_scale(
    overlap_keyframes: list,
    depth_np: np.ndarray,
    depth_conf_np: np.ndarray,
    n_overlap: int,
) -> float:
    """
    Estimate scale = median(stored_depth / curr_depth) over overlap frames.
    Returns a scalar in [0.1, 10.0].

    overlap_keyframes : list of keyframe dicts, each has "depth" (H, W) in global scale
    depth_np          : (N, H, W, 1) or (N, H, W)  raw from current pass
    depth_conf_np     : (N, H, W)                   raw confidence from current pass
    n_overlap         : how many overlap frames to use (≤ len(overlap_keyframes))
    """
    if n_overlap == 0 or not overlap_keyframes:
        return 1.0

    n = min(n_overlap, len(overlap_keyframes), depth_np.shape[0])
    ratios = []

    for i in range(n):
        kf = overlap_keyframes[i]
        stored_depth = kf["depth"]  # (H, W) globally calibrated

        raw_d = depth_np[i]
        curr_depth = raw_d[..., 0] if raw_d.ndim == 3 else raw_d

        conf_raw = depth_conf_np[i]
        curr_conf = conf_raw[..., 0] if conf_raw.ndim == 3 else conf_raw

        if stored_depth.shape != curr_depth.shape:
            continue

        pos_conf = curr_conf[curr_conf > 0]
        if pos_conf.size == 0:
            continue
        conf_thr = float(np.percentile(pos_conf, 75))

        mask = (
            (curr_conf > conf_thr)
            & (stored_depth > 0)
            & (curr_depth > 0)
            & np.isfinite(stored_depth)
            & np.isfinite(curr_depth)
        )
        if mask.sum() < 50:
            continue

        ratio = float(np.median(stored_depth[mask] / curr_depth[mask]))
        if np.isfinite(ratio) and ratio > 0:
            ratios.append(ratio)

    if not ratios:
        return 1.0

    return float(np.clip(np.median(ratios), 0.1, 10.0))
