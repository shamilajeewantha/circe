import numpy as np
import open3d as o3d
import open3d.core as o3c

from visual_util import depth_edge

# Per-axis resolution of the voxel grid used for genuine cross-candidate
# coverage overlap in the greedy set-cover (so two candidates both showing
# the same real spot don't each get full, double-counted credit for it).
_VOXEL_GRID_N = 20


def filtered_keyframe_points(kf: dict) -> np.ndarray:
    """
    This keyframe's own (X, Y, Z) points, filtered to finite + confident +
    non-depth-edge. Depth-edge rejection strips "flying pixel" artifacts at
    depth discontinuities that would otherwise inflate coverage scores with
    points that don't represent real geometry.

    Confidence filter matches pipeline.py's 20th-percentile pattern (used for
    every submap/global-map) instead of a near-zero absolute threshold —
    otherwise points that would be rejected as "bottom 20% weakest confidence"
    everywhere else in the codebase could still count as real coverage here.
    """
    pts = kf["world_points"].reshape(-1, 3)
    conf = kf["conf"].reshape(-1)
    finite = np.isfinite(pts).all(axis=1) & np.isfinite(conf)
    mask = finite.copy()
    if finite.any():
        thr = np.percentile(conf[finite], 20.0)
        mask &= conf >= thr
    edges = depth_edge(kf["depth"], rtol=0.03).reshape(-1)
    mask &= ~edges
    return pts[mask]


_CUBE_SIZE_PERCENTILE = 75.0  # excludes far-background content a single frame happens to see


def compute_cube(frame_points: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Cube center = centroid of frame_points (already filtered).
    Cube half-size = the _CUBE_SIZE_PERCENTILE-th percentile of each point's
    largest single-axis distance from the centroid (|x-cx|, |y-cy|, |z-cz|,
    take the max of the three per point, then the percentile across all
    points).

    Using the true max is not robust: even after depth-edge rejection, a
    single camera image genuinely sees both near content (a few inches away)
    and far background (a wall/floor stretching meters into the distance) at
    once — real points, not noise, but including that far background inflates
    the cube well past the actual local area. Percentile clipping (75th)
    trades the "100% containment" guarantee for reflecting the near/local
    area instead of everything a single frame happens to see.
    """
    center = frame_points.mean(axis=0)
    per_point_max_axis_dist = np.abs(frame_points - center[None, :]).max(axis=1)
    half_size = float(np.percentile(per_point_max_axis_dist, _CUBE_SIZE_PERCENTILE))
    return center, half_size


def build_global_point_cloud(keyframe_db: list):
    """
    Merge every keyframe's own filtered points into one Open3D tensor point
    cloud, each point carrying its owning frame_idx as a native per-point
    attribute. Rebuilt fresh every pass by the caller — no incremental index
    maintenance, so refresh-on-participation can never leave it stale.
    Returns None if there are no valid points anywhere yet.
    """
    all_pts = []
    all_owners = []
    for kf in keyframe_db:
        pts = filtered_keyframe_points(kf)
        if len(pts) == 0:
            continue
        all_pts.append(pts)
        all_owners.append(np.full(len(pts), kf["frame_idx"], dtype=np.int32))

    if not all_pts:
        return None

    points_np = np.concatenate(all_pts, axis=0).astype(np.float32)
    owners_np = np.concatenate(all_owners, axis=0)

    pcd = o3d.t.geometry.PointCloud()
    pcd.point.positions = o3c.Tensor(points_np)
    pcd.point.frame_idx = o3c.Tensor(owners_np)
    return pcd


_CUBE_COVERAGE_TARGET = 0.90        # keep selecting cube-candidates until this fraction of voxels covered
_SAFETY_CEILING = 20                # absolute cap on cube-candidates, purely to avoid pathological/unbounded growth
_DIMINISHING_RETURNS_FLOOR = 0.80   # only allow an early stop once at least this much is covered
_DIMINISHING_RETURNS_MIN_MARGINAL = 0.02  # ...and the best remaining candidate adds less than this fraction
_MAX_DISTANCE_RATIO = 5.0           # hard cutoff: candidates beyond this many cube_half_sizes are never eligible
_MIN_CONTRIBUTION_FRACTION = 0.01   # minimum marginal contribution (of universe) to be selectable at all


def select_covisible_keyframes(
    global_pcd,
    cube_center: np.ndarray,
    cube_half_size: float,
    keyframe_db_by_idx: dict,
    exclude_frame_indices: set,
):
    """
    Query global_pcd for points inside the cube (real Open3D box query, not
    custom distance math), then greedily select cube-candidate keyframes.

    Coverage universe is voxelized (not per-candidate-disjoint) so two
    candidates showing the same real spot genuinely compete for the same
    coverage credit, instead of always both scoring "full points" regardless
    of overlap between them.

    IMPORTANT: the universe is built from ALL in-cube points, including
    recent-overlap (guaranteed-included) frames — those frames aren't
    selectable as candidates (excluded below), but their coverage still
    counts. Excluding them from the accounting entirely (an earlier bug)
    made the coverage target measure a fake, artificially-incomplete universe
    that ignored what the guaranteed frames already show — causing the
    greedy loop to reach for far, low-value candidates chasing a target that
    was already effectively met.

    Candidate eligibility (both applied BEFORE scoring, not just penalizing):
      - _MAX_DISTANCE_RATIO: a candidate whose camera is beyond this many
        cube_half_sizes is never eligible, full stop — the exponential score
        penalty alone never actually reaches zero, so without a hard cutoff a
        far candidate can still win by default if every closer candidate's
        marginal contribution has collapsed to zero from redundancy.
      - _MIN_CONTRIBUTION_FRACTION: a candidate's (initial) marginal
        contribution must be at least this fraction of the universe to be
        selectable at all — independent of the 80%/2% diminishing-returns
        rule below, which only applies once coverage is already high. Without
        this, a literal handful of stray/noise voxels is enough to win below
        80% coverage just for being nonzero.

    Each iteration ranks remaining eligible candidates by:
        marginal_new_voxels_covered * exp(-camera_distance_to_center / half_size)
    Exponential (not linear/hyperbolic) distance decay so a frame many
    cube-widths away needs an overwhelming marginal contribution to ever be
    selected, rather than a merely large one.

    Selection continues until _CUBE_COVERAGE_TARGET (90%) of the cube's
    voxels are covered, OR — once at least _DIMINISHING_RETURNS_FLOOR (80%)
    is already covered — the best remaining candidate would add less than
    _DIMINISHING_RETURNS_MIN_MARGINAL (2%) of the universe, OR the
    _SAFETY_CEILING is hit (pathological/adversarial case).

    Returns (selected_keyframes, universe_size, uncovered_count,
             coverage_fraction, candidate_details) where candidate_details is
    a list of dicts — one per candidate considered (selected or not) — with
    frame_idx, initial marginal voxel count, distance, distance ratio, score,
    and whether it was excluded by the distance/contribution gates or
    ultimately selected. This makes any future pass fully explainable from
    the log alone.
    """
    if global_pcd is None or cube_half_size <= 0:
        return [], 0, 0, 0.0, []

    bbox = o3d.geometry.AxisAlignedBoundingBox(
        min_bound=cube_center - cube_half_size,
        max_bound=cube_center + cube_half_size,
    )
    # get_point_indices_within_bounding_box belongs to the legacy (non-tensor)
    # open3d.geometry API and expects a legacy Vector3dVector — NOT the
    # o3d.core.Tensor that the tensor-API PointCloud stores its points as.
    # Convert at this query boundary only; storage stays on the tensor API
    # (per-point frame_idx attribute).
    legacy_points = o3d.utility.Vector3dVector(global_pcd.point.positions.numpy())
    indices = bbox.get_point_indices_within_bounding_box(legacy_points)
    if len(indices) == 0:
        return [], 0, 0, 0.0, []

    positions = global_pcd.point.positions.numpy()[indices]
    owners = global_pcd.point.frame_idx.numpy()[indices]

    # Voxelize using ALL in-cube points (recent-overlap included) — the true
    # universe, not one that pretends guaranteed frames cover nothing.
    voxel_size = (2.0 * cube_half_size) / _VOXEL_GRID_N
    voxel_ids = np.floor(
        (positions - (cube_center - cube_half_size)) / voxel_size
    ).astype(np.int64)
    voxel_keys = (
        voxel_ids[:, 0] * _VOXEL_GRID_N * _VOXEL_GRID_N
        + voxel_ids[:, 1] * _VOXEL_GRID_N
        + voxel_ids[:, 2]
    )

    universe_size = len(set(voxel_keys.tolist()))
    if universe_size == 0:
        return [], 0, 0, 0.0, []

    # Recent-overlap (guaranteed) frames aren't selectable, but their coverage
    # counts — pre-cover their voxels before the greedy loop even starts.
    excluded_mask = np.array([o in exclude_frame_indices for o in owners])
    uncovered = set(voxel_keys.tolist()) - set(voxel_keys[excluded_mask].tolist())

    candidate_mask = ~excluded_mask
    cand_owners = owners[candidate_mask]
    cand_voxel_keys = voxel_keys[candidate_mask]
    unique_owners = np.unique(cand_owners)
    candidate_voxels = {
        int(o): set(cand_voxel_keys[cand_owners == o].tolist()) for o in unique_owners
    }

    coverage_fraction = (universe_size - len(uncovered)) / universe_size
    if len(unique_owners) == 0:
        return [], universe_size, len(uncovered), coverage_fraction, []

    # --- Build per-candidate details up front (Fix J), applying the hard
    # distance cutoff (Fix L) and minimum-contribution floor (Fix M) as
    # eligibility gates, not just score penalties. ---
    candidate_details: list = []
    eligible_owners: list = []
    for o in unique_owners:
        o = int(o)
        kf = keyframe_db_by_idx[o]
        dist = float(np.linalg.norm(kf["camera_center"] - cube_center))
        dist_ratio = dist / cube_half_size
        initial_marginal = len(candidate_voxels[o] & uncovered)
        marginal_fraction = initial_marginal / universe_size

        excluded_reason = None
        if dist_ratio > _MAX_DISTANCE_RATIO:
            excluded_reason = "distance_cutoff"
        elif marginal_fraction < _MIN_CONTRIBUTION_FRACTION:
            excluded_reason = "min_contribution"

        score = (
            initial_marginal * float(np.exp(-dist_ratio))
            if excluded_reason is None else 0.0
        )

        candidate_details.append({
            "frame_idx": o,
            "initial_marginal_voxels": initial_marginal,
            "distance": round(dist, 6),
            "distance_ratio": round(dist_ratio, 4),
            "score": round(score, 6),
            "excluded_reason": excluded_reason,
            "selected": False,  # updated below if actually picked
        })
        if excluded_reason is None:
            eligible_owners.append(o)

    details_by_owner = {d["frame_idx"]: d for d in candidate_details}

    target_covered = int(np.ceil(universe_size * _CUBE_COVERAGE_TARGET))
    selected: list = []

    while uncovered and len(selected) < _SAFETY_CEILING:
        covered_so_far = universe_size - len(uncovered)
        if covered_so_far >= target_covered:
            break

        best_owner, best_score, best_marginal = None, -1.0, 0
        for o in eligible_owners:
            if o in selected:
                continue
            marginal = len(candidate_voxels[o] & uncovered)
            if marginal == 0:
                continue
            marginal_fraction = marginal / universe_size
            if marginal_fraction < _MIN_CONTRIBUTION_FRACTION:
                continue
            kf = keyframe_db_by_idx[o]
            dist = float(np.linalg.norm(kf["camera_center"] - cube_center))
            score = marginal * np.exp(-dist / cube_half_size)
            if score > best_score:
                best_score = score
                best_owner = o
                best_marginal = marginal

        if best_owner is None:
            break

        covered_fraction_now = covered_so_far / universe_size
        marginal_fraction = best_marginal / universe_size
        if (
            covered_fraction_now >= _DIMINISHING_RETURNS_FLOOR
            and marginal_fraction < _DIMINISHING_RETURNS_MIN_MARGINAL
        ):
            break

        selected.append(best_owner)
        details_by_owner[best_owner]["selected"] = True
        uncovered -= candidate_voxels[best_owner]

    coverage_fraction = (universe_size - len(uncovered)) / universe_size

    return (
        [keyframe_db_by_idx[o] for o in selected],
        universe_size,
        len(uncovered),
        coverage_fraction,
        candidate_details,
    )
