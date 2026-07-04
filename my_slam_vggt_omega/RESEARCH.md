# Research: Stitching VGGT-Omega Submaps into a Consistent Global Map

*Audit + literature review, July 2026. Motivated by a broken `global_map.glb`: appeared
mirrored/upside-down, structures duplicated, submaps mutually inconsistent.*

---

## 1. Problem statement

VGGT-Omega is a feed-forward batch model. Every inference pass over N frames produces a
reconstruction in a **fresh arbitrary coordinate system with a fresh arbitrary scale**
(the model normalizes scene scale per batch). A streaming pipeline that runs many passes
must therefore solve, per pass:

1. **Rotation + translation** between the pass-local frame and the global frame (SE(3)).
2. **Scale** between the pass-local units and global units (the missing 7th DOF → Sim(3)).
3. **Merging** the aligned geometry into one map without duplicates or stale ghosts.

Getting any one of these wrong produces exactly the artifacts we observed.

---

## 2. Audit of our implementation — four bugs found

### Bug 1 — Scale never entered the geometry (root cause of inconsistency)

The base paper we follow (arXiv:2511.16282) applies the estimated per-block scale
*"to all VGGT depths **and extrinsics** in the block"* **before** computing the alignment
Delta. Our code computed `scale` from depth ratios but applied it **only to the stored
`depth` field** of the keyframe DB — never to the depth used for unprojection, never to
the extrinsic translations.

Consequences:
- Every submap landed in the global frame at its **own random scale** → the same door
  appears twice at two different sizes; the trajectory bends nonsensically.
- The keyframe DB became internally inconsistent: `kf["depth"]` was scale-corrected but
  `kf["world_points"]` was not, so the covisibility sphere check and the next pass's
  scale estimate consumed mismatched data → compounding drift.

### Bug 2 — Global map exported without the viewer-convention transform ("mirrored")

Per-pass submap GLBs are transformed by `inv(ext[0]) @ opengl_conv` where
`opengl_conv = diag(1, -1, -1, 1)` — a 180° rotation about X that converts the
camera convention (+Y down, +Z forward) to the GLB viewer convention (+Y up, +Z toward
viewer). The global map was exported in **raw camera-convention coordinates with no
transform**, so it rendered upside-down and back-to-front relative to every submap.

Note: a true mirror is mathematically impossible in this pipeline — all rotations come
from quaternions (proper rotations, det = +1) and SE(3)/Sim(3) transforms preserve
handedness. The "mirrored" appearance was Bug 2 compounded by Bug 1.

### Bug 3 — No confidence or edge filtering in the global map

Submaps are filtered at the 20th confidence percentile plus depth-edge rejection (inside
`predictions_to_glb`). The global map kept every point with `conf > 1e-5` → sky junk,
depth-edge smears, and far-point streaks dominate the visual. VGGT-SLAM prunes points
below 25 % of the mean confidence per submap; comparable filtering is required at merge
time.

### Bug 4 — Single-anchor Delta (accuracy, not correctness)

Delta was computed from **one** anchor frame per pass. With up to 11 known frames in a
batch, per-frame Deltas can be averaged in closed form (rotation averaging via SVD
projection to SO(3), median translation), and the spread between per-frame Deltas is a
free **drift diagnostic**. The base paper also uses a single anchor, so this is an
upgrade rather than a bug fix — but the residual logging matters for observability.

---

## 3. Literature review

### VGGT-SLAM — arXiv:2505.12549 (MIT, NeurIPS 2025)

The closest system to ours: builds submaps with VGGT, each sharing **one frame** with the
previous submap, and aligns them using the dense pointmap correspondences of that shared
frame.

- **Key theoretical result:** with *uncalibrated* cameras, feed-forward reconstructions
  are only defined up to a **15-DOF projective transform** (Projective Reconstruction
  Theorem). Sim(3) alignment cannot absorb the shear/perspective distortions, especially
  at small inter-frame disparity. They therefore estimate **SL(4) homographies**
  (5-point RANSAC on the shared-frame pointmaps) and optimize a factor graph (GTSAM)
  with loop-closure constraints.
- **Practical settings:** submap width 8–32 frames; keyframe admission at ≥ 25 px
  Lucas-Kanade disparity; point pruning below 25 % of mean confidence.
- **Relevance to us:** we deliberately do **not** adopt SL(4)/GTSAM (project rule:
  consistency must emerge from joint inference, not a separate optimizer). The SL(4)
  result tells us what error to *expect* to remain: slow projective drift that Sim(3)
  cannot absorb. Our covisibility-sphere revisits + refresh-on-participation are the
  mechanism that counteracts it (re-inference replaces drifted geometry instead of
  optimizing it).

### Base paper — arXiv:2511.16282 ("Temporally coherent 3D maps with VGGT")

The design our pipeline follows most directly:

- Sliding window over the stream; the next block's frame list = previous block's
  keyframes + current frames (k overlapping keyframes).
- **Sim(3) alignment**: a scale factor per block is computed by least-squares against
  LiDAR depth (median over frames), then applied **to all VGGT depths and extrinsics**;
  afterwards a **single-anchor SE(3) Delta** `Δ = inv(E_local_ref) @ E_global_ref` is
  right-multiplied onto all block extrinsics.
- Since we have no LiDAR, our substitute is the **median depth-ratio** between the
  current pass's raw depth and the stored (already-global-scale) depth of the
  overlapping frames — same estimator shape, self-referential instead of sensor-anchored
  (accepting slow metric drift as the price of being sensor-free).

### EC3R-SLAM (arXiv:2510.02080), SING3R-SLAM (arXiv:2511.17207), MASt3R-Fusion (arXiv:2509.20757)

All 2025 feed-forward-pointmap SLAM systems; all estimate **Sim(3)** submap constraints
via the **(weighted) Umeyama algorithm** on corresponding points, then feed them into a
pose graph. Confirms: when intrinsics are *predicted* by the network (as VGGT-Omega
does), Sim(3) + anchor frames is the standard alignment class; the projective residual
that worried VGGT-SLAM is handled by (a) calibrated-ish intrinsics and (b) graph
optimization — or in our case, (b') periodic re-inference of revisited regions.

### MASt3R-SLAM (CVPR 2025), SLAM3R, Spann3R / CUT3R (context)

- MASt3R-SLAM: two-view pointmap priors + second-order optimization; calibration-free.
- SLAM3R / Spann3R / CUT3R: replace explicit alignment with **learned memory/implicit
  state** — the model itself carries the global frame. This is the "philosophically pure"
  end-state (no alignment code at all) but requires a model trained for streaming, which
  VGGT-Omega is not.

---

## 4. Chosen approach (and why)

**Sim(3)-style alignment, implemented as: scale → then SE(3) Delta — plus
refresh-on-participation for consistency over time.**

Per pass:
1. Estimate scale `s` = median over known batch frames of median(stored_depth / raw_depth)
   on high-confidence pixels, clipped to [0.1, 10].
2. Apply `s` to the geometry: `depth ← s · depth`, `E_local ← [R | s·t]` (this was Bug 1).
3. Compute per-known-frame Deltas `Δ_i = inv(E_local_scaled[i]) @ E_global_stored[i]`;
   average rotations (SVD projection to SO(3)), median translations; log the rotation
   spread as `delta_residual_deg`.
4. `E_global = E_local_scaled @ Δ`; world points via unprojection **from the global
   extrinsics directly** (points and cameras cannot disagree by construction).
5. Every batch frame's DB entry is overwritten with the fresh reconstruction
   (refresh-on-participation). Covisibility-sphere revisits pull old keyframes into the
   batch, so returning to a mapped area re-reconstructs old + new views **jointly in one
   inference** — the feed-forward analogue of loop closure, per project philosophy.

Rejected alternatives:
- **SL(4) + factor graph (VGGT-SLAM):** banned by project rules (no GTSAM, no separate
  optimizer); its benefit (absorbing projective drift) is partially recovered by
  re-inference of revisited regions.
- **Umeyama on pointmaps:** more robust than pose-anchor alignment in principle, but
  needs correspondence weighting and RANSAC to resist depth outliers; pose-based
  multi-anchor averaging gets most of the benefit with none of the machinery. Revisit if
  `delta_residual_deg` logs show pose-only alignment is the bottleneck.
- **Learned-memory models (CUT3R et al.):** requires a different model; out of scope.

Known remaining limitations:
- **Metric drift**: scale is chained pass-to-pass with no absolute anchor (no LiDAR/IMU).
- **Projective drift**: Sim(3) cannot absorb shear/perspective error (VGGT-SLAM's
  argument); mitigated but not eliminated by revisit re-inference.
- **Ghosting**: stale points from keyframes that overlap a revisited region but were not
  selected into the batch. Candidate mitigations (deferred): invalidate non-participant
  points inside the current sphere; voxel-grid dedup at export.

---

## 5. References

| System | Link | Alignment | Notes |
|---|---|---|---|
| VGGT-SLAM | [arXiv:2505.12549](https://arxiv.org/abs/2505.12549) | SL(4) homography + GTSAM | 1 shared frame/submap; 25 px disparity keyframing; 25 % conf pruning |
| Base paper | [arXiv:2511.16282](https://arxiv.org/abs/2511.16282) | Sim(3): LiDAR scale + 1-anchor SE(3) | Scale applied to depths **and** extrinsics; our template |
| EC3R-SLAM | [arXiv:2510.02080](https://arxiv.org/abs/2510.02080) | weighted Umeyama Sim(3) + pose graph | |
| SING3R-SLAM | [arXiv:2511.17207](https://arxiv.org/abs/2511.17207) | Sim(3) submap registration | Gaussian-splat map |
| MASt3R-Fusion | [arXiv:2509.20757](https://arxiv.org/abs/2509.20757) | Sim(3) visual + SE(3) factor graph | IMU/GNSS anchored |
| MASt3R-SLAM | [CVPR 2025](https://www.semanticscholar.org/paper/0022d6667d72a07ddcc29c0f2e22e1068df6d58c) | pointmap matching, 2nd-order opt | calibration-free |
