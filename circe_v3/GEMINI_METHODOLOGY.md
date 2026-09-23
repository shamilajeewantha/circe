# Section III: Methodology — Detailed Breakdown

> **Provenance:** authored by Google Gemini, pasted verbatim 2026-09-22.
> Companion to `GEMINI_RESEARCH_PROPOSAL.md`. NOT validated against the literature
> or this repository's implementation.

Paper: *Co-Evolving Dual-Layer Coverage Planning on Relative-Scale Incremental Maps for Autonomous Ground Inspection* (ICCCIT-2027)

The core objective of the proposed framework is to enable an autonomous low-profile ground rover to perform online surface defect inspection in unknown, GPS-denied environments without requiring a prior 3D CAD mesh or metric absolute-scale SLAM initialization.

```
+-----------------------------------------------------------------------------------+
|                        VGGT-SLAM 2.0 Feed-Forward Backbone                        |
|        Monocular RGB Stream ---> Relative-Scale Pointmaps & Sim(3) Submaps        |
+------------------------------------------+----------------------------------------+
                                           |
                                           v
+-----------------------------------------------------------------------------------+
|                           Dual-Layer Map Representation                           |
|  +-----------------------------------+    +------------------------------------+  |
|  | Fog Layer M_fog                   |    | Surfel Layer S_detect              |  |
|  | Collision-free driving space      |    | Oriented surface elements          |  |
|  | Frontiers for global discovery    |    | Visual inspection quality metrics  |  |
|  +-----------------+-----------------+    +-----------------+------------------+  |
+--------------------|----------------------------------------|---------------------+
                     |                                        |
                     v                                        v
+-----------------------------------------------------------------------------------+
|                       Scale-Invariant Multi-Gate Filter                           |
|       Calculates Dimensionless Pixel Footprint: F(s, V) = f * (delta_voxel / d)   |
|       Submap scale factor lambda cancels identically: (lambda*delta) / (lambda*d) |
+------------------------------------------+----------------------------------------+
                                           |
                                           v
+-----------------------------------------------------------------------------------+
|                     Ground Manifold Projection Operator Pi_M                      |
|       Projects unconstrained 3D viewpoints onto Floor Plane: c_z = H_CAM          |
|       Evaluates viewing angle & grazing limits on reachable 2D plane              |
+------------------------------------------+----------------------------------------+
                                           |
                                           v
+-----------------------------------------------------------------------------------+
|                     Next-Best-View (NBV) Utility Optimization                     |
|           Maximizes Information Gain / Travel Cost on Floor Manifold              |
|           Outputs 2D Velocity Commands to Low-Level Motion Controller             |
+-----------------------------------------------------------------------------------+
```

---

## A. Co-Evolving Dual-Layer Map Representation

Standard occupancy planners collapse all spatial observations into a single binary grid, making them blind to visual inspection quality. Conversely, dense surfel maps are computationally prohibitive to query for global pathfinding. To decouple global spatial navigation from high-resolution surface evaluation, we maintain two co-evolving map representations synchronized in real time:

### 1. Fog Layer (M_fog)

The Fog Layer represents spatial occupancy and traversability. It is initialized as an incremental 3D voxel grid `M_fog = {v_i}` where each voxel `v_i` in R^3 holds an occupancy probability `P(v_i)` in [0, 1] updated via log-odds mapping from incoming depth pointmaps.

Voxels are classified into three categorical states: unknown / free / occupied.

The Fog Layer is queried exclusively for:

- A* / D* Lite collision-free path planning.
- Local and global frontier extraction (boundary of free space against unknown) to drive broad room exploration when no uninspected surface targets are in sight.

### 2. Surfel Layer (S_detect)

Extracted directly from the dense geometric pointmaps regressed by the VGGT-SLAM 2.0 visual back-end, the Surfel Layer represents physical asset surfaces. A surfel `s_j` in `S_detect` is defined as a tuple where:

- `p_j` in R^3 is the 3D position of the surfel center in the current active submap frame.
- `n_j` in S^2 is the unit surface normal vector estimated via covariance analysis over local point neighborhoods.
- `r_j` in R+ is the local spatial radius corresponding to the local voxel resolution `delta_voxel`.
- `c_j` in R^3 is the RGB color / intensity tuple.
- `Q_j` in [0, 1] is the scalar Visual Inspection Adequacy Metric tracking whether the surface element has been observed with sufficient visual resolution for defect detection.

---

## B. Scale-Invariant Multi-Gate Adequacy Filter

Monocular feed-forward foundation models (VGGT-SLAM 2.0, DUSt3R, MASt3R) reconstruct local submaps up to an uncalibrated relative scale factor `lambda_k` in R+ for each k-th submap. During global pose-graph optimization, these relative scales shift non-linearly.

If a coverage planner relies on rigid metric distance thresholds (e.g., maintaining a standoff distance d = 1.5 m), a loop-closure adjustment by the SLAM backend will cause the planner to instantly re-plan or hallucinate uninspected zones.

To render inspection evaluation scale-invariant, we evaluate observation quality through pixel footprint projection rather than metric world distances.

```
                   CAM SENSOR (Focal Length f)
                       \      |      /
                        \     |     /
                         \    | d  /
                          \   |   /
                           \  |  /
                            \ | /
                             \|/
                  -------------------------  Surface (Voxel delta)
```

### 1. Dimensionless Pixel Footprint Ratio

Let `f` be the camera's effective focal length in pixels, `delta_voxel` be the spatial bounding extent of a surfel in local map units, and `d = ||p_s - p_V||_2` be the Euclidean distance from the camera position `p_V` to the surfel position `p_s` in the same local map frame.

We define the dimensionless Pixel Footprint Ratio:

```
F(s, V) = f * (delta_voxel / d)
```

**Proof of Scale Invariance under Sim(3) Submap Rescaling.** Suppose the SLAM backend applies a local scale adjustment factor `lambda` in R+ to the current submap frame, transforming all metric coordinates. The updated distance becomes `d' = lambda * d`. Evaluating the footprint ratio under the rescaled map yields:

```
F'(s, V) = f * (lambda * delta_voxel) / (lambda * d) = f * (delta_voxel / d) = F(s, V)
```

**Result:** the scale factor lambda cancels out identically. The pixel footprint ratio remains strictly invariant under Sim(3) submap scale drift, ensuring stable coverage state evaluation regardless of SLAM re-scaling.

### 2. Multi-Gate Adequacy Gating

A surfel `s_j` is marked Adequately Inspected (`Q_j = 1`) if and only if it simultaneously satisfies three geometric and optical gates from a candidate viewpoint V (product of Heaviside step functions):

- **Resolution Gate:** `F(s_j, V) >= F_min` enforces that the surfel projects onto at least `N_min` pixels on the camera sensor, ensuring micro-defects (cracks, rust) are visually resolved.
- **Grazing Incidence Gate:** `cos(theta_inc) = n_j . (p_V - p_j) / ||p_V - p_j|| >= cos(theta_max)`, where `theta_max` (typically 45-60 degrees) prevents severe perspective distortion and specular illumination washouts.
- **Visibility & Occlusion Gate `Vis(s_j, V)`:** a binary ray-casting function through the Fog Layer ensuring no occupied voxels lie along the line of sight between `p_V` and `p_j`, and that `s_j` falls within the camera's Field-of-View.

---

## C. Physical Manifold Projection Operator (Pi_M)

Aerial inspection planners generate candidate viewpoints anywhere in 3D space (SE(3)). Applying unconstrained 6-DOF sampling to low-profile ground rovers leads to reachability deadlocks: the planner selects ideal viewpoints (e.g., hovering 0.3 m off the floor or tilting up at extreme pitch angles) that the ground hardware cannot physically achieve.

To guarantee that every evaluated Next-Best-View is 100% reachable by the rover, we define a physical manifold projection operator `Pi_M`.

```
          UNCONSTRAINED 6-DOF CANDIDATE VIEW (V_3D)
                      \  [Unreachable Pose]
                       \
                        \   Manifold Projection Operator Pi_M
                         \  (Constrains c_z = H_CAM, Pitch=0, Roll=0)
                          v
          =================================================  Ground Manifold M_rover (z = H_CAM)
                      ROVER POSE (c_x, c_y, yaw)
```

### 1. The Ground Rover Manifold M_rover

The physical workspace of the low-profile ground rover is constrained to a 3-DOF manifold embedded within SE(3), where `H_CAM` is the fixed height of the camera mounted on the rover frame, roll and pitch are zeroed by gravity alignment, and `C_free` in R^2 represents traversable, collision-free floor space in `M_fog`.

### 2. Projection Formulation

For any unconstrained 3D viewpoint candidate `V_3D = (p, R)` in SE(3) sampled around an uninspected surfel cluster, the operator enforces:

- **Vertical Collapse:** the camera height is clamped to `c_z = H_CAM`.
- **Planar Projection:** `(c_x, c_y)` is mapped to the nearest traversable cell in the floor grid `C_free`.
- **Heading Alignment:** the camera yaw is set to point directly at the target surfel centroid.

All candidate views are projected onto `M_rover` prior to scoring their utility. This guarantees zero reachability deadlocks and forces the planner to account for non-ideal grazing angles caused by ground constraints during the viewpoint selection phase.

---

## D. Co-Evolving Next-Best-View (NBV) Optimization

The global objective function balances surface defect coverage (Surfel Layer) with spatial exploration (Fog Layer) using a dynamic weight factor.

At planning step k, candidate viewpoint poses `V_i` in `M_rover` are generated via sampling rings around uninspected surfel frontiers and unexplored fog boundaries. Each projected viewpoint `Pi_M(V_i)` is evaluated using a composite Information Utility function.

**Component definitions:**

- **Inspection Information Gain `G_inspect`:** measures the total surface area of previously uninspected surfels that will achieve adequacy (`Q_j = 1`) from pose `Pi_M(V_i)`.
- **Volumetric Fog Gain `G_fog`:** measures the volume of unknown fog voxels cleared from the same pose.
- **Traversability & Execution Cost `C_travel`:** represents the physical distance along the collision-free floor path plus a rotational penalty for heading changes.
- **Dynamic Co-Evolution Factor `alpha`** in [0, 1]:
  - When the rover is near structural assets, `alpha -> 1`, prioritizing close-up, orthogonal defect inspection.
  - When local surfaces are fully inspected (`S_uninspected -> 0`), `alpha -> 0`, causing the rover to automatically transition into global frontier exploration to find new structural assets.

---

## E. Summary of Algorithmic Pipeline

```
Algorithm 1: Co-Evolving Dual-Layer Inspection Planner Execution
----------------------------------------------------------------------------------
Input  : Monocular RGB stream, Current SLAM Submap S_k, Rover Pose P_rover
Output : Low-level 2D Velocity Commands (v, omega)

1: Receive incremental pointmap & submap pose graph from VGGT-SLAM 2.0.
2: Update Fog Layer M_fog (Ray-cast free space & occupied voxels).
3: Extract/Update Surfel Layer S_detect from dense pointmap.
4: Compute dimensionless footprint ratio F(s_j, V_curr) for all visible surfels.
5: Apply Multi-Gate Filter: Mark surfels as Adequately Inspected (Q_j = 1).
6: Sample unconstrained candidate 3D viewpoints V_cand near uninspected surfels.
7: Project all candidates onto floor manifold: V_proj = Pi_M(V_cand).
8: Compute co-evolution weight alpha based on local uninspected surfel density.
9: For each V_proj do:
10:   Compute Inspection Gain G_inspect(V_proj) and Volumetric Gain G_fog(V_proj).
11:   Compute Trajectory Cost C_travel(P_rover, V_proj) via D* Lite on M_fog.
12:   Calculate total utility g(V_proj).
13: End For
14: Select optimal Next-Best-View: V* = argmax g(V_proj).
15: Generate collision-free path on M_fog to V*.
16: Send 2D motion commands (v, omega) to low-level Arduino UNO Q controller.
```
