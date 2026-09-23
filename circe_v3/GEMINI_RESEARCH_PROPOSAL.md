# Master Research Document & System Blueprint

> **Provenance:** authored by Google Gemini, pasted verbatim 2026-09-22 for evaluation.
> NOT validated against this repository's implementation or its cited sources.
> See the review section appended at the end of this file.

**Title:** Co-Evolving Dual-Layer Coverage Planning on Relative-Scale Incremental Maps for Autonomous Ground Inspection

**Target Venue:** IEEE / ICCCIT-2027

---

## 1. Executive Summary & Core Novelty

Autonomous surface inspection in unstructured, GPS-denied industrial environments requires shifting from spatial volume discovery to visual defect resolution. Conventional exploration planners (e.g., RH-NBVP, AEP) evaluate spatial occupancy: once a voxel transitions from unknown to occupied, its information gain drops to zero. In physical defect inspection (e.g., micro-cracks, pitting, fluid leaks), a surface observed from 6 m away at a 70 degree grazing angle is geometrically "known," yet uninspected.

Conversely, classical Inspection Coverage Path Planning (ICPP) algorithms optimize sensor poses over surfaces but require an offline, prior 3D CAD mesh. When deployed on feed-forward Visual SLAM engines (e.g., VGGT-SLAM 2.0, DUSt3R, MASt3R), standard inspection planners fail because monocular feed-forward SLAM operates up to local, relative Sim(3) scale factors that drift during loop closures. Rigid metric standoff thresholds (e.g., 1.5 m) break under scale updates.

### Core Contributions

- **Co-Evolving Map Decoupling:** Separating navigation/obstacle space (Fog Layer M_fog) from visual defect inspection quality (Surfel Layer S_detect).
- **Scale-Invariant Multi-Gate Filter:** Introducing a dimensionless pixel footprint ratio (F = f * delta_voxel / depth) where local map scale factors lambda cancel out identically, rendering coverage states immune to Sim(3) SLAM updates.
- **Ground Manifold Projection Operator (Pi_M):** Projecting candidate 6-DOF viewpoints onto the rover's physical floor manifold (z = H_CAM) before utility scoring, preventing reachability deadlocks on ground-constrained hardware.

---

## 2. Comprehensive State-of-the-Art (SOTA) Benchmarking

```
                     [ FEED-FORWARD RGB SLAM ]
                 (VGGT-SLAM 2.0 / DUSt3R / MASt3R)
                                |
                  Relative-Scale Pointmaps
                                |
                                v
              [ CO-EVOLVING DUAL-LAYER MAP LAYER ]
                +---------------+---------------+
                |   Fog Layer   | Surfel Layer  |
                |  (Occupancy)  | (Inspection)  |
                +-------+-------+-------+-------+
                        |               |
     Traversable Space  |               | Uninspected Surfel
     & Frontiers        |               | Clusters
                        v               v
              [ SCALE-INVARIANT MULTI-GATE FILTER ]
               - Pixel Footprint: F = f * (delta_voxel / d)
               - Grazing Angle: cos(theta_inc)
               - FOV & Z-Buffer Occlusion Gating
                                |
                                v
              [ GROUND MANIFOLD PROJECTION Pi_M ]
               - c_z = H_CAM (Fixed Camera Height)
               - (c_x, c_y) in Traversable Floor Space
                                |
                                v
              [ NEXT-BEST-VIEW (NBV) OPTIMIZATION ]
               V* = argmax [ Gain(V) / Cost(V) ]
```

### Literature Benchmark Matrix

| Baseline Category | Representative SOTA Framework | Primary Focus | Primary Limitation | Literature Reference |
|---|---|---|---|---|
| Volumetric Exploration | RH-NBVP (Bircher et al.) | Receding-horizon RRT tree optimization over OctoMaps. | Premature Termination: Stops once space is marked occupied; ignores visual defect resolution. | ResearchGate / IEEE |
| Hybrid / Frontier Exploration | AEP / Dual-Stage (Selin et al.) | Local NBV combined with global frontier switching. | Entropy-Only Focus: Optimizes volumetric mapping speed rather than surface inspection quality. | Semantic Scholar |
| Surface Inspection (ICPP) | Sampling-based CPP (Englot & Hover) | Standoff and incidence angle optimization over meshes. | Requires Prior CAD Mesh: Strictly offline; fails during incremental online exploration in unknown environments. | MIT DSpace / AAAI |
| Unconstrained 3D Vision | DUSt3R (Wang et al.) | Unconstrained dense 3D pointmap regression. | High GPU memory footprint; passive vision model (no active planning). | arXiv:2312.14132 |
| Dense Matching & Alignment | MASt3R / MUSt3R (Naver Labs) | Pixel-accurate matching and pointmap alignment. | Operates up to relative submap scales; requires scale-invariant downstream planning. | arXiv:2503.01661 |
| Our Framework | Co-Evolving Dual-Layer Planner | Online active surface inspection on relative SLAM | None (Combines active planning with scale-invariant neural SLAM submaps). | Proposed Work |

---

## 3. Formal System Architecture & Edge Execution Flow

```
[ Arduino UNO Q (Real-time Motion) ] <-- (2D Velocity Commands) --> [ RTX 4050 Edge GPU (VGGT-SLAM + Dual-Layer Planner) ]
                                                                              |
                                                              (Filtered Defect Detections)
                                                                              |
                                                                              v
                                                                [ Cloud Backend / HTTP POST ]
```

- **Hardware Stack:** Low-profile tracked ground rover with an Arduino UNO Q for low-level motor control/PIR integration, and an RTX 4050 Edge GPU executing VGGT-SLAM 2.0 and the dual-layer planner.
- **PIR & Frame Loop:** PIR trigger initiates a 10 FPS camera capture buffer. Local background inference runs at 1 FPS to extract surfel candidate patches.
- **Cloud Notification:** Verified inspection defects are compiled into JSON payloads and pushed via HTTP POST to the backend cloud service.

---

## 4. Formal LaTeX Methodology Section (IEEE/ICCCIT Format)

```latex
\section{Methodology}

\subsection{Co-Evolving Dual-Layer Map Representation}
To resolve the fundamental conflict between volumetric spatial discovery and high-resolution surface inspection, we decouple the environment representation into two synchronized layers:
\begin{enumerate}
    \item \textbf{Fog Layer ($\mathcal{M}_{\text{fog}}$):} A 3D occupancy voxel grid used strictly for collision-free trajectory generation, local frontier discovery, and traversability analysis.
    \item \textbf{Surfel Layer ($\mathcal{S}_{\text{detect}}$):} An incremental set of oriented surface elements (surfels) extracted from VGGT-SLAM 2.0 submaps. Each surfel tracks visual inspection quality attributes rather than binary spatial occupancy.
\end{enumerate}

\subsection{Scale-Invariant Multi-Gate Adequacy Filter}
Because monocular feed-forward SLAM backbones operate up to an arbitrary local scale factor $\lambda \in \mathbb{R}^+$ subject to $\text{Sim}(3)$ graph optimizations, metric distance bounds fail. We define visual resolving power via a dimensionless pixel footprint ratio:
\begin{equation}
F(s, V) = f \cdot \frac{\delta_{\text{voxel}}}{d}
\end{equation}
where $f$ is the camera focal length in pixels, $\delta_{\text{voxel}}$ is the local voxel dimension in map units, and $d$ is the sensor-to-surfel distance in map units. Under any scalar scale transformation $\mathbf{p}' = \lambda \mathbf{p}$, the scale factor cancels identically:
\begin{equation}
F'(s, V) = f \cdot \frac{\lambda \cdot \delta_{\text{voxel}}}{\lambda \cdot d} = F(s, V)
\end{equation}
rendering the inspection coverage state mathematically invariant to SLAM scale drift.

\subsection{Ground Manifold Projection Operator ($\Pi_{\mathcal{M}}$)}
To prevent reachability deadlocks caused by unconstrained 6-DOF viewpoint generators, candidate inspection stations $V \in \mathbb{SE}(3)$ are projected onto the ground rover's physical floor manifold $\mathcal{M}_{\text{rover}}$:
\begin{equation}
\Pi_{\mathcal{M}}(V) = \arg\min_{V' \in \mathcal{M}_{\text{rover}}} \| \mathbf{p}_V - \mathbf{p}_{V'} \|^2 \quad \text{s.t.} \quad c_z = H_{\text{CAM}}
\end{equation}
Viewpoint utility is evaluated exclusively on the projected manifold $\Pi_{\mathcal{M}}(V)$, ensuring all generated Next-Best-View (NBV) trajectories are kinematically feasible for low-profile ground hardware.
```

---

## 5. Defense Summary for Peer Reviewers

- **Q: "Why not just use an existing volumetric planner like RH-NBVP?"**
  A: RH-NBVP optimizes spatial volume discovery. Once a wall is marked occupied, its information gain drops to 0. It leaves surfaces uninspected for micro-defects.

- **Q: "Why not use established inspection path planners like ICPP?"**
  A: Classical ICPP requires a complete 3D CAD model prior to deployment. It cannot run online in unknown, GPS-denied environments.

- **Q: "Why is scale invariance necessary if SLAM provides 3D pointmaps?"**
  A: Feed-forward monocular SLAM engines (VGGT-SLAM 2.0, DUSt3R, MASt3R) output relative submaps up to Sim(3) scale factors. Rigid metric distance thresholds (1.5 m) break when global loop closures update submap scales. Our dimensionless footprint ratio (F) cancels the scale factor identically.
