# Autonomous Factory-Floor Inspection Rover — v2 Proposal (Ground-Only)

**Status:** v2 revision. The original **Mothership-Scout** concept (ground rover + companion drone, eye-to-hand visual servoing) is **superseded**. The drone is **dropped** — it was not stably steerable and the aerial half was unreliable. This document is the **ground-only** redesign: a single low-profile tracked rover that explores and inspects a factory floor using vision-based SLAM. All aerial elements, docking, and drone servoing are removed from scope.

**Target contest:** Arduino UNO Q + App Lab — *Industrial IoT / Smart Predictive Maintenance* scenario.

**Document type:** Build / architecture spec. A section on the contest pitch framing is included (§9). Items are explicitly marked **[SETTLED]** or **[OPEN / DEFERRED]** so the doc reflects the real state.

---

## 1. Concept

A low-profile autonomous tracked rover inspects an industrial factory floor for early-failure and safety defects. It navigates GPS-denied, LiDAR-free, using a single camera and a feed-forward visual-SLAM backbone (VGGT-SLAM 2.0) run off-board. It explores autonomously, stops at intervals, photographs its surroundings, and runs defect detection **on-board** the UNO Q. Detected defects are logged with their location on the reconstructed map.

**Why a rover (not a human walk-through):** the low-profile chassis reaches under/around equipment (under conveyors, between machine bases, tight equipment rows) that are tedious or awkward for humans to inspect repeatedly, and it can run the same route frequently to track slow degradation over time — the predictive-maintenance payoff.

**Height constraint (explicit):** the chassis is low, so tall machine bodies, gauges, panels, and high walls are **out of view by design**. Inspection targets are therefore **floor-level and base-level**: floor/plinth condition, machine bases, and low fixtures. This constraint shapes the whole class list and capture pattern.

---

## 2. Hardware

| Part | Component | Notes |
|---|---|---|
| Chassis | TP101 tracked chassis (low profile) | Skid-steer, open-loop |
| Motors | DC track motors, **no encoders** | Open-loop only; coarse moves |
| Compute (on-rover) | Arduino UNO Q (4GB) | Dual-brain: STM32U585 MCU + Qualcomm MPU (Debian) |
| Compute (off-board) | RTX 4050 6GB, Ubuntu | Remote, runs VGGT-SLAM 2.0 |
| Camera | Raspberry Pi Camera v2 (IMX219), **62.2° H × 48.8° V**, 8MP | Connected via RPi Zero 2W bridge to the UNO Q |
| IMU | **MPU-6500** (6-axis: gyro + accel, **no magnetometer**) | Optional MPU-9250 upgrade considered, low value in this environment |
| Obstacle safety | Forward ToF sensor | Hardware stop reflex |
| Link | WiFi | Rover ↔ RTX (frames out, poses back) |

**Note on IMU:** MPU-6500 has no magnetometer, so no absolute compass heading. This is largely irrelevant here because (a) VGGT-SLAM provides absolute heading correction at every stop, and (b) a factory floor's motors/steel corrupt magnetometers anyway. Gyro-integrated *relative* heading between SLAM updates is all that's needed.

---

## 3. Perception & Mapping — [SETTLED]

- **Backbone:** VGGT-SLAM 2.0 (Maggio & Carlone, MIT SPARK, 2026). Feed-forward RGB SLAM: dense point cloud + per-frame camera poses, incremental submap alignment with attention-based loop closure. Real-time demonstrated on a ground robot (Jetson Thor) in the paper; the remote RTX 4050 exceeds that for the SLAM compute.
- **Runs off-board** on the RTX 4050. Rover streams frames; poses/map come back.
- **Output is relative scale** (no metric). This is a core property, not a bug — the whole pipeline is designed to be scale-free where possible (see §5).
- **Map representation:** relative occupancy / elevation grid derived by down-projecting the point cloud onto the floor plane and height-filtering. Obstacles = cells clearly above floor height. No metric needed for "avoid tall cells" (relative height suffices).

---

## 4. Navigation & Exploration — [SETTLED]

- **Exploration model:** rover heads toward outstanding coverage gaps (see §7A). VGGT reconstructs large dense chunks per view, so unknown regions are primarily **occlusion-bound** (behind machines, around corners) rather than sensor-range-bound. The rover repositions to reveal occluded pockets *and* to re-image already-mapped surfaces that lack detection-grade views.
- **Incremental, single pass.** VGGT-SLAM 2.0 is incremental — the map grows *as the rover moves*. There is **no upfront "map first, then inspect" phase**; mapping and inspection happen in the same traversal, with two coverage layers co-evolving on the growing map (§7A).
- **No explicit goal.** Full coverage is the emergent objective — continue until *both* coverage layers close (§7A).
- **Stop trigger:** next station chosen by gap-driven selection (§7A); between stations, motion is in self-calibrated command-time units (§5). Drift corrected by VGGT re-localization at each stop.
- **Obstacle avoidance:** two layers —
  1. **Map-based (planning):** avoid cells with relative height above floor.
  2. **ToF reflex (hardware):** forward ToF is a hard interrupt that overrides any motor command, independent of the (possibly stale) map. This is the real collision backstop.

---

## 5. Motion Self-Calibration (scale bootstrapping) — [SETTLED]

The rover has no encoders and the map is relative-scale. Solution: **use the rover's own motion as a self-calibrating ruler.**

- Issue a known open-loop command for a known time (e.g. both tracks max, 1 s).
- Read VGGT pose before/after → translation delta **in map units** (`‖t_after − t_before‖`).
- This gives a running conversion: **map-units per command-second**.
- **Online, not once:** VGGT relative scale can vary across submaps and shift on loop closure, so re-estimate on every known move. Maintain a running estimate (EMA or small Kalman filter) — self-heals as scale drifts.

**Notes / pitfalls:**
- Get displacement from **VGGT pose delta**, not the point cloud.
- **Translation calibrates cleaner than rotation** — skid-steer turn slip makes turn-rate noisier; calibrate separately, weight rotation as less trustworthy.
- **Measure in steady state:** a 1 s move includes accel/stiction transients; short moves undershoot. Use consistent move lengths or the steady-state middle of a longer move.

**What needs scale vs. what doesn't:**
- **Scale-free:** exploration, frontier/occlusion logic, "go toward unknown," "avoid tall cell" (relative height). Most of the system.
- **Wants metric:** fine "climb vs go-around" step-height decisions — **avoided entirely** by treating any above-floor cell as go-around. For a low rover meant to go under/around (not over) things, this is the right call anyway.

---

## 6. Localization Loop (IMU ↔ SLAM) — [SETTLED]

Two-rate estimation to bridge slow, laggy remote SLAM with continuous motion:

- **Fast layer (on-rover, ~100 Hz+):** MPU-6500 gyro+accel dead-reckoning. Runs locally on the STM32 side, no network. Tracks motion since the last known-good pose.
- **Slow layer (remote, on arrival):** each fresh VGGT pose snaps the estimate back to absolute truth, killing accumulated IMU drift.
- Start simple: integrate IMU, hard-reset to VGGT pose on each arrival. Upgrade to complementary/Kalman only if inter-update drift is too large.
- **Stop-and-inspect makes this easy:** a stationary rover barely drifts, so the pose is rock-solid exactly when photographing. The inspection design and the localization problem solve each other. Each stop is also the clean moment to reset gyro heading drift.

---

## 7. Inspection Cycle — [SETTLED, with one deferred sub-item]

### 7.1 Capture pattern
- **8 photos per full rotation** as the **baseline** ring. Rationale: RPi Cam v2 horizontal FOV ≈ 62°; with ~25% overlap, step ≈ 45°; 360° ÷ 45° = **8 shots**. The formal name for this stop-turn-shoot cycle is a **station–viewpoint** inspection pattern (stations = stops, viewpoints = the ring shots); the fixed 8-ring is the simplest (rule-based, degenerate) case of viewpoint planning — see §7A for the coverage-aware upgrade.
- **Full 8-ring early / gap-subset later:** once the coverage layer (§7A) is populated, only shoot the sectors that still contain uncovered surface, skipping directions the previous stations already covered. Deterministic zero-extra-compute coverage is the reason to keep the ring as the fallback.
- **Step-and-settle:** turn ~45°, **stop**, let settle, capture. **Never shoot while rotating** — motion blur kills detection. (Also keeps the localization pose clean.)
- One horizontal ring, aimed slightly down toward floor/base-level targets (matches the height constraint). **Floor surfaces need a downward-looking DOF** — see §7A and Open Item #6.
- Then move to the next station (§7A) and repeat the drill.

### 7.2 Two-tier detection with mid-ring escalation
- **Continuous detection** runs on the live video stream (not only on the 8 stills), enabling frame-to-frame **tracking**.
- **Still HD photos** are captured at each of the 8 settle points for high-quality confirmation/records.
- **Escalation trigger (both must hold):** a detection is **low-confidence AND small in frame** → "something might be there but too far to tell." (Large low-conf = distance isn't the issue, don't approach; high-conf = no need.)
- **Interrupt mid-ring, not after:** escalate the instant the trigger fires, *before* continuing the ring — otherwise the low-conf target can't be relocated once the rover moves on.
  1. At shot *k*, small low-conf detection fires.
  2. Store ring state: shot index + the **VGGT pose of the ring center** (the return anchor).
  3. Approach: turn to center the object, drive forward in small steps, re-detecting, until big/confident enough to confirm or dismiss. Capture close-up HD frames.
  4. **Return to the stored ring-center pose** and re-orient to shot-*k* heading (closed-loop on VGGT pose; approach was short, so return error is bounded).
  5. Resume ring from shot *k*+1.
- **Re-detection guard (avoid infinite re-escalation):** because detection is continuous, a **tracker (ByteTrack / IOU-centroid)** holds a persistent ID per object; a track already inspected is marked handled and won't re-trigger. **Backup:** map-location suppression — mark the inspected object's approximate map location handled; suppress escalation on detections pointing at handled locations (tolerance radius = tuning parameter). Track ID covers the short continuous timescale; map-location covers "lost and re-saw later."
- **Fallbacks / edge cases:**
  - If return-to-ring-center can't be re-established within tolerance → **re-do the whole ring** from current position rather than stitching a misaligned one. *(Return accuracy is the main risk point in this behavior.)*
  - Multiple simultaneous low-conf blobs → handle **one per interrupt** (nearest/lowest-conf first); chase any still-present others after the ring completes. Do not queue mid-ring.

---

## 7A. Coverage-Aware Station & Viewpoint Planning — [SETTLED design / OPEN tuning]

This is the professional formulation of the "dancing-angel" hop-rotate pattern. It sits in the **Next-Best-View (NBV) / station–viewpoint coverage** literature. The pieces below are standard NBV + coverage-tracking machinery, specialized to this rover's constraints (fixed low camera height, relative scale, incremental SLAM). It is an assembly of known methods, not a novel algorithm.

### 7A.1 Two coverage layers (the key idea)

**Mapped ≠ inspected.** Two *separate* coverage concepts are tracked on the persistent incremental map, on two different structures:

1. **Fog layer (mapping coverage)** — on the occupancy voxel grid. A voxel is `unknown / free / occupied`. "Fog" = `unknown`. Frontier = free↔unknown boundary. Answers: *has this space been reconstructed at all?* Drives exploration.
2. **Detection-quality layer (inspection coverage)** — on the surfels. Answers: *has this real mapped surface been imaged at detection-grade quality?* Drives inspection.

These do **not** coincide. A region can be fully out of fog (geometry reconstructed) yet have zero detection-grade images (VGGT mapped it fine from far away at a grazing angle — useless for spotting a hairline crack). The **detection layer clears slower** and is what gates completion. Because both layers co-evolve on one incrementally-growing map, **revisits are emergent**: a surface mapped early-but-not-inspected stays an uncovered gap and gets selected later when its gain/cost wins — no special "come back" bookkeeping.

### 7A.2 State carried between cycles

```
Occupancy voxel grid:  each voxel ∈ {unknown, free, occupied}     # fog layer
Surfel set (one per occupied voxel):                              # detection layer
    p_s      : 3D position          (map units, relative scale)
    n_s      : unit surface normal  (PCA of k-NN; sign toward observed side)
    covered  : bool                 (init false)
    q_best   : float                (best observation quality; init 0)
```

### 7A.3 Adequacy test (core primitive)

Is surfel `s` inspection-covered by view `V=(c,R,K)`? Four gates:

```
adequate(s, V):
    u = project(s.p_s → V)
    if u ∉ image_bounds:            return false   # (1) in FOV
    if occluded(s, V):              return false   # (2) not blocked (z-buffer vs. surfels)
    depth      = ‖s.p_s − c‖
    footprint  = f · (voxel_size / depth)          # pixels the surfel spans
    if footprint < F_MIN:           return false   # (3) close/sharp enough
    ray        = (s.p_s − c)/depth
    cos_inc    = dot(−ray, s.n_s)                   # 1 head-on, 0 grazing
    if cos_inc < COS_INCIDENCE_MIN: return false   # (4) square enough
    return true
```

**Scale-freeness:** (1) projective, (2) ordering-only, (4) unit-vector dot — all scale-free. (3) `footprint = f·voxel_size/depth`: `voxel_size` and `depth` are both map units, so the ratio is scale-invariant → real pixel count comparable to `F_MIN`. So the whole test works on relative scale. Quality score `q = footprint · cos_inc`.

### 7A.4 Per-cycle sequence (one station selection, end to end)

1. **Extend map** from the new VGGT submap: insert points (fog shrinks), voxel-downsample new occupied region into surfels, estimate normals by PCA (sign toward the observing camera). New surfels start `covered=false` → auto-enter the inspection backlog.
2. **Update coverage** from the frames captured at the previous station: `for s in frustum_cull(V): if adequate(s,V): s.covered=true; s.q_best=max(...)`.
3. **Extract gaps:** `detection_gaps = euclidean_cluster({s: ¬covered}, R_CLUSTER)`; `fog_frontier = free voxels adjacent to unknown`.
4. **Plan a viewpoint per gap** — project the ideal NBV onto the rover's reachable manifold (§7A.5).
5. **Score & pick** — greedy `argmax(gain/cost)` (§7A.6); if no reachable detection gap → head to nearest fog frontier; if neither remains → `DONE`.
6. **Execute** — drive (calibrated + IMU/SLAM + ToF) → settle → ring or gap-subset capture → onboard detect/track/escalate → live `update_coverage`.

### 7A.5 Viewpoint projection onto the reachable manifold (the constraint that bites)

The camera cannot reach an arbitrary 6-DOF pose. Reachable views are constrained:

```
c.z   = H_CAM              # FIXED — camera bolted at one low height
c.xy  ∈ traversable floor  # free to choose
yaw   ∈ [0, 2π)            # skid-steer body rotation
tilt  ∈ [tilt_min,tilt_max]   if a tilt servo exists, else FIXED_TILT
```

For gap cluster `G` (centroid `C`, mean normal `n̄`): ideal standoff view is `c_ideal = C + d*·n̄` (look head-on from distance `d*` giving footprint just above `F_MIN`). Then **project**: force `c.z = H_CAM`, place `c.xy` along the horizontal part of `n̄` at distance `d_h`, set `yaw` to face `C`, set `tilt` (or fixed), `snap_to_traversable(c.xy)`, and **re-verify adequacy after the snap** (snapping can degrade the view). If no surfel of `G` is adequate from the snapped pose → mark `G` unreachable.

**Geometric consequence (forced, not optional):**
- **Vertical-ish surfaces** (machine sides, base plates, walls): `n̄` horizontal → low camera at horizontal standoff faces them head-on → **close cleanly**.
- **Floor surfaces** (concrete-floor cracks, `n̄ ≈ +z`): a low forward-looking camera sees them near-grazing → gate (4) fails from every reachable `(x,y)`. **Without a downward DOF, floor surfels are unsatisfiable and never clear.** Floor crack is in the class list → this forces a decision (Open Item #6): tilt servo, permanent down-cant, or drop floor-crack.

### 7A.6 Station scoring

A station serves *all* gaps co-visible from its pose, not just one:

```
for G in detection_gaps:
    V = plan_viewpoint(G);  if V == NONE: continue
    gain = |{ s : adequate(s, V) }|             # every surfel V covers, across all gaps
    cost = path_len(rover → V.c) + W_TURN·|Δyaw|
pick argmax(gain / cost)
```
Greedy gain-per-cost; no global optimization (fine for the edge budget). `cost` includes a turn penalty because skid-steer turning is slow and slip-prone. Distances are relative but consistent, so ratios hold without metric scale.

### 7A.7 Robustness notes

- **Occlusion prediction can be wrong on a partial map** — an unmapped object may block a shot `adequate()` predicted. Self-correcting: the blocked surfels simply stay `¬covered` and resurface as a gap next cycle. No special handling.
- **Escalation (§7.2) is a reactive override**, not part of planning; its close-up frames also run `update_coverage`, so escalated surfaces get flagged.
- **Termination:** both `detection_gaps` and `fog_frontier` empty over the reachable floor.

### 7A.8 Tunables (all empirical; scale-free except distances)
`F_MIN` (footprint→YOLO-works), `COS_INCIDENCE_MIN` (~cos 60° start), `d*`/`d_h` (standoff, map units), `R_CLUSTER`, `W_TURN`, `H_CAM`, tilt range. Calibrate `F_MIN` by placing a known defect at increasing map-distances and finding where YOLO recall collapses.

---

## 8. Defect Detection Model — [SETTLED (model) / OPEN (dataset)]

- **Model:** YOLO26n (Ultralytics), fine-tuned from COCO-pretrained. Chosen as the lightest/fastest current YOLO (nano: ~1.7 ms T4 TensorRT, 2.4M params, NMS-free end-to-end) — suits real-time on-board inference on the UNO Q. Fine-tune, don't train from scratch.
- **Classes (floor/base-level, visually distinct, stageable, tied to real early-failure signals):**
  - Oil / fluid leak
  - Rust / corrosion
  - Concrete crack
  - *(optional)* Concrete spalling
- **Data situation — [OPEN, unstarted]:**
  - Oil leak: a real Roboflow industrial set exists (~311 imgs).
  - Crack / spalling: transferable from civil datasets (CODEBRIM / SDNET2018) — a floor crack ≈ a bridge crack physically.
  - Rust + factory-specific classes: likely **staged in-house** (Sri Lanka), shot from **rover-eye height/angle** to match deployment.
  - Target ~150–200 imgs/class minimum, varied angle/lighting/distance, augmentation on. 70/20/10 split.

---

## 9. Compute Placement / Contest Pitch — [SETTLED]

**The claim is autonomy, not efficiency.** (Efficiency framing invites "just offboard it to the RTX." Autonomy framing makes offboarding the thing that *fails*.)

- **Off-board (RTX):** heavy, **intermittent**, latency-tolerant geometry → VGGT-SLAM. A late map update is survivable. The GPU is a **prototype scaffold** — the product vision is a rover with no GPU umbilical.
- **On-board (UNO Q):** light, **continuous**, latency-critical perception → YOLO26n live detection + tracking driving real-time motion/escalation decisions **every frame**. This loop **physically cannot run over the WiFi link**: continuous HD video streaming is bandwidth-prohibitive, variable round-trip latency breaks tracking (dropped/late frames = lost IDs) and stutters the approach loop, and it would depend on the single most fragile part of the system. The RTX *can't* do this job, not merely *shouldn't*.
- This split maps directly onto the UNO Q's **dual-brain (MCU + MPU)** value proposition — the contest's core selling point.
- **Centerpiece demo:** mid-run, **physically cut the WiFi**. SLAM map goes stale, but the rover keeps detecting, tracking, and approaching defects on-board — visibly proving the on-board loop is real and independent. Undeniable because judges *watch it happen*.

---

## 10. Open / Deferred Items

| # | Item | State | Risk |
|---|---|---|---|
| 1 | **Dataset** — sourcing/staging images | **Unstarted.** Longest-lead task. | High (blocks detection) |
| 2 | **MCU/MPU software split + Bridge RPC data flow** on UNO Q | Deferred ("decide later"). Concept clear (real-time motor/IMU/ToF on STM32; vision on MPU). | Medium |
| 3 | **Defect → map pinning** | Deferred. Plan: re-run full VGGT over captured images; defect inherits its image's pose. | Low |
| 4 | **Escalation return-accuracy** tuning + suppression tolerance radius | Design settled (§7.2); real-world tuning pending. | Medium (main behavioral risk) |
| 5 | **Camera lens** | Confirmed **stock v2, 62°, 8 shots**. | Closed |
| 6 | **Downward DOF for floor surfaces** — tilt servo vs. permanent down-cant vs. drop floor-crack class | **Undecided.** Forced by §7A.5 geometry: a fixed low forward camera cannot inspection-cover the floor (grazing incidence). Floor-crack is currently a class. | Medium (blocks floor-crack coverage) |
| 7 | **Coverage-planning tunables** (`F_MIN`, `COS_INCIDENCE_MIN`, standoff, `R_CLUSTER`, `W_TURN`) | Design settled (§7A); empirical calibration pending. | Medium |

**Biggest risks, honestly:** (a) the real-time loop between remote heavy SLAM and a moving rover over WiFi — mitigated by stop-and-inspect + IMU bridge, but still the most likely demo-day failure; (b) the escalation return-to-ring accuracy; (c) the unstarted dataset.

**Cut from scope (v1 → v2):** companion drone, aerial servoing, autonomous docking/return-to-base, cloud dashboard/web UI.

---

## 11. Core Loop Summary

```
incremental VGGT-SLAM 2.0 (remote RTX, relative scale) — map grows as we move
   → extend two co-evolving layers on the persistent map:
        • fog layer      (voxel grid: unknown/free/occupied)   — mapping coverage
        • detection layer (surfels: covered? + quality)        — inspection coverage
   → update detection coverage from last station's frames (adequacy test)
   → extract gaps:  detection-gaps (uncovered surfel clusters) + fog frontier
   → choose next STATION = argmax(gain / cost) over reachable gap viewpoints
        (viewpoint = ideal NBV projected onto {traversable xy, yaw, tilt, z=H_CAM})
        if no reachable detection gap → head to nearest fog frontier
        if neither remains → DONE
   → drive to station  [motion self-calibration + IMU/SLAM localization]
   → STOP
        → ring (8-shot) or gap-subset settle-and-shoot (continuous detection + tracking)
             ├─ small + low-conf detection?
             │     → interrupt ring, store pose
             │     → approach (re-detect, close-up HD)
             │     → return to ring center, resume
             │     └─ tracker + map-location suppression prevent re-escalation
             └─ confirmed defect → log + pin to map pose
             └─ update detection coverage live (escalation frames included)
   → repeat until BOTH layers close (detection layer gates completion)
   +  ToF hard-interrupt collision reflex (always on)
   +  onboard AI survives WiFi loss (autonomy demo)
   +  revisits to mapped-but-uninspected surfaces are emergent from the gap logic
```