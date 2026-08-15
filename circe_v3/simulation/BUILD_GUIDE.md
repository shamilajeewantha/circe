# circe_v3 Simulation — BUILD GUIDE (Milestone 1: Full Coverage-Aware NBV, *no detector*)

> **Two machines.** VGGT-SLAM runs on the **Windows laptop** (in WSL, with the GPU) — that half is
> already **authored, run, and validated** (see §4). The **Gazebo sim + rover brain** runs on the
> **Ubuntu + RTX box** — that's **your** half if you are the sim-laptop agent. They talk over a network
> API (this mirrors project.md §9: rover ⇄ off-board SLAM, so the *same code runs on the real robot*).
>
> Source of truth for intended design is `circe_v3/project.md` (refs like **§7A.6** point there). This
> guide is milestone-1 scope. Items only resolvable on the sim box are marked **[SIM-BOX: decide + document]**.

---

## 1. Mission and the loop

A low-profile skid-steer rover autonomously inspects an indoor space by **imaging every reachable surface
at detection-grade quality**, using **real VGGT-SLAM 2.0** (off-board, relative scale) as the mapping/pose
backbone. It drives station-to-station; at each station it spins a **full 8-shot ring**. It finishes when
it has both *mapped* everywhere and *adequately imaged* every reachable surface.

**No object detector runs this milestone** — that is the only thing cut (plus the behaviors a detector
triggers). Everything about *where to go and where to look* is kept: the adequacy test (§7A.3) that decides
"imaged well enough" is **pure geometry** (project / occlude / footprint / incidence) — no detector needed.

```
LOOP (project.md §11, detector removed):
  VGGT-SLAM map grows (real, OFF-BOARD on the SLAM laptop, relative scale)
  → extend BOTH coverage layers on the persistent map:
       • fog layer      — voxel grid {unknown, free, occupied}      (mapping coverage, §7A.1)
       • detection layer — surfels {p, n, covered?, q_best}         (inspection coverage, §7A.2)
  → update the detection layer from the LAST station's 8 frames via the adequacy test (§7A.3, geometric)
  → extract gaps:  detection-gaps (clusters of ¬covered surfels)  +  fog frontier (free↔unknown)
  → next STATION = argmax(gain/cost) over reachable gap viewpoints (§7A.5+§7A.6);
       else nearest fog frontier; else DONE
  → ROTATE to face it, DRIVE FORWARD (open-loop command-time × self-calibrated scale, closed on fused pose)
  → STOP → FULL 8-SHOT RING (rotate 45°, settle, capture ×8) → frames to the coverage node
  → repeat until BOTH layers close (detection layer gates completion)
  +  ToF hard-interrupt reflex — always on, overrides any motor command
```

**Horizontal camera consequence (§7A.5, kept):** floor surfels (normal ≈ +z) are unsatisfiable from any
horizontal viewpoint → they stay permanently-uncovered gaps. **Exclude near-+z-normal surfels from the
completion set** (`circe_coverage/coverage_node.py`'s `floor_normal_cos` param, `|n_z| > 0.8`). This is a
soft heuristic, not a hard guarantee: a shallow-angle floor surfel near a wall can still occasionally clear
the adequacy test from a horizontal view. **When judging a test run, "floor coverage stays small/near-zero"
is the expectation — don't treat any nonzero floor coverage as a bug**, and don't treat "some floor patches
never clear" as a failure either; both are correct given a fixed-height horizontal camera.

**ToF consequence (§4, kept as-is):** the sensor is a **single forward-facing ray** (URDF: `gpu_lidar` with
`samples=1`, zero angular spread, mounted on the front centerline pointing along +x) — not a scanning arc.
It sees only what's directly ahead and cannot detect anything to the side, behind, or during the 8-shot
ring's in-place rotation. `driver_node.py` takes `min()` of the LaserScan ranges (which, with one sample,
is just that one range) and force-zeros forward velocity below `tof_stop` — this is a literal last-line
safety backstop, not collision avoidance; it only ever protects the robot from driving straight into
something already in front of it.

**Deviations from project.md:** cut = YOLO26n inference (§8), mid-ring escalation (§7.2), object tracking,
defect→map pin/logging (Open #3), WiFi-cut demo (§9), tilt servo (Open #6). Kept = everything else.

---

## 2. Deployment topology (matches the real robot, unchanged sim→real)

```
┌───────────────────────────────┐  POST /frames (JPEG keyframes)  ┌────────────────────────────────────┐
│ SLAM laptop (Windows, WSL,GPU)│ ───────────────────────────────▶│ SIM laptop (Ubuntu + RTX)          │
│ = off-board VGGT-SLAM 2.0     │                                  │ = Gazebo depot world + rover BRAIN │
│   slam_host/slam_server.py    │ ◀─────────────────────────────── │   circe_* ROS 2 nodes + circe_viz  │
│   + viser raw-map viewer :8080│    GET /map (poses+cloud, rel)   │   circe_vggt_client = HTTP client  │
│ env: WSL `vggt` (py3.11)      │                                  │ env: system py3.12 (ROS 2 Jazzy)   │
└───────────────────────────────┘                                  └────────────────────────────────────┘
```

The rover only ever pushes frames and pulls poses+map. VGGT-SLAM ingests frames through its `Camera`
backend, so **it never knows whether frames came from Gazebo or a real RPi camera** — same server, sim or
real. On the real robot the SLAM box is native Linux on the LAN (no WSL networking caveat).

---

## 3. Evidence-backed component choices (don't re-litigate)

| Block | Choice | Evidence |
|---|---|---|
| Feed-forward RGB SLAM | **VGGT-SLAM 2.0** | arXiv [2601.19887](https://arxiv.org/abs/2601.19887); RSS 2026 [paper 51](https://roboticsconference.org/program/papers/51/); [MIT-SPARK/VGGT-SLAM](https://github.com/MIT-SPARK/VGGT-SLAM). Monocular → **relative scale** (§5 self-cal required). Real-time code released (`main_realtime.py`, pluggable `Camera` backends). |
| Sim stack | **Gazebo Harmonic + ROS 2 Jazzy** + `ros_gz` | Official Jazzy pairing; matches repo's `gazebo/`. |
| Indoor world | **[SIM-BOX: your choice]** — TurtleBot4 `depot.sdf` is one evidence-backed candidate | [turtlebot4_simulator](https://github.com/turtlebot/turtlebot4_simulator) has occlusion pockets good for frontier testing. Not mandated — pick whatever actually runs on your install; see §8 for selection criteria. Other candidates: `warehouse.sdf`, `maze.sdf`, any Gazebo Harmonic-compatible indoor SDF. |
| Fog voxel occupancy | **OctoMap**-style or in-node grid | Standard, §7A.1. |
| Coverage-aware NBV (two-layer) | **Custom** per §7A | Existing planners ([ethz-asl/nbvplanner](https://github.com/ethz-asl/nbvplanner)) are ROS 1, no two-layer surfel formulation. |
| Map/decision viewer | **Gradio**, reusing the repo's `gradio_demo.py::_export_glb` | frustums, OpenCV→glTF axis flip, voxel/max_points. |
| Motion / nav | **Custom skid-steer driver** (not Nav2) | Faithful to §2/§5/§6 (encoder-free, relative-scale, ToF backstop). |

---

## 4. SLAM laptop (this half is DONE + validated) — `slam_host/`

**Verified working on the authoring laptop** (WSL `vggt` env, Python 3.11.15, torch 2.3.1+cu121, RTX 4050):

- **VGGT-SLAM runs:** `main.py` on a 50-frame `office_loop` subset → 1 submap, real trajectory written,
  VGGT ≈0.33 s/frame, SALAD loop-closure active.
- **The server runs:** `slam_server.py` smoke test — POSTed 48 `office_loop` frames, got back
  `num_submaps=1`, a **5873-point cloud** + a 4×4 pose over HTTP. Contract proven end-to-end.

### Files (authored here)
| File | Purpose |
|------|---------|
| `network_camera.py` | `NetworkCamera(vggt_slam.cameras.Camera)` — frames arrive over HTTP instead of RealSense |
| `slam_server.py` | Loads VGGT+Solver (like `main_realtime.py`), runs the submap loop, serves the API below |
| `requirements-lock.txt` | Exact verified deps frozen from the working env |
| `environment.yml` | Conda recreate spec |
| `README.md` | Deps, run steps, **WSL2 networking** (portproxy / mirrored) |

**No edits to vendored VGGT-SLAM source** — imports the installed `vggt_slam`, so `git pull`s stay clean.

### Verified deps + checkpoints (the two gotchas)
- `vggt_slam@35327ac` (real-time code), `vggt(VGGT_SPARK)@6e6e161`, `salad@33ca9c0`,
  `gtsam-develop==4.3a1.dev202606262021`, torch `2.3.1+cu121`. `perception_models`+`sam3` **not needed**.
- **Two checkpoints must be present** at `$TORCH_HOME/hub/checkpoints/` (both were missing; fetched).
  **Check `echo $TORCH_HOME` first** — it's overridden on this machine (`~/.bashrc`:
  `TORCH_HOME=/mnt/d/others_github/model_cache`), so the real location is
  `/mnt/d/others_github/model_cache/hub/checkpoints/`, not torch's default `~/.cache/torch/hub/...`
  (confirmed live: a run's log showed `Loaded model from /mnt/d/others_github/model_cache/hub/checkpoints/
  dino_salad.ckpt`). See `slam_host/README.md` for the full explanation.
  - `model.pt` — VGGT-1B (~4.7 GB) from `https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt`
  - `dino_salad.ckpt` — SALAD loop-closure (~336 MB) from
    `https://github.com/serizba/salad/releases/download/v1.0.0/dino_salad.ckpt`
  - (DINOv2 backbone auto-downloads on first run.)
- **This is a big external dependency pinned by commit, not vendored** — a future `git pull` on
  `D:\others_github\VGGT-SLAM` can move past the verified commit above. Before trusting it again, follow
  `slam_host/README.md`'s **"Re-verifying after an upstream VGGT-SLAM pull"** section (checks the
  sub-dependency pins, re-runs the smoke test, re-freezes `requirements-lock.txt`, and requires updating
  the commit hash consistently in that file, `requirements-lock.txt`, and here).

### Run
```bash
conda activate vggt
cd .../circe_v3/simulation/slam_host
python slam_server.py --port 8000 --submap_size 8 --vis_map   # viser raw map at :8080
```
See `slam_host/README.md` for the office_loop sanity replay and the WSL2 port-forwarding.

### HTTP API (the wire contract — real-robot-plausible, unchanged sim→real)
| Method / path | Body / query | Returns |
|---|---|---|
| `POST /session` | `{intrinsics,width,height,submap_size?}` | config ack (relative scale) |
| `POST /frames` | multipart JPEG file(s) | `{received, queued}` |
| `GET /pose/latest` | — | latest `T_cam_world` 4×4 (**relative scale**) |
| `GET /map` | `after_submap, known_loops, voxel, max_points` | `{full_refresh, num_submaps, num_loops, submaps:[{submap_id,poses}], cloud:{n,xyz_f32_b64,rgb_u8_b64}}` |
| `GET /status` | — | worker + camera counters |

**Loop closure:** re-optimises *all* poses → when the loop count grows, `/map` returns `full_refresh:true`
and the whole map; the client rebuilds and re-runs motion self-cal (§5). All output is **relative scale**.

---

## 5. Sim laptop (YOUR half) — `ros2_ws/src/`

Runs on Ubuntu + ROS 2 Jazzy + Gazebo Harmonic. **Every package below is already authored as working,
syntax-clean code** (`python3 -m compileall` passes clean) — none of this is a spec for you to implement
from scratch. `circe_vggt_client` is additionally **validated** against the SLAM server's live HTTP contract
(§4); the rest are authored but **not yet built/run** (no ROS 2/Gazebo on the authoring machine) — your job
is `colcon build`, run, and tune/fix, not write. All Python, `ament_python`, per-node **JSONL logging** (a
run must be reconstructable), loops log `[i/N]`.

```
circe_vggt_client/   # AUTHORED. HTTP client: /rover/camera/image → POST /frames ; poll GET /map
                     #   → republish /vggt/pose (Odometry) + /vggt/cloud (PointCloud2), relative scale.
                     #   Pure client — no torch/gtsam. Point slam_url at the SLAM laptop (WSL2 caveat).
circe_rover_description/  # URDF/xacro: low chassis, HORIZONTAL cam at H_CAM (RPi v2 intrinsics 62.2°×48.8°),
                         #   IMU (gyro+accel), forward Range (ToF). Tracks ≈ skid-steer DiffDrive plugin.
circe_sim_gazebo/    # [SIM-BOX: pick the world, see §3/§8] indoor world + spawn + ros_gz bridges:
                     #   /rover/camera/image, /rover/imu, /rover/tof, /rover/cmd_vel.
circe_localization/  # §5 motion self-cal: known-move VGGT pose deltas → /circe/scale (map-units per cmd-sec, EMA).
                     # §6 two-rate: IMU dead-reckon ~100Hz, hard-reset to /vggt/pose on arrival → /circe/pose_fused.
circe_mapping/       # /vggt/cloud → (a) fog voxel grid {unknown/free/occupied} + frontier → /circe/frontiers;
                     #   (b) surfels {p, PCA normal n, covered=false, q_best=0} → /circe/surfels.   [§7A.1-2]
circe_coverage/      # geometric adequacy(s,V) [§7A.3, NO detector]: for each captured view (K,R,c) frustum-cull
                     #   + 4 gates (in-FOV / z-buffer occlusion / footprint≥F_MIN / incidence≥COS_MIN);
                     #   set covered, q_best; publish /circe/detection_gaps (¬covered clusters). Exclude +z-normal.
circe_explore/       # §7A.5 plan_viewpoint (ideal standoff C+d*·n̄ → project to {z=H_CAM, xy traversable, yaw→C,
                     #   snap, re-verify}) ; §7A.6 argmax(gain/cost). IMPORTANT: gain is NOT the gap-cluster
                     #   size — it is computed by actually re-running the §7A.3 adequacy test (project + FOV +
                     #   footprint + incidence) against every known surfel from the CANDIDATE pose before that
                     #   candidate is even considered; candidates with gain=0 are discarded outright. Do not
                     #   replace this with a cheaper heuristic — it changes termination behavior.
                     #   cost=path+W_TURN·|Δyaw|; else nearest frontier; else DONE → /circe/goal_station.
circe_driver/        # rotate-to-heading then drive-forward (cmd-time × /circe/scale, closed on /circe/pose_fused);
                     #   at station: FULL 8-SHOT RING (rotate 45°, settle, capture ×8; never shoot moving) → views to
                     #   circe_coverage. ToF hard-interrupt: /rover/tof < thresh → neutral cmd_vel, overrides all.
circe_viz/           # Gradio real-time viewer (the asked-for app) — see §6.
circe_bringup/       # one launch (Gazebo + client + all nodes) + params yaml (H_CAM, voxel, F_MIN, COS_MIN,
                     #   d*/d_h, R_CLUSTER, scale-EMA gain, ToF thresh, W_TURN, ring step 45°).
```

### Topic graph
```
Gazebo depot → /rover/camera/image, /rover/imu, /rover/tof ; /rover/cmd_vel → skid plugin
circe_vggt_client  ⇄  SLAM laptop HTTP  →  /vggt/pose (Odometry), /vggt/cloud (PointCloud2)   [both rel-scale]
circe_localization → /circe/pose_fused, /circe/scale
circe_mapping      → /circe/frontiers, /circe/surfels
circe_coverage     → /circe/detection_gaps  (marks surfels covered from ring views)
circe_explore      → /circe/goal_station
circe_driver       → /rover/cmd_vel (+ 8-shot ring views to coverage; ToF override)
circe_viz          → live 3D (subscribes cloud/pose/surfels/frontiers/gaps/goal/state)
```

---

## 6. `circe_viz` — the real-time viewer

Gradio app (rclpy node) on the sim laptop showing **the map the robot has AND its decisions**, live:
- Reuse `gradio_demo.py::_export_glb` patterns — camera-frustum wireframes, OpenCV→glTF axis flip, voxel
  downsample, `max_points` cap.
- `gr.Model3D` refreshed by a `gr.Timer` (~1 Hz, submap cadence). Overlays: point cloud (map); robot pose
  frustum + trajectory; **surfels colored covered=green / uncovered=red** (or `q_best` heat); **fog
  frontier** markers; **detection-gap** clusters; **next-goal station** marker.
- Side panel: coverage-% gauges (fog + detection), #frontier, #gaps, current `/circe/scale`, robot state
  (DRIVING / RINGING / DONE), log tail. Layer toggles + perf sliders.
- **Honest limitation:** `gr.Model3D` resets the camera on each reload → keep the refresh modest. VGGT-SLAM's
  **viser on the SLAM laptop** already gives a continuous raw-map stream; `circe_viz` is the circe
  **superset** (decisions + coverage + robot), not a viser replacement.
- Launch: `python -m circe_viz.app --port 7860` or via `circe_bringup`; SSH-tunnel like `controller/`.

---

## 7. Verification

**SLAM laptop — DONE (see §4):** VGGT-SLAM runs on office_loop; `slam_server` smoke test returns a
5873-point map + pose over HTTP.

**Sim laptop — staged (no pytest; RViz + rosbag):**
1. **Model** — `display.launch`: chassis + horizontal cam + IMU + ToF in RViz; teleop; four `/rover/*` topics live.
2. **Client ⇄ SLAM** — start `slam_server` on the SLAM laptop (reachable via WSL2 mirrored/portproxy);
   `circe_vggt_client` streams frames and republishes `/vggt/pose` + `/vggt/cloud` as the rover moves.
3. **Self-cal / mapping / coverage** — `/circe/scale` converges; fog + surfels grow in RViz; after a ring,
   in-view surfels flip covered with sensible `q_best`; `/circe/detection_gaps` shrinks; floor never clears.
4. **Full loop** — `circe_bringup`; rover plans stations over gaps+frontier, drives, rings, closes both
   layers to DONE; unmapped obstacle → ToF reflex halts it.
5. **circe_viz** — live viewer shows map + coverage + robot pose/goal updating in real time.
6. Record a rosbag of a full run.

### Known issue from the first live joint test (2026-08-15) — needs your input

First real cross-machine run worked at the transport level (`POST /frames`/`GET /map` all `200 OK`,
1665+ frames received, 0 dropped) but **produced zero submaps**. `slam_server.py`'s new per-25-frame
progress log (`slam_server_logs/slam_server_*.log` — see `slam_host/README.md` "Logging") showed the
cause precisely: `keyframes_pending=1/9`, **stuck at exactly 1 for ~1950 straight frames**. The keyframe
gate (`compute_disparity(img, min_disparity=50.0, ...)` in `slam_server.py`'s `slam_worker()`) accepted
only the very first frame and rejected every frame after it as "not different enough" — so nothing ever
reached the `submap_size(8)+overlapping_window_size(1)=9` threshold needed to trigger a submap.

**Two possible causes — only checkable from the sim side:**
1. **The rover genuinely wasn't moving / the camera view wasn't changing** during that run — in which
   case 0 disparity is *correct*, and there's no server-side bug: the fix is making sure the rover is
   actually driving (or the 8-shot ring is actually rotating) while frames are being streamed.
2. **`min_disparity=50.0` (the server's default) is too high** for whatever motion the Gazebo camera
   actually produces — a real tuning issue, fixed by lowering `--min_disparity` on the SLAM laptop's
   launch command.

**Action:** confirm whether the rover was actually moving during that test. If yes and this persists,
report back with that confirmation so `--min_disparity` can be tuned down from here.

---

## 8. Settled + delegated

**Settled:** split topology; **8-shot ring**; **no subtree** (VGGT-SLAM stays at its own repo, `pip -e`
installed); SLAM env = WSL `vggt` (py3.11 — *no* 3.12/cp312/RoboStack needed, the split removed that);
transport = FastAPI/HTTP with the §4 contract; loop-closure `full_refresh`; **SLAM-laptop networking is
done** — WSL mirrored networking + two scoped Hyper-V/Windows Firewall rules for TCP/8000, verified
(`slam_host/README.md`'s "Networking" section has the exact commands + current LAN IP). Point
`slam_url` at that IP; only real cross-machine reachability is unverified until the sim laptop connects.

**[SIM-BOX: decide + document]:**
- **Gazebo world.** No world is hardcoded (`circe_sim_gazebo/launch/sim.launch.py` takes it as a required
  `world` arg). Pick one and document it here + in that file's docstring. Criteria: (a) enough occluded /
  non-convex structure that fog frontier + detection-gap clustering aren't trivial to close, (b) small
  enough that a full coverage run finishes in a reasonable wall-clock time for iteration, (c) actually
  installs cleanly on your Gazebo Harmonic setup (no missing-mesh/model-path fights). `depot.sdf` (§3) is
  a reasonable first try, not a requirement — `warehouse.sdf`/`maze.sdf` or anything else that meets the
  criteria is equally acceptable.
- OctoMap vs in-node fog grid; seed `F_MIN`/`COS_INCIDENCE_MIN` defaults
  (`cos 60°`, an `F_MIN` guess — no YOLO recall curve to calibrate against yet); frame transport tuning (JPEG
  quality + `/map` poll rate); WSL2 reachability of the SLAM laptop (mirrored networking or `netsh portproxy`).
