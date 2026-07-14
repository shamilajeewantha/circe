# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Hard rules (non-negotiable)

1. **Verify against official sources; attach evidence.** Do not present a claim, diagnosis, or
   instruction as fact from memory. Confirm it against a credible, official source (official docs,
   the project's own repo/issues, the vendor, a reproducible command on this machine) and **cite the
   evidence** — a URL, a command + its output, a file + line. This applies to root-cause claims and
   error diagnoses too: show the proof (e.g. the exact error, the registry value, the signature check),
   not an assertion. If you cannot verify it, do **not** state it as fact and do **not** dress it up as
   a "hypothesis" to make it sound legitimate — an unverified claim asserted as if true is a
   hallucination. Say plainly "I have not verified this / I don't know," then either go verify it before
   proceeding or stop and ask. Never let an unverified guess drive a decision or an instruction.

2. **Surface blockers and prerequisites UP FRONT — before implementing.** Before starting a plan,
   identify everything that must be true for the WHOLE plan to run end-to-end (env state, installs,
   permissions, security policies, restarts, credentials). If any prerequisite needs the user to do
   something, **ask them to run it first and confirm** — do not start implementing and then stop
   halfway to ask. Never bury a required user action mid-execution. Transparency over confidence:
   state what could break the plan and why, plainly, at the start.

3. **Never abandon a plan or task list midway.** Once there is an active plan/task list, keep executing
   every item through to completion — do not stop to "report progress," hand back, or wait around. The
   **only** acceptable reason to halt before all tasks are done is to **ask the user a genuine blocking
   question**: a specific, answerable question whose answer you actually need to proceed correctly (and
   whose consequences — e.g. hours of compute, an irreversible delete — justify pausing). "I'm not sure
   what to do next," reporting status, or laziness are NOT blockers. If you can keep going, keep going;
   if you are truly blocked, ask one precise question and resume the moment it is answered.

## Repository overview

This is **not a single application** — it's a collection of loosely related, independently-runnable
Python subprojects exploring WiFi/vision-controlled drones and rovers. There is no shared build system,
no root-level dependency manifest, and no test suite (verification throughout is ad-hoc: small scripts
run manually, not pytest). Each subproject below has its own run command and its own dependencies.

**Naming collision to watch for** (this has caused real confusion in past sessions — check the path,
not just the word "controller"):
- `controller/` (dir) — a FastAPI web app that teleoperates a **simulated** PX4/Gazebo drone + rover.
- `controller_simulation.py` (root file) — a standalone matplotlib script simulating an **adaptive
  IBVS control law** (image-based visual servoing with the image Jacobian learned via gradient
  descent). Not wired to any camera, drone, or the `controller/` app — pure control-theory math demo.
- `DM002HW_controller_tests/` — reverse-engineering scripts + a reference Gradio app for the **real**
  DM002HW WiFi toy drone.
- `circe_v1/optical_flow_control/` — the actively-developed Gradio controller for that same real
  DM002HW drone (evolved from `DM002HW_controller_tests/gradio_app.py`).

## Subprojects

### `circe_v1/optical_flow_control/` — DM002HW hover-hold Gradio app (active development)

The main app. Controls a real DM002HW WiFi toy drone and can run an optical-flow "Hold Position"
feature using its forward-facing video.

```bash
cd circe_v1/optical_flow_control
python main.py    # Gradio UI at http://localhost:7860
```

No `requirements.txt` in this subtree — needs `gradio`, `opencv-python`, `numpy` in the active env.

Architecture (read these together, in this order, to understand a change):
- `drone.py` — UDP control socket to the drone. `_loop` (daemon thread, 50 Hz) continuously
  re-transmits whatever `set_controls(roll, pitch, throttle, yaw)` last set — every axis is a byte
  0–255, neutral = 128, magnitude = distance from 128. On a send error it sets `_armed=False` and
  stops — nothing else on its own tears down video/hold, see `main.py`'s lost-link handling below.
- `video_stream.py` / `flow_sources.py` — receive/decode the drone's proprietary UDP JPEG video, or
  substitute a local webcam ("Demo" mode) for development without a real drone.
- `flow_stabilizer.py` — `FlowStabilizer.update(frame) -> FlowCorrection`. Dense optical flow
  (OpenCV DIS, ultrafast) between consecutive frames, median over a centered ROI, deadband + EMA
  smoothing, then a PD + slew-rate-limited controller producing `roll_delta`/`throttle_delta`. This
  is **velocity damping** (frame-to-frame), not position-hold against a fixed target — see the
  module's own docstring for why a forward-facing camera can't do better without IMU fusion (rotation
  and translation both look like flow). Pitch/yaw are never touched by hover — only roll (horizontal
  drift) and throttle (vertical drift).
- `flow_processor.py` — `FlowProcessor` owns the background loop that reads frames, calls the
  stabilizer, and actuates the drone (continuous every frame, or an older "pulsed" mode). Includes a
  stale-frame watchdog that forces neutral if frames stop arriving — treat this as a hard safety
  invariant, not a tunable.
- `flow_log.py` — one JSONL-per-session logger. Every tunable param, every frame's control internals,
  and every UI param change are logged — this project has a standing requirement that *nothing* about
  a hover session is unreconstructable from its log file.
- `main.py` — the Gradio UI: connect/disconnect, start/stop video, engage/disengage hold, tuning
  sliders. Single-source-of-truth state (`CONNECTED`/`LIVE`/`HOLDING`) recomputed on both click and
  poll tick. Has a full `atexit`/signal `shutdown()` that tears down every thread/socket/server —
  this exists because leaked instances previously held ports and log-file handles open on Windows.
- `applog.py` — app-wide logging setup; `close_logging()` releases the Windows file lock on exit.

Runtime output (`flow_debug/`, `video_debug/`, `app_logs/`) is gitignored — never delete it manually to
"fix" a stuck folder; if deletion fails, a stray process still has the file open (kill it, don't force-delete).

### `DM002HW_controller_tests/` — protocol reverse-engineering

The original RE work and reference implementation for the DM002HW's UDP protocol (port 8800 control,
port 1234 video). `PROTOCOL_NOTES.md` and the "How the protocol was reverse-engineered" section of the
root `README.md` document the full process (packet capture via laptop-as-relay, pcapng parsing from
scratch, byte-offset analysis). `gradio_app.py` here is the **visual/layout reference** that
`circe_v1/optical_flow_control/main.py` was designed to match (plain default Gradio theme, two-column
layout, colored flight buttons, no custom CSS beyond a few color overrides) — check here first before
changing that app's look. `*.pcapng` files and derived `capture_analysis.txt`/`drone_packets.txt` are
gitignored (raw traffic, potentially sensitive).

### `controller/` + `gazebo/` + `drone_detection/` — PX4/Gazebo SITL simulation stack (Linux/ROS 2 only)

An entirely separate project: a **simulated** drone + rover in Gazebo, controlled over PX4 offboard
control, with YOLO-based detection of the drone from the rover's camera. Requires ROS 2 Jazzy, a built
PX4-Autopilot, and Gazebo — not runnable as-is on a plain Windows/Python setup. See `README.md` (root)
and `gazebo/README.md` for full launch order; summary:

```bash
bash gazebo/launch_baylands.sh   # PX4 SITL + Gazebo + GCS heartbeat + camera bridge
bash controller/run.sh           # FastAPI web controller on :8080
# open http://localhost:8080
```

- `gazebo/takeoff_hover.py` — offboard control, hovers at a **fixed hardcoded NED setpoint**.
- `gazebo/drone_detector.py` — YOLO (+ HSV fallback) computes the detected drone's pixel centroid and
  publishes it to `/drone_pixel` — **nothing in this repo subscribes to that topic**; it's a display/
  debug publisher only, not part of any closed control loop. (Do not confuse with the working closed
  loop below — this file's output is a dead end.)
- `controller/server.py` — **`_servo_loop()` is a real, working closed-loop IBVS controller**: takes
  the highest-confidence YOLO bbox from `camera_node`, computes pixel error against a fixed image-center
  target, and drives `drone_node.apply_increment(...)` to null it — same adaptive-Jacobian-via-gradient-
  descent control law as `controller_simulation.py` (there: `J_bar`; here: `theta_hat`, coupled across
  both axes into one scalar instead of two independent per-axis estimates). One-shot-then-settle timing
  (`LOOP_WAIT=1.25s` between steps). Exposed via `POST /servo/start`, `POST /servo/stop`,
  `GET /servo/status` — **but `static/index.html` has no UI for these endpoints yet**, so it can
  currently only be triggered by calling the API directly, not from the web page.
- `controller/drone_node.py` / `rover_node.py` — manual open-loop teleop (discrete position/velocity
  increments from keyboard input via a websocket) is the normal path; `drone_node.apply_increment` is
  also what `_servo_loop` calls, so the same node serves both manual and closed-loop-servo commands.
- `controller/camera_node.py` + `detect_worker.py`/`detect_client.py` — MJPEG-over-websocket video to
  the browser, with YOLO bounding boxes (run locally or offloaded to a second GPU laptop over
  `/ws/detect`) drawn as an overlay only — detection does not feed back into control here either.
- `drone_detection/best.pt` — the YOLO model weights (Git LFS), `drone_detect.py` is a standalone
  batch-inference test script.

### `controller_simulation.py` (root) — adaptive IBVS control-law simulation

Standalone, no dependencies on the rest of the repo. Simulates a 1-D image-based visual servo: drive a
tracked pixel `p` to a target `p_star` via `u = alpha * (1/J_bar) * e`, where `J_bar` (the estimated
image Jacobian, pixels-per-command) is itself updated online by gradient descent on prediction error
each step, since the true depth (and therefore true Jacobian) is unknown. This is the reference control
law for any future "drive a tracked point to an arbitrary target" feature — note it has no camera/
tracking component; a real implementation needs to add point tracking (e.g. template matching) that
this script doesn't address.

### `circe_v1/docs/mothership-scout.md` — design doc, no code yet

Design for a future rover (Arduino Uno Q) that would port `DM002HW_controller_tests/drone.py` and
`video_stream.py` to act as a companion-drone stabilizer alongside rover autonomy and off-board
VGGT-SLAM. Explicitly "design stage, no code written yet" — treat as context/background, not as a
description of current code.

## Verification (no test suite exists)

There is no pytest/unit-test setup anywhere in this repo. Changes are verified with small one-off
scripts: write a script that imports the target module, feed it synthetic data (e.g. synthetic frames
for `flow_stabilizer.py`), assert expected behavior, and write results to a UTF-8 text file rather than
printing directly — `print()` of certain characters can crash Windows' cp1252 console. Read the output
file back afterward to confirm.
