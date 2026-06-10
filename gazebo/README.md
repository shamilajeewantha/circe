# PX4 + Gazebo + ROS 2 Simulation Stack — Launch Guide

## Reference Guides (official sources)
- PX4 ROS 2 User Guide: https://docs.px4.io/main/en/ros2/user_guide.html
- PX4 ROS 2 Offboard Control: https://docs.px4.io/main/en/ros2/offboard_control.html
- Official offboard Python example: https://github.com/PX4/px4_ros_com/blob/main/src/examples/offboard_py/offboard_control.py
- Gazebo SITL: https://docs.px4.io/main/en/sim_gazebo_gz/index.html
- Simulate failsafes / battery: https://docs.px4.io/main/en/simulation/failsafes.html

---

## Scripts in this folder

| File | Purpose |
|------|---------|
| `launch_default.sh` | Launch full stack with the flat default world |
| `launch_baylands.sh` | Launch full stack with the Baylands terrain world |
| `takeoff_hover.py` | Arms drone and hovers at 2 m via ROS 2 offboard control |
| `gcs_heartbeat.py` | Sets battery params + sends MAVLink GCS heartbeats so PX4 allows arming |
| `drone_detector.py` | Runs YOLOv8 on rover camera feed, publishes drone pixel position to `/drone_pixel` |

See also: [`../controller/`](../controller/) — web-based drone + rover controller with embedded camera feed.

`launch_default.sh` and `launch_baylands.sh` are **identical** — only the `GZ_WORLD` variable at the top differs.

---

## What This Does

Full perception pipeline:

```
PX4 drone (hovering at 10 m)
        ↓  Gazebo camera
Explorer R2 rover (ground robot, static)
        ↓  ros_gz_bridge
ROS 2 topic: /rover_camera/image
        ↓  YOLOv8
/drone_pixel  (x, y pixels + confidence)
```

### Rover model
The rover is the **Explorer R2** from the DARPA SubT Challenge, loaded from Gazebo Fuel:
- URI: `https://fuel.gazebosim.org/1.0/OpenRobotics/models/explorer_r2_sensor_config_2`
- Local cache: `~/.gz/fuel/fuel.gazebosim.org/openrobotics/models/explorer_r2_sensor_config_2/`
- Sensors: 4 RGBD cameras (front, left, back, right), LiDAR, IMU
- Only the **front** camera (`rs_front`) is bridged to ROS 2

### Two worlds

| World | SDF file | Rover placement |
|-------|----------|-----------------|
| `default` | `~/PX4-Autopilot/Tools/simulation/gz/worlds/default.sdf` | Static, `z=0.4` (flat ground plane at z=0) |
| `baylands` | `~/PX4-Autopilot/Tools/simulation/gz/worlds/baylands.sdf` | Non-static — drops from z=10 onto terrain surface |

In Baylands the rover is non-static because the terrain is a 3D mesh (`.DAE` files) with no flat ground plane. Spawning at z=10 and letting physics drop it is the only reliable way to land it on the actual surface.

---

## Prerequisites — verify these first

```bash
# PX4 is built
ls ~/PX4-Autopilot/build/px4_sitl_default/bin/px4

# MicroXRCEAgent is installed
which MicroXRCEAgent

# ROS 2 Jazzy + px4_msgs workspace built
ls ~/ws_px4/install/

# Explorer R2 is in both world SDFs
grep -c "explorer_r2" ~/PX4-Autopilot/Tools/simulation/gz/worlds/default.sdf    # must print 1
grep -c "explorer_r2" ~/PX4-Autopilot/Tools/simulation/gz/worlds/baylands.sdf   # must print 1
```

---

## Clean Start — run these if anything was previously running

```bash
pkill -f "gz sim" 2>/dev/null
pkill -f MicroXRCEAgent 2>/dev/null
pkill -f "bin/px4" 2>/dev/null
pkill -f "takeoff_hover\|drone_detector\|gcs_heartbeat\|parameter_bridge" 2>/dev/null
sleep 2
```

---

## Quick Launch — single command

```bash
bash ~/github_desktop/circe/gazebo/launch_default.sh    # flat default world
bash ~/github_desktop/circe/gazebo/launch_baylands.sh   # Baylands terrain world
```

Starts everything in order, waits for each step, shuts everything down cleanly on Ctrl+C.
Logs: `/tmp/xrce.log`, `/tmp/px4.log`, `/tmp/gcs.log`, `/tmp/bridge.log`

---

## Manual Launch — 5 terminals (default world)

Use this if you need separate terminals to inspect each process. For Baylands, replace Terminal 2 as noted.

---

### Terminal 1 — MicroXRCE-DDS Agent
*(per official docs: https://docs.px4.io/main/en/ros2/user_guide.html)*

```bash
MicroXRCEAgent udp4 -p 8888
```

Wait for: `running... | port: 8888`

---

### Terminal 2 — PX4 SITL + Gazebo
*(per official docs: https://docs.px4.io/main/en/sim_gazebo_gz/index.html)*

```bash
# Default world
cd ~/PX4-Autopilot
make px4_sitl gz_x500

# Baylands world
cd ~/PX4-Autopilot
PX4_GZ_WORLD=baylands make px4_sitl gz_x500
```

- **Do NOT add `PX4_GZ_MODEL_NAME`** — causes "Accel/Gyro/Baro missing" sensor errors
- Spawns `x500_0` drone with full sensor bridges (IMU, barometer, GPS, compass)

Wait for:
```
INFO  [commander] Ready for takeoff!
```

---

### Terminal 3 — Battery fix + GCS heartbeat

Replaces QGroundControl for terminal-only setups.
Run **after "Ready for takeoff!" appears** in Terminal 2. Keep it running.

```bash
python3 ~/github_desktop/circe/gazebo/gcs_heartbeat.py
```

Expected output:
```
Connected to PX4 (system 1)
  Set COM_LOW_BAT_ACT = 0
  Set BAT_LOW_THR = 0.0
  ...
Sending GCS heartbeats — keep this running while flying...
```

---

### Terminal 4 — Rover camera bridge (Gazebo → ROS 2)

The inline `rover_cam` model is fixed-jointed to the rover front. It publishes a 1280×720 image.
The bridge remaps Gazebo's auto-generated topic to `/rover_camera/image`.

```bash
source /opt/ros/jazzy/setup.bash
source ~/ws_px4/install/local_setup.bash

# Default world
ros2 run ros_gz_bridge parameter_bridge \
  "/world/default/model/rover_cam/link/link/sensor/camera/image@sensor_msgs/msg/Image[gz.msgs.Image" \
  --ros-args \
  -r "/world/default/model/rover_cam/link/link/sensor/camera/image:=/rover_camera/image"

# Baylands world (only world name changes)
ros2 run ros_gz_bridge parameter_bridge \
  "/world/baylands/model/rover_cam/link/link/sensor/camera/image@sensor_msgs/msg/Image[gz.msgs.Image" \
  --ros-args \
  -r "/world/baylands/model/rover_cam/link/link/sensor/camera/image:=/rover_camera/image"
```

Verify:
```bash
ros2 topic hz /rover_camera/image
# Expected: ~30 Hz
```

---

### Terminal 5 — Offboard takeoff and hover
*(adapted from https://github.com/PX4/px4_ros_com/blob/main/src/examples/offboard_py/offboard_control.py)*

```bash
source /opt/ros/jazzy/setup.bash
source ~/ws_px4/install/local_setup.bash
python3 ~/github_desktop/circe/gazebo/takeoff_hover.py
```

Expected output:
```
Switching to offboard mode
Arm command sent
Hovering — current z=-10.00m (target -10.0m NED)
```

---

## Gazebo UI Controls

| Action | Control |
|--------|---------|
| Rotate view | Left-click + drag |
| Zoom | Scroll wheel |
| Pan | Shift + left-click + drag |
| Select model | Left-click on model |
| Orbit mode | Press `Escape` to exit any other mode, then left-drag |

Mouse clicks not working → you are on Wayland. The launch scripts fix this with `export QT_QPA_PLATFORM=xcb`.

---

## Verify

```bash
# Camera flowing at 30 Hz
ros2 topic hz /rover_camera/image

# FMU topics present
ros2 topic list | grep fmu

# Drone z position (should be ~-10 NED = 10 m altitude)
gz topic -e -n 1 -t /world/default/dynamic_pose/info | grep -A 6 x500

# View camera feed
source /opt/ros/jazzy/setup.bash
ros2 run rqt_image_view rqt_image_view
# Select /rover_camera/image
```

---

## Step 6 — YOLO drone detection

Runs while the rest of the stack is up. Subscribes to `/rover_camera/image`, detects the drone, publishes pixel coordinates.

```bash
# Install ultralytics (one-time)
pip3 install ultralytics --break-system-packages

# Run detector (in a new terminal, with ROS 2 sourced)
source /opt/ros/jazzy/setup.bash
source ~/ws_px4/install/local_setup.bash
python3 ~/github_desktop/circe/gazebo/drone_detector.py
```

Verify:
```bash
# Pixel position of drone (x, y = pixels; z = confidence)
ros2 topic echo /drone_pixel
# Expected: x: 320.x  y: 240.x  z: 0.8x

# Debug image with bounding box (saved every 30 frames)
eog /tmp/rover_cam_debug.jpg

# Live debug feed in rqt
ros2 run rqt_image_view rqt_image_view
# Select /rover_camera/debug
```

Detection logic:
1. **YOLOv8n** — primary detector (`yolov8n.pt`, auto-downloaded on first run)
2. **HSV bright-object fallback** — kicks in if YOLO misses the drone (detects bright/white objects)

---

## Web Controller

A separate FastAPI web controller lives in `../controller/`. It lets you move the drone and rover
from a browser page with embedded camera feed — no QGroundControl needed.

```bash
# Start the sim first, then in a second terminal:
bash ~/github_desktop/circe/controller/run.sh
# Open http://localhost:8080
# Over SSH: ssh -L 8080:localhost:8080 user@host  → http://localhost:8080
```

Controls:
- **Drone**: WASD = forward/left/back/right, R/F = up/down, Q/E = yaw. Step size dropdown.
- **Rover**: IJKL = drive, Space = brake toggle. Starts braked — press BRAKE button to release.
- **Camera**: Live MJPEG feed from `/rover_camera/image` embedded in the page.

The rover requires the DiffDrive plugin injected into both world SDFs (already done in `baylands.sdf` and `default.sdf`).

---

## Known Issues

| Error | Cause | Fix |
|-------|-------|-----|
| `Accel/Gyro/Baro missing` | Used `PX4_GZ_MODEL_NAME=x500_0` | Remove it — use plain `make px4_sitl gz_x500` |
| `param types mismatch` | INT32 param sent as FLOAT | Fixed in `gcs_heartbeat.py` |
| `No connection to GCS` | No QGroundControl running | Run `gcs_heartbeat.py` (Terminal 3) |
| `Battery unhealthy` | SITL battery drained instantly | `gcs_heartbeat.py` sets all battery thresholds to 0 |
| `Sensors frozen / no IMU data` | Gazebo in lockstep with dead PX4 | Run clean start commands, restart from Terminal 1 |
| Port 8888 in use | Old MicroXRCEAgent still alive | `kill $(fuser 8888/udp)` then restart Terminal 1 |
| Camera topic not found | Wrong world name in bridge | Topic is `/world/<world_name>/model/rover_cam/link/link/sensor/camera/image` — match the world |
| Rover buried in ground (default) | Wrong z pose in SDF include | Must be `z=0.4` — matches the model's canonical pose (wheel radius 0.31 m + joint offset) |
| Rover buried in ground (baylands) | Baylands has non-flat terrain mesh | Rover is non-static and drops from z=10 onto actual terrain surface |
| Camera image dark / black trees | Baylands ambient was blue-purple `0.8 0.5 1` | Fixed: ambient changed to neutral `0.8 0.8 0.8 1` in baylands.sdf |
| Camera image pixelated (320×240) | Fuel Camera model hardcodes 320×240, can't be overridden via `<include>` | Fixed: replaced with inline `<model>` at 1280×720 in both world SDFs |
