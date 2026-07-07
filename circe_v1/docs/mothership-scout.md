# Mothership-Scout — Heterogeneous Recon on a Single Arduino Uno Q

**One-line pitch:** A tank-tracked rover whose Arduino Uno Q runs the entire
ground-side stack of a UGV-drone disaster-recon team — deterministic
self-driving on the MCU, AI perception and companion-drone stabilization on
the MPU — while off-board VGGT-SLAM handles navigation.

Status: design stage, no code written yet. Milestone 1 (below) is the
starting point.

## Hardware inventory

| Component | Role | Notes |
|---|---|---|
| Arduino Uno Q 4GB | Rover brain (MCU + MPU) | Qualcomm Dragonwing QRB2210 (quad-core Cortex-A53, Debian Linux) + STM32U585 (Cortex-M33, Zephyr). WCBN3536A WiFi 5 dual-band + BT, single USB-C, no onboard Ethernet, no CSI connector. |
| Mini TP101 tank chassis | Locomotion | 2x 33GB-520 motors, no encoders — driving must be open-loop. |
| Motor driver | Needed, not yet chosen | TB6612FNG or L298N — pick before Milestone 1. |
| Pan/tilt servos | Structural scan head aiming | Points the RPi 2W's CSI camera (see below) for crack detection scans. |
| ToF distance sensor (forward) | Obstacle reflex | Read by MCU. |
| MPU-9250 9-axis IMU | Heading/impact sensing | I2C, read by MCU. |
| DM002HW drone | Companion aerial scout | Compute-less, WiFi flight controller, streams camera. Protocol already reverse-engineered — see [Reusable assets](#reusable-assets-from-dm002hw_controller_tests) below. |
| RTX 4050 host | Off-board heavy compute | Runs DA3 / VGGT-SLAM for navigation and pose. Reachable over local WiFi/LAN only — no internet/cellular required. |
| Raspberry Pi Zero 2W | Camera capture + cloud uplink | Rover-mounted. Carries the CSI camera (replaces the USB rover camera) and is the sole path to the RTX 4050 — see [Networking topology](#networking-topology) below. |
| RPi Camera Module v2 (CSI) | Rover's own vision input | Attached to the RPi 2W. Replaces the USB rover camera in the original hardware list — single camera now serves both crack detection and VGGT-SLAM mapping. |

## Architecture: the brain split

The Uno Q's two processors are treated as genuinely separate systems with
different real-time guarantees, bridged by a fast RPC link.

### MCU (STM32U585, Zephyr) — real-time, deterministic

- Track motor PWM — differential drive, open-loop (no encoders available).
- Pan/tilt servo PWM — camera scanning head.
- IMU read (I2C) — gyro for straight-line/turn control, accel for tip/impact
  detection, mag for coarse heading.
- Forward ToF — obstacle stop reflex ("too close → halt").
- Watchdog — if MPU commands go stale, stop motors unconditionally.

This side owns anything safety-critical or timing-critical. It never waits
on the MPU or the network to decide whether to stop.

### MPU (Qualcomm, Debian/Python) — AI + comms

- Crack detection on rover camera (structural damage assessment).
- Optical-flow hover corrections, streamed to the drone's flight controller
  over WiFi.
- Drone detection/tracking on the drone's incoming video.
- High-level motion logic — decides where the rover should go, sends motion
  commands down to the MCU.

The MPU decides; the MCU actuates. The MPU has no direct access to motors,
servos, or sensors — everything crosses the bridge.

### Bridge RPC (MPU ↔ MCU)

~8ms round-trip target. This is the seam between "AI decides" and "hardware
moves." Not yet designed in detail — needs a concrete transport (UART?
shared memory? Zephyr IPC?) and message schema before Milestone 3.

### Off-board (RTX 4050)

Runs VGGT-SLAM for navigation/pose and heavy DA3 perception. Confirmed
**not real-time** (VGGT-SLAM is inherently batch/offline) — so the link to
this host can tolerate normal WiFi/network latency and occasional
dropouts with no local fallback required. This significantly simplifies the
networking design compared to a hard real-time control loop.

## Networking topology

Resolved: the single-radio conflict on the Uno Q is avoided by splitting
the two WiFi jobs across two boards, both rover-mounted:

- **Uno Q's radio** joins the DM002HW drone's own WiFi AP
  (`192.168.169.1`, closed local network, no internet) — receives its
  video, sends control/hover corrections. This is the MPU's job per the
  [brain split](#mpu-qualcomm-debianpython--ai--comms) above.
- **RPi 2W's radio** joins the local WiFi/LAN and reaches the RTX 4050 —
  confirmed **local network only**, no internet/cellular needed, since the
  RTX 4050 is expected to be on the same WiFi/LAN during demos.
- The RPi 2W also carries the CSI camera (rover's own vision input) and
  ships **both** the rover's own frames **and** the drone's frames
  (relayed to it from the Uno Q) to the RTX 4050 for VGGT-SLAM/DA3.

This means the Uno Q's MPU receives the drone's video directly (own radio)
but must **hand drone frames off to the RPi 2W** for the cloud leg, since
the Uno Q itself has no path to the RTX 4050. Whatever local processing the
MPU does on drone frames (detection/tracking, optical-flow hover
corrections) still happens on-device before/alongside that hand-off — only
the cloud-bound copy needs to reach the RPi 2W.

### Still open: Uno Q ↔ RPi 2W physical link

Not yet decided. The two boards are on the same chassis, so this should be
a direct wired connection rather than another wireless hop (and the Uno
Q's one radio is already spoken for by the drone link). Whatever is chosen
needs enough throughput to move both the RPi 2W's own CSI frames and the
relayed drone frames without starving the ~8ms MPU↔MCU bridge:

- **USB (gadget-Ethernet)** — enough bandwidth for camera frames, uses the
  Uno Q's single USB-C port (check for contention with any other USB
  peripherals).
- **UART/serial** — simple, but likely too slow for image data if it's
  also carrying camera frames both directions; would only work if the
  RPi 2W's frames go straight to the RTX 4050 without transiting the Uno Q
  question in reverse.

Decide before Milestone 5 (drone stream + optical flow, since that's the
first milestone that needs frames crossing this link). Doesn't block
Milestones 1-4, which are all local to the rover.

## Reusable assets from `DM002HW_controller_tests/`

The DM002HW drone's WiFi protocol has already been fully reverse-engineered
and implemented earlier in this repo, and should be **ported/adapted, not
reimplemented**:

- `drone.py` — confirmed-working UDP control protocol (port 8800): arm/
  disarm, roll/pitch/throttle/yaw, idle-mode keepalive, all the timing
  quirks (decoupled counters, connect-time state reset) already debugged.
- `video_stream.py` — receives and reconstructs the drone's proprietary
  UDP JPEG-fragment video (port 1234), including the JPEG header synthesis
  (DHT tables, etc.) needed since the drone omits standard headers.
- `pcap_analyzer.py` / `PROTOCOL_NOTES.md` — full protocol documentation and
  analysis tooling, useful if the MPU-side integration surfaces new
  protocol edge cases.

For Mothership-Scout, the MPU's "drone detection/tracking" and "optical-flow
hover corrections" milestones consume `video_stream.py`'s decoded frames and
reuse `drone.py`'s control channel to send corrections back — this is
largely a porting/integration task (Python → same Python, different host),
since the Uno Q's own radio is the one joining the drone's AP (see
[Networking topology](#networking-topology) above). The cloud-bound copy of
those frames then needs to cross the still-undecided Uno Q ↔ RPi 2W link to
reach the RTX 4050.

## Demo storyline (what a judge sees)

1. Rover drives itself through an indoor course — MCU handling motors + IMU
   straight-line + ToF obstacle stops, all deterministic.
2. Rover reaches a wall; pan/tilt head scans it; MPU runs crack detection
   and flags damage.
3. Companion drone launches; MPU processes its WiFi video stream, computes
   optical-flow drift, and streams hover corrections back — a compute-less
   drone stabilized entirely by the rover's board.
4. Throughout: both brains working concurrently — the thing only this
   board (Uno Q) does on its own.

## Build order (milestones)

Each milestone is standalone and should be demoable on its own before
moving to the next.

1. **MCU motor + ToF + watchdog** — rover drives and stops safely.
   *(Start here. Pick the motor driver — TB6612FNG or L298N — first.)*
2. **MCU + IMU** — straight-line driving and controlled turns without
   encoders.
3. **Bridge RPC** — MPU sends a motion command, MCU executes it.
4. **MPU crack detection** — model runs on rover camera at a few FPS.
5. **MPU drone stream + optical flow** — hover corrections over WiFi.
   *(Blocked on resolving the [networking topology question](#open-question-networking-topology).)*
6. **Integration** — full concurrent demo.
