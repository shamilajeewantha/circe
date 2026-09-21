# Circe: an autonomous floor-level inspection rover with on-board defect detection

## Overview

Circe is a low-profile tracked rover that autonomously inspects an industrial floor for early-failure and safety defects. It navigates without GPS, LiDAR, or wheel encoders, using a single camera and a feed-forward visual SLAM backbone running off-board. It explores on its own, stops at selected stations, photographs its surroundings, and runs defect detection on-board the Arduino UNO Q. Confirmed defects are logged against their position on the reconstructed map.

The target application is repeatable inspection of areas that are awkward for a person to check often: under conveyors, between machine bases, along tight equipment rows. Running the same route frequently is what makes slow degradation visible, and that is the predictive-maintenance payoff.

The chassis is deliberately low, so tall machine bodies, wall panels, and high-mounted gauges are outside the camera's view by design. Inspection targets are therefore floor-level and base-level: floor and plinth condition, machine bases, and low fixtures. This single constraint defines the class list, the capture pattern, and the coverage model described below.

## How the system works

### The operating loop

1. The rover streams camera frames to an off-board SLAM host, which returns camera poses and a growing dense map.
2. Two coverage layers are updated on that map: which space has been reconstructed, and which reconstructed surfaces have been photographed well enough to detect defects on.
3. The planner selects the next station as the reachable viewpoint with the best expected coverage gain per unit of travel cost.
4. The rover drives there using self-calibrated open-loop motion, bridged by on-board IMU dead-reckoning and corrected by SLAM on arrival.
5. It stops, settles, and captures a ring of images while detection runs continuously on the live video.
6. Any uncertain detection triggers an approach-and-confirm interrupt before the ring continues. Confirmed defects are logged against their map position.

The loop repeats until both coverage layers close.

### On-board versus off-board

Work is divided along the UNO Q's dual-brain architecture according to latency tolerance, not raw throughput:

- **Off-board (remote GPU):** heavy, intermittent, latency-tolerant geometry, specifically SLAM. A late map update is survivable.
- **On-board (UNO Q):** light, continuous, latency-critical perception, specifically detection and tracking driving motion and escalation decisions on every frame.

The on-board loop cannot be relocated to the remote GPU. Continuous HD video streaming is bandwidth-prohibitive, variable round-trip latency breaks tracking through dropped and late frames, and it would make the most fragile component in the system load-bearing for its most time-critical function.

This is demonstrable rather than asserted: cutting the Wi-Fi link mid-run stops map updates, while the rover keeps detecting, tracking, and approaching defects on-board.

## Mapping and localization

Mapping uses [VGGT-SLAM 2.0](https://arxiv.org/abs/2601.19887) (Maggio and Carlone, MIT SPARK), a feed-forward RGB SLAM method that produces a dense point cloud and per-frame camera poses, with incremental submap alignment and attention-based loop closure. It runs off-board on a remote GPU laptop: the rover streams frames and receives poses and map data back.

Mapping is incremental, so there is no separate "map first, inspect second" phase. Mapping and inspection happen in a single traversal.

### Working without metric scale

Feed-forward monocular SLAM reconstructs at **relative scale**, with no metric units. Combined with the absence of wheel encoders, the rover has no direct measurement of how far a given motor command moves it.

This is resolved by using the rover's own motion as a self-calibrating ruler. The rover issues a known open-loop drive command for a known duration, reads its SLAM pose immediately before and after, and takes the length of the translation between them. The result is a running conversion factor: map units per command-second.

The factor is re-estimated on every known move rather than once at startup, because relative scale can differ between submaps and shift on loop closure. A running exponential average absorbs that drift. Translation calibrates more reliably than rotation, since skid-steer turn slip makes turn rate noisy, so the two are estimated separately and rotation is weighted as less trustworthy.

Most of the system never needs metric units at all. Exploration, frontier logic, and obstacle rejection by relative height are all scale-free. The one decision that would genuinely want metres, judging step height for climb-versus-go-around, is removed from scope: any cell above floor height is treated as go-around, which is the correct behaviour for a low rover intended to pass under and around obstacles rather than over them.

### Two-rate localization

On-board, a six-axis IMU dead-reckons at roughly 100 Hz on the microcontroller side with no network dependency. Each fresh SLAM pose then resets the estimate to absolute truth and clears accumulated drift. Because the rover stops to inspect, pose accuracy is highest exactly when images are captured.

A forward time-of-flight sensor sits underneath both layers as a hard interrupt that overrides any motor command, independent of a possibly stale map.

## Coverage: mapped is not inspected

The planner tracks two separate coverage layers on the same incrementally growing map:

1. **Fog layer (mapping coverage).** An occupancy grid in which each cell is unknown, free, or occupied. The frontier is the boundary between free and unknown space. This answers whether a volume has been reconstructed at all, and drives exploration.
2. **Detection-quality layer (inspection coverage).** State carried per small surface patch on the reconstructed geometry. This answers whether a mapped surface has been imaged at detection-grade quality, and drives inspection.

The two do not coincide, which is the central point of the design. SLAM can reconstruct a machine base accurately from several metres away at a grazing angle: the geometry is complete, but the imagery is useless for detecting a hairline crack. The detection layer clears more slowly than the fog layer, and it is what gates run completion.

A surface patch counts as inspection-covered by a given view only if it passes four gates:

- It projects inside the image.
- It is not occluded by nearer geometry.
- Its projected footprint exceeds a minimum pixel size.
- It is viewed squarely enough rather than at a grazing angle.

All four tests are scale-free. The footprint gate evaluates focal length times patch size divided by depth, and since patch size and depth are both expressed in map units, the ratio yields a true pixel count that can be compared against a fixed threshold despite the map having no metric scale.

Station selection is then a greedy next-best-view step. Uncovered patches are clustered, an ideal standoff viewpoint is computed for each cluster, and that viewpoint is projected onto the poses the rover can actually reach: fixed camera height, traversable floor positions, and free heading. Candidates are ranked by expected gain per unit cost, where gain counts every patch the view would cover and cost includes a turn penalty, since skid-steer rotation is slow and slip-prone. Distances are relative but internally consistent, so the ratio holds without metric scale.

Revisits then require no special bookkeeping. A surface mapped early but not yet inspected simply remains an open gap, and is selected later when its gain-to-cost ratio wins.

### A constraint that falls out of the geometry

Vertical surfaces such as machine sides and base plates have horizontal normals, so a low camera at horizontal standoff views them head-on and they clear cleanly.

Floor surfaces have normals pointing up, and are seen at near-grazing incidence from every reachable position on the floor, so the incidence gate fails everywhere. With a purely forward-looking camera, floor patches are unsatisfiable by construction: no amount of driving would ever clear them. That result is what put a downward degree of freedom on the camera, and it was the coverage model that exposed the need rather than a failed test run.

## Inspection at each station

The rover captures a ring of **eight images per full rotation**. The step follows from the optics: a 62° horizontal field of view with roughly 25% overlap gives a 45° step, and 360° divided by 45° is 8.

The rover turns, stops, allows the chassis to settle, then captures. Images are never taken while rotating, because motion blur degrades small-defect detection and corrupts the localization pose. Once the coverage layer is populated, later stations capture only the sectors that still contain uncovered surface, with the full eight-image ring retained as the deterministic fallback.

Detection runs continuously on the live video stream rather than only on the eight stills, which enables frame-to-frame tracking and the escalation behaviour below.

### Escalation: approach and confirm

Escalation triggers when a detection is both **low-confidence and small in frame**, indicating a possible defect too distant to classify. Both conditions are required: a large low-confidence detection means distance is not the limiting factor, so approaching will not help, and a high-confidence detection needs no second look.

1. At ring shot *k*, the trigger fires.
2. The ring state is stored: shot index, plus the SLAM pose of the ring centre as the return anchor.
3. The rover turns to centre the object, drives forward in small steps while re-detecting, and captures close-up frames until the detection is confirmed or dismissed.
4. It returns to the stored ring-centre pose, closed-loop on SLAM pose, and re-orients to the shot-*k* heading. Return error is bounded because the approach is short.
5. The ring resumes at shot *k*+1.

The interrupt fires mid-ring rather than after it, because once the rover moves on, a small unlocalized low-confidence target cannot reliably be recovered. Repeat escalation on the same object is suppressed by a persistent tracker ID over the short timescale, backed by map-location suppression for objects lost and re-observed later. If the return pose cannot be re-established within tolerance, the ring is re-shot in full from the current position rather than stitched together from misaligned segments.

## Defect detection model

Detection uses [YOLO26n](https://docs.ultralytics.com/models/yolo26/) from Ultralytics, fine-tuned from COCO-pretrained weights. It was selected as the lightest current YOLO variant, at 2.4 M parameters with native NMS-free end-to-end inference, for real-time on-board execution on the UNO Q.

It detects oil and fluid leaks, rust and corrosion, and concrete cracks: floor-level defects that are visually distinct and tied to genuine early-failure signals. Training images are captured at rover-eye height and angle, so the model is trained on the same viewpoint it runs at.

## Results

The rover runs the full cycle end to end with no operator input. From a cold start on an unmapped floor it explores, builds the map as it moves, selects its own inspection stations, captures the ring at each one, interrupts itself to approach and confirm uncertain detections, and logs confirmed defects against their position on the map. The run ends when both coverage layers close.

The navigation and inspection-planning stack was developed and validated in Gazebo simulation first, with SLAM hosted on a separate machine so the simulated system matched the physical topology exactly, then brought onto the rover. That order was deliberate: iterating on coverage-planning behaviour against real skid-steer slip and a real eight-image ring would have cost days per defect found.

The coverage thresholds are calibrated against real detection performance rather than guessed. The minimum pixel footprint is set by placing a known defect at increasing distances and finding the point where detection recall collapses, which ties the planner's definition of an adequate view directly to what the detector can actually resolve.

Cutting the Wi-Fi link mid-run leaves the map stale while the rover carries on detecting, tracking, and approaching defects on-board. The autonomy claim is demonstrated rather than asserted.

**One deviation from the original proposal:** the companion drone was dropped. The unit available was a dirt-cheap toy-grade quadcopter that never flew stably enough to steer reliably, and an unstable aerial half would have contributed nothing but an additional failure mode. The build is ground-only.
