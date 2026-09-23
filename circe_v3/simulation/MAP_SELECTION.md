# Simulation world selection — mandatory checklist + record of every failure

**Read this before adopting or changing the Gazebo world.** Every world listed under
"Rejected" was adopted at some point, looked fine, and silently destroyed a whole
debugging session. The checklist exists because the failures were *not* visible in
logs, topic rates, or node health — only in the reconstructed map.

`circe_sim_gazebo/launch/sim.launch.py` takes the world as a required `world:` arg,
so switching is cheap. Choosing wrong is not.

---

## The checklist — ALL must pass before adopting a world

Run these against the world SDF **and every model it `<include>`s** (a world file that
is 26 lines of `<include>` tells you nothing; the geometry lives in the Fuel models).

### 1. Walls must be SOLID — check collision *geometry type*, not count
```bash
grep -c '<collision' model.sdf          # count alone is NOT the test
sed -n '/<collision/,/<\/collision>/p' model.sdf | head -12   # look at the GEOMETRY
```
| What you see | Verdict |
|---|---|
| `<mesh><uri>..._colision.stl</uri>` | **PASS** — one mesh can be an entire building |
| `<box><size>...` per wall | **PASS** — real wall bodies |
| `<plane><normal>0 0 1</normal><size>100 100</size>` **only** | **FAIL** — that is a bare FLOOR. The rover drives through every wall. |

> A previous version of this guidance said "collisions must be >> 1". **That is wrong**
> and would have rejected the warehouse, which is correct with a single mesh collision.
> The type matters, the count does not.

**Live confirmation** (never trust the SDF alone — drive at a wall and use GROUND TRUTH):
```bash
gz model -m circe_rover -p          # ground truth pose. NEVER use /rover/odom (see §2)
```

### 2. Verify motion with `gz model -p`, never `/rover/odom`
`gz-sim-diff-drive-system` integrates odometry from **commanded wheel rotation** and
has no notion of slip. A rover pinned against a wall with spinning wheels reports
phantom motion forever. Measured, same instant:

| source | x |
|---|---|
| `gz model -m circe_rover -p` (truth) | **2.82 m** (pinned on a wall) |
| `/rover/odom` | **44.61 m** |

Several conclusions in this project were wrong because of this, including "the rover is
outside the building".

### 3. The world must be VISUALLY FEATURE-RICH — this is the one that hurt most
VGGT-SLAM is a monocular feed-forward reconstructor. A geometrically perfect world with
flat, dark, repeating surfaces produces a **garbage map while every health metric looks
green**. Check before adopting:
```bash
grep -oE '<diffuse>[^<]+</diffuse>' world.sdf | sort -u | wc -l   # colour variety
find <model dir> -iname '*.png' -o -iname '*.jpg' | head           # real textures?
```
And measure what the camera actually sees:
```python
# mean should NOT be ~dark; corners are what optical flow tracks
cv2.goodFeaturesToTrack(gray, 2000, 0.01, 5)   # want hundreds, not tens
```

### 4. ALWAYS look at the reconstructed map against ground truth
Scalar counters lie. Use `scratchpad/map_vs_truth.py` (renders the SLAM cloud's PCA
floor-plan beside the true floorplan parsed from the SDF). On the maze it reported
**planarity 0.044 and plan aspect 1.09 vs a true 1.00 — both "good"** — while the actual
render was an amorphous fan of points with radial streaking and no walls at all.
**A blob is also planar and also square.** Look at the picture.

### 5. Sensor plugins are usually DISABLED in stock worlds
Every turtlebot4 world ships `Sensors` commented out and has no `Imu` system at all.
Sensors then advertise their gz topics and **never publish** — bridge looks healthy,
ROS topics exist, all silent. Every fork in `circe_sim_gazebo/worlds/` must carry:
```xml
<plugin name="gz::sim::systems::Sensors" filename="gz-sim-sensors-system">
    <render_engine>ogre2</render_engine>
</plugin>
<plugin name="gz::sim::systems::Imu" filename="gz-sim-imu-system" />
```

---

## Record of rejected worlds

### `depot.sdf` — REJECTED: no wall collisions
Fuel `Depot` model: **16 `<visual>`, 1 `<collision>`, and that collision is a
100×100 m ground plane.** No wall or crate collision geometry at all.

Cost: the rover ghosted through the building and wandered to the edge of the plane.
Presented as: unbounded map growth, ~150 sparse surfels, `covered` stuck at 0 (8-shot
rings firing in open air), and bizarre ToF behaviour — `gpu_lidar` raycasts the
**render** scene, so it *saw* walls that physics let the rover pass through.

### `maze.sdf` — REJECTED: geometrically sound, visually degenerate
41 visuals / 41 matching inline box collisions, zero Fuel downloads. Collision-wise it
is the *best* of the three. It still had to be abandoned:

- **42 materials, only 3 distinct colours**, walls `diffuse 0 0.01 0.05` (near-black).
  Camera mean pixel **25.5/255**.
- VGGT accepted **1 keyframe in 4.3 hours out of 278,000 frames** — a blank dark wall
  gives optical flow nothing to track.
- **19 "loop closures" in a 20×20 m maze** — near-certainly false matches, because every
  corridor is identical. A false loop re-optimises *all* poses and scrambles the map.
- Map span thrashed `0.6 → 3.9 → 9.5 → 2.3` rel-units (diverging).
- Final reconstruction: an amorphous fan of points, no walls, no corridors.

Adding per-wall hues + a generated albedo texture was attempted
(`worlds/textures/feature_wall.png`, 42 materials → 43 distinct colours). Kept in tree
for reference, but a synthetic texture pasted on toy geometry is not worth pursuing when
a genuinely realistic world exists.

**Lesson: "has collisions" and "is a good SLAM world" are independent properties.
Check both, separately.**

---

## Current choice: `warehouse_sensors.sdf` (fork of turtlebot4 `warehouse.sdf`)

Why it passes where the others failed:

| Criterion | Evidence |
|---|---|
| Solid walls | Fuel `Warehouse` collision = **`meshes/warehouse_colision.stl` (156K mesh)** — the whole building, not a plane |
| Real textures | `Rough_Square_Concrete_Block.jpg`, `Terrazzo005_1K_Color.jpg`, `Asphalt010_1K_Color.jpg`, `Tape001_1K_Color.png` — photographic, not flat colour |
| Visual variety | 27 included models: 8 `shelf`, 5 `shelf_big`, 4 `Jersey Barrier`, `pallet_box_mobile`, chairs, tables, people — non-repeating structure, so loop closure has something real to match |
| Domain match | project.md §1 is **industrial factory-floor inspection**. A warehouse is the actual target domain; a maze never was. |
| Occlusion for §7A | Shelving creates genuine occlusion pockets, which is what fog frontier + detection-gap clustering need to be non-trivial |

Caveat: it pulls 27 models from Fuel on first launch, so the first run needs network and
takes time. After that they are cached under `~/.gz/fuel/`.

---

## If you adopt a new world, append its result here — pass or fail.
The point of this file is that the next agent does not re-learn any of the above the
expensive way.
