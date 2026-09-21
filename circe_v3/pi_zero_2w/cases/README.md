# Raspberry Pi Zero 2 W — candidate cases (measured, not trusted)

Downloadable case models for the Zero 2 W, **every one measured against the official board
drawing** before being kept. Pick one; the measurements below are the reason to pick it.

Nothing here was kept on the strength of its title. Each STL was voxelised and its board pocket
and wall openings measured; the method and the raw numbers are at the bottom.

## Reference dimensions (official)

From [`raspberry-pi-zero-2-w-mechanical-drawing.pdf`](raspberry-pi-zero-2-w-mechanical-drawing.pdf)
(kept here; source: [datasheets.raspberrypi.com](https://datasheets.raspberrypi.com/rpizero2/raspberry-pi-zero-2-w-mechanical-drawing.pdf)):

| Feature | Value |
|---|---|
| Board outline | **65.0 × 30.0 mm** |
| Mounting holes | **3.5 mm in from every edge** → 58 × 23 mm pattern |
| mini-HDMI centreline | **x = 12.4 mm** |
| micro-USB (OTG/data) centreline | **x = 41.4 mm** |
| micro-USB (power) centreline | **x = 54.0 mm** |
| microSD | left short edge, socket on PCB **underside** |
| CSI-2 camera FFC | right short edge |
| GPIO | 40-pin header footprint, centred on the long edge (unpopulated from the factory) |

Port list confirmed in the
[product brief](https://datasheets.raspberrypi.com/rpizero2/raspberry-pi-zero-2-w-product-brief.pdf):
microSD slot, CSI-2 camera connector, USB 2.0 OTG, mini HDMI, HAT-compatible 40-pin I/O header
footprint. Form factor stated as 65 mm × 30 mm.

## Onshape-native models (editable in Onshape, no import needed)

These are public Onshape documents, not meshes - open the link, change parameters, export.

| Model | Onshape document | Notes |
|---|---|---|
| **Pi Zero 2W case, parametric** by Petros | [cad.onshape.com/documents/ab669ee7...](https://cad.onshape.com/documents/ab669ee7052b08fe565fff49/w/37b43e480373b4c8ea7a09e1/e/670d9b9fafee8b605de035d7) | **Configurable cutouts**: HDMI yes/no, SD yes/no, pin headers no/serial, power (power only / power + USB), print-hole supports. **No Onshape account needed** - follow the link, change the config, right-click -> export STL. Listing: [Printables 1483036](https://www.printables.com/model/1483036-raspberry-pi-zero-2w-case-parametric) |
| **Pi Zero case (Onshape source)** by cfunseth | [cad.onshape.com/documents/c993e969...](https://cad.onshape.com/documents/c993e969a52a4661883c1765/w/7dbff4f8027c426091c92394) | Original Zero, same 65 x 30 footprint. Listing: [Thingiverse 1227867](https://www.thingiverse.com/thing:1227867/files) |

I could not measure either one - an Onshape document is not a file I can download and voxelise, and
the public library itself is a JS app behind a login that returns an empty shell to my fetcher.
Both links came from web search, not from browsing Onshape. Treat the parameter lists above as the
designers' own claims, not as something I verified.

Everything else in this folder is a mesh that imports into Onshape via **Insert -> Import**.

## The verdict

**`jmtodaro-slotted` covers all six openings.** Corrected - an earlier version of this file said it
had no camera slot. That was a measurement bug on my side, not a property of the model. See
[Correction](#correction) below.

| Model | Board pocket | HDMI | USB (OTG) | Power | microSD | Camera (CSI) | GPIO |
|---|---|---|---|---|---|---|---|
| [jmtodaro-slotted](jmtodaro-slotted/) | 66.0 x 30.0 | ✅ | ✅ | ✅ | ✅ 12.0 wide | ✅ **17.1 wide** | ✅ 53 wide |
| [opcow-honeycomb](opcow-honeycomb/) | 66.9 x 31.8 | ✅ 14.7 wide | ✅ merged 24 | ✅ merged | ✅ 15.9 wide | ✅ 20.4 wide | ❌ |
| [anemonix-sleeve](anemonix-sleeve/) | 68.1 x 33.0 | ⚠️ one 54.6 mm slot spans all three | ⚠️ | ⚠️ | ✅ 19.5 wide | ✅ 19.5 wide | ❌ solid lid |
| [onshape_custom_case.fs](onshape_custom_case.fs) | 66.0 x 31.0 | ✅ 12.4 | ✅ merged | ✅ merged | ✅ 18 wide | ✅ 20 wide | ✅ 54 wide |

Measured port centrelines are in the model's own frame; compare against 12.4 / 41.4 / 54.0 above.

## Correction

The first version of this file reported `jmtodaro-slotted` and `opcow-honeycomb` as having **no
camera slot**. Both were wrong.

The cause was a guard in the first scanner: when the wall band between the board pocket and the
part's outer edge came out degenerate, it returned "no openings" instead of reporting that it could
not measure - a silent false negative. It also sampled only 14 z-levels, too coarse for openings
confined to a narrow height band.

The rescan uses 60 z-levels at 0.3 mm and builds a **per-wall opening map**: for each wall, an image
of along-wall position x height, marked where no material exists anywhere through the wall band.
jmtodaro's `top_header.stl` shows clear openings on **both** short edges at h = 3.4-7.5 mm:
**17.1 mm** on one (CSI camera FFC, connector is ~17 mm) and **12.0 mm** on the other (microSD).

Rescanned results for the other two are in the table above; both also have two open short edges.

### 1. `jmtodaro-slotted/` — best dimensional match, all six openings ⭐

**[github.com/jmtodaro/Pi-Zero-2-Simple-Slotted-Case](https://github.com/jmtodaro/Pi-Zero-2-Simple-Slotted-Case)**

Outer 69.2 × 34.2 mm, bottom tray 5.5 mm, tops 7.7 mm. Board pocket **66.0 × 30.0 mm** — 1.0 mm
of slack on length, 0.0 mm on width, the tightest of any model here.

Port slots measured on the long edge (mirrored frame, so read right-to-left):
`7.6–16.0`, `20.4–28.8`, `47.6–59.2` → centrelines **54.2 / 41.4 / 12.6**. Against the official
54.0 / 41.4 / 12.4 that is a **worst-case error of 0.2 mm**. This designer worked from the real
drawing.

- `bottom.stl` — tray, effectively closed (all cuts live in the top)
- `top.stl` / `top2_v2.stl` — no GPIO slot
- `top_header.stl` / `top2_header.stl` — **use these**: adds a 53.2 mm GPIO slot (header is 50.8 mm)
- **Both** short edges open at h = 3.4–7.5 mm: **17.1 mm** (CSI camera ribbon) and **12.0 mm**
  (microSD). Confirmed on the per-wall opening maps

**Choose this** — it is the only model here with all six openings and the tightest fit.

### 2. `opcow-honeycomb/` — camera + ports, no GPIO

**[github.com/opcow/Honeycomb-Raspberry-Pi-Zero-W-Case](https://github.com/opcow/Honeycomb-Raspberry-Pi-Zero-W-Case)**

Outer 75.9 × 40.9 mm, 10 mm deep. Pocket **66.8 × 32.0 mm**. Modelled lying on its side — the
part height runs along **Y**, not Z; re-orient before slicing.

- Long edge: `6.4–21.2` (centre **13.8** ≈ HDMI) and `36.8–60.8` — a single 24 mm slot spanning
  both micro-USBs, same merged approach as the custom case
- **Both** short edges open: rescan gives **15.9 mm** and **20.4 mm** — the narrower is microSD,
  the wider takes a camera ribbon
- Four decorative `top_N.stl` lids, all with only a 6.4 mm opening → **no GPIO slot**

**Choose this if** the camera matters more than the GPIO header.

### 3. `anemonix-sleeve/` — crude, open on three sides

**[github.com/anemonix/rPiZeroCase](https://github.com/anemonix/rPiZeroCase)**

472 triangles — a plain rectangular sleeve. Pocket **68.0 × 33.2 mm** (3 mm slack: loose). One
long edge carries a single **54.8 mm** slot spanning x 6.4–61.2, which covers all three of
12.4 / 41.4 / 54.0 at once. Both short edges open 20 mm.

`RpiLid.stl` is a solid 80 × 45 × 5 plate with no openings at all.

**Choose this if** you want something to modify. Not a finished case.

### 4. `onshape_custom_case.fs` — the one with everything

Parametric FeatureScript written against the official drawing. Live in Onshape:
<https://cad.onshape.com/documents/c8d3e196f246a31f2cfb5f75/w/bf4b1e50044852634d518d81/e/07c922c7e875590308432e3f>

Outer 70 × 35 × 15 mm, 2 mm walls, base + lip-fit lid, 4 standoffs on the 58 × 23 pattern with
Ø2.1 pilot holes for M2.5 self-tappers. All six openings present. Side openings are open-topped
slots capped by the lid, so it prints with no bridging. Parameters: wall, clearance, standoff,
headroom, lid thickness.

Two caveats, stated plainly: the GPIO/microSD/CSI **y-positions are not dimensioned on the
official drawing** — I scaled them off it, so those three openings are deliberately oversized
rather than precise. PCB thickness 1.4 mm is an assumption and only affects internal headroom.

## Rejected, and why

| Model | Reason |
|---|---|
| [outdoorbits Tiny_case](https://github.com/outdoorbits/case-for-little-backup-box) | Pocket fine (67.2 × 32.8) but the three long-edge slots sit at 14.8/31.8/48.8 — they serve a **Zero4U USB hub HAT**, not the Pi's own ports. Useless without that HAT |
| outdoorbits Battery_case | 81 × 69 mm, no board-sized pocket found — battery compartment |
| [darkparrott PiScout PoE](https://github.com/darkparrott/Pi-Zero-PiScout-Case) | 51 mm tall, PoE-specific |
| [eat-sleep-code timelapse](https://github.com/eat-sleep-code/3d-print-garden-timelapse-enclosure) | 103 × 169 × 74 mm, no board pocket — weatherproof box |
| [plaursen piwebcam](https://github.com/plaursen/piwebcam) | 106 × 44 mm, no board pocket |

## How these were measured

No mesh library was used. Each STL is voxelised by **z-ray parity** — for every grid column the
triangle crossings are collected, sorted, and a point is inside when an odd number of crossings
lies below it. 0.4 mm grid, 12–14 slices through the part.

Per slice: scan rows and columns to find the inner faces of the walls (the board pocket), then
check each of the four wall bands for material. A band with no material along a run of the wall
is an **opening**, reported as an interval in mm from the pocket corner.

Models are modelled in arbitrary orientations, so all three axes are tried and the one yielding a
65 × 30-ish pocket wins — several of these are modelled on their side.

Scripts live in the session scratchpad, not committed. Re-derivable from this description; the
per-slice numbers quoted above are the output.

## Sources not reachable from here

- <https://www.printables.com/model/753906-raspberry-pi-zero-2-w-with-camera-module-v-3-case> —
  Printables returns **HTTP 403** to non-browser clients, so nothing there could be downloaded or
  verified. Same for every other printables.com model
- [GrabCAD Pi Zero 2 W board model](https://grabcad.com/library/raspberry-pi-zero-2-w-1) and
  [with 40-pin header](https://grabcad.com/library/raspberry-pi-zero-2-w-with-40-pin-male-connector-1)
  — needs a login. Worth grabbing manually: importing the board STEP into Onshape lets you
  test-fit any of these cases directly
- Raspberry Pi publishes **no official STEP/3D model** for the Zero 2 W, only the 2D mechanical
  drawing kept here
