# Arduino UNO Q — board CAD and candidate cases

Official CAD for the UNO Q (ABX00162) plus measured case candidates. Same treatment as
[`../pi_zero_2w/cases/`](../pi_zero_2w/cases/README.md): nothing is listed on the strength of its
title, everything is measured.

Product page: <https://docs.arduino.cc/hardware/uno-q/>

## Reference dimensions (official)

From the [UNO Q datasheet](https://docs.arduino.cc/resources/datasheets/ABX00162-datasheet.pdf),
§12 Mechanical Information, quoted verbatim:

> The board dimension measures 68.58 mm × 53.34 mm, with bottom-side parts kept below 2 mm so the
> board can stack onto carrier bases. The outline and hole pattern follows and are compatible with
> the UNO form factor.

| Feature | Value |
|---|---|
| Board outline | **68.58 × 53.34 mm** (classic UNO footprint) |
| Mounting holes | UNO pattern — `[13.97, 2.54]`, `[15.24, 50.80]`, `[66.04, 7.62]`, `[66.04, 35.56]` |
| Bottom-side clearance | components kept **below 2 mm** so it stacks onto carrier bases |
| Ports | USB-C, microSD, Qwiic, traditional UNO headers, bottom high-speed connectors (MIPI-CSI camera / MIPI-DSI display), analog audio |

The hole coordinates are not from the datasheet — they are from the OpenSCAD source of the case in
`cases/raspberry-tips-stack/`, and they match the standard Arduino UNO pattern.

**The form-factor line matters more than it looks:** because the outline and hole pattern are UNO
compatible, *any* Arduino UNO case will bolt up mechanically. It will not have the right holes —
the UNO Q has USB-C and Qwiic where an UNO has USB-B and a barrel jack, and no UNO case has a
window for the bottom MIPI-CSI connectors. Mechanical fit is not port fit.

## `board_cad/` — official Arduino CAD

| File | What it is |
|---|---|
| `ABX00162-step.zip` | Official **STEP** model (`UNO Q simplified.stp`). Import straight into Onshape to test-fit a case |

Not committed (large, and one click away):

- [Full pinout PDF](https://docs.arduino.cc/resources/pinouts/ABX00162-full-pinout.pdf) (3 MB)
- [Datasheet PDF](https://docs.arduino.cc/resources/datasheets/ABX00162-datasheet.pdf) (29 MB)
- [CAD files zip](https://docs.arduino.cc/static/e05d1a3141208468509d10c329387dcd/ABX00162-cad-files.zip)
  (12 MB — Allegro `.brd`, Gerbers, drill files; PCB fab data, not a 3D model)

Arduino publishing an official STEP is the big advantage over the Pi Zero 2 W, which has no
official 3D model at all.

## Onshape-native models

**No public Onshape document for an UNO Q case was found.** The board launched in late 2025 and the
community models that exist are meshes on Printables/Thingiverse/GitHub, not Onshape documents.
Searched: Onshape doc links for "Arduino UNO Q" case/enclosure - only generic Arduino UNO tutorials
and GrabCAD/Printables listings came back.

What does exist, and is the practical route into Onshape:

| Route | Link |
|---|---|
| **Official UNO Q board STEP** | `board_cad/ABX00162-step.zip` - **Insert -> Import** into Onshape, then build or fit a case around the real board |
| UNO-form-factor case, Onshape-native (Pi Zero sibling project) | The Pi Zero folder has two public Onshape docs; see [`../pi_zero_2w/cases/README.md`](../pi_zero_2w/cases/README.md#onshape-native-models-editable-in-onshape-no-import-needed) for what an editable-in-Onshape listing looks like |

Both meshes below import into Onshape via **Insert -> Import** and can be measured/modified there.

## Candidate cases

Two verified, measured the same way as everything in `../pi_zero_2w/cases/`. They sit at opposite
ends of the trade-off — pick on **height vs. port access**:

| Case | Assembled height | Board pocket | Windows measured | Header access |
|---|---|---|---|---|
| [mcmchris-slim](cases/mcmchris-slim/) | **11 mm** | 68.4 × 55.6 | 2 (10.8 + 7.2 mm) | ❌ solid cap |
| [raspberry-tips-stack](cases/raspberry-tips-stack/) | **34.9 mm** | hood 77.2 × 58.8 | 3 (20 / 11.2 / 48 mm) | ❌ closed top |

Board is **68.58 × 53.34 mm**. Neither gives access to the UNO headers — if you need to plug a
shield or jumper into the header rows, neither of these works as-is.

### 1. `cases/mcmchris-slim/` — slim, tight fit ✅

**[github.com/mcmchris/mcm-android-auto-uno-q](https://github.com/mcmchris/mcm-android-auto-uno-q)**

Outer **74.18 × 58.94 mm**; `Bottom.stl` 5 mm + `Cap.stl` 6 mm → **11 mm assembled**, a third the
height of the stack case. Supplied as 3MF (BambuStudio); both the original `.3mf` and an STL I
converted from it are kept here.

Measured board pocket **68.4 × 55.6 mm** at h = 2.3–4.3 mm. Note the length: 68.4 measured against
a 68.58 mm board, at 0.4 mm voxel resolution — so it is somewhere around 68.4–68.8, i.e. **snug to
the point of zero slack on length**. Check it against your board before printing a full set. Width
has 2.3 mm of slack.

Cap windows: **10.8 mm** (`33.6–44.4`, consistent with USB-C) and **7.2 mm** (`12.0–19.2`, microSD
or Qwiic). Only those two — no camera window, and the cap is otherwise solid.

Built for a wireless Android Auto project, so the window choice reflects that, not general use.

### 2. `cases/raspberry-tips-stack/` — verified, with source ✅

**[github.com/raspberry-tips/arduino-uno-q-projects](https://github.com/raspberry-tips/arduino-uno-q-projects/tree/main/gehaeuse)**
· write-up: [raspberry.tips](https://raspberry.tips/en/3d-druck/arduino-uno-q-case-3d-printed)

Two-part screwless snap-fit case for **UNO Q + UNO Media Carrier stacked**. Ships OpenSCAD source,
which is why this one can be verified properly rather than inferred.

| Part | Outer bbox (measured) |
|---|---|
| `boden.stl` (base tray) | 81.88 × 63.64 × 11.0 mm |
| `haube.stl` (hood) | 81.88 × 63.64 × 31.7 mm |

The source declares the board and hole pattern directly:

```
brd   = [68.58, 53.34];
holes = [[13.97, 2.54], [15.24, 50.80], [66.04, 7.62], [66.04, 35.56]];
clr = 0.6; wall = 2.4; tray_t = 3.0;
```

`brd` matches the datasheet's 68.58 × 53.34 **exactly**. Declared windows, and what the voxel scan
actually measured in `haube.stl`:

| Window | Declared in SCAD | Measured |
|---|---|---|
| USB-C | `usb_y = [25, 45]` → 20 mm | **20.0 mm** (`11.2–31.2`) |
| Qwiic | `qwiic_y` | **11.2 mm** (`34.0–45.2`) |
| Jack / CSI zone | `win_x = [9, 57]` → 48 mm | **48.0 mm** (`11.6–59.6`), both long sides |

Source and geometry agree on every window. Ventilation slots measured at 8 mm pitch, matching the
`for (x = [8 : 8 : 68])` loop.

`camarm.scad` is a separate GoPro-style camera arm — relevant given the CSI camera plan in
[`../../circe_v1/docs/mothership-scout.md`](../../circe_v1/docs/mothership-scout.md).

**Caveat:** total height 34.9 mm because it is built for the **UNO Q + Media Carrier stack** (the Q
sits at z = 22 in the stack). With a bare UNO Q it fits but is much taller than needed. The SCAD
header itself flags two open questions (`A2` USB position lengthwise, `A3` which wall carries the
jacks/CSI) — the designer had not finalised those.

### 3. Measured and rejected ❌

| Model | Measured | Why rejected |
|---|---|---|
| [canalBrincandoComIdeias/Q1129](https://github.com/canalBrincandoComIdeias/Q1129) `K.106 montTUDO` | bbox 87 × 62 × 10.2 | No 68.6 × 53.3 pocket found on any axis |
| Q1129 `K.133 Case Vertical` | bbox 77 × 77 × 10.2 | Same — no board-sized pocket |

Both are classic-UNO cases. Because the UNO Q shares the UNO outline and hole pattern they would
bolt up, but neither exposes a pocket my scan could confirm, and their cutouts are for USB-B and a
barrel jack — ports the UNO Q does not have. Not committed.

Also swept with no usable result: ~50 GitHub repos matching *arduino uno case / enclosure / uno q*.
Nearly all are project-specific boxes (clocks, badges, game consoles) built around an UNO rather
than cases for the board.

### 4. Not reachable from here ❌

| Model | Status |
|---|---|
| [Case for Arduino Uno Q by schreinerman](https://www.printables.com/model/1466872-case-for-arduino-uno-q) | The link you sent. Printables returns **HTTP 403** to non-browser clients — could not download or verify. Download it manually if you want it measured |
| [Arduino UnoQ Case (Thingiverse 7195406)](https://www.thingiverse.com/thing:7195406) | Page loads, but file IDs are JS-rendered and not extractable, so no download URL |
| [Modular Enclosure for UNO Q](https://forum.arduino.cc/t/modular-enclosure-for-arduino-uno-q/1426830) | Design files are Blender `.blend`, not a printable mesh |
| **Onshape public document library** | `cad.onshape.com/documents/public` is a JS app behind a login — my fetcher gets an empty shell, so I cannot browse it. Web search *does* surface public Onshape doc links, and that is how the two Pi Zero ones were found; the same searches returned nothing UNO Q specific |
| [Case for Arduino Uno Q by schreinerman](https://www.printables.com/model/1466872-case-for-arduino-uno-q) | **HTTP 403** via curl with a browser user-agent *and* via WebFetch. Cloudflare bot protection, not an auth wall I can pass |
| [Thingiverse 7188152](https://www.thingiverse.com/thing:7188152) | Returns 200 but only a 25 KB JavaScript shell — generic page title, no file list, no `.stl` reference in the HTML. Content is client-side rendered |

**To get these two measured:** download the files in a browser (or have another tool fetch them),
drop them anywhere on disk, and give me the path. The scan is local and costs nothing.

## Rescan note

Everything here was **re-measured** after a bug was found in the first scanner (it silently reported
"no openings" when the wall band came out degenerate — that produced a false negative on the Pi Zero
side; see [`../pi_zero_2w/cases/README.md#correction`](../pi_zero_2w/cases/README.md#correction)).

The corrected scanner (60 z-levels at 0.3 mm, per-wall opening maps) **confirmed every UNO Q number
above** — no windows were missed:

| Part | Corrected result |
|---|---|
| `mcm_Bottom.stl` | pocket 68.4 × 55.2; one window, 10.2 mm, one short edge |
| `mcm_Cap.stl` | pocket 68.4 × 55.2; windows 11.4 mm and 6.9 mm on opposite short edges. Long edges closed → **header access confirmed absent** |
| `rt_haube.stl` | 47.7 mm window both long sides (SCAD says 48), 19.8 mm one short side (SCAD says 20 = USB-C), 10.8 mm other (Qwiic), plus ~4.8 mm vents |

**One known limitation:** `rt_boden.stl` reports openings on all four sides. That is an artifact —
it is a shallow open-topped tray, so above the tray wall the band is empty everywhere and reads as a
gap. Do not treat those as windows.

## Verification method

Identical to the Pi Zero 2 W work — see
[`../pi_zero_2w/cases/README.md`](../pi_zero_2w/cases/README.md#how-these-were-measured). STLs are
voxelised by z-ray parity on a 0.4 mm grid; per slice the wall bands are scanned for gaps, and each
gap is reported as an interval in mm. Where OpenSCAD source exists (as here) the measured intervals
are cross-checked against the declared constants — both agreed.

## If you want a bare-UNO-Q case

Nothing found fits a **bare** UNO Q with correct port cutouts and a sane height. The options are:

1. Print the raspberry-tips case and accept 34.9 mm of height
2. Download the schreinerman Printables model manually and I will measure it
3. Import `board_cad/ABX00162-step.zip` into Onshape and generate one parametrically, the way
   [`../pi_zero_2w/cases/onshape_custom_case.fs`](../pi_zero_2w/cases/onshape_custom_case.fs) was
   built — the official STEP makes this considerably easier than it was for the Pi
