# Factory-floor inspection datasets

Source datasets for training the ground-rover inspection model. Each subfolder is one
downloaded dataset, kept in its **original YOLO export** (`train/valid/test` + `data.yaml`).

> **This folder is READ-ONLY.** It is the immutable source pool. The curation/merge pipeline in
> `../pipeline/` (FiftyOne-based) reads these by reference and copies pixels into `../merged/` — it
> never modifies anything here. Only this `README.md` is editable. The merged 11-class training set and
> the analysis report (`../reports/index.html`) are produced by `../pipeline/` (see its README).

## Folder naming convention

```
<category>__<source-slug>
```

- **`<category>`** — our target class(es), hyphen-joined when a dataset covers more than one
  (e.g. `crack-corrosion`). This is *our* taxonomy, **not** the dataset's raw class names.
- **`__`** — double underscore, separates our category from the source.
- **`<source-slug>`** — the Roboflow **workspace** slug (first path segment of the Universe URL),
  so the origin collection is identifiable at a glance. External (non-Roboflow) sources use
  `ext-<name>`.

Example: `gauge-face__dataset-qxuat` = a *gauge_face* dataset from Roboflow workspace `dataset-qxuat`.

## How these numbers were verified

- **Image counts** — counted from the actual `*/images/` files on disk.
- **Per-class box counts** — counted by parsing every `*/labels/*.txt` file (first token per line).
  These are real annotation counts, not README/`data.yaml` claims.
- **Cross-checked** against Roboflow (MCP `projects_get`) for splits and class totals.

The **Verification** column below records the level of checking performed on each dataset:

- **Visual + counts** — sample images were rendered with bounding boxes and inspected to confirm the
  labels match the imagery, in addition to the count audit. **All YOLO datasets here reached this level.**
- **N/A** — non-YOLO external dataset (`bolt-defects__ext-sdnet2025`); source images were viewed but it
  has no YOLO labels to audit.

> `crack__kx3zs` was **deleted** (2026-07): visual inspection showed whole-frame classification
> boxes (no localization) and at least one flatly mislabeled tile. Unusable for detection.

## At a glance

Class rows are `name: box-count`.

| Folder | Category | Classes (box counts) | Images (tr/va/te) | Total boxes | Verification |
|---|---|---|---|---|---|
| `corrosion__meee-aegk9` | corrosion | `Corroded: 693` | 456/102/53 = 611 | 693 | Visual + counts |
| `crack-corrosion__codebrim` | crack + corrosion | `CorrosionStain: 1549`, `Crack: 2937`, `Efflorescence: 808`, `ExposedBars: 1500`, `Spallation: 1883` | 721/206/103 = 1030 | 8677 | Visual + counts |
| `fire-smoke__ak-rfpt5` | fire + smoke | `Fire: 6041`, `Other: 5069`, `Smoke: 5782` | 6541/1492/1299 = 9332 | 16892 | Visual + counts |
| `fire-smoke__yolov5-u5mjv` | fire + smoke | `fire: 9377`, `flame: 274`, `smoke: 12891` | 5783/1630/817 = 8230 | 22542 | Visual + counts |
| `fluid-patch__cv-6rgre` | fluid_patch | `Spill: 28814` | 17121/1667/818 = 19606 | 28814 | Visual + counts |
| `fluid-patch__kuanhan-fu` | fluid_patch | `[leak]: 934`, `object: 1` | 639/53/60 = 752 | 935 | Visual + counts |
| `fluid-patch__project-5v4jq` | fluid_patch | `leak: 345` | 141/41/19 = 201 | 345 | Visual + counts |
| `fluid-patch__welding-defects` | fluid_patch | `oil_spill: 8754` | 6660/—/— = 6660 | 8754 | Visual + counts |
| `gauge-face__dataset-qxuat` | gauge_face | `center: 9223`, `gauge: 9262`, `max: 9194`, `min: 9182`, `tip: 9468` | 6396/1829/914 = 9139 | 46329 | Visual + counts |
| ~~`gauge-face__kaushal-bhide`~~ | — | **DELETED** (needle endpoints only, no dial box) | — | — | — |
| `gauge-face__yolo-lxxyl` | gauge_face | `Center: 429`, `Gauge: 429`, `Needle: 429` | 390/33/0 = 423 | 1287 | Visual + counts |
| `kashita1crack-spall-detector` | concrete_crack + spalling | `cracks: 3326`, `spalling: 1913` | 1520/—/371 = 1891 | 5239 | Visual + counts |
| `uminho-gya54crack-detection-concrete-gphbn` | concrete_crack | `crack: 1551` | 791/215/111 = 1117 | 1551 | Visual + counts |
| `bolt-defects__ext-sdnet2025` | loose_bolt + missing_bolt | `Loosen: 529` → loose_bolt, `Missing: 439` → missing_bolt (+302 no-defect bg) | COCO, re-split | 968 | Visual + counts |

> `[leak]` in `fluid-patch__kuanhan-fu` is the class literally named
> `"Spill Detection - v2 2024-01-24 3-05pm"` in its `data.yaml` — a leaked project string, not a real
> class name. See Known problems.

## Known problems (data quality) — read before training

These are real issues found by inspecting the downloaded content. Each needs handling at merge/train
time; nothing here is fixed automatically.

| Folder | Problem | Action needed |
|---|---|---|
| `fluid-patch__kuanhan-fu` | Class 0 name is a leaked project string (`"Spill Detection - v2 2024-01-24 3-05pm"`); class `1 object` is 1 stray box on a motorbike (`74_mp4-15`), not a spill. | Map class `0` → `fluid_patch`; **drop class `1`** (or delete that one image). |
| `fire-smoke__ak-rfpt5` | `Other` class (5069 boxes) is junk — boxes on people, watermarks, logos. Heavy outdoor/wildfire bias. | **Drop `Other`**; filter/curate for indoor-plant relevance. |
| `fire-smoke__yolov5-u5mjv` | `flame` class tiny (274). 871 background images with no boxes. Outdoor bias. | Fold `flame` → `fire`; decide whether to keep background imgs; filter bias. |
| `fluid-patch__welding-defects` | **Train split only** (no valid/test); augmentation baked in (2220 raw → 6660). Visual check found a **likely mislabel**: a black mouse pad boxed as `oil_spill` (same CCTV domain as `kuanhan-fu`) — dark blobs may be conflated with oil. | **Re-split** into train/valid/test; **spot-review labels** before trusting. |
| `fluid-patch__cv-6rgre` | Augmentation baked into export (~19.6k imgs vs ~8k raw on Universe). | Aware only — may over-weight this source in the merge. |
| `gauge-face__dataset-qxuat` | 5 classes; only `gauge` is the dial bbox. On-disk box counts ~2–14% below Roboflow (export clipped boxes). | For `gauge_face` keep class `1 gauge`; drop needle/tick classes unless doing reading. |
| ~~`gauge-face__kaushal-bhide`~~ | **DELETED** — needle endpoints only, no dial box. | Removed; not in the merge. |
| `gauge-face__yolo-lxxyl` | 3-class export vs 5 on Universe; **no test split**; small (423). | Fine to use (dial bbox = class `1 Gauge`); add a test split if needed. |
| `corrosion__meee-aegk9` | Source is instance-segmentation exported as bbox; ~40% empty label files. | Aware only — empties act as clean-fastener negatives. |
| `crack-corrosion__codebrim` | Contains 3 extra defect classes (Efflorescence/ExposedBars/Spallation) beyond our taxonomy. | Keep or drop classes 2–4 depending on final class list. |
| `bolt-defects__ext-sdnet2025` | COCO JSON (has segmentation → import bboxes only); 3 phantom image refs in the Loosen JSON. | **Now included** via COCO → loose_bolt / missing_bolt; missing files skipped + logged. |
| `kashita1crack-spall-detector`, `uminho-…-concrete-gphbn` | New; raw slug folder names kept (datasets/ immutable). | Mapped in `config.py`; strengthen concrete_crack (now 3-source) + spalling. |
| ~~`crack__kx3zs`~~ | **DELETED** — whole-frame classification boxes (no localization) + mislabeled tiles. | Removed 2026-07; do not re-download. |

**Cross-cutting:** several sources are augmented at export time, so raw-image counts differ from
Universe and box counts are inflated — weight sources deliberately when merging. Both `fire-smoke__*`
sets carry outdoor/wildfire bias that won't match a factory floor.

## Per-folder detail & re-download

Class rows are `id name: box-count`. All Roboflow sets exported in **YOLO** format; re-download from
the URL, pick the version, export as YOLOv8/YOLOv11 (or use the SDK at the bottom).

### corrosion__meee-aegk9  →  `corrosion`   [Verification: Visual + counts]
- **Source:** https://universe.roboflow.com/meee-aegk9/nut-and-bolt-corrosion-2-qk6z0 (v1, 2024-10-17)
- **Classes:** `0 Corroded: 693`
- **Content:** rusted nuts/bolts close-ups. ~40% of label files are empty (clean-fastener negatives).
- **Note:** Roboflow project is *instance-segmentation*; this export is bbox.

### crack-corrosion__codebrim  →  `concrete_crack` + `corrosion`   [Verification: Visual + counts]
- **Source:** https://universe.roboflow.com/codebrim-jhs57/code-uibqb (v1, 2024-09-24 — CODEBRIM benchmark)
- **Classes:** `0 CorrosionStain: 1549` · `1 Crack: 2937` · `2 Efflorescence: 808` ·
  `3 ExposedBars: 1500` · `4 Spallation: 1883`
- **Content:** concrete structural defects. Use class 1 (`concrete_crack`) + class 0 (`corrosion`);
  classes 2–4 are bonus defects — keep or drop per final taxonomy. **Counts match Roboflow exactly.**

### fire-smoke__ak-rfpt5  →  `fire` + `smoke`   [Verification: Visual + counts]
- **Source:** https://universe.roboflow.com/ak-rfpt5/fire-and-smoke-detection-idbvj (v5, 2023-09-30)
- **Classes:** `0 Fire: 6041` · `1 Other: 5069` · `2 Smoke: 5782`
- **Content:** fire/smoke boxes are correct and localized (verified). **`Other` is junk** — its
  boxes land on people/watermarks/logos; drop it. Heavy **outdoor/wildfire bias** — filter for
  indoor/plant relevance.
- **Note:** Universe project actually has **14 raw classes**; this v5 export is a 3-class remap
  (Fire/Other/Smoke). Box totals reconcile (~16.9k).

### fire-smoke__yolov5-u5mjv  →  `fire` + `smoke`   [Verification: Visual + counts]
- **Source:** https://universe.roboflow.com/yolov5-u5mjv/smoke-fire-wsde7 (v4, 2024-08-12)
- **Classes:** `0 fire: 9377` · `1 flame: 274` · `2 smoke: 12891`
- **Content:** `flame` is tiny — fold into `fire`. 871 background images (no boxes). Same outdoor bias.

### fluid-patch__cv-6rgre  →  `fluid_patch`   [Verification: Visual + counts]
- **Source:** https://universe.roboflow.com/cv-6rgre/spills-ax5xv (v2, 2024-10-01)
- **Classes:** `0 Spill: 28814`
- **Content:** largest set here, floor-spill focused. Box count inflated by augmentation baked into
  the export (Universe raw project is ~8k imgs / 12k boxes).

### fluid-patch__kuanhan-fu  →  `fluid_patch`   [Verification: Visual + counts]
- **Source:** https://universe.roboflow.com/kuanhan-fu-mri8i/leakage-fw7iy (v1, 2024-02-28)
- **Classes:** `0 "Spill Detection - v2 2024-01-24 3-05pm": 934` · `1 object: 1`
- **Content:** class 0 = **real floor leak/puddle boxes** (verified good). Problems are cosmetic only:
  the class-0 **name is a leaked project string**, and class `1 object` is **one stray box on a
  parked motorbike** (image `74_mp4-15`). **Fix at merge:** map `0` → `fluid_patch`, drop class `1`.

### fluid-patch__project-5v4jq  →  `fluid_patch`   [Verification: Visual + counts]
- **Source:** https://universe.roboflow.com/project-5v4jq/leaks (v2, 2024-03-25)
- **Classes:** `0 leak: 345`
- **Content:** small, pipe/joint leaks. **Counts match Roboflow exactly.**

### fluid-patch__welding-defects  →  `fluid_patch`   [Verification: Visual + counts]
- **Source:** https://universe.roboflow.com/other-welding-defects/oil_spill-rk2z7 (v2, 2024-06-27)
- **Classes:** `0 oil_spill: 8754`
- **Content:** **train split only** (no valid/test) and augmented (2220 raw → 6660 train).
  Re-split before training.

### gauge-face__dataset-qxuat  →  `gauge_face`   [Verification: Visual + counts]
- **Source:** https://universe.roboflow.com/dataset-qxuat/analog-gauge-wj1r6 (v2, 2024-12-11)
- **Classes:** `0 center: 9223` · `1 gauge: 9262` · `2 max: 9194` · `3 min: 9182` · `4 tip: 9468`
- **Content:** largest gauge set. For `gauge_face` bbox use class `1 gauge` (whole dial); the rest are
  needle/tick keypoints for dial reading. On-disk box counts run ~2–14% below Roboflow (export
  likely clipped out-of-bounds boxes); splits/images match exactly.

### gauge-face__kaushal-bhide  →  gauge reading (supplementary)   [Verification: Visual + counts]
- **Source:** https://universe.roboflow.com/kaushal-bhide-fgktg/gauge-ielsn (v4, 2023-09-26)
- **Classes:** `0 base: 2884` · `1 tip: 2955`
- **Content:** needle endpoints only — **no whole-dial box**. Useful for needle-angle reading, not for
  `gauge_face` detection on its own. Augmented (1129 raw → 2897).

### gauge-face__yolo-lxxyl  →  `gauge_face`   [Verification: Visual + counts]
- **Source:** https://universe.roboflow.com/yolo-lxxyl/analog-gauge-meter (v13 "v06-crop", 2022)
- **Classes:** `0 Center: 429` · `1 Gauge: 429` · `2 Needle: 429`
- **Content:** real industrial pressure gauges on piping; class `1 Gauge` = whole-dial bbox (verified,
  correct id mapping). Small (423 imgs), **no test split**.
- **Note:** Universe project has **5 classes** (Center/Head/Needle/Tail/Gauge); this export is a
  reduced 3-class scheme. Content is coherent and usable regardless.

### bolt-defects__ext-sdnet2025  →  (external, not in taxonomy) — not Roboflow, not YOLO   [Verification: N/A — non-YOLO]
- **Source:** external download (SDNET2025). Not on Roboflow; not retrievable via the SDK.
- **Structure:** `Defected/` (`Annotated Loosen bolt & nuts`, `Annotated Missing bolt & nuts`,
  `Original Image` ~524 imgs) and `Fixed/` (`640-640` ~302, `Original image` ~302). Annotations are
  **CSV/JSON + pre-annotated images** (bbox and polygon), *not* YOLO `.txt` labels.
- **Content:** bolt loosening / missing detection — outside the current 10-class taxonomy. Convert to
  YOLO first if adopting. Left as-is.

## Mapping to the 11-class target taxonomy

`0 concrete_crack, 1 corrosion, 2 fluid_patch, 3 fire, 4 smoke, 5 gauge_face, 6 efflorescence,
7 exposed_rebar, 8 spalling, 9 loose_bolt, 10 missing_bolt` — all covered by the sources here.
The full per-source class map lives in `../pipeline/config.py` (single source of truth).

Notes:
- The two newest folders keep their raw Roboflow slug names (`kashita1crack-spall-detector`,
  `uminho-gya54crack-detection-concrete-gphbn`) — `datasets/` is immutable, so they are not renamed;
  they are mapped by their real name in `config.py`.
- `bolt-defects__ext-sdnet2025` is now **included** via its COCO JSON (`Loosen`→loose_bolt,
  `Missing`→missing_bolt; the `Fixed` set becomes no-defect background negatives).
- `gauge-face__kaushal-bhide` was deleted (needle endpoints only, no dial box).

## Reproducing this folder (Roboflow SDK)

```python
# pip install roboflow   ;   set ROBOFLOW_API_KEY in a .gitignore'd .env
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_KEY")
# example: gauge-face__dataset-qxuat
rf.workspace("dataset-qxuat").project("analog-gauge-wj1r6").version(2).download("yolov8")
```
Take `workspace`/`project`/`version` from each **Source** line above.
`bolt-defects__ext-sdnet2025` is an external download, not available via the Roboflow SDK.
