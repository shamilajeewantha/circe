# merged_v2/ — 13-class merged dataset (generated, not source)

Everything in this folder except this README is **generated** by `pipeline/03_export.py` and is
gitignored. To rebuild: `01_import.py` → `02_curate.py` → `03_export.py` in the WSL `yolo_det` env.

## What it is

The successor to `merged/` (11 classes, what `runs/detect/circe_yolo26n` trained on). `merged/` is
deliberately **kept** alongside this so the two runs stay comparable.

**55,385 images** — 38,752 train / 11,068 valid / 5,565 test — `nc: 13`:

| id | class | instances | id | class | instances |
|---|---|---|---|---|---|
| 0 | concrete_crack | 7,292 | 7 | exposed_rebar | 1,353 |
| 1 | corrosion | 2,104 | 8 | spalling | 3,552 |
| 2 | fluid_patch | 35,076 | 9 | missing_bolt | 434 |
| 3 | fire | 14,826 | 10 | bolt_ok | 1,755 |
| 4 | smoke | 18,096 | 11 | bolt_defective | 133 |
| 5 | gauge_face | 9,298 | 12 | bolt_corroded | 621 |
| 6 | efflorescence | 700 | | | |

## What changed vs `merged/`

- **`loose_bolt` dropped.** SDNET's "Loosen" boxes mark the same physical fastener that the new
  `dataset_annotation/` pipeline now labels `bolt_defective`/`bolt_ok` — two contradictory positive
  labels on identical pixels, which Ultralytics' multi-label BCE trains both heads on rather than
  erroring.
- **`missing_bolt` kept.** It marks an EMPTY HOLE, a *different* object in the same photo, so it does
  not conflict. Measured over the 199 overlapping SDNET images: of 437 `missing_bolt` boxes, 392
  (90%) have IoU < 0.3 against any new box and 378 have exactly zero overlap (median best-IoU 0.000).
- **`bolt_ok` / `bolt_defective` / `bolt_corroded` added**, from the `dataset_annotation/` pipeline
  (Gemini `gemini-robotics-er-2-preview`, 3-pass majority-vote consensus, SAM3 box tightening).
- **The 200 SDNET "Missing" and 302 "Fixed" images are byte-identical in two source trees** (the
  read-only `datasets/` copy and the `dataset_annotation/` working copy). `01_import.py`'s
  `merge_same_image_sources()` unions their boxes onto one sample; left unmerged, `02_curate.py`'s
  near-dup pass would drop one copy and silently delete half the annotations for those images.

## Split integrity

Split membership is pinned by `reports/split_manifest.csv` (`stem → split`), honoured by
`03_export.stratified_split()`. On this export, **54,446 images were pinned from the manifest and only
939 new ones were stratified**, and a direct diff against `merged/` confirmed **0 images changed
split**. Without that manifest, re-running the pipeline after a taxonomy change silently
re-randomizes *every* split (an image's bucket is keyed on its primary label, so relabelling one
source shifts every subsequent RNG draw), putting the previous run's test images into the new
training set.
