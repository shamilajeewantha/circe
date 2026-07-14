# Dataset curation & merge pipeline

Turns the 12 read-only source datasets in `../datasets/` into ONE clean, de-duplicated,
9-class YOLO dataset in `../merged/`, plus a visual analysis report in `../reports/`.
Built on **FiftyOne** (import, near-duplicate detection, embeddings/UMAP) so the whole
process is reproducible — no hand-rolled merging.

## Hard rule
`../datasets/` is **read-only**. The pipeline imports images by reference and copies pixels
into `../merged/`; it never modifies a source file. The only writable file in `datasets/`
is its own `README.md`.

## Taxonomy (9 classes)
`0 concrete_crack, 1 corrosion, 2 fluid_patch, 3 fire, 4 smoke, 5 gauge_face,
6 efflorescence, 7 exposed_rebar, 8 spalling` — defined in `config.py`, the single source of
truth for the per-source class map, dropped classes, and excluded folders.

## Setup (once, in the `drone_detect` conda env)
```bash
conda activate drone_detect
pip install fiftyone umap-learn        # torch/CLIP already present in this env
```

## Run order
```bash
# in drone_detect (needs FiftyOne + GPU):
python pipeline/01_import.py     # sources -> one FiftyOne dataset (unified labels)
python pipeline/02_curate.py     # CLIP embeddings: near-dup tagging + UMAP + stats
python pipeline/03_export.py     # de-dup + stratified 70/20/10 -> ../merged/ (YOLO)

# report can run in base env (matplotlib/cv2) or drone_detect:
python pipeline/make_report.py   # -> ../reports/index.html  (publishable as an Artifact)

# optional interactive review (drone_detect, opens a browser):
python pipeline/launch_app.py
```
Then train with the existing `../train.py` (already points at `../merged/data.yaml`).

## What each step guarantees
- **01_import** — asserts every unified label is within the 9-class taxonomy; excluded folders'
  images are logged (with reason) to `../reports/dropped_images.csv`.
- **02_curate** — tags near-duplicates (never deletes); writes `stats.json` + `embedding.json`;
  appends duplicate images to the drop manifest.
- **03_export** — drops duplicates, caps background/negative images per split (logs any cut),
  regenerates a stratified split, writes `../merged/` + `data.yaml (nc: 9)`.
- **make_report** — composition, class balance, UMAP domain/bias figures, per-class sample
  mosaics, the drop-manifest summary, and recommendations.

## Every dropped image has a reason
`../reports/dropped_images.csv` has one row per dropped image: `filepath, source, stage, reason`
(excluded folder / near-duplicate / background-cap). Box-level class drops (junk classes, gauge
keypoints) are summarized in the report.

## Notes
- `ENABLE_MISTAKENNESS` in `config.py` is off (it needs a model-prediction pass). Turn on after a
  first training run to auto-surface likely mislabels.
- `merge_datasets.py` and `analyze_dataset.py` (old hand-rolled scripts) were removed and replaced
  by this pipeline.
