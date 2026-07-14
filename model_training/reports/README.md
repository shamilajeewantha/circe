# model_training/reports/

Dataset-analysis output for the merged 11-class dataset (class balance, source
contribution, UMAP domain map, per-class mosaics, near-duplicate / dropped-box audit).

**Tracked in git** (the lightweight, viewable deliverable):
- `figures/*.png` — all analysis plots (class balance, composition, source
  contribution, UMAP by source/class, per-class mosaics, dropped boxes)
- `stats.json` — summary statistics
- `preliminary_report.html` — small standalone summary (~12 KB)

**Gitignored** (heavy and fully regenerable — see root `.gitignore`):
- `index.html` (~12 MB, images embedded as base64)
- `dropped_images.csv` (~11 MB near-dup / drop audit)
- `embedding.json` (~4 MB UMAP coordinates)
- `*_run.log` (pipeline/curate run logs)

**Regenerate everything:** `python model_training/pipeline/make_report.py`.
