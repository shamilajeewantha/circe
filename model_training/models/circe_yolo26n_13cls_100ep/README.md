# circe_yolo26n_13cls_100ep

YOLO26-nano structural-inspection detector, **13 classes**, trained 100 epochs on this laptop
(RTX 4050 Laptop GPU, 6 GB). This is the trained artifact; everything here is a copy from
`runs/detect/train-2/` (which is gitignored) so the model and its provenance survive in git.

| | |
|---|---|
| **best.pt** | epoch 98, **mAP50 0.5508**, mAP50-95 0.3970 |
| Previous model (`circe_yolo26n`, 11-class, 10 epochs) | mAP50 0.3998 |
| **Improvement** | **+37.8%** |
| Dataset | `merged_v2/` — 55,385 images (38,752 train / 11,068 valid / 5,565 test) |
| Base weights | `yolo26n.pt` (transfer learning) |
| Training time | 6.2 GPU-hours over 100 epochs |
| Trained | 2026-09-18 → 2026-09-20 |

## Per-class results (validation split, 11,068 images / 18,882 instances)

| class | instances | precision | recall | mAP50 | mAP50-95 |
|---|---|---|---|---|---|
| gauge_face | 1,856 | 0.961 | 0.998 | **0.995** | 0.975 |
| fluid_patch | 6,933 | 0.805 | 0.892 | **0.921** | 0.697 |
| bolt_corroded | 132 | 0.593 | 0.682 | **0.681** | 0.492 |
| bolt_ok | 362 | 0.523 | 0.785 | **0.696** | 0.512 |
| fire | 2,991 | 0.615 | 0.651 | 0.660 | 0.457 |
| bolt_defective | 24 | 0.605 | 0.667 | **0.616** | 0.492 |
| spalling | 701 | 0.621 | 0.502 | 0.525 | 0.331 |
| smoke | 3,511 | 0.708 | 0.372 | 0.478 | 0.349 |
| missing_bolt | 85 | 0.601 | 0.425 | 0.455 | 0.288 |
| concrete_crack | 1,423 | 0.529 | 0.340 | 0.373 | 0.218 |
| exposed_rebar | 278 | 0.480 | 0.356 | 0.346 | 0.171 |
| efflorescence | 151 | 0.530 | 0.185 | 0.209 | 0.080 |
| corrosion | 435 | 0.436 | 0.190 | 0.199 | 0.092 |
| **all** | **18,882** | **0.616** | **0.540** | **0.551** | **0.396** |

## How to read these numbers

**The 0.551 aggregate is not the useful number.** It averages across a 264:1 class imbalance
(fluid_patch 35,076 training instances vs bolt_defective 133), so per-class AP is what decides
whether the model is fit for a given job.

- **Fastener inspection works.** All three Gemini-annotated bolt classes land at 0.62-0.70 mAP50
  with 0.67-0.79 recall. `bolt_defective` is the notable one: it had **0.000 recall at epoch 14**
  and was expected to need re-annotation, but `cls_pw=0.5` class weighting plus full training
  brought it to 0.667 recall. 133 instances turned out to be enough.
- **Diffuse texture defects do not work.** `corrosion` (0.199) and `efflorescence` (0.209) are the
  weakest despite `corrosion` having 2,104 instances - more than every bolt class combined. These
  are boundary-less, texture-like defects that a box detector handles poorly; more epochs will not
  fix this, and more data probably will not either. Segmentation is the likelier path.
- `smoke` has decent precision (0.708) but poor recall (0.372) - it finds few instances but is
  usually right when it does.

## Training configuration

Full config in `args.yaml`. The non-default choices that mattered:

- `cls_pw: 0.5` — class-weight power for the imbalance (0.0 = off, 1.0 = full inverse frequency).
  Ultralytics applies this as `bce_loss *= class_weights` (verified in the installed 8.4.93
  `utils/loss.py:437`). At 0.5 the computed weights ranged from 0.202 (fluid_patch) to 3.231
  (bolt_defective). **This is an engineering judgement, not a cited default** - it is the first
  knob to sweep if rare-class recall needs improving.
- `patience: 25` — real early stopping (never triggered; the model was still improving at epoch 98).
- `close_mosaic: 10` — mosaic augmentation disabled for the final 10 epochs. Clearly visible in
  `results.csv`: mAP50 crawled ~0.0005/epoch over epochs 77-89, then gained 0.0089 across epochs
  90-98 once training switched to un-augmented images.
- `save_period: 1`, `seed: 0`, `deterministic: true`, batch 16 @ 640.

## Taxonomy note (13 classes, not 11 or 14)

`loose_bolt` was **dropped** and `bolt_ok`/`bolt_defective`/`bolt_corroded` added; `missing_bolt`
was **kept**. SDNET's "Loosen" boxes mark the same physical fastener the new scheme labels
`bolt_defective`/`bolt_ok` — two contradictory positive labels on identical pixels, which
Ultralytics' multi-label BCE trains both heads on rather than erroring. `missing_bolt` marks an
EMPTY HOLE, a different object, so it does not conflict: measured over the 199 overlapping SDNET
images, 392 of 437 `missing_bolt` boxes have IoU < 0.3 against any new box and 378 have exactly
zero overlap. See `../../pipeline/config.py` for the full reasoning.

## Files

- `best.pt` — the model (Git LFS, per `.gitattributes` `*.pt filter=lfs`)
- `args.yaml` — complete training configuration as actually run
- `results.csv` — per-epoch metrics for all 100 epochs
- `results.png` — loss and mAP curves
- `BoxPR_curve.png` — precision-recall curves per class
- `confusion_matrix_normalized.png` — per-class confusion

## Reproducing / continuing

```bash
# rebuild the dataset (WSL yolo_det env)
cd model_training/pipeline && python 01_import.py && python 02_curate.py && python 03_export.py

# train (edit constants at the top of train.py first)
cd model_training && python train.py

# evaluate this model
python val.py    # point WEIGHTS at models/circe_yolo26n_13cls_100ep/best.pt
```

Note the run was interrupted several times (manual stops overnight, plus two transient
`cudaErrorUnknown` faults from WSL2 GPU passthrough). Every interruption resumed cleanly from
`last.pt` with optimizer state intact — `results.csv` shows no discontinuity at any resume boundary.
