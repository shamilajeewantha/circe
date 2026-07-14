# model_training/merged/

The **merged 11-class YOLO detection dataset** (~4 GB, ~55,068 images) produced by
the merge pipeline — the actual training input for `train.py`.

Layout (standard YOLO): `train/ valid/ test/` each with `images/` + `labels/`, plus
`data.yaml` (`path/train/val/test/nc/names`, `nc: 11`).

11 classes: concrete_crack, corrosion, fluid_patch, fire, smoke, gauge_face,
efflorescence, exposed_rebar, spalling, loose_bolt, missing_bolt.

**Contents are gitignored** — everything except this README is excluded (too large
for git). The folder is kept so paths resolve.

**Regenerate:** run `model_training/pipeline` (FiftyOne) over the immutable
`model_training/datasets/` sources. For training, a WSL-native copy is used for I/O
speed (`~/datasets/circe_merged`), with `data.yaml`'s `path:` pointed at that copy.
