# datasets/

Local YOLO datasets directory. Ultralytics auto-downloads sample datasets here
(e.g. `coco8`) when a config references them.

**Contents are gitignored** — everything except this README is excluded (see the
root `.gitignore`). The folder is kept in git so tooling that expects the path to
exist still works. Sample datasets are re-downloaded automatically on demand; the
project's real training data lives under `model_training/`.
