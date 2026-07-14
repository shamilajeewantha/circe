# model_training/sample_images/

A small set (~36) of sample images used as a quick `SOURCE` for `predict.py` smoke
tests — a fast way to eyeball a trained model's detections without running full
validation.

**Contents are gitignored** — everything except this README is excluded. The folder
is kept so `predict.py`'s default `SOURCE="sample_images"` resolves. Drop any images
you want to test on into this folder.
