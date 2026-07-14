"""Train YOLO26 on the merged 11-class inspection dataset — official Ultralytics layout.

Edit the constants below and run:  python train.py

Checkpointing & resume (official Ultralytics flow)
--------------------------------------------------
Ultralytics writes BOTH `last.pt` and `best.pt` at the END OF EVERY EPOCH by default (save=True), so a
crash loses at most the current epoch. Resuming is an EXPLICIT, manual action — there is no auto-resume
(https://docs.ultralytics.com/modes/train/#resuming-interrupted-trainings):

    model = YOLO("path/to/last.pt")   # explicitly load the interrupted checkpoint
    model.train(resume=True)          # resume=True; all other args come from the run's args.yaml

So to continue after a crash: set `RESUME = True` below (RESUME_CKPT points at the run's last.pt) and
re-run. `resume` defaults to False; a normal run is a fresh run. As a safety net, if you set RESUME=True
against a checkpoint that is NOT genuinely mid-training (Ultralytics strips the optimizer + sets epoch=-1
when a run FINISHES), this script ERRORS OUT — because `resume=True` on such a file makes Ultralytics
silently restart a brand-new run on its BUILT-IN DEFAULT dataset+config (NOT this project's data).
Docs: https://docs.ultralytics.com/usage/cfg/

Output layout is the official Ultralytics convention: runs/detect/<RUN_NAME>/ with weights/{best,last}.pt,
args.yaml, results.csv, results.png, confusion matrix, etc.

Live dashboard + later analysis
-------------------------------
TensorBoard logging is enabled in Ultralytics settings (`yolo settings tensorboard=True`), so training
streams event files into the run dir. In a SECOND terminal (same yolo_det env):
    tensorboard --logdir runs/detect          # then open http://localhost:6006
It plots, live per epoch: train/val box_loss, cls_loss, dfl_loss; metrics/precision, metrics/recall,
metrics/mAP50, metrics/mAP50-95; and the LR schedule. For offline analysis later, everything is also in
`runs/detect/<RUN_NAME>/results.csv` (one row per epoch) plus the auto-saved results.png / PR / F1 /
confusion-matrix plots. Docs: https://docs.ultralytics.com/integrations/tensorboard/
"""
from pathlib import Path

from ultralytics import YOLO

# ---- settings (edit these) --------------------------------------
MODEL   = "yolo26n.pt"     # pretrained base -> training from it is transfer-learning/fine-tuning.
                           #   variants: yolo26n/s/m/l/x.pt (bump for accuracy; drop BATCH if VRAM-limited)
DATA    = "/home/shamila/datasets/circe_merged/data.yaml"   # WSL-native copy (fast per-epoch reads)
EPOCHS  = 10               # first shakedown run
IMGSZ   = 640
BATCH   = 16               # yolo26n @ 640 fits ~6 GB VRAM; use 8 for yolo26s, or -1 for AutoBatch
DEVICE  = "0"              # "0" = first GPU, "cpu" = force CPU
RUN_NAME = "train"         # Ultralytics' OFFICIAL DEFAULT name -> runs/detect/train/ (auto-increments
                           #    train2, train3, ... on re-runs). NOTE: the default-dataset-restart footgun
                           #    a generic `train/` dir once caused is now guarded by the explicit RESUME
                           #    flag + is_resumable() below, NOT by the folder name.

# ---- resume (explicit — the official Ultralytics way) ----
RESUME      = False        # False = fresh run. True = CONTINUE an interrupted run (see RESUME_CKPT).
RESUME_CKPT = ""           # "" = auto-resolve the LATEST runs/detect/<RUN_NAME>*/weights/last.pt (the run
                           #  that just crashed — correct even though exist_ok=False auto-increments the
                           #  dir name). Set an explicit path here to resume a SPECIFIC run instead.

# ---- fine-tuning knobs (defaults = standard full fine-tune from the pretrained weights) ----
FREEZE      = None         # freeze first N layers for small-data transfer (e.g. 10 = backbone); None = all trainable
# NOTE: with OPTIMIZER="auto" (below), Ultralytics IGNORES LR0 and momentum and auto-derives them
# (the run log prints: "optimizer=auto ... ignoring 'lr0=0.01' ... AdamW(lr=0.000667...)"). To actually
# control LR0/LRF, set OPTIMIZER to a concrete optimizer (e.g. "SGD" or "AdamW"). LRF (the final-LR
# fraction) still applies under auto.
LR0         = 0.01         # initial learning rate (ONLY used when OPTIMIZER != "auto")
LRF         = 0.01         # final LR = LR0 * LRF (scheduler end)
COS_LR      = False        # True = cosine LR schedule (often smoother than the default linear)
PATIENCE    = 100          # early-stop after this many epochs w/o val improvement (100 = effectively off for 10 ep)
OPTIMIZER   = "auto"       # "auto" (recommended; auto-picks optimizer+lr0+momentum) or SGD/Adam/AdamW/...
SAVE_PERIOD = 1            # 1 = ALSO keep a checkpoint every epoch (epoch0.pt, epoch1.pt, ...) alongside
                           #   best.pt/last.pt (which save every epoch regardless). ~6 MB/epoch for yolo26n;
                           #   raise to 5/10 for very long runs if disk is tight. -1 = only best.pt/last.pt.
CACHE       = False        # "ram"/"disk" to cache images; keep False when training from the WSL-native copy
WORKERS     = 8            # dataloader worker processes (Ultralytics default)
SEED        = 0            # fixed seed + deterministic=True below -> reproducible, comparable runs
# NOTE close_mosaic: Ultralytics default is 10 = disable mosaic aug for the LAST 10 epochs. On a 10-epoch
# run that turns mosaic OFF for the whole run. Set to e.g. 3 to keep mosaic on for the first 7 epochs of a
# short run, or leave 10 for long runs. Kept at the official default here; change if you want mosaic aug.
CLOSE_MOSAIC = 10
# -----------------------------------------------------------------

HERE = Path(__file__).parent
PROJECT = HERE / "runs" / "detect"                       # official runs/detect/<name>/ tree
RUN_DIR = PROJECT / RUN_NAME


def is_resumable(ckpt: Path) -> bool:
    """True only if `ckpt` is a genuinely mid-training checkpoint for THIS dataset.

    Ultralytics strips the optimizer and sets epoch=-1 when a run FINISHES, so those two fields
    distinguish an interrupted run (resumable) from a finished/smoke checkpoint. We also require the
    saved dataset to match DATA. Used only to REFUSE a bad `resume=True` (which would otherwise make
    Ultralytics silently restart on its built-in default dataset) — not to auto-resume anything.
    """
    import torch
    # weights_only=False is required: an Ultralytics checkpoint holds a pickled model object (not just
    # tensors), so weights_only=True cannot load it. Safe here because `ckpt` is our OWN local run file
    # (a trusted artifact), and it is how Ultralytics itself loads checkpoints. Never point at an
    # untrusted/downloaded .pt.
    try:
        ck = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"[resume] could not read {ckpt}: {e}")
        return False
    args = ck.get("train_args") or {}
    epoch = ck.get("epoch", -1)
    mid_training = ck.get("optimizer") is not None and isinstance(epoch, int) and epoch >= 0
    same_data = str(args.get("data", "")) == str(DATA)
    return mid_training and same_data


def resume():
    if RESUME_CKPT:
        ckpt = HERE / RESUME_CKPT
    else:
        # auto-resolve the most-recently-modified last.pt across all <RUN_NAME>* runs, so we resume the run
        # that actually just crashed (not a stale older dir with the same base name). Printed for transparency.
        cands = sorted(PROJECT.glob(f"{RUN_NAME}*/weights/last.pt"), key=lambda p: p.stat().st_mtime)
        if not cands:
            raise SystemExit(f"RESUME=True but no runs/detect/{RUN_NAME}*/weights/last.pt exists. "
                             f"Set RESUME=False to start a fresh run.")
        ckpt = cands[-1]
        print(f"[resume] auto-resolved latest run -> {ckpt}")
    if not ckpt.exists():
        raise SystemExit(f"RESUME=True but no checkpoint at {ckpt}. Set RESUME=False to start fresh.")
    if not is_resumable(ckpt):
        raise SystemExit(
            f"RESUME=True but {ckpt} is not a resumable checkpoint (it is finished/stripped, or belongs "
            f"to a different dataset). Ultralytics `resume=True` on such a file silently RESTARTS a new "
            f"run on its built-in default dataset — refusing. Start a fresh run with RESUME=False, or point RESUME_CKPT "
            f"at a genuinely interrupted last.pt.")
    print(f"[resume] official resume -> {ckpt} (weights + optimizer + epoch restored from the checkpoint)")
    model = YOLO(str(ckpt))
    model.train(resume=True)   # official: explicit checkpoint + resume=True, no other args
    return model


def train_fresh():
    print(f"[fresh] starting new run -> {RUN_DIR} (auto-increments to {RUN_NAME}2, ... if it already exists)")
    model = YOLO(MODEL)
    model.train(
        data=DATA, epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH, device=DEVICE,
        project=str(PROJECT), name=RUN_NAME, exist_ok=False,   # official default: auto-increment, never clobber a prior run
        freeze=FREEZE, lr0=LR0, lrf=LRF, cos_lr=COS_LR, patience=PATIENCE,
        optimizer=OPTIMIZER, save_period=SAVE_PERIOD, cache=CACHE,
        workers=WORKERS, seed=SEED, deterministic=True, close_mosaic=CLOSE_MOSAIC,
        plots=True,   # save results.png / PR / F1 / confusion-matrix for later analysis
    )
    return model


def main():
    model = resume() if RESUME else train_fresh()
    # Report the ACTUAL run dir Ultralytics used, not the static RUN_NAME — with exist_ok=False the real
    # dir may be train2, train3, ... on re-runs. `model.trainer.{best,last,save_dir}` are
    # the true paths.
    t = model.trainer
    print("\ntraining done.")
    print(f"  run dir      : {t.save_dir}")
    print(f"  best weights : {t.best}")
    print(f"  last weights : {t.last}")
    print(f"  results csv  : {t.save_dir / 'results.csv'}")
    print(f"\n  -> for predict.py / val.py / export.py, set WEIGHTS to:\n     {t.best}")


if __name__ == "__main__":
    main()
