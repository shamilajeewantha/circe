"""Plot training curves from a YOLO run's results.csv.

Ultralytics already saves results.png automatically; this is a cleaner
custom version (losses on one axis, mAP on another).

Run:  python plot_results.py
"""
from pathlib import Path
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- settings (edit these) --------------------------------------
RUN  = "runs/merged_full"     # folder holding results.csv
OUT  = "loss_curve.png"       # saved inside RUN
# -----------------------------------------------------------------

HERE = Path(__file__).parent


def main():
    run = HERE / RUN
    csv_path = run / "results.csv"
    rows = list(csv.DictReader(csv_path.open()))
    col = {k.strip(): [float(r[k]) for r in rows] for k in rows[0]}
    ep = col["epoch"]

    fig, ax1 = plt.subplots(figsize=(10, 5))

    # losses (left axis)
    for key, style in [
        ("train/box_loss", "-"), ("train/cls_loss", "-"), ("train/dfl_loss", "-"),
        ("val/box_loss", "--"), ("val/cls_loss", "--"), ("val/dfl_loss", "--"),
    ]:
        if key in col:
            ax1.plot(ep, col[key], style, label=key)
    ax1.set_xlabel("epoch"); ax1.set_ylabel("loss"); ax1.legend(loc="upper left", fontsize=8)

    # metrics (right axis)
    ax2 = ax1.twinx()
    for key in ("metrics/mAP50(B)", "metrics/mAP50-95(B)"):
        if key in col:
            ax2.plot(ep, col[key], ":", linewidth=2, label=key)
    ax2.set_ylabel("mAP"); ax2.set_ylim(0, 1); ax2.legend(loc="lower right", fontsize=8)

    plt.title(f"Training curves — {run.name}")
    plt.tight_layout()
    out = run / OUT
    plt.savefig(out, dpi=120)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
