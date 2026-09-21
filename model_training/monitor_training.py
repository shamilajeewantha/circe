"""Background watchdog for a long training run: logs GPU temp/util/VRAM, free disk on both
filesystems, and training progress every INTERVAL seconds, and shouts when anything crosses a
threshold.

Why this exists: a 100-epoch run on this laptop is ~18 h of SUSTAINED GPU load - unlike the
network-bound annotation runs, which idled the GPU at 41-74 C. Two things can kill a run that long
silently: thermal throttling/shutdown, and either filesystem filling up (D: has run to 100% full in
this project before, and checkpoints land there).

Runs from Windows (reads nvidia-smi + WSL df via `wsl`), independent of the training process, so
killing/restarting training never disturbs it and vice versa.

Usage
-----
    python monitor_training.py --run-dir runs/detect/train
    python monitor_training.py --run-dir runs/detect/train --interval 600 --temp-warn 85

Output: <run-dir>/monitor.log - one CSV row per sample, plus WARNING lines (also echoed to stdout).
"""
import argparse
import csv
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).parent

FIELDS = ["timestamp", "gpu_temp_c", "gpu_util_pct", "vram_used_mb", "vram_total_mb",
          "disk_d_free_gb", "wsl_free_gb", "last_epoch", "last_map50"]


def nvidia_smi():
    """(temp, util, vram_used, vram_total) from nvidia-smi; None fields if unavailable."""
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=temperature.gpu,utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30,
        ).stdout.strip().splitlines()[0]
        return [p.strip() for p in out.split(",")]
    except Exception as e:
        print(f"  nvidia-smi read failed: {e}")
        return ["", "", "", ""]


def wsl_free_gb():
    """Free GB on the WSL filesystem, where the training dataset copy lives."""
    try:
        out = subprocess.run(
            ["wsl", "-e", "bash", "-lc", "df -BG --output=avail /home | tail -1"],
            capture_output=True, text=True, timeout=60,
        ).stdout.strip()
        return out.replace("G", "").strip()
    except Exception as e:
        print(f"  wsl df read failed: {e}")
        return ""


def training_progress(run_dir: Path):
    """(last_epoch, last_mAP50) from Ultralytics' own results.csv - '' before the first epoch lands."""
    results = run_dir / "results.csv"
    if not results.exists():
        return "", ""
    try:
        rows = [r for r in csv.DictReader(results.open()) if r.get("epoch")]
        if not rows:
            return "", ""
        last = rows[-1]
        key = next((k for k in last if "mAP50(B)" in k and "95" not in k), None)
        return last["epoch"].strip(), (last[key].strip() if key else "")
    except Exception as e:
        print(f"  results.csv read failed: {e}")
        return "", ""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, required=True,
                    help="Ultralytics run dir, e.g. runs/detect/train (monitor.log is written here)")
    ap.add_argument("--interval", type=int, default=600, help="Seconds between samples (default 600 = 10 min)")
    ap.add_argument("--temp-warn", type=int, default=85,
                    help="WARN above this GPU temp in C. Conservative default: laptop GPUs generally "
                         "throttle in the mid-to-high 80s; this is not a vendor-published number for "
                         "this exact card.")
    ap.add_argument("--disk-warn-gb", type=int, default=5, help="WARN when either filesystem drops below this")
    args = ap.parse_args()

    run_dir = (HERE / args.run_dir).resolve() if not args.run_dir.is_absolute() else args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "monitor.log"
    new_file = not log_path.exists()

    print(f"Monitoring -> {log_path} every {args.interval}s "
          f"(warn: >{args.temp_warn}C, <{args.disk_warn_gb}GB free)")

    with log_path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if new_file:
            writer.writerow(FIELDS)
            fh.flush()

        while True:
            temp, util, vram_used, vram_total = nvidia_smi()
            d_free_gb = round(shutil.disk_usage("D:\\").free / 1024**3, 1)
            wsl_free = wsl_free_gb()
            epoch, map50 = training_progress(run_dir)
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            writer.writerow([ts, temp, util, vram_used, vram_total, d_free_gb, wsl_free, epoch, map50])
            fh.flush()   # flush every sample - a crash/power loss must not lose the history
            print(f"{ts}  gpu={temp}C util={util}% vram={vram_used}/{vram_total}MB  "
                  f"D:={d_free_gb}GB wsl={wsl_free}GB  epoch={epoch} mAP50={map50}")

            if temp and temp.isdigit() and int(temp) > args.temp_warn:
                msg = f"{ts}  WARNING: GPU temp {temp}C exceeds {args.temp_warn}C"
                print(msg); writer.writerow([msg]); fh.flush()
            if d_free_gb < args.disk_warn_gb:
                msg = f"{ts}  WARNING: D: free {d_free_gb}GB below {args.disk_warn_gb}GB"
                print(msg); writer.writerow([msg]); fh.flush()
            if wsl_free and wsl_free.isdigit() and int(wsl_free) < args.disk_warn_gb:
                msg = f"{ts}  WARNING: WSL free {wsl_free}GB below {args.disk_warn_gb}GB"
                print(msg); writer.writerow([msg]); fh.flush()

            time.sleep(args.interval)


if __name__ == "__main__":
    main()
