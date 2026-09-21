"""AUTONOMOUS thermal/disk kill-switch for a long training run. Takes action ITSELF - it does not
notify anyone and does not need Claude, a chat session, or a human to be awake.

Why this exists: the other two layers both ultimately depend on something outside the machine
staying alive. monitor_training.py only writes rows to a log nobody reads. alert_watch.sh raises an
alarm, but the alarm goes to a Claude session that may be busy, closed, or gone. A cron check only
fires while that session is idle. NONE of those protect the hardware at 3am with the laptop
unattended. This does: it polls the GPU directly and KILLS the training process if the machine is
in danger, then exits.

Killing training is cheap here: train.py checkpoints every epoch (SAVE_PERIOD=1), so a kill costs at
most the current epoch and the run resumes with RESUME=True.

Thresholds: WARN_C only logs. KILL_C must be sustained for KILL_STREAK consecutive polls before
acting, so a single spurious reading never aborts an 18-hour run. NOTE: NVIDIA GPUs also throttle
and shut down in firmware well before damage - this is defence in depth, not the only protection.

Usage - run it from WSL so it can signal the training process. --pid is REQUIRED (this guard
never pattern-matches a target; see kill_training):
    python thermal_guard.py --pid $(pgrep -f 'python train.py' | head -1)
    python thermal_guard.py --pid 1234 --warn-c 75 --kill-c 80 --poll 30
    python thermal_guard.py --pid 1234 --kill-c 1 --dry-run   # test thresholds, kills nothing

Defaults: warn 75C, kill 80C, disk floor 3GB, poll 30s, kill-streak 3. Both temp defaults were chosen
by the user against this run's measured temperature distribution - see the --warn-c/--kill-c help text.
"""
import argparse
import subprocess
import time
from datetime import datetime
from pathlib import Path

LOG = Path(__file__).parent / "reports" / "thermal_guard.log"


def log(msg: str) -> None:
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S}  {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def gpu_temp():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20,
        ).stdout.strip().splitlines()[0]
        return int(out.strip())
    except Exception as e:
        log(f"WARN could not read GPU temp: {e}")
        return None


def disk_free_gb(path="/mnt/d"):
    try:
        out = subprocess.run(["df", "-BG", "--output=avail", path],
                             capture_output=True, text=True, timeout=20).stdout.strip().splitlines()[-1]
        return int(out.replace("G", "").strip())
    except Exception as e:
        log(f"WARN could not read disk free: {e}")
        return None


def kill_training(reason: str, pid: int, dry_run: bool) -> None:
    """Kill ONE explicitly-targeted PID. Never pattern-matches.

    This used to `pkill -f train.py`, which is how this guard killed the very run it was meant to
    protect: a threshold TEST fired the kill path, the pattern matched the live training process,
    and 17 epochs of work were SIGTERMed. A guard that can take down an unrelated process on a bad
    pattern is more dangerous than the thing it guards against - so the target is now a PID the
    caller must pass in explicitly, and --dry-run exercises the whole path while killing nothing.
    """
    if dry_run:
        log(f"[DRY-RUN] would kill pid {pid}: {reason} (no signal sent)")
        return
    log(f"!!! KILLING TRAINING pid {pid}: {reason}")
    try:
        subprocess.run(["kill", "-TERM", str(pid)], timeout=20)
    except Exception as e:
        log(f"  kill {pid} failed: {e}")
    log("!!! training killed. Resume with RESUME=True in train.py (loses at most one epoch).")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--warn-c", type=int, default=75,
                    help="Log a warning above this. USER-CHOSEN 75C sits just under this run's measured p95 "
                         "(76C, max 78C over 52 samples), so it flags the hot tail of normal operation - "
                         "early notice while there is still headroom before the 80C kill.")
    ap.add_argument("--kill-c", type=int, default=80,
                    help="Kill above this (sustained). Grounded in this GPU's OWN reported limits, "
                         "not a guess: nvidia-smi reports Slowdown at ~2C above the ~86C limit "
                         "(~88-89C) and Shutdown ~12C above it (~98C). USER-CHOSEN 80C is far BELOW that "
                         "limit - a deliberately conservative ceiling to keep the laptop cool rather than to "
                         "avoid damage. Note this sits only ~2C above the 78C max this run actually reaches, "
                         "so a hot epoch WILL trip it; that is accepted, and training resumes from the "
                         "last checkpoint. NOTE the card protects itself in firmware regardless - this guard "
                         "exists to stop prolonged hot running, not to prevent a meltdown.")
    ap.add_argument("--kill-streak", type=int, default=3,
                    help="consecutive over-KILL_C polls required before killing (debounce)")
    ap.add_argument("--disk-floor-gb", type=int, default=3)
    ap.add_argument("--poll", type=int, default=30, help="seconds between polls")
    ap.add_argument("--pid", type=int, required=True,
                    help="PID of the training process to kill. REQUIRED and explicit - this guard "
                         "never pattern-matches for a target (see kill_training's docstring).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Exercise every code path but never send a signal. USE THIS to test "
                         "thresholds - testing with real thresholds against a live run is how this "
                         "script once killed 17 epochs of training.")
    args = ap.parse_args()

    log(f"thermal_guard ARMED{' [DRY-RUN]' if args.dry_run else ''}: target pid {args.pid}, "
        f"warn>{args.warn_c}C, KILL>{args.kill_c}C x{args.kill_streak} polls, "
        f"disk floor {args.disk_floor_gb}GB, every {args.poll}s")

    streak = 0
    while True:
        t = gpu_temp()
        d = disk_free_gb()

        if t is not None:
            if t > args.kill_c:
                streak += 1
                log(f"CRITICAL GPU {t}C (>{args.kill_c}C) streak {streak}/{args.kill_streak}")
                if streak >= args.kill_streak:
                    kill_training(f"GPU {t}C sustained over {args.kill_c}C", args.pid, args.dry_run)
                    return
            else:
                if streak:
                    log(f"GPU back to {t}C - streak reset")
                streak = 0
                if t > args.warn_c:
                    log(f"WARN GPU {t}C (>{args.warn_c}C)")

        if d is not None and d < args.disk_floor_gb:
            kill_training(f"disk free {d}GB below floor {args.disk_floor_gb}GB", args.pid, args.dry_run)
            return

        time.sleep(args.poll)


if __name__ == "__main__":
    main()
