import json
import logging
import os
from datetime import datetime

log = logging.getLogger("vggt_omega.pipeline")


class RunLogger:
    def __init__(self, run_dir: str):
        self.run_dir = run_dir
        self.log_path = os.path.join(run_dir, "run_log.jsonl")
        os.makedirs(run_dir, exist_ok=True)

    def log_pass(self, entry: dict) -> str:
        n_old = len(entry.get("old_covisible_indices_used", []))
        ts = datetime.now().strftime("%H:%M:%S")
        msg = (
            f"[{ts}] [PASS {entry['pass']:04d}] frame={entry['frame_idx']:04d} "
            f"batch={entry.get('batch_size', '?')} "
            f"(ctx={entry.get('n_context', '?')}"
            f"[old_revisit={n_old}] new=1) "
            f"cube_cov={entry.get('cube_coverage_pct', 0):.1f}% "
            f"Δ={entry.get('delta_norm', 0):.4f} "
            f"Δres={entry.get('delta_residual_deg', 0):.2f}° "
            f"scale={entry.get('scale_estimated', 1):.4f} "
            f"inf={entry['inference_time_s']:.2f}s "
            f"GPU={entry['gpu_mem_after_GB']:.2f}GB"
        )

        # Persist the human-readable line too, so a past run can be reloaded
        # and its log replayed verbatim (see main.py _load_run).
        entry["ts"] = datetime.now().isoformat()
        entry["display"] = msg
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

        log.info(msg)
        return msg

    def info(self, msg: str) -> str:
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] [INFO] {msg}"
        with open(self.log_path, "a") as f:
            f.write(json.dumps({"info": msg, "display": line, "ts": datetime.now().isoformat()}) + "\n")
        log.info(line)
        return line
