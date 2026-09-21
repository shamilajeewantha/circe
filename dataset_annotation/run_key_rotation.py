"""Rotates GEMINI_API_KEY_1..4 (from .env) across repeated invocations of
04_annotate_with_gemini.py for one --src/--out/--passes job, so a job bigger than one key's
daily RPD can still finish using multiple Google accounts' separate quotas.

Why a wrapper instead of a code change to 04_annotate_with_gemini.py: that script already
swallows every API error INSIDE call_gemini_batch (broad except, logged, never re-raised - see
its docstring) and keeps marching through the rest of its chunks even after a key's quota is
exhausted, rather than stopping early. So there's no exception to catch here to detect
"this key is out of quota" - detection instead means: run the whole job with a key, then check
how many images are STILL not cached across every pass dir, and switch to the next key only if
that leaves anything unfinished. Per-image caching (unchanged in the target script) means a
second key's run automatically skips everything the first key already finished.

Usage
-----
    python run_key_rotation.py --src circe_datasets/npu_bolt/working_images \\
        --out circe_datasets/npu_bolt --passes 3 --model gemini-robotics-er-2-preview
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from dotenv import dotenv_values

HERE = Path(__file__).parent
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def pending_count(src: Path, out: Path, passes: int, model: str, prompt_version: int) -> int:
    """How many images in `src` are NOT yet cached under `model`+`prompt_version`, across every
    pass dir 1..passes (or cache/raw_gemini/ if passes==1) - mirrors the cache-hit check inside
    04_annotate_with_gemini.py's run_annotation_pass, so this wrapper's notion of "done" matches
    the target script's own."""
    images = sorted(f for f in src.iterdir() if f.is_file() and f.suffix.lower() in IMG_EXTS)
    if passes == 1:
        raw_dirs = [out / "cache" / "raw_gemini"]
    else:
        raw_dirs = [out / "cache" / f"raw_gemini_pass{p}" for p in range(1, passes + 1)]
    pending = 0
    for img in images:
        for raw_dir in raw_dirs:
            raw_path = raw_dir / f"{img.stem}.json"
            if not raw_path.exists():
                pending += 1
                break
            data = json.loads(raw_path.read_text(encoding="utf-8"))
            if data.get("model") != model or data.get("prompt_version") != prompt_version:
                pending += 1
                break
    return pending


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--passes", type=int, default=1)
    parser.add_argument("--model", type=str, default="gemini-robotics-er-2-preview")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--prompt-version", type=int, default=4,
                         help="Must match PROMPT_VERSION in 04_annotate_with_gemini.py (currently "
                              "4) - kept as a plain flag rather than importing that module by a "
                              "filename starting with a digit, which needs importlib.util, not a "
                              "normal import.")
    args = parser.parse_args()

    src = args.src.resolve()
    out = args.out.resolve()
    prompt_version = args.prompt_version

    keys_env = dotenv_values(HERE / ".env")
    keys = [(name, val) for name, val in keys_env.items()
            if name.startswith("GEMINI_API_KEY_") and val]
    keys.sort(key=lambda kv: kv[0])
    if not keys:
        print("No GEMINI_API_KEY_N entries found in .env - nothing to rotate.", file=sys.stderr)
        sys.exit(1)

    total_images = len([f for f in src.iterdir() if f.is_file() and f.suffix.lower() in IMG_EXTS])
    print(f"Job: {src} -> {out}, passes={args.passes}, model={args.model}, "
          f"{total_images} image(s) total, {len(keys)} key(s) available for rotation.")

    for key_name, key_val in keys:
        remaining = pending_count(src, out, args.passes, args.model, prompt_version)
        if remaining == 0:
            print(f"[{key_name}] Nothing pending - job already fully annotated. Stopping rotation.")
            return
        print(f"[{key_name}] {remaining} image(s) still pending across {args.passes} pass(es) - "
              f"running with this key now.")

        cmd = [
            sys.executable, str(HERE / "04_annotate_with_gemini.py"),
            "--src", str(src), "--out", str(out),
            "--passes", str(args.passes), "--model", args.model,
            "--retries", str(args.retries),
        ]
        if args.limit:
            cmd += ["--limit", str(args.limit)]

        env = os.environ.copy()
        env["GEMINI_API_KEY"] = key_val
        result = subprocess.run(cmd, env=env, cwd=str(HERE))
        print(f"[{key_name}] invocation exited with code {result.returncode}.")

    remaining = pending_count(src, out, args.passes, args.model, prompt_version)
    if remaining == 0:
        print("All keys tried (or job finished early) - job fully annotated.")
    else:
        print(f"All {len(keys)} key(s) tried - {remaining} image(s) still pending "
              f"(likely today's combined quota exhausted). Re-run tomorrow to continue.")


if __name__ == "__main__":
    main()
