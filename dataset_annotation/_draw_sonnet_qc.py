"""One-off script: draw QC visualizations for annotations/cache/raw_sonnet/*.json (Claude's own
model predictions, kept in a separate folder from cache/raw_gemini/ per explicit instruction) into
annotations/qc/sonnet_raw/. Reuses draw_visualization from annotate_common.py - same drawing code
already used for qc/gemini_raw/, so the two are visually comparable at a glance. Not part of the
numbered pipeline - a manual comparison aid only, run directly, not via WSL (no torch/sam/genai
deps needed for this)."""
import json
from pathlib import Path

from annotate_common import BoltAnnotation, BoltBox, draw_visualization

base = Path(__file__).parent
raw_dir = base / "annotations" / "cache" / "raw_sonnet"
src_dir = base / "npu_bolt"
qc_dir = base / "annotations" / "qc" / "sonnet_raw"
qc_dir.mkdir(parents=True, exist_ok=True)

for json_path in sorted(raw_dir.glob("*.json")):
    stem = json_path.stem
    data = json.loads(json_path.read_text(encoding="utf-8"))
    ann = BoltAnnotation(boxes=[BoltBox(**b) for b in data["boxes"]])
    matches = list(src_dir.glob(f"{stem}.*"))
    if not matches:
        print(f"SKIP {stem}: no source image found in npu_bolt/")
        continue
    img_path = matches[0]
    out_path = qc_dir / img_path.name
    draw_visualization(img_path, ann, out_path)
    print(f"{stem}: {len(ann.boxes)} box(es) -> {out_path}")

print(f"Done. QC images written to {qc_dir}")
