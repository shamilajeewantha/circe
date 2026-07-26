"""Majority-vote merge of 04_annotate_with_gemini.py's --passes > 1 output into a single consensus
annotation. Local and free (no API calls) - reads cache/raw_gemini_pass{1..N}/<stem>.json for every
pass found on disk for each image, clusters boxes across passes by IoU (Union-Find, not an
order-dependent greedy match), majority-votes presence + class label per cluster, and writes the
winning consensus straight into cache/raw_gemini/<stem>.json - the SAME schema
05_tighten_boxes.py already reads, so stage 5 needs ZERO changes to consume multi-pass output.

Only relevant after 04_annotate_with_gemini.py was run with --passes > 1 (each pass writes to its
own cache/raw_gemini_pass{N}/ + qc/gemini_raw_pass{N}/ - see that script's docstring). Auto-detects
how many pass folders actually exist on disk (glob cache/raw_gemini_pass*/) and, per image, how
many of those actually have that image's JSON (a pass that failed/was skipped for one specific
image just lowers that image's own vote denominator) - neither needs to be told to this script.

Voting algorithm
-----------------
1. For each image, collect every box from every pass found: (pass_number, box) pairs.
2. Union-Find over all box pairs FROM DIFFERENT PASSES with IoU >= --iou-threshold (default 0.5) -
   connected components become clusters. Boxes within the SAME pass never merge with each other
   (Gemini's own single-pass output already applies a "one box per fastener, never merge" rule -
   see SYSTEM_PROMPT in 04_annotate_with_gemini.py) - only cross-pass matching happens here.
3. Per cluster: presence_count = number of DISTINCT passes contributing to it. If one pass
   contributes 2+ boxes into the same cluster (a within-pass near-duplicate that happens to land
   here) that pass still counts once - logged as a warning, first such box kept for geometry, never
   inflating the vote.
4. final_present = presence_count >= --vote-threshold (default: a real majority,
   ceil(num_passes_found / 2), computed PER IMAGE against that image's own pass count).
5. Label: plurality vote among the labels of the boxes that voted for this cluster; ties broken by
   the same bolt_corroded > bolt_defective > bolt_ok priority SYSTEM_PROMPT already uses for a
   single fastener showing multiple issues, for consistency.
6. Final box_2d: mean of the matched boxes' coordinates - the simplest defensible merge;
   05_tighten_boxes.py's SAM3 tightening pass still runs downstream regardless and refines geometry
   further, so this is a pre-tightening estimate, not the final word on box tightness.
7. Clusters below --vote-threshold are recorded (for QC - "2/5 passes suggested this, excluded")
   but excluded from the consensus annotation actually written to cache/raw_gemini/.

Usage
-----
    python 06_vote_consensus.py                              # auto-detects however many passes exist
    python 06_vote_consensus.py --vote-threshold 4 --iou-threshold 0.6
    python 06_vote_consensus.py --clean                       # free - regenerates from pass cache

Output
------
Overwrites cache/raw_gemini/<stem>.json (same {model, prompt_version, boxes} schema as
04_annotate_with_gemini.py's single-pass output, tagged with a synthetic model string like
"consensus(gemini-robotics-er-1.6-preview,5-pass,thresh=3)" and prompt_version="consensus" -
deliberately never equal to any real DEFAULT_MODEL/int PROMPT_VERSION, so if 04 is later re-run in
plain single-pass mode over the same --out, its own staleness check correctly treats a consensus
result as stale and re-annotates rather than silently trusting a coincidental match).

qc/gemini_consensus/<stem>.<ext> - vote-tally QC image (draw_consensus_visualization): each
INCLUDED box drawn with its vote tally as the label, e.g. "4/5 detected - corroded:3 defective:0
ok:1", so vote confidence is visible at a glance.
qc/gemini_consensus/<stem>.json - full per-cluster vote breakdown, INCLUDING excluded (below-
threshold) clusters, for manual inspection of what almost made the cut.
"""
import argparse
import json
import logging
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from annotate_common import BoltBox, IMG_EXTS, draw_consensus_visualization

log = logging.getLogger("vote_consensus")

# Same tie-break priority SYSTEM_PROMPT (04_annotate_with_gemini.py) already uses when a single
# fastener shows multiple issues - reused here for label-vote ties, for consistency.
LABEL_PRIORITY = {"bolt_corroded": 0, "bolt_defective": 1, "bolt_ok": 2}


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def _iou(box_a: List[int], box_b: List[int]) -> float:
    """box_a/box_b: [ymin, xmin, ymax, xmax], same 0-1000 normalized space both boxes already
    share (both came from the same image), so no unit conversion is needed to compare them."""
    ay0, ax0, ay1, ax1 = box_a
    by0, bx0, by1, bx1 = box_b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    area_a = max(0, ax1 - ax0) * max(0, ay1 - ay0)
    area_b = max(0, bx1 - bx0) * max(0, by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def vote_boxes_for_image(
    per_pass_boxes: Dict[int, List[BoltBox]], num_passes: int, iou_threshold: float,
    vote_threshold: int, image_name: str,
) -> List[dict]:
    """Core voting algorithm - see module docstring for the full spec. per_pass_boxes:
    {pass_number: [BoltBox, ...]} for whichever passes were actually found for THIS image.
    num_passes = len(per_pass_boxes), passed in explicitly rather than recomputed so it's
    unambiguous this is "passes found for this image", not "--passes 04 was run with" (they can
    differ if a pass failed/was skipped for one specific image).

    Returns a list of cluster dicts (box_2d, label, presence_count, num_passes, label_votes,
    included) for ALL clusters, both included and excluded, so QC output can show what almost made
    the cut. The caller filters by `included` for the actual consensus annotation."""
    entries: List[Tuple[int, BoltBox]] = []
    for pass_num, boxes in per_pass_boxes.items():
        for box in boxes:
            entries.append((pass_num, box))

    n = len(entries)
    uf = _UnionFind(n)
    for i in range(n):
        for j in range(i + 1, n):
            if entries[i][0] == entries[j][0]:
                continue  # never merge two boxes from the SAME pass
            if _iou(entries[i][1].box_2d, entries[j][1].box_2d) >= iou_threshold:
                uf.union(i, j)

    clusters: Dict[int, List[int]] = defaultdict(list)
    for i in range(n):
        clusters[uf.find(i)].append(i)

    results: List[dict] = []
    for idxs in clusters.values():
        seen_passes: Dict[int, BoltBox] = {}
        for i in idxs:
            pass_num, box = entries[i]
            if pass_num in seen_passes:
                log.warning("  %s: pass %d contributed more than one box to the same vote cluster "
                            "(within-pass near-duplicate) - counting that pass once, keeping the "
                            "first box for geometry.", image_name, pass_num)
                continue
            seen_passes[pass_num] = box

        presence_count = len(seen_passes)
        label_votes = Counter(box.label for box in seen_passes.values())
        max_votes = max(label_votes.values())
        tied = [label for label, count in label_votes.items() if count == max_votes]
        winning_label = min(tied, key=lambda label: LABEL_PRIORITY[label])

        boxes_2d = [box.box_2d for box in seen_passes.values()]
        avg_box = [round(sum(coord) / len(coord)) for coord in zip(*boxes_2d)]

        results.append({
            "box_2d": avg_box,
            "label": winning_label,
            "presence_count": presence_count,
            "num_passes": num_passes,
            "label_votes": dict(label_votes),
            "included": presence_count >= vote_threshold,
        })
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path, default=Path(__file__).parent / "circe_datasets" / "npu_bolt" / "working_images")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "circe_datasets" / "npu_bolt")
    parser.add_argument("--iou-threshold", type=float, default=0.5,
                         help="Minimum IoU for two boxes from DIFFERENT passes to be considered "
                              "the same real fastener and merged into one vote cluster.")
    parser.add_argument("--vote-threshold", type=int, default=None,
                         help="Minimum number of DISTINCT passes that must agree a fastener is "
                              "present for it to survive into the final consensus. Default: a real "
                              "majority, ceil(num_passes_found / 2), computed PER IMAGE - not every "
                              "image necessarily has the same number of passes found on disk.")
    parser.add_argument("--limit", type=int, default=None,
                         help="Only consider the first N images from --src this run.")
    parser.add_argument("--clean", action="store_true",
                         help="Wipe qc/gemini_consensus/ before running, then regenerate it (and "
                              "re-overwrite cache/raw_gemini/) from the existing "
                              "cache/raw_gemini_pass*/ files - FREE, no API calls, since the "
                              "expensive part (the per-pass Gemini cache) is kept.")
    args = parser.parse_args()

    src = args.src.resolve()
    out = args.out.resolve()
    raw_dir = out / "cache" / "raw_gemini"  # WRITE target - same path 05_tighten_boxes.py reads
    consensus_qc_dir = out / "qc" / "gemini_consensus"

    if args.clean:
        shutil.rmtree(consensus_qc_dir, ignore_errors=True)

    raw_dir.mkdir(parents=True, exist_ok=True)
    consensus_qc_dir.mkdir(parents=True, exist_ok=True)

    # FileHandler flushes every record as it's emitted, so the log survives a crash/Ctrl-C
    # partway through - appends to the same run_log.txt every stage writes to.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(out / "run_log.txt", encoding="utf-8"), logging.StreamHandler()],
        force=True,  # a package imported before this point may already have attached its own
                     # root-logger handler, which makes a plain basicConfig() a silent no-op -
                     # force=True always (re)configures regardless (see 03_propose_regions_concept.py
                     # for the real crash that surfaced this).
    )
    log.info("Run started. Source: %s", src)
    if args.clean:
        log.info("--clean: wiped qc/gemini_consensus/ (regenerated from cache/raw_gemini_pass*/, "
                  "free - no API calls)")

    pass_dirs = sorted(
        (out / "cache").glob("raw_gemini_pass*"),
        key=lambda p: int(p.name.replace("raw_gemini_pass", "")),
    )
    if not pass_dirs:
        raise SystemExit(f"No cache/raw_gemini_pass*/ folders found under {out} - run "
                          f"04_annotate_with_gemini.py --passes N first.")
    log.info("Found %d pass folder(s): %s", len(pass_dirs), ", ".join(d.name for d in pass_dirs))

    images = sorted(f for f in src.iterdir() if f.is_file() and f.suffix.lower() in IMG_EXTS)
    if args.limit:
        images = images[: args.limit]
    total = len(images)
    log.info("Images found: %d", total)

    processed, skipped, total_included, total_excluded = 0, 0, 0, 0

    for i, img_path in enumerate(images, start=1):
        per_pass_boxes: Dict[int, List[BoltBox]] = {}
        models_seen = set()
        for pass_dir in pass_dirs:
            pass_num = int(pass_dir.name.replace("raw_gemini_pass", ""))
            raw_path = pass_dir / f"{img_path.stem}.json"
            if not raw_path.exists():
                continue
            data = json.loads(raw_path.read_text(encoding="utf-8"))
            per_pass_boxes[pass_num] = [BoltBox(**b) for b in data["boxes"]]
            models_seen.add(data.get("model", "unknown"))

        if not per_pass_boxes:
            skipped += 1
            log.warning("[%d/%d] %s - no cached annotation in ANY pass folder; skipping.",
                        i, total, img_path.name)
            continue

        num_passes = len(per_pass_boxes)
        vote_threshold = (args.vote_threshold if args.vote_threshold is not None
                           else math.ceil(num_passes / 2))
        clusters = vote_boxes_for_image(
            per_pass_boxes, num_passes, args.iou_threshold, vote_threshold, img_path.name
        )
        included = [c for c in clusters if c["included"]]
        excluded = [c for c in clusters if not c["included"]]
        total_included += len(included)
        total_excluded += len(excluded)

        model_tag = models_seen.pop() if len(models_seen) == 1 else "+".join(sorted(models_seen))
        consensus_model = f"consensus({model_tag},{num_passes}-pass,thresh={vote_threshold})"
        (raw_dir / f"{img_path.stem}.json").write_text(
            json.dumps(
                {
                    "model": consensus_model,
                    # Deliberately a STRING, never equal to the real int PROMPT_VERSION 04 uses for
                    # its own staleness check - guarantees a consensus result always reads as
                    # stale/mismatched if 04 is later re-run in plain single-pass mode, so it
                    # correctly re-annotates rather than silently trusting a coincidental match.
                    "prompt_version": "consensus",
                    "boxes": [{"box_2d": c["box_2d"], "label": c["label"]} for c in included],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        (consensus_qc_dir / f"{img_path.stem}.json").write_text(
            json.dumps(clusters, indent=2), encoding="utf-8"
        )
        draw_consensus_visualization(img_path, clusters, consensus_qc_dir / img_path.name)

        processed += 1
        log.info("[%d/%d] %s -> %d box(es) included, %d excluded (out of %d/%d pass folder(s) "
                  "found for this image)", i, total, img_path.name, len(included), len(excluded),
                  num_passes, len(pass_dirs))

    log.info("Done. %d image(s) processed, %d skipped (no pass data found for that image), "
              "%d total boxes included, %d excluded (below vote threshold).",
              processed, skipped, total_included, total_excluded)
    log.info("Consensus written to %s. QC (vote tallies) at %s. Run 05_tighten_boxes.py next.",
              raw_dir, consensus_qc_dir)


if __name__ == "__main__":
    main()
