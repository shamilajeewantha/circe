"""Step 2 — curation: duplicate detection + embedding domain map + composition stats.

Two duplicate passes, both only TAG (nothing is deleted here; 03_export drops the tag):
  1. exact-duplicate  — MD5 content hash, catches byte-identical copies instantly.
  2. near-duplicate    — mobilenet-v2 image embeddings + fiftyone.brain similarity, catches
                         visually-redundant frames (e.g. adjacent video frames) above DUP_THRESH.

Then a UMAP visualization of the same embeddings is computed to expose domain gaps
(source/class clusters) for the report.

Writes:
  - reports/stats.json      (per-source / per-class counts, splits, drops, n_duplicates)
  - reports/embedding.json  (2-D UMAP points + source + primary class, for the report's domain map)
Appends every tagged duplicate to reports/dropped_images.csv with a reason.

Embeddings / similarity / visualization are cached as FiftyOne brain runs, so re-runs skip the
heavy recompute unless FORCE_EMBED is set. Runs in `drone_detect` (needs torch + umap-learn):
    python pipeline/02_curate.py
"""
import hashlib
import json
import logging
from collections import Counter, defaultdict

import fiftyone as fo
import fiftyone.brain as fob
import fiftyone.zoo as foz

from config import (CLASS_MAP, DROP_CLASS_REASON, DUP_THRESH, EMBED_MODEL,
                    FO_DATASET_NAME, REPORTS_DIR, SPLITS_ON_DISK, TAXONOMY)
from _util import DropLog, iter_split_images, read_yolo, source_names

logger = logging.getLogger(__name__)

EMBED_FIELD = "embedding"
SIM_KEY = "img_sim"
VIZ_KEY = "img_viz"
UMAP_FIELD = "umap"
FORCE_EMBED = False   # set True to recompute embeddings/similarity/visualization from scratch


def file_hash(filepath, chunk_size=1 << 20):
    """Returns the MD5 hex digest of a file, or None if it cannot be read."""
    h = hashlib.md5()
    try:
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                h.update(chunk)
    except OSError as e:
        logger.warning("could not hash %s: %s", filepath, e)
        return None
    return h.hexdigest()


def compute_dropped_boxes():
    """Counts boxes dropped per (source, reason) by re-reading sources — for the report."""
    dropped = Counter()
    for folder, class_map in CLASS_MAP.items():
        for split in SPLITS_ON_DISK:
            for _, lbl in iter_split_images(folder, split):
                for cid, *_ in read_yolo(lbl):
                    if class_map.get(cid) is None:
                        reason = DROP_CLASS_REASON.get((folder, cid), "dropped class")
                        dropped[f"{folder} :: {reason}"] += 1
    return dict(dropped)


def primary_label(labels):
    """Returns the most common ground-truth label in a list, or 'background'."""
    if not labels:
        return "background"
    return Counter(labels).most_common(1)[0][0]


def tag_exact_duplicates(dataset, drops):
    """MD5-hash every image; tag byte-identical copies 'duplicate'. Returns the set of tagged ids."""
    print(f"exact-duplicate pass: hashing {dataset.count()} images ...")
    seen, dup_ids = {}, []
    for sample in dataset.select_fields(["filepath", "source"]):
        digest = file_hash(sample.filepath)
        if digest is None:
            continue
        if digest in seen:
            dup_ids.append(sample.id)
            drops.add(sample.filepath, sample.source, "curate:exact-duplicate",
                      f"byte-identical to {seen[digest]}")
        else:
            seen[digest] = sample.filepath
    if dup_ids:
        dataset.select(dup_ids).tag_samples("duplicate")
    print(f"  tagged {len(dup_ids)} exact-duplicate images")
    return set(dup_ids)


def compute_embeddings_and_viz(dataset):
    """Compute (or reuse cached) mobilenet embeddings, similarity index, and UMAP points."""
    brains = set(dataset.list_brain_runs())
    have = (not FORCE_EMBED and EMBED_FIELD in dataset.get_field_schema()
            and SIM_KEY in brains and VIZ_KEY in brains)
    if have:
        print("embeddings/similarity/UMAP already cached — reusing")
        return
    if FORCE_EMBED:
        for k in (SIM_KEY, VIZ_KEY):
            if k in brains:
                dataset.delete_brain_run(k)

    print(f"loading zoo model {EMBED_MODEL} ...")
    model = foz.load_zoo_model(EMBED_MODEL)

    if FORCE_EMBED or EMBED_FIELD not in dataset.get_field_schema():
        print(f"computing embeddings into '{EMBED_FIELD}' (num_workers=0) — this is the slow step ...")
        dataset.compute_embeddings(model, embeddings_field=EMBED_FIELD,
                                   num_workers=0, skip_failures=True)

    print("building similarity index for near-duplicate detection ...")
    fob.compute_similarity(dataset, embeddings=EMBED_FIELD, brain_key=SIM_KEY)

    print("computing UMAP visualization ...")
    fob.compute_visualization(dataset, embeddings=EMBED_FIELD, method="umap", num_dims=2,
                              points_field=UMAP_FIELD, brain_key=VIZ_KEY,
                              num_workers=0, skip_failures=True)


def tag_near_duplicates(dataset, drops, already):
    """Use the similarity index to tag near-duplicate images above DUP_THRESH."""
    sim = dataset.load_brain_results(SIM_KEY)
    sim.find_duplicates(thresh=DUP_THRESH)
    dup_ids = [i for i in sim.duplicate_ids if i not in already]
    if dup_ids:
        view = dataset.select(dup_ids)
        for sample in view.select_fields(["filepath", "source"]):
            drops.add(sample.filepath, sample.source, "curate:near-duplicate",
                      f"embedding distance < {DUP_THRESH} to a kept image")
        view.tag_samples("duplicate")
    print(f"  tagged {len(dup_ids)} additional near-duplicate images")
    return set(dup_ids)


def write_embedding_json(dataset):
    """Write 2-D UMAP points (+ source, primary class) for kept (non-duplicate) samples."""
    kept = dataset.match_tags("duplicate", bool=False)
    ids, sources, points, det_labels = kept.values(
        ["id", "source", UMAP_FIELD, "ground_truth.detections.label"])
    x, y, src, primary = [], [], [], []
    for pt, s, labels in zip(points, sources, det_labels):
        if pt is None:
            continue
        x.append(float(pt[0])); y.append(float(pt[1]))
        src.append(s); primary.append(primary_label(labels))
    emb = {"x": x, "y": y, "source": src, "primary": primary, "n": len(x)}
    (REPORTS_DIR / "embedding.json").write_text(json.dumps(emb), encoding="utf-8")
    print(f"wrote {REPORTS_DIR / 'embedding.json'} ({len(x)} points)")


def write_stats(dataset, n_dup):
    """Per-source / per-class composition + splits + dropped boxes -> reports/stats.json."""
    src_cls = defaultdict(Counter)
    split_counts, bg_counts = Counter(), Counter()
    for sample in dataset.select_fields(
            ["source", "orig_split", "tags", "ground_truth"]):
        src_cls[sample.source].update(d.label for d in sample.ground_truth.detections)
        split_counts[sample.orig_split] += 1
        if "background" in sample.tags:
            bg_counts[sample.orig_split] += 1

    class_totals = Counter()
    for counter in src_cls.values():
        class_totals.update(counter)

    stats = {
        # split-count sum is an exact aggregation; len()/count() can be a stale Mongo estimate.
        "n_samples": sum(split_counts.values()),
        "taxonomy": TAXONOMY,
        "class_totals": {k: class_totals.get(k, 0) for k in TAXONOMY},
        "per_source_class": {s: dict(c) for s, c in src_cls.items()},
        "orig_split_counts": dict(split_counts),
        "background_counts": dict(bg_counts),
        "n_duplicates": n_dup,
        "dropped_boxes": compute_dropped_boxes(),
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print("\nclass totals:", stats["class_totals"])
    print(f"wrote {REPORTS_DIR / 'stats.json'}")


def main():
    dataset = fo.load_dataset(FO_DATASET_NAME)
    dataset.untag_samples("duplicate")   # idempotent re-run

    drops = DropLog(reset=False)
    exact = tag_exact_duplicates(dataset, drops)
    compute_embeddings_and_viz(dataset)
    near = tag_near_duplicates(dataset, drops, exact)
    drops.close()

    n_dup = len(exact | near)
    print(f"total duplicate-tagged: {n_dup} ({len(exact)} exact + {len(near)} near)")

    write_embedding_json(dataset)
    write_stats(dataset, n_dup)


if __name__ == "__main__":
    main()
