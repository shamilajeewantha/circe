"""Packages one dataset's dataset/ + qc/visualized/ into a single, self-contained, checksummed
zip under circe_datasets/exports/ - a frozen snapshot for handing off, not the live pipeline
output (which stays untouched in place). See circe_datasets/exports/README.md for what's
included/excluded and why.

Usage
-----
    python build_dataset_export.py --name npu_bolt
    python build_dataset_export.py --name bolt-defects__ext-sdnet2025
"""
import argparse
import hashlib
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).parent


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True, help="Dataset folder name under circe_datasets/")
    args = parser.parse_args()

    ds_dir = HERE / "circe_datasets" / args.name
    dataset_dir = ds_dir / "dataset"
    visualized_dir = ds_dir / "qc" / "visualized"
    exports_dir = HERE / "circe_datasets" / "exports"
    exports_dir.mkdir(parents=True, exist_ok=True)

    staging = exports_dir / f"_staging_{args.name}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    print(f"[{args.name}] copying images/ + labels/ ...")
    shutil.copytree(dataset_dir / "images", staging / "images")
    shutil.copytree(dataset_dir / "labels", staging / "labels")
    print(f"[{args.name}] copying qc/visualized/ -> visualized/ ...")
    shutil.copytree(visualized_dir, staging / "visualized")

    n_images = len(list((staging / "images").iterdir()))
    n_labels = len(list((staging / "labels").iterdir()))
    n_boxes = sum(
        1 for lbl in (staging / "labels").iterdir() for _ in lbl.read_text(encoding="utf-8").splitlines()
    )

    data_yaml_src = (dataset_dir / "data.yaml").read_text(encoding="utf-8")
    data_yaml_lines = [
        line for line in data_yaml_src.splitlines()
        if not line.strip().startswith("path:")
    ]
    (staging / "data.yaml").write_text("\n".join(data_yaml_lines) + "\n", encoding="utf-8", newline="")

    readme = f"""# {args.name} - bolt-defect dataset (frozen export)

Generated {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")} by build_dataset_export.py.
This is a snapshot, not the live pipeline output - re-running the annotation pipeline does not
change this archive.

## Contents

- `data.yaml` - class list (nc=3: bolt_ok, bolt_defective, bolt_corroded). `path:` removed from the
  original so this resolves relative to wherever this archive is extracted, not the source repo.
- `images/` - {n_images} source images.
- `labels/` - {n_labels} YOLO-format label files (one per image, empty file = no fasteners found),
  {n_boxes} total box annotations.
- `visualized/` - the same images with final (SAM3-tightened) boxes drawn on, for human QC.
- `checksums.sha256` - one line per file above; verify with `sha256sum -c checksums.sha256` to
  confirm nothing in this archive was edited after creation.

## Annotation method

Gemini (`gemini-robotics-er-2-preview`), 3 independent passes per image, merged by majority vote
(box included only if >=2/3 passes agree, `06_vote_consensus.py`), then boxes were tightened to
the fastener's actual pixel boundary via SAM3 (`05_tighten_boxes.py`). No train/val/test split -
that happens later if/when this feeds into model_training/pipeline.
"""
    (staging / "README.md").write_text(readme, encoding="utf-8", newline="")

    print(f"[{args.name}] computing checksums ...")
    checksum_lines = []
    for f in sorted(staging.rglob("*")):
        if f.is_file():
            checksum_lines.append(f"{sha256_of(f)}  {f.relative_to(staging).as_posix()}")
    (staging / "checksums.sha256").write_text(
        "\n".join(checksum_lines) + "\n", encoding="utf-8", newline=""
    )

    zip_path = exports_dir / f"{args.name}_dataset.zip"
    print(f"[{args.name}] writing {zip_path} ...")
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(staging.rglob("*")):
            if f.is_file():
                zf.write(f, f.relative_to(staging))

    shutil.rmtree(staging)
    size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"[{args.name}] Done. {zip_path} ({size_mb:.1f} MB), {n_images} images, "
          f"{n_labels} labels, {n_boxes} boxes.")


if __name__ == "__main__":
    main()
