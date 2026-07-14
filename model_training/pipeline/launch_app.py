"""Open the FiftyOne App for interactive human review of the curated dataset.

Browse/filter by `source`, `duplicate`, `background` tags (and `review` if
mistakenness was enabled). Nothing is modified — this is for eyeballing the
cleanup before/after export.

Run in `drone_detect`:  python pipeline/launch_app.py
(Then open the printed URL; press Ctrl-C to stop.)
"""
import fiftyone as fo

from config import FO_DATASET_NAME


def main():
    dataset = fo.load_dataset(FO_DATASET_NAME)
    print(f"{FO_DATASET_NAME}: {len(dataset)} samples")
    print("tags:", dataset.count_sample_tags())
    session = fo.launch_app(dataset)
    session.wait()   # blocks until you close it


if __name__ == "__main__":
    main()
