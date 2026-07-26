# dataset_annotation/circe_datasets/

Working copies + pipeline output for the `dataset_annotation` re-annotation pipeline
(`01`/`03`/`04`/`05`/`06` + `_visualize_original_annotations.py`). Everything under here is
**regenerable** and gitignored (except this README) — never hand-edit anything in this tree, and
never treat it as source.

One subfolder per dataset name, mirroring `../datasets/<name>/` (the read-only source layout).
`npu_bolt` is the first; more will be added the same way as this pipeline is pointed at other
NPU-BOLT-style datasets.

```
circe_datasets/
  <dataset_name>/
    working_images/    # CAD-*-excluded working copy of ../datasets/<name>/images/, built by
                        # 01_remove_cad.py. The read-only source is never written to directly -
                        # this is the only copy every downstream stage (03/04/05/06) reads from.
    cache/              # per-image JSON, the source of truth everything else regenerates from
      sam_regions_concept/
      raw_gemini/                  # single-pass (--passes 1, the default)
      raw_gemini_pass{1..N}/       # multi-pass (--passes N > 1) - one folder per independent pass
    qc/                 # human-viewable visualizations, all disposable
      sam_proposals_concept/
      sam_proposals_concept_for_gemini/
      gemini_raw/
      gemini_raw_pass{1..N}/
      gemini_consensus/            # 06_vote_consensus.py's majority-vote QC (vote tallies drawn per box)
      visualized/                  # 05_tighten_boxes.py's final QC
      sam_regions_debug/
      original_labels/             # _visualize_original_annotations.py's output - the dataset's OWN
                                    # original XML ground truth drawn as-is, for comparison only
    dataset/             # final YOLO-format output (images/ + labels/ + data.yaml)
    run_log.txt / cleanup_cad_log.txt
```

**Regenerate:** run the numbered pipeline in order against `../datasets/<name>/` - see
`../README.md` for the full pipeline diagram and each stage's flags.
