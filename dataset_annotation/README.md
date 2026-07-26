# dataset_annotation

Re-annotates the NPU-BOLT dataset (`datasets/npu_bolt/`, read-only) into a clean 3-class YOLO-format dataset using
Gemini, because the classes NPU-BOLT ships with (bolt head / bolt side / bolt nut / blur bolt)
are bolt-*localization* categories, not defect states - not usable for a defect-detection robot.

## Classes annotated

Exactly 3 - no more, no less (see `SYSTEM_PROMPT` in `04_annotate_with_gemini.py` for the full
wording Gemini is given):

| class | meaning |
|---|---|
| `bolt_ok` | fastener present, undamaged, fully seated, no visible defect, no corrosion |
| `bolt_defective` | a non-corrosion mechanical problem: rotation, protrusion, backing-out, a gap from the mating surface, OR the head/thread/body is visibly bent, sheared, cracked, or stripped |
| `bolt_corroded` | visible rust, pitting, or corrosion discoloration - wins the tie-break over `bolt_defective` even if the fastener is also loose/damaged (see priority below) |

**`bolt_defective` merges what were originally two separate classes, `bolt_loose` and
`bolt_damaged`.** Dropped that split: both symptoms boil down to "the fastener doesn't sit flush,"
for different root causes, and distinguishing them from a single static photo is an unreliable call
for both Gemini and a nano-capacity YOLO model to make consistently - training on noisy,
overlapping ground truth hurts more than collapsing the distinction. See conversation history for
the full reasoning.

**Deliberately excluded**: a "missing bolt" / empty-hole class. Decided against it this round -
an empty hole has no consistent local visual signature an object detector can learn (it's
context-dependent on whether a bolt *should* be there), and the robot this feeds is an
explore/inspection robot with no prior reference image of any given site to diff against. See
conversation history for the full reasoning; revisit only with a proper localize-the-fastener-site
-first architecture, not by just adding a 4th label.

If a fastener shows multiple issues at once, Gemini is instructed to pick one label by priority:
**`bolt_corroded` > `bolt_defective` > `bolt_ok`**. Corrosion wins the tie-break because it's
usually the more visually-certain call, and it's frequently the *root cause* of a mechanical
symptom (a corroded bolt needs replacement, not just tightening - labeling it `bolt_defective`
would imply the wrong fix). Fasteners Gemini can't classify confidently are skipped rather than
guessed at.

### Box scope (what exactly gets boxed)

A fastener isn't one blob - it's a head, a nut, a shank, and a thread, and "bolt" is ambiguous
between them. The rule Gemini is given: box the **head/cap** (the part a wrench or socket turns -
a bolt's hex head, or a nut). If a length of exposed shank/thread is visibly part of that *same*
fastener (e.g. it's backed out and sticking up), that goes in the *same* box, not a separate one.
A bare shank/thread with **no** head or nut visible anywhere in frame doesn't get boxed at all -
which fastener it belongs to would be a guess, same reasoning as the excluded missing-bolt case.

## Pipeline: four staged scripts

Annotation is split across four scripts, each owning one cost/resource concern, so nothing
expensive is ever re-run just to test a downstream stage:

| stage | script | package | costs | what it does |
|---|---|---|---|---|
| 1 | `02_propose_regions_dumb.py` | official `sam2` | local/free | SAM2 promptless/automatic region-proposal pass |
| 2 | `03_propose_regions_concept.py` | official `sam3` | local/free | SAM3 text-prompted concept region-proposal pass |
| 3 | `04_annotate_with_gemini.py` | Gemini API | **real quota** | the one Gemini batch call |
| 4 | `05_tighten_boxes.py` | official `sam3` | local/free | SAM3 box-tightening + final dataset/qc output |

Each stage reads an *earlier* stage's `cache/` output and writes its own. Concretely: changing
`--concepts` and re-running stage 2 never touches `cache/raw_gemini/` or spends a Gemini call;
tweaking tightening/QC-drawing code and re-running stage 4 never re-calls Gemini or re-runs either
SAM proposal pass. Both proposal stages also skip recompute for any image whose cache already
matches the current settings (see each stage's `--force`), so re-running them costs nothing for
unchanged images either.

Both SAM roles use the **official Meta packages directly** ([`facebookresearch/sam3`](https://github.com/facebookresearch/sam3),
[`facebookresearch/sam2`](https://github.com/facebookresearch/sam2)), not the `ultralytics` wrapper
this pipeline used earlier. Switched after two real problems hit with the ultralytics path
specifically: a cascading CUDA out-of-memory error on a 100-image run that was never fully
explained, and repeated friction with ultralytics' own CLI-style argument whitelist, which made
SAM2's grid-density tuning (`points_per_side`) unreachable. The official `SAM2AutomaticMaskGenerator`
class takes that as a genuine, direct constructor argument - no whitelist in the way.

## Setup

### 1. Get a Gemini API key

1. Go to **https://aistudio.google.com/apikey**.
2. Sign in with a Google account.
3. Click "Create API key" (pick/create a Google Cloud project if prompted - the free tier needs
   no billing enabled).
4. Copy the key.

### 2. Put the key in `.env`

Open `dataset_annotation/.env` (already exists, already gitignored - never gets committed) and
replace the placeholder:

```
GEMINI_API_KEY=paste-your-real-key-here
```

`.env.example` is the committed template documenting this variable name for anyone who clones the
repo fresh - it holds no real key. Only needed for stage 3 (`04_annotate_with_gemini.py`).

### 3. WSL environment - Python 3.12, both official SAM packages

Stages 1, 2, and 4 need a local SAM model; stage 3 (Gemini) needs neither. Run everything in a WSL
conda env with **Python >= 3.12** (SAM3's requirement) - this repo's `yolo_det_py312` env:

```bash
conda create -n yolo_det_py312 python=3.12
conda activate yolo_det_py312
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128

git clone https://github.com/facebookresearch/sam3.git && cd sam3
pip install -e ".[notebooks]"   # the plain extras-free install is missing einops/pycocotools that
cd ..                           # sam3's own __init__.py unconditionally imports - confirmed this
                                 # session; the notebooks extra is the one that has both

git clone https://github.com/facebookresearch/sam2.git && cd sam2
pip install --no-build-isolation -e .   # --no-build-isolation avoids pip re-downloading a whole
cd ..                                   # separate torch/CUDA stack into an isolated build env -
                                         # hit a real "No space left on device" from this once
                                         # (WSL's /tmp is a small RAM-backed tmpfs, not real disk)

pip install google-genai opencv-python pydantic python-dotenv
pip install "numpy<2"   # sam3 pins numpy<2; opencv-python's newest wheels want numpy>=2 - installing
                         # the notebooks extra above already downgrades opencv-python to a
                         # numpy<2-compatible version, but pin numpy explicitly too if anything
                         # upgrades it later
```

**SAM3** (`facebook/sam3` on Hugging Face) is **gated**. `03_propose_regions_concept.py` and
`05_tighten_boxes.py` both auto-detect `sam3.pt` sitting next to the script and default
`--sam3-checkpoint` to it - if you already have the file (from a manual/browser download after
accepting the gated terms), just drop it in `dataset_annotation/` and nothing else is needed, no
flag to pass. That file being on disk does **not** mean `huggingface_hub` has a cached token - those
are independent - so if you *don't* have it locally, the auto-download fallback requires
`hf auth login` (a token from https://huggingface.co/settings/tokens tied to an account with
granted access) or it 401s. **SAM2** is **not** gated - `facebook/sam2.1-hiera-tiny` (the default
here) auto-downloads with no access request either way.

Verify the whole env in one shot:
```bash
python -c "import torch, cv2, sam3, sam2, pydantic, dotenv, google.genai; \
  from sam3.model_builder import build_sam3_image_model; \
  from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator; \
  print('cuda:', torch.cuda.is_available())"
```

## Scripts (run in order)

### `01_remove_cad.py` - strip synthetic CAD renders

NPU-BOLT ships 3 image groups by filename prefix: `AUT-*` (204 field photos), `WEB-*` (116
internet photos), `CAD-*` (17 synthetic CAD renders - not real photographs, must not be annotated
as if they were). `datasets/npu_bolt/` is **read-only** (never written to), so this script COPIES
the real (`AUT-*`/`WEB-*`) images into a working copy at `circe_datasets/npu_bolt/working_images/`,
excluding `CAD-*`.

```bash
python 01_remove_cad.py
```

Idempotent - safe to re-run any time; already-copied files are skipped.

Logs to `circe_datasets/npu_bolt/cleanup_cad_log.txt`.

### `02_propose_regions_dumb.py` - SAM2 promptless region proposals (stage 1, local/free)

```bash
python 02_propose_regions_dumb.py --limit 1     # speed is UNVERIFIED - test one image first
python 02_propose_regions_dumb.py                # full pass over everything in circe_datasets/npu_bolt/working_images/
```

Key flags:

| flag | default | purpose |
|---|---|---|
| `--src` | `circe_datasets/npu_bolt/working_images` | folder of images to process |
| `--out` | `circe_datasets/npu_bolt` | output folder (shared with all stages) |
| `--sam2-model-id` | `facebook/sam2.1-hiera-tiny` | Hugging Face repo ID (not gated, auto-downloads) |
| `--points-per-side` | 32 | grid density - a real, directly-reachable tuning knob (lower = faster/coarser) |
| `--limit N` | none | only consider the first N images from `--src` this run |
| `--force` | off | recompute even for images whose cache already matches the current model+points-per-side |

No text prompt, no concept understanding at all - flags every visually-distinct region SAM2 finds
by low-level structure. Writes `cache/sam_regions_dumb/<stem>.json` + a QC visualization (real mask
overlay, not a simplified outline) to `qc/sam_proposals_dumb/<stem>.<ext>`.

This is a genuinely different failure mode from stage 2's concept-targeted pass: that pass and
Gemini both have to understand what the *word* "bolt" visually means, so a fastener ambiguous
enough to fool one has a real chance of fooling both. This pass has zero notion of "bolt" at all,
so it can catch a fastener that's visually distinct-but-unusual even when both language-grounded
passes miss it.

### `03_propose_regions_concept.py` - SAM3 concept-targeted region proposals (stage 2, local/free)

```bash
python 03_propose_regions_concept.py --limit 5      # smoke test a handful first
python 03_propose_regions_concept.py                # full pass over everything in circe_datasets/npu_bolt/working_images/
```

Key flags:

| flag | default | purpose |
|---|---|---|
| `--src` | `circe_datasets/npu_bolt/working_images` | folder of images to process |
| `--out` | `circe_datasets/npu_bolt` | output folder (shared with all stages) |
| `--concepts` | `bolt,screw,nut,fastener,rivet` | comma-separated text concepts SAM3 searches for |
| `--sam3-checkpoint` | `sam3.pt` next to the script, if present, else auto-download | override to point elsewhere |
| `--limit N` | none | only consider the first N images from `--src` this run |
| `--force` | off | recompute even for images whose cache already matches the current `--concepts` |

Writes `cache/sam_regions_concept/<stem>.json` + a QC visualization (real mask overlay) to
`qc/sam_proposals_concept/<stem>.<ext>`. Open both proposal folders before running stage 3 - this
is the "dry run" from an earlier single-script design; there's no separate `--dry-run` flag any
more, since running stages 1-2 alone already stops before any Gemini call exists to make.

**Resumable**: safe to Ctrl-C and re-run any time - already-proposed images (same settings) are
skipped for free in both proposal stages.

### `04_annotate_with_gemini.py` - the Gemini batch call (stage 3, **costs real API quota**)

```bash
python 04_annotate_with_gemini.py --limit 10   # smoke test first
python 04_annotate_with_gemini.py              # full run over everything in circe_datasets/npu_bolt/working_images/
```

Key flags:

| flag | default | purpose |
|---|---|---|
| `--src` | `circe_datasets/npu_bolt/working_images` | folder of images to annotate |
| `--out` | `circe_datasets/npu_bolt` | output folder (shared with all stages) |
| `--model` | `gemini-robotics-er-1.6-preview` | Gemini model ID - switch any time, no code edit needed |
| `--skip-region-hints` | off | ignore both proposal caches even if stages 1-2 were run - send images with no hint text |
| `--limit N` | none | only consider the first N images from `--src` this run |
| `--retries` | 3 | retry attempts for the whole run's single API call (not per-image) on failure |
| `--reannotate-stale` | off | see "Changing the model" below |

Reads **both** `cache/sam_regions_concept/<stem>.json` (stage 2) and `cache/sam_regions_dumb/<stem>.json`
(stage 1) and sends both as separate hint text alongside each image - if either upstream stage
wasn't run for some/all pending images, those are just sent with no hints from that source (a
warning is logged, never blocking). Writes `cache/raw_gemini/<stem>.json` (the resumability cache)
and `qc/gemini_raw/<stem>.<ext>` (Gemini's boxes exactly as returned, before stage 4 touches
geometry).

**Resumable**: safe to Ctrl-C and re-run any time. Already-annotated images are never re-sent to
the API (see "Changing the model or the class taxonomy" below). Every image gets an explicit log
line saying exactly what happened to it and why (`ANNOTATE (single batch API call...)`,
`SKIP: already annotated by current model+taxonomy`, etc.) - nothing is silently skipped.

### `05_tighten_boxes.py` - SAM3 box-tightening + final output (stage 4, local/free)

```bash
python 05_tighten_boxes.py --limit 10   # smoke test first
python 05_tighten_boxes.py              # full pass over everything cached in cache/raw_gemini/
```

Key flags:

| flag | default | purpose |
|---|---|---|
| `--src` | `circe_datasets/npu_bolt/working_images` | folder of images to process |
| `--out` | `circe_datasets/npu_bolt` | output folder (shared with all stages) |
| `--sam3-checkpoint` | `sam3.pt` next to the script, if present, else auto-download | override to point elsewhere |
| `--skip-tightening` | off | skip the SAM box-tightening pass; still produces `dataset/`+`qc/visualized/`, just with Gemini's boxes exactly as cached |
| `--limit N` | none | only consider the first N images from `--src` this run |
| `--clean` | off | wipe `dataset/` + `qc/visualized/` + `qc/sam_regions_debug/`, regenerate from `cache/raw_gemini/` - FREE, no API calls |

Reads `cache/raw_gemini/<stem>.json` for every image that has one (skips + warns on images missing
that cache - run stage 3 first). Does **not** know about `--model`/taxonomy staleness at all - that
is stage 3's concern; this stage just reads whatever is currently cached and applies
`migrate_boxes()` unconditionally (a safe no-op on already-current labels). Writes the final
`dataset/` (trainable YOLO output) + `qc/visualized/` + `qc/sam_regions_debug/`.

## Batching (stage 3)

Every image that still needs annotating after `--limit` and the cache are applied goes into **one
single Gemini API call** for the whole run - each image sent as an `"Image: <filename>"` text part
immediately followed by its bytes, with Gemini asked to return one result per image tagged with
that same filename so results match back up to the right file.

There's no automatic chunking and no automatic retry-at-smaller-size on a partial miss. `--limit`
is the only batch-size control - pick a number you've confirmed works against your own account and
run the script again with a different `--limit` to cover the rest; already-annotated images are
skipped for free, so repeat runs just continue where the last one left off. If an image comes back
missing from the response, it's logged and written to `failures.txt`, not silently retried - just
re-run.

TPM (tokens/minute) isn't a real constraint here - it resets every 60 seconds, so a single
oversized request just needs you to wait it out before the next call. **RPD (requests/day) is the
actual fixed ceiling** - and since one run = one request regardless of `--limit`, RPD is only worth
tracking if you're re-running the script many times in a single day. Google's hard per-request
ceilings (100MB inline payload, 3,600 images/request, and for `gemini-3.5-flash` specifically
1,048,576 input / 65,536 output tokens - see
[file-input-methods](https://ai.google.dev/gemini-api/docs/file-input-methods),
[image-understanding](https://ai.google.dev/gemini-api/docs/image-understanding),
[models/gemini-3.5-flash](https://ai.google.dev/gemini-api/docs/models/gemini-3.5-flash)) aren't
enforced by this script - a `--limit` that pushes past them just fails at the API and shows up as
a failed batch call in `run_log.txt`.

**Known unverified risk**: Google's docs don't document how reliable per-image filename
attribution is inside a multi-image request. Spot-check `qc/gemini_raw/`/`qc/visualized/` output,
especially on large `--limit` runs.

## SAM region hints and box tightening

Real testing found Gemini's classification quality good, but its bounding boxes loose/imprecise,
and some fasteners missed outright (especially on blurred/cluttered backgrounds) - a documented
weak point of VLMs generally for precise localization (Google's own ASIMOV benchmark shows Gemini
3.0 Flash beating Robotics-ER specifically on box accuracy). The fix, without spending a second
Gemini call: local SAM brackets the one API call, as four separate scripts -

```
02_propose_regions_dumb.py:    SAM2 (local, promptless/automatic)      --  polygon TEXT hints
03_propose_regions_concept.py: SAM3 (local, text-prompted concept)     --  polygon TEXT hints
                                                            |
                                                            v
04_annotate_with_gemini.py:                   ONE Gemini batch API call (image + both hint sources)
                                                            |
                                                            v
05_tighten_boxes.py:            SAM3 (local, box-prompted mode)  -->  each returned box tightened
```

**Two independent proposal sources, chosen deliberately for uncorrelated failure modes**:

1. **Concept-targeted** (stage 2, SAM3 `Sam3Processor.set_text_prompt`, `--concepts`, default
   `bolt, screw, nut, fastener, rivet`) - searches the whole image specifically for fastener-like
   concepts. This is semantic - the model has to understand what the *word* "bolt" visually means,
   the same way Gemini does, so a fastener ambiguous enough to fool Gemini's language-grounding has
   a real chance of also fooling this pass. Still generally useful (catches things Gemini's own
   scan might independently miss), just not a fully independent check.
2. **Geometric** (stage 1, SAM2 `SAM2AutomaticMaskGenerator.generate()`, promptless) - zero text,
   zero notion of "bolt" at all, flags any visually-distinct region by low-level structure
   (edges, texture). Broader and noisier (expect background/clutter), but its mistakes come from a
   genuinely different cause than a language model's - it can catch a fastener that's visually
   unusual enough to fool language-grounded judgment, purely because it's still a distinct region.

An earlier version of this pipeline tried to get this second signal via SAM3's own automatic mode
through `ultralytics` - catastrophically slow (7.5-8+ min/image) and, more fundamentally, not a
real SAM3 capability at all: Meta's own official `facebookresearch/sam3` repo has no
`automatic_mask_generator.py`, no `points_per_side`, nothing - SAM3 is a promptable-only model by
design. SAM2 is the correct, natively-supported tool for a promptless pass; its own
`SAM2AutomaticMaskGenerator` takes `points_per_side`/`crop_n_layers` as real, direct constructor
arguments (confirmed by reading `sam2/automatic_mask_generator.py`), so grid-density tuning is
finally actually reachable if a run is too slow, unlike the ultralytics dead end hit earlier.

Both proposal passes send **polygons, not boxes** - each candidate is a simplified outline (list of
`[x,y]` points, 0-1000 normalized, capped at `MAX_POLYGON_POINTS`), preserving real segmented shape
instead of collapsing it to a rectangle, while still staying compact text (not a second image).

**Box tightening** (stage 4): each box Gemini returns gets re-prompted through SAM3
(`Sam3Processor.add_geometric_prompt`, the actual Grounded-SAM pattern: use a rough detection to
prompt a segmentation model for a pixel-precise boundary) to tighten its geometry - falling back to
the original box if SAM3 can't segment that region, never discarding a detection.

Skip any of these independently: `--skip-region-hints` on stage 3 (ignore both proposal caches),
`--skip-tightening` on stage 4, or simply don't run stage 1 and/or stage 2 at all - each producer
stage's absence is tolerated gracefully by stage 3, logged as a warning, never blocking.

**Full transparency**: every pipeline stage gets its own separate `qc/` folder - not just a
combined before/after overlay - so each stage is independently viewable and directly comparable
against the others, and every proposal is drawn as a **real alpha-blended mask overlay** on the
actual segmented pixels (not a simplified polygon outline, which was found to visually distort
irregular real masks too much to trust for QC by eye):

1. `qc/sam_proposals_dumb/` - SAM2's geometric candidates (stage 1), before Gemini sees anything
2. `qc/sam_proposals_concept/` - SAM3's concept-targeted candidates (stage 2), before Gemini sees anything
3. `qc/gemini_raw/` - Gemini's boxes exactly as returned (stage 3), before SAM tightening touches geometry
4. `qc/visualized/` - after SAM tightening (stage 4, FINAL) - this is what `dataset/labels/` contains
5. `qc/sam_regions_debug/` - bonus (stage 4): before vs. after tightening overlaid on one image, plus the raw JSON

## Output layout

Three-way split: `cache/` is the source of truth everything else regenerates from, `dataset/` is
the actual trainable YOLO output - standard Ultralytics/YOLO26 layout, same shape as
`model_training/merged/` in this repo, nothing else in this folder so a future importer can point
straight at it - and `qc/` is human-facing diagnostics, never consumed by training code. Each
script only wipes the `cache/`/`qc/`/`dataset/` subfolders it owns (see each script's `--clean`
above), never another stage's output.

```
circe_datasets/npu_bolt/
  working_images/                       CAD-*-excluded working copy of datasets/npu_bolt/images/
                                         (built by 01_remove_cad.py; datasets/ itself never modified)
  cache/
    sam_regions_dumb/<stem>.json        (stage 1) SAM2's raw geometric candidates actually sent as
                                         hints - not read back except by stage 3
    sam_regions_concept/<stem>.json     (stage 2) SAM3's raw concept-targeted candidates actually
                                         sent as hints - not read back except by stage 3
    raw_gemini/<stem>.json              (stage 3) raw Gemini result + which model/taxonomy version
                                         produced it - still written one file per image even though
                                         the API call is batched
  dataset/                              (stage 4)
    data.yaml                           nc: 3, names 0-2, train/val both point at images/
    images/<stem>.jpg                   copy of the original photo (datasets/npu_bolt/ never modified)
    labels/<stem>.txt                   YOLO label: "class_id cx cy w h" normalized 0-1, one line
                                         per box (empty file = valid "no defects found" background)
  qc/
    sam_proposals_dumb/<stem>.jpg       (stage 1) real mask overlay - SAM2 geometric candidates
    sam_proposals_concept/<stem>.jpg    (stage 2) real mask overlay - SAM3 concept candidates
    gemini_raw/<stem>.jpg               (stage 3) Gemini's boxes exactly as returned, no tightening
    visualized/<stem>.jpg               (stage 4, FINAL) after SAM tightening - QC quality here
    sam_regions_debug/<stem>.jpg        (stage 4) gray = before tightening, class color = after
    sam_regions_debug/<stem>.json       (stage 4) same, as numbers
  failures.txt                          (stage 3) images that failed after retries, for manual re-run
  run_log.txt                           full timestamped log, appended to by ALL FOUR stages -
                                         one interleaved history per `circe_datasets/npu_bolt/` output folder
```

`dataset/data.yaml`'s `train:`/`val:` both point at the same `images/` folder on purpose - this is
one unsplit source. Real train/valid/test splitting happens later, in `model_training/pipeline`
when this gets merged with the rest of the datasets (point it at `circe_datasets/npu_bolt/dataset/`
specifically - that's the clean YOLO-shape folder, `cache/`/`qc/` aren't part of the dataset).

## Rate limits and real token usage

Google doesn't publish a fixed limits table anymore - check **your own account's actual numbers**
at **https://aistudio.google.com/rate-limit?timeRange=last-28-days** (pick "Free tier" / your
project). Look up RPM (requests/minute), TPM (tokens/minute), and RPD (requests/day) for whichever
model `--model` is currently pointed at.

Every real API call logs its actual token usage (`tokens: prompt=X output=Y total=Z`, straight from
`response.usage_metadata` - real measured numbers, not an estimate), and the end-of-run summary
totals them up. Check `run_log.txt` after a run for ground truth on what a given batch actually
cost, rather than guessing from image resolution.

As checked this session (free tier): `gemini-3.5-flash` = 5 RPM / 250K TPM / **20 RPD**;
`gemini-3.1-flash-lite` = 15 RPM / 250K TPM / 500 RPD; `gemini-robotics-er-1.6-preview` = 5 RPM /
250K TPM / 20 RPD; `gemini-robotics-er-1.5-preview` = 10 RPM / 250K TPM / 20 RPD. `--model` currently
defaults to `gemini-robotics-er-1.6-preview` - purpose-built for spatial/object-detection reasoning,
tested with good classification quality (see "SAM region hints and box tightening" above for how
its box-geometry weakness is addressed without a second API call). See the model-options comment
block above `DEFAULT_MODEL` in `04_annotate_with_gemini.py` for the full list tried and their
tested/untested status. That trade means **RPD is the one that matters** on the free tier (only
20/day) - TPM isn't a practical constraint since it resets every 60 seconds, so an oversized request
just costs you a wait, not a redesign. Since each run makes exactly one request regardless of
`--limit`, a full `circe_datasets/npu_bolt/working_images/` run split across a handful of manually-sized `--limit` calls stays
well inside the daily 20-request budget.

## Changing the model or the class taxonomy

Pass `--model` to `04_annotate_with_gemini.py` (no code edit needed), or edit `CLASSES`/
`SYSTEM_PROMPT` in that same script (bump `PROMPT_VERSION` when you do the latter - it exists
specifically so a taxonomy edit is detected as staleness too, not just a model swap). Each cached
result in `raw_gemini/*.json` records which model *and* which taxonomy version produced it. On the
next run of stage 3:

- Images already annotated by the **current** model+taxonomy: skipped, free, as always.
- Images annotated under a **different** model or an **older taxonomy**: **kept as-is and left
  alone by default** - you get a warning log line, nothing is re-sent to the API automatically.
  Re-annotating already-paid-for images is a real cost decision, not something this script will
  ever do without being told.
  - If the old labels can be losslessly remapped into the current class list (e.g. the
    `bolt_loose`/`bolt_damaged` -> `bolt_defective` merge), that remap happens automatically for
    free via `LABEL_MIGRATIONS` in `annotate_common.py`, so old cache still produces valid, correct
    output under the new taxonomy without an API call (applied unconditionally by both stage 3 and
    stage 4). Known gap: this can't fix the corrosion-priority tie-break retroactively - a fastener
    that's both corroded and loose/damaged, annotated back when `bolt_damaged`/`bolt_loose`
    outranked `bolt_corroded`, stays labeled `bolt_defective` instead of the current-priority-correct
    `bolt_corroded` until re-run with `--reannotate-stale`.
- To actually redo stale images with the current model/taxonomy (costs new API calls), pass
  `--reannotate-stale` to `04_annotate_with_gemini.py` explicitly.

Stage 4 (`05_tighten_boxes.py`) doesn't need to know about any of this - it just reads whatever's
currently in `cache/raw_gemini/*.json` and applies `migrate_boxes()` unconditionally.

## Cost awareness

Only `04_annotate_with_gemini.py` (stage 3) costs Gemini API quota (and, above the free tier,
money). It will never make a call you didn't ask for: new images are only annotated when you run
it pointed at them, and stale-model re-annotation requires the explicit `--reannotate-stale` flag.
`--limit N` (all four stages) is there to let you sanity-check output on a handful of images before
committing to a full run. Stages 1, 2, and 4 cost only local GPU time, no matter how many times you
re-run them.
