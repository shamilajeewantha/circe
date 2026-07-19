"""Stage 3 of 4: annotate bolt/nut images with Gemini into exactly 3 classes:
    bolt_ok, bolt_defective, bolt_corroded

Reads region proposals from TWO independent local SAM passes and sends them as polygon text hints
alongside each image:
  - cache/sam_regions_concept/<stem>.json (written by 03_propose_regions_concept.py, SAM3
    text-prompted) - semantic, may share Gemini's own language-grounding blind spots (both have to
    understand what the WORD "bolt" visually means).
  - cache/sam_regions_dumb/<stem>.json (written by 02_propose_regions_dumb.py, SAM2 promptless) -
    zero notion of "bolt" at all, a genuinely different failure mode.
Does NOT load any SAM model itself - if either upstream stage wasn't run for some/all pending
images, those images are just sent with no hints from that source (a warning is logged), never
blocked. This is the ONLY stage that spends real Gemini API quota - 02, 03, 05 are all local/free
and can be re-run freely.

bolt_defective merges "loose" and "damaged" - kept as separate classes originally, but they share
the same underlying visual symptom (fastener doesn't sit flush) for different root causes, which
makes them an unreliable distinction for both Gemini and a nano-capacity YOLO model to draw
consistently. See conversation history for the full reasoning.

Priority when a fastener shows multiple issues: bolt_corroded > bolt_defective > bolt_ok. Corrosion
wins because it's usually the more visually-certain call AND is frequently the root cause of a
mechanical symptom (a corroded bolt needs replacement, not just tightening).

(No missing/empty-hole class this round - see conversation history: that class
needs a different pipeline entirely and is deliberately out of scope here.)

BATCHING
--------
Every image that needs annotating in a given run (i.e. not already cached under the current
model+taxonomy) goes into ONE Gemini API call - no automatic chunking/splitting. Each image is
sent as an "Image: <filename>" text part immediately followed by its bytes, and the model is
asked to return one result per image tagged with that same filename so results can be matched
back up. Per-image caching (cache/raw_gemini/<stem>.json) is unchanged either way.

There is no built-in cap on how many images go into that one call - batch size is controlled
entirely by --limit (how many images from --src are even considered this run) plus how many of
those are already cached. If you want to control how large a single request is (payload size,
token usage, blast radius of one failed call), set --limit yourself and run the script multiple
times - the resumable cache means a second run just picks up wherever the first stopped. See
Google's own docs for the actual hard per-request ceilings if you need to reason about how far you
can push --limit: 100MB inline payload cap
(https://ai.google.dev/gemini-api/docs/file-input-methods), 3,600 images/request
(https://ai.google.dev/gemini-api/docs/image-understanding), and for gemini-3.5-flash specifically
1,048,576 input / 65,536 output tokens per request
(https://ai.google.dev/gemini-api/docs/models/gemini-3.5-flash) - none of these are enforced by
this script; a request that exceeds them fails at the API and shows up as a FAILED batch call in
run_log.txt.

If the response is missing an entry for some image in the call (truncated output, or the model
dropping/mis-tagging one), that image is logged and written to failures.txt - it is NOT retried
automatically at a smaller size. Just re-run the script (with a smaller --limit if you want) and
the missing image gets picked up as still-pending.

Setup
-----
1. Get an API key at https://aistudio.google.com/apikey (sign in with a Google account,
   "Create API key").
2. Put it in dataset_annotation/.env (already created, gitignored - never gets committed):
     GEMINI_API_KEY=paste-your-key-here
3. Install deps:
     pip install google-genai opencv-python pydantic python-dotenv

Usage
-----
    python 02_propose_regions_dumb.py            # stage 1 (optional but recommended)
    python 03_propose_regions_concept.py          # stage 2 (optional but recommended)
    python 04_annotate_with_gemini.py --limit 10  # smoke test first
    python 04_annotate_with_gemini.py             # full run over everything in npu_bolt/

Rate limits: check YOUR actual free-tier numbers at
https://aistudio.google.com/rate-limit?timeRange=last-28-days (they're account-specific, not
published as a fixed table). As checked this session: gemini-3.5-flash = 5 RPM / 250K TPM / 20 RPD;
gemini-3.1-flash-lite = 15 RPM / 250K TPM / 500 RPD. RPD is the tight one on gemini-3.5-flash -
each run is exactly one request regardless of --limit, so RPD only becomes a concern if you're
running the script many times in one day (e.g. testing different --limit values back to back).

Output: writes cache/raw_gemini/<stem>.json (resumability cache - {model, prompt_version, boxes}),
qc/gemini_raw/<stem>.<ext> (Gemini's boxes exactly as returned, structurally-invalid ones dropped -
before 05_tighten_boxes.py touches geometry at all), and failures.txt (images missing from the
batch response after retries). Run 05_tighten_boxes.py next to produce the final dataset/ + qc/.

CURRENTLY ON TRIAL: gemini-robotics-er-1.6-preview, not gemini-3.5-flash. History: 3.1-flash-lite
was tried first (500 RPD is very attractive) but its label quality was unreliable - visibly wrong
labels and missed fasteners. Switched to full 3.5-flash for label quality (20 RPD trade-off).
Robotics-ER is now being trialed instead because it's purpose-built by Google for spatial/embodied
reasoning - object detection and bounding-box localization specifically - which is a closer match
to this task than a general-purpose chat model. Confirmed from Google's docs
(https://ai.google.dev/gemini-api/docs/robotics-overview) this session: it uses the same
box_2d=[ymin,xmin,ymax,xmax]/1000 format and supports response_schema structured output, and its
free-tier limits (5 RPM / 250K TPM / 20 RPD) are no worse than 3.5-flash's.

Tested so far (real npu_bolt/ photos, small --limit): classification/label quality was good
("super great" per direct feedback), so system_instruction IS apparently being honored well enough
- that unverified risk looks resolved in practice, though not against a large batch yet. BUT
bounding box tightness was reported inaccurate/loose - which matches an independent source, not
just this one test: Google's own ASIMOV benchmark shows Gemini 3.0 Flash beating Robotics-ER
specifically on bounding-box accuracy (https://ai.google.dev/gemini-api/docs/robotics-overview) -
so this looks like a real, documented weak point of this model family, not a fluke. Two mitigations
are now in the code: an explicit box-tightness rule was added to SYSTEM_PROMPT (edges must touch
the fastener on every side, no padding), and filter_valid_boxes() drops/logs any box that's
structurally degenerate (inverted, out of [0,1000], wrong shape) rather than corrupting the label
file - neither of these can fix imprecise-but-structurally-valid geometry (05_tighten_boxes.py's
SAM tightening pass is the mitigation for that; QC-by-eye via qc/visualized/ can catch what's left).

Multi-image batching (this script's whole call_gemini_batch approach) is still unverified for this
model specifically - Google's examples are all single-image. Watch for filename-attribution issues
if running a large --limit on Robotics-ER.

See the model options list above DEFAULT_MODEL for other models tried/considered and their status.
Switch models any time via --model, no code edit needed - see main()'s argparse setup below. If a
call fails with a "model not found" error, check
https://ai.google.dev/gemini-api/docs/robotics-overview (or the equivalent Gemini model docs page)
for the current model ID.
"""
import argparse
import json
import logging
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv
from google import genai
from google.genai import types

from annotate_common import (
    CLASSES, IMG_EXTS, BoltAnnotation, BatchAnnotation, draw_visualization, filter_valid_boxes,
    migrate_boxes, _mime_for,
)

load_dotenv(Path(__file__).parent / ".env")

log = logging.getLogger("gemini_annotate")

# Models tried/considered this session - reference only (not live code, nothing here is imported).
# Pass any of these via --model at any time; no code edit needed. Free-tier limits as checked
# against aistudio.google.com/rate-limit this session; (tested)/(untested) reflects what's actually
# been run against real npu_bolt/ photos, not assumed.
#   "gemini-3.1-flash-lite"           # 15 RPM/250K TPM/500 RPD - (tested) label quality unreliable,
#                                      # visibly wrong + missed fasteners
#   "gemini-3.5-flash"                # 5 RPM/250K TPM/20 RPD  - (tested, infra only) ran a 75-image
#                                      # single batch call with no infra problems; label/box quality
#                                      # vs the two ER models below not directly compared
#   "gemini-robotics-er-1.5-preview"  # 10 RPM/250K TPM/20 RPD - (untested) purpose-built for
#                                      # spatial/object detection; not yet run against npu_bolt/
#   "gemini-robotics-er-1.6-preview"  # 5 RPM/250K TPM/20 RPD  - (tested) classification quality
#                                      # reported "super great", but bounding boxes reported
#                                      # inaccurate/loose - matches Google's own ASIMOV benchmark,
#                                      # which shows Gemini 3.0 Flash beating Robotics-ER specifically
#                                      # on box accuracy (ai.google.dev/gemini-api/docs/robotics-overview)
#
# Default only - override per-run with --model, no code edit needed (e.g. --model gemini-3.5-flash
# to go back to Flash). Whichever model actually ran is what gets tagged into cache/raw_gemini/*.json
# and compared against on the next run - see the --reannotate-stale gating in main().
DEFAULT_MODEL = "gemini-robotics-er-1.6-preview"

# Bump this whenever CLASSES/SYSTEM_PROMPT changes meaningfully. Cached results record which
# version produced them (alongside MODEL) so a taxonomy change is detected as staleness too, not
# just a model change - see the --reannotate-stale gating in main().
PROMPT_VERSION = 8  # v2: merged bolt_loose+bolt_damaged -> bolt_defective (3-class taxonomy)
                    # v3: explicit box-scope rule (head/cap + attached shank/thread only; never a
                    # bare shank with no head/nut in frame) - class names unchanged, only box
                    # geometry/consistency, so LABEL_MIGRATIONS has nothing to remap for this bump
                    # (switching the API call itself to a multi-image batch didn't change the
                    # per-image label schema or class definitions, so no bump needed for that)
                    # v4: systematic grid-scan + second-pass self-check rules, to reduce missed
                    # fasteners - class names/geometry rules unchanged, so LABEL_MIGRATIONS still
                    # applies unmodified; existing cache is just flagged stale, not auto-redone
                    # v5: SAM region-hint text now part of every request (see SYSTEM_PROMPT's
                    # "Candidate regions" rule) - input format changed, existing cache is stale;
                    # geometry/labels unaffected so LABEL_MIGRATIONS still applies unmodified
                    # v6: hints are now TWO polygon lists (concept-targeted SAM3 + generic SAM2),
                    # not one box list - SYSTEM_PROMPT's "Candidate regions" rule fully rewritten to
                    # describe both sources and the polygon format; existing cache stale again
                    # v7: generic (blind) hints pass REMOVED entirely (SAM3 had no native automatic
                    # mode) - back to ONE polygon list (concept-targeted only)
                    # v8: TWO polygon lists again, but correctly framed this time and via different
                    # official packages - "concept-targeted" (SAM3, text-prompted, official
                    # facebookresearch/sam3) + "geometric" (SAM2, promptless, official
                    # facebookresearch/sam2, genuinely no language-grounding bias unlike the earlier
                    # generic-via-SAM3 attempt) - SYSTEM_PROMPT's "Candidate regions" rule rewritten
                    # again to describe both sources with this corrected framing

SYSTEM_PROMPT = """You are labeling images of bolts, nuts, and other threaded fasteners for a
robot inspection vision training set. For every individual fastener (bolt head, nut, or bolt+nut
assembly) visible in the image, output one bounding box using exactly one of these three labels -
no other labels are allowed:

- bolt_ok: fastener present, undamaged, fully seated, no visible defect, no corrosion.
- bolt_defective: a NON-corrosion mechanical problem is visible - rotation, protrusion, backing-out,
  a gap between the fastener and the mating surface (not fully tightened), OR the head/thread/body
  is visibly bent, sheared, cracked, stripped, or otherwise mechanically deformed. Covers both
  "loose" and "damaged" as one class - do not try to distinguish them further.
- bolt_corroded: visible rust, pitting, or corrosion discoloration on the fastener's surface, EVEN
  IF the fastener also looks loose/damaged (corrosion is frequently the root cause of the
  mechanical symptom, and correct remediation differs - a corroded bolt should not simply be
  tightened, it likely needs replacement).

Rules:
- Only box actual fastener hardware (bolt heads, nuts, screws, rivets used as fasteners). Do NOT
  box empty holes, washers alone, brackets, or surrounding structure - this dataset intentionally
  excludes a "missing bolt" class this round.
- Box scope: box the head/cap of the fastener - the part a wrench or socket would turn (a bolt's
  hex head, or a nut). If a length of shank or exposed thread is visibly part of that SAME fastener
  (e.g. it's backed out and sticking up), include that shank/thread inside the same box rather than
  giving it a separate one. Do NOT box a bare shank or exposed thread with no head or nut visible
  anywhere in the frame - which fastener it belongs to is ambiguous, same reasoning as the excluded
  missing-bolt case; skip it.
- If a fastener shows multiple issues, pick the single label by this priority (corrosion wins ties,
  since it's usually the more visually-certain call and the more safety-relevant one):
  bolt_corroded > bolt_defective > bolt_ok.
- If you cannot see a fastener clearly enough to classify it confidently, skip it rather than
  guessing.
- If an image contains no fasteners at all, give it an empty boxes list - do not omit it.
- Scan the image in a systematic grid, not just where fasteners are obvious: divide it into
  top-left, top-right, bottom-left, bottom-right, and center regions, and deliberately look for
  fasteners in EACH region in turn, including ones that are small, distant, partially occluded, at
  the image edge, or in a repeated/patterned row where it's tempting to box only a few examples.
- After you have an initial list of boxes, do a second independent pass over the whole image
  before finalizing your answer, specifically hunting for anything you missed the first time -
  treat the first pass as a draft, not a final answer.
- Do not box shadows, dirt spots, or textures that merely resemble a bolt head - only box hardware
  you are confident is a genuine fastener.
- Do not box wire fasteners, cable clamps, cable ties, or wire-rope clips - these resemble bolts but
  secure a wire/cable, not a structural joint.
- Do not box small cone-shaped or tapered fittings used as guy-wire anchors, pole braces, or
  turnbuckle-style tensioners - these are tension hardware, not bolts, even though their shape can
  look bolt-like.
- Do not default to bolt_ok when unsure - actively check for corrosion and mechanical defect before
  assigning bolt_ok; partial or localized rust still counts as bolt_corroded.
- One box per fastener, never merge: if several fasteners are clustered or in a row, output one
  separate tightly-cropped box per fastener - never a single large box spanning multiple fasteners
  or a whole assembly/panel.
- Box tightness: every box's four edges must each touch the visible extent of that fastener (head/
  cap, plus any same-fastener shank/thread per the box scope rule above) on that side - no padding,
  margin, or slack on any edge. A box that is visibly looser than the fastener it's drawn around,
  or that clips off part of the fastener, is wrong even if the label is correct.
- Candidate regions: some images come with up to TWO separate lists of candidate regions, from two
  independent automated segmentation passes that ran before you saw the image, using two different
  models with two genuinely different failure modes. NEITHER list is authoritative - both are hints
  only, to help you notice things, not a checklist to trust or reproduce blindly. You must still
  verify every candidate independently against every rule above, and you must still detect any
  genuine fastener that has no matching candidate in either list.
  1. "Concept-targeted candidate regions" - from a pass that searched the image specifically for
     fastener-like concepts (bolt, screw, nut, fastener, rivet) using a text-prompted segmentation
     model. This model has to understand what the WORD "bolt" visually means, similar to how you
     do - so it can still miss the same atypical/ambiguous fasteners you might miss, but it is
     generally relevant since it was looking for the same kind of object you are.
  2. "Geometric candidate regions" - from a completely different, blind pass with NO text prompt
     and NO understanding of what a fastener is at all: it flags any visually-distinct region by
     low-level image structure alone (edges, texture boundaries), with no language involved.
     Expect this list to contain background, brackets, shadows, dirt, and other non-fastener
     clutter - but because it has no notion of "bolt," it can catch a fastener that looks unusual
     or ambiguous enough to fool a language-based judgment (yours or the concept-targeted pass's),
     purely because it's still a visually distinct region. Treat it as a broader, noisier
     complement to list 1, not a replacement for it.
  Both lists describe each candidate as a POLYGON, not a box: a list of [x, y] point pairs
  (normalized 0-1000, same scale as box_2d) that trace the approximate outline of whatever that
  pass segmented, in order around the shape. Use the polygon's actual outline - not just its rough
  extent - to judge whether something looks fastener-shaped and to see where its real edges are;
  this is richer information than a plain rectangle. When you do detect a fastener, your own output
  box_2d must still be a tight bounding box per the box tightness rule above, regardless of whether
  it came from a candidate polygon or was found independently - you are not asked to output
  polygons yourself, only to use the polygons you're given as visual aids."""

# Batch call prompt: images arrive as repeated (text-label, image-bytes) pairs, this final text
# part tells the model how to report results back per-image.
BATCH_USER_PROMPT = """You have been given multiple images in this request, each one immediately
preceded by a text part reading "Image: <filename>". Treat each image independently - do not let
fasteners or context in one image influence another. Apply the SAME full thoroughness (systematic
grid scan, second verification pass) to every image in this batch, including the last ones - do
not become less careful as you work through the list.

Return exactly one entry per image, in the "images" list, in the same order the images were given.
Each entry's "file" field must be copied EXACTLY (byte-for-byte, including extension and case) from
its "Image: <filename>" label. Do not skip an image even if it has no fasteners - give it an empty
boxes list instead. Do not invent an entry for a filename that wasn't shown to you."""


def call_gemini_batch(
    client: genai.Client, image_paths: List[Path], model: str, retries: int, backoff: float,
    concept_hints: Optional[Dict[str, List[List[List[int]]]]] = None,
    dumb_hints: Optional[Dict[str, List[List[List[int]]]]] = None,
) -> Tuple[Dict[str, BoltAnnotation], int]:
    """One API call annotates ALL of image_paths together. Returns ({filename: BoltAnnotation},
    total_tokens_used - 0 if the call never went through). Any filename the model doesn't return
    an entry for is simply absent from the dict; the caller treats that as a failure for that
    image (same as the old per-image None-return convention, just resolved per-file afterward
    instead of per-call).

    concept_hints/dumb_hints: {filename: [polygon, ...]} read from cache/sam_regions_concept/ and
    cache/sam_regions_dumb/ respectively, or None if no cache file existed for ANY pending image in
    this batch from that source (treated as that source being disabled entirely). None means no
    hint text of that kind is sent at all; an empty list for a given filename (that image has a
    cache file, but the pass found nothing) is distinguished in the sent text from "no cache file
    for this image" so the model isn't misled either way. Each polygon is
    [[x,y], [x,y], ...] normalized 0-1000 - deliberately NOT simplified to a box (see
    SYSTEM_PROMPT's "Candidate regions" rule for exactly how these are described to the model)."""
    contents: list = []
    for p in image_paths:
        contents.append(f"Image: {p.name}")
        contents.append(types.Part.from_bytes(data=p.read_bytes(), mime_type=_mime_for(p)))
        if concept_hints is not None:
            polygons = concept_hints.get(p.name, [])
            if polygons:
                contents.append(
                    f"Concept-targeted candidate regions for {p.name}, from a text-prompted "
                    f"segmentation pass that searched specifically for fastener-like concepts - "
                    f"each is a polygon outline (list of [x,y] points, 0-1000 normalized, NOT a "
                    f"bounding box), hints only, not authoritative: {polygons}"
                )
            else:
                contents.append(f"No concept-targeted candidate regions were found for {p.name}.")
        if dumb_hints is not None:
            polygons = dumb_hints.get(p.name, [])
            if polygons:
                contents.append(
                    f"Geometric candidate regions for {p.name}, from a blind, concept-agnostic "
                    f"segmentation pass with no text prompt and no understanding of what a "
                    f"fastener is - broader and noisier than the concept-targeted list, expect "
                    f"background/clutter - each is a polygon outline (list of [x,y] points, "
                    f"0-1000 normalized, NOT a bounding box), hints only, not authoritative: "
                    f"{polygons}"
                )
            else:
                contents.append(f"No geometric candidate regions were found for {p.name}.")
    contents.append(BATCH_USER_PROMPT)

    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        response_mime_type="application/json",
        response_schema=BatchAnnotation,
    )

    last_err = None
    for attempt in range(1, retries + 1):
        try:
            response = client.models.generate_content(model=model, contents=contents, config=config)
            usage = getattr(response, "usage_metadata", None)
            tokens = getattr(usage, "total_token_count", 0) or 0
            if response.parsed is not None:
                if usage is not None:
                    log.info("  batch tokens: prompt=%s output=%s total=%s",
                              getattr(usage, "prompt_token_count", "?"),
                              getattr(usage, "candidates_token_count", "?"), tokens)
                by_file: Dict[str, BoltAnnotation] = {}
                for img in response.parsed.images:
                    if img.file in by_file:
                        log.warning("  batch response had a duplicate entry for %s; keeping the "
                                    "first one", img.file)
                        continue
                    by_file[img.file] = BoltAnnotation(boxes=img.boxes)
                return by_file, tokens
            last_err = f"response.parsed was None; raw text: {response.text[:500]!r}"
        except Exception as e:  # noqa: BLE001 - deliberately broad, this is a best-effort batch job
            last_err = repr(e)
        if attempt < retries:
            time.sleep(backoff * attempt)
    log.error("Batch call FAILED for %d image(s) after %d attempts: %s",
              len(image_paths), retries, last_err)
    return {}, 0


def _load_hints_for(
    pending: List[Path], cache_dir: Path, source_label: str
) -> Optional[Dict[str, List[List[List[int]]]]]:
    """Shared logic for both hint sources: a missing cache file for a given image just means "no
    hint for this image" (warned once per image, never blocking); if NONE of the pending images
    have a cache file at all, hints are treated as fully disabled (returns None) so the model
    doesn't get a wall of "No candidate regions were found" filler text for a batch that never had
    the producing stage run at all."""
    found_any_cache = False
    hints: Dict[str, List[List[List[int]]]] = {}
    for img_path in pending:
        cache_path = cache_dir / f"{img_path.stem}.json"
        if cache_path.exists():
            found_any_cache = True
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            hints[img_path.name] = cached.get("polygons", [])
        else:
            log.warning("%s - no %s cache found. Sending with no hints of this kind for this "
                        "image.", img_path.name, source_label)
    return hints if found_any_cache else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path, default=Path(__file__).parent / "npu_bolt")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "annotations")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                         help=f"Gemini model ID to call (default {DEFAULT_MODEL!r}). Switching "
                              "models never needs a code edit - just pass a different --model. "
                              "Images cached under a different model are handled exactly like a "
                              "PROMPT_VERSION change: kept as-is by default, see --reannotate-stale.")
    parser.add_argument("--skip-region-hints", action="store_true",
                         help="Ignore both cache/sam_regions_concept/ and cache/sam_regions_dumb/ "
                              "even if those stages were run - send images with no hint text at "
                              "all. Use this to A/B-test hint impact without deleting that cache.")
    parser.add_argument("--limit", type=int, default=None,
                         help="Only consider the first N images from --src this run. This is the "
                              "ONLY thing that controls batch size - every image that still needs "
                              "annotating after --limit and the cache are applied goes into ONE "
                              "Gemini API call. Pick N yourself based on what you've confirmed "
                              "works against your own account (payload size, TPM); re-run with a "
                              "different --limit to cover the rest - already-annotated images are "
                              "never re-sent, so repeat runs just pick up where the last stopped.")
    parser.add_argument("--retries", type=int, default=3,
                         help="Retries apply to the WHOLE run's single API call, not per-image - "
                              "if it fails outright it's retried (same request) up to this many "
                              "times before the whole batch is treated as failed. This does not "
                              "make extra calls to cover partial misses - it only re-sends an "
                              "already-failed request.")
    parser.add_argument("--backoff", type=float, default=3.0, help="Base seconds for retry backoff")
    parser.add_argument("--reannotate-stale", action="store_true",
                         help="Re-call the API (costs new calls) for images already annotated under "
                              "a DIFFERENT model or class taxonomy (MODEL/PROMPT_VERSION) than the "
                              "current ones. Without this flag, such images are left as-is (old "
                              "annotation kept) and only logged as a warning - re-annotating never "
                              "happens silently.")
    parser.add_argument("--clean", action="store_true",
                         help="Wipe qc/gemini_raw/ before running, then regenerate it from "
                              "cache/raw_gemini/ - FREE, no API calls, since the expensive part "
                              "(the cache) is kept.")
    parser.add_argument("--wipe-cache", action="store_true",
                         help="DANGER: also delete cache/raw_gemini/ (the Gemini annotation "
                              "cache), forcing EVERY image to be re-sent to the API on this run - "
                              "real cost, not just a relabel. Requires --clean to also be set, as "
                              "a deliberate extra step so this can't be triggered by accident.")
    args = parser.parse_args()

    if args.wipe_cache and not args.clean:
        raise SystemExit("--wipe-cache requires --clean too (deliberately - this is the expensive, "
                          "re-annotate-everything option, not a casual one).")

    src = args.src.resolve()
    out = args.out.resolve()
    raw_dir = out / "cache" / "raw_gemini"
    sam_regions_concept_cache_dir = out / "cache" / "sam_regions_concept"
    sam_regions_dumb_cache_dir = out / "cache" / "sam_regions_dumb"
    gemini_raw_dir = out / "qc" / "gemini_raw"

    if args.clean:
        shutil.rmtree(gemini_raw_dir, ignore_errors=True)
        (out / "failures.txt").unlink(missing_ok=True)
        if args.wipe_cache:
            shutil.rmtree(raw_dir, ignore_errors=True)

    for d in (raw_dir, gemini_raw_dir):
        d.mkdir(parents=True, exist_ok=True)

    # FileHandler flushes every record as it's emitted, so the log survives a crash/Ctrl-C
    # partway through a long batch - unlike accumulate-then-write-once-at-the-end. Appends to the
    # same run_log.txt every stage writes to, so the full pipeline history for a given `out/`
    # interleaves in one place.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(out / "run_log.txt", encoding="utf-8"), logging.StreamHandler()],
    )
    log.info("Run started. Source: %s", src)
    if args.clean:
        log.info("--clean: wiped qc/gemini_raw/ (regenerated from cache/, free)%s",
                  " AND cache/raw_gemini/ (--wipe-cache set - full re-annotation ahead)"
                  if args.wipe_cache else "")

    client = genai.Client()

    images = sorted(f for f in src.iterdir() if f.is_file() and f.suffix.lower() in IMG_EXTS)
    if args.limit:
        images = images[: args.limit]
    log.info("Images found: %d", len(images))

    failures: List[str] = []
    done, cached, stale_kept, total_tokens = 0, 0, 0, 0

    # --- Pass 1: resolve cache status for every image up front. Images that are already usable
    # (current-model-and-taxonomy cache hit, or a stale cache we're keeping as-is) get their
    # result immediately. Everything else goes into `pending`, sent in ONE batch API call below.
    results: Dict[Path, BoltAnnotation] = {}
    pending: List[Path] = []

    for img_path in images:
        raw_path = raw_dir / f"{img_path.stem}.json"
        if raw_path.exists():
            cached_data = json.loads(raw_path.read_text(encoding="utf-8"))
            cached_model = cached_data.get("model", "unknown")
            cached_version = cached_data.get("prompt_version", "unknown")
            is_current = cached_model == args.model and cached_version == PROMPT_VERSION

            if is_current:
                results[img_path] = BoltAnnotation(boxes=cached_data["boxes"])
                cached += 1
                log.info("%s - SKIP: already annotated by current model+taxonomy (%s, v%d)",
                          img_path.name, args.model, PROMPT_VERSION)
                continue
            elif not args.reannotate_stale:
                results[img_path] = BoltAnnotation(
                    boxes=migrate_boxes(cached_data["boxes"], img_path.name)
                )
                stale_kept += 1
                log.warning("%s - SKIP: cached under model=%s taxonomy=v%s (current is model=%s "
                            "taxonomy=v%d); keeping the old annotation as-is. Pass "
                            "--reannotate-stale to redo it.",
                            img_path.name, cached_model, cached_version, args.model, PROMPT_VERSION)
                continue
            else:
                log.info("%s - RE-ANNOTATE: was model=%s taxonomy=v%s, redoing with model=%s "
                          "taxonomy=v%d (--reannotate-stale set)",
                          img_path.name, cached_model, cached_version, args.model, PROMPT_VERSION)
        pending.append(img_path)

    # --- Build hints for `pending` from both upstream stages' cache output. See
    # _load_hints_for's docstring for the None-vs-empty-dict-vs-per-image-missing semantics.
    concept_hints: Optional[Dict[str, List[List[List[int]]]]] = None
    dumb_hints: Optional[Dict[str, List[List[List[int]]]]] = None
    if pending and not args.skip_region_hints:
        concept_hints = _load_hints_for(pending, sam_regions_concept_cache_dir,
                                         "SAM concept-proposal (run 03_propose_regions_concept.py)")
        dumb_hints = _load_hints_for(pending, sam_regions_dumb_cache_dir,
                                      "SAM geometric-proposal (run 02_propose_regions_dumb.py)")
        if concept_hints is None:
            log.warning("No SAM concept-proposal cache found for ANY pending image - did you run "
                        "03_propose_regions_concept.py first?")
        if dumb_hints is None:
            log.warning("No SAM geometric-proposal cache found for ANY pending image - did you "
                        "run 02_propose_regions_dumb.py first?")
    elif pending:
        log.info("--skip-region-hints set - sending %d image(s) to %s with no segmentation hints",
                  len(pending), args.model)

    # --- Pass 2: exactly one API call covering every pending image. No chunking, no automatic
    # retry-at-smaller-size on partial misses - --retries only re-sends this same request if it
    # fails outright. Batch size is entirely up to --limit; run again with a different --limit
    # (already-annotated images are skipped for free) if you want to cover more in a second call.
    if pending:
        log.info("ANNOTATE (single batch API call covering %d image(s)): %s",
                  len(pending), ", ".join(p.name for p in pending))
        by_file, tokens = call_gemini_batch(
            client, pending, args.model, args.retries, args.backoff, concept_hints, dumb_hints
        )
        total_tokens += tokens

        for img_path in pending:
            ann = by_file.get(img_path.name)
            if ann is None:
                log.error("FAILED %s: not present in batch response", img_path.name)
                failures.append(img_path.name)
                continue
            raw_path = raw_dir / f"{img_path.stem}.json"
            raw_path.write_text(
                json.dumps(
                    {"model": args.model, "prompt_version": PROMPT_VERSION,
                     "boxes": [b.model_dump() for b in ann.boxes]},
                    indent=2,
                ),
                encoding="utf-8",
            )
            log.info("OK: %s -> %d box(es)", img_path.name, len(ann.boxes))
            results[img_path] = ann
            done += 1
    else:
        log.info("Nothing to annotate - all images already have usable cached results.")

    # --- Pass 3: regenerate qc/gemini_raw/ for every image that has a result (cheap, no API
    # calls, applies uniformly to fresh/cached/stale-kept results) - Gemini's boxes exactly as
    # returned (structurally valid ones only), before 05_tighten_boxes.py touches geometry at all.
    for img_path in images:
        result = results.get(img_path)
        if result is None:
            continue
        result = filter_valid_boxes(result, img_path.name)
        draw_visualization(img_path, result, gemini_raw_dir / img_path.name)

    if failures:
        (out / "failures.txt").write_text("\n".join(failures), encoding="utf-8")

    log.info("Done. %d newly/re-annotated, %d cached (current model+taxonomy), %d cached (stale, "
              "kept as-is), %d failed, %d tokens used this run (real API calls only - "
              "cached/skipped images cost nothing).",
              done, cached, stale_kept, len(failures), total_tokens)
    if stale_kept:
        log.info("%d image(s) still hold annotations from an older model or class taxonomy - "
                  "re-run with --reannotate-stale if you want them redone.", stale_kept)
    if failures:
        log.info("Failures logged to %s", out / "failures.txt")
    log.info("Raw results cached at %s. Run 05_tighten_boxes.py next to produce dataset/ + final qc/.",
              raw_dir)


if __name__ == "__main__":
    main()
