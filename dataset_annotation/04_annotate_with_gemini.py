"""Stage 3 of 4: annotate bolt/nut images with Gemini into exactly 3 classes:
    bolt_ok, bolt_defective, bolt_corroded

Reads region proposals from cache/sam_regions_concept/<stem>.json (written by
03_propose_regions_concept.py, SAM3 text-prompted) and sends them alongside each image both as a
polygon text hint AND as a visual overlay image (qc/sam_proposals_concept/<stem>.<ext>, the same
numbered mask overlay already drawn for QC). The SAM2 promptless/geometric pass
(02_propose_regions_dumb.py) is no longer sent to Gemini at all as of this session - real
experience this session was that SAM3's concept-targeted results were consistently strong once its
own bugs were fixed, and the geometric pass added noise/cost without enough benefit to justify
including it here (02_propose_regions_dumb.py itself is untouched and still useful standalone -
just not fed into this stage anymore).
Does NOT load any SAM model itself - if 03_propose_regions_concept.py wasn't run for some/all
pending images, those images are just sent with no hints (a warning is logged), never blocked.
This is the ONLY stage that spends real Gemini API quota - 02, 03, 05 are all local/free and can be
re-run freely.

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
    python 03_propose_regions_concept.py          # stage 2 (recommended - this stage's real input)
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

# Matches whichever SYSTEM_PROMPT_V<N> constant is currently active below (SYSTEM_PROMPT = ...).
# Change this to the same number whenever you switch which one is active.
PROMPT_VERSION = 3
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
                    # v9: real miss found via visual QC on a 30-image gemini-3.1-flash-lite run -
                    # AUT-0001.jpg (a hex nut + 2 Phillips screws in a shadowed, top-down-angle
                    # channel) got 0 boxes even though the concept-targeted SAM3 pass correctly
                    # found all of them (confirmed via qc/sam_proposals_concept/AUT-0001.jpg).
                    # Clarified the "skip if not confident" rule: it's for "is this a fastener at
                    # all" uncertainty, not "which of the 3 classes" uncertainty - the latter should
                    # never suppress a detection. Geometry/labels unaffected, LABEL_MIGRATIONS still
                    # applies unmodified.
                    # v10: real regression found via visual QC re-inspecting the SAME image
                    # (AUT-0000.jpg) after the v9 re-run: two separate bolts sharing one cable-clamp
                    # bracket, previously two correctly-separated tight boxes, came back as a single
                    # box spanning both - violating the existing "one box per fastener, never merge"
                    # rule. No temperature was pinned, so this run-to-run difference wasn't cleanly
                    # attributable to the v9 prompt edit vs. plain sampling noise; added
                    # temperature=0 to GenerateContentConfig (structured-extraction task, standard
                    # practice, makes future prompt changes actually testable) and strengthened the
                    # "never merge" rule with an explicit same-bracket example either way, since it's
                    # a real failure mode regardless of root cause. Geometry/labels unaffected.
                    # v11: temperature=0 did NOT fix the v10 regression - re-running the SAME image
                    # at temperature=0 gave the IDENTICAL merged result both times, proving this is a
                    # reproducible model behavior on this hardware shape, not sampling noise. Added a
                    # concrete, dataset-specific example to the "never merge" rule describing the
                    # exact shape (a wire-rope/cable clamp with 1-2 threaded studs+nuts, common in
                    # this dataset, that looks like "one part" at a glance) and stating explicitly
                    # that a clamp with 2 visible nuts must produce 2 boxes. Geometry/labels
                    # unaffected.
                    # v12: explicit user request - suspected Gemini was silently ignoring correct
                    # SAM3 concept-targeted hints (consistent with the earlier AUT-0001.jpg miss).
                    # Added mandatory per-candidate accounting: response schema now requires a
                    # concept_candidate_dispositions entry for EVERY numbered concept-targeted
                    # candidate (accepted=true + which box, or accepted=false + a specific reason) -
                    # no silent drops. Concept hint text is now numbered by index so the model can
                    # reference candidates unambiguously. Gemini can still invent boxes with no
                    # matching candidate (this only adds accounting, not a ceiling on detections).
                    # Discarded reasons are logged per-image and persisted in cache/raw_gemini/.
                    # Only applies to list 1 (concept-targeted/SAM3) per explicit instruction - list
                    # 2 (geometric/SAM2) stays advisory-only, unchanged.
                    # v13: real user concern - a comparison run on gemini-robotics-er-1.6-preview
                    # under v12 produced severe hallucination (~13 phantom boxes on empty
                    # background on one image, while missing the one obvious real bolt) - a much
                    # bigger drop than ER's previously-reported "super great" quality before the
                    # SAM hint text existed at all. Hypothesis: asking a model to mentally
                    # re-project dozens of raw [[x,y],...] numbers back onto the photo is a much
                    # harder channel than seeing them. Now also sends the ALREADY-COMPUTED numbered
                    # mask overlay image (qc/sam_proposals_concept/<file>, drawn by
                    # 03_propose_regions_concept.py) as an additional image right after the raw
                    # photo and text candidate list, so the model can see candidates visually
                    # instead of only reasoning from coordinate text. No new SAM computation - pure
                    # reuse of an existing QC artifact. Geometry/labels unaffected; input format
                    # changed (extra image per call when a concept-search QC overlay exists), so
                    # cache is bumped stale like every prior hint-format change.
                    # v14: two explicit user calls, both real simplifications: (1) the geometric/
                    # SAM2 candidate list is REMOVED entirely, not just left advisory - real
                    # experience this session was SAM3's concept-targeted results were consistently
                    # strong once its own bugs were fixed, and the geometric pass added noise/cost
                    # without enough benefit (02_propose_regions_dumb.py itself is untouched, just
                    # no longer fed into this stage). (2) the raw polygon coordinate text for
                    # concept-targeted candidates is now the FALLBACK, not sent alongside the
                    # overlay image - when the overlay exists (the normal case), it alone is the
                    # candidate list; sending both was redundant and plausibly made the model work
                    # harder reconciling two representations of the same thing instead of just
                    # looking at the image. Geometry/labels unaffected.
                    # v15: explicit user hypothesis, worth taking seriously - burning index numbers
                    # onto the overlay image (candidate 0, 1, 2, ...) may bias the model toward
                    # treating each number as something it needs to draw a box FOR, rather than
                    # judging the highlighted region on its own visual merits. Removed numbers
                    # entirely from what Gemini sees: the overlay image sent to Gemini is now drawn
                    # with draw_numbers=False (03_propose_regions_concept.py writes a SEPARATE
                    # unnumbered copy to qc/sam_proposals_concept_for_gemini/ specifically for this,
                    # keeping the original numbered qc/sam_proposals_concept/ unchanged for OUR OWN
                    # human QC inspection, which still benefits from numbers). The
                    # ConceptCandidateDisposition schema also lost its `index` field - dispositions
                    # are now matched back to regions purely by position/order, never by a number
                    # the model has to read or produce. SYSTEM_PROMPT reworded throughout to say
                    # "highlighted regions" generically instead of "numbered candidates", and now
                    # states plainly that every highlighted region could be a bolt and, if it's not,
                    # a reason is mandatory. Geometry/labels unaffected.
                    # v16: three real problems reported after the v15 run. (1) "boxes in the air" -
                    # Gemini apparently treating a highlighted region's mere existence as evidence a
                    # fastener is there, rather than independently confirming it - added an explicit
                    # statement that a highlighted region is a suggestion to LOOK, never evidence by
                    # itself, and that rejecting one is a normal, expected, frequent outcome. (2)
                    # close-together fasteners still sometimes merged into one box despite the v11
                    # fix - added an explicit line that physical closeness is NEVER, by itself, a
                    # reason to merge two fasteners. (3) bolt_defective confirmed via real evidence
                    # (grepped all of cache/raw_gemini/*.json and dataset/labels/*.txt - 0
                    # occurrences anywhere this session) to have never been produced - class
                    # definition itself verified intact/unchanged in both annotate_common.py and
                    # SYSTEM_PROMPT, so not a narrowing bug; added an explicit reminder that it's a
                    # real independent category to check for on every fastener, same as corrosion,
                    # BEFORE the (unchanged, deliberately-kept) corrosion-wins-ties rule applies.
                    # Geometry/labels unaffected.
                    # v17: real regressions found via visual QC on the v16 run, cross-checked
                    # against the actual SAM3 masks (qc/sam_proposals_concept/*.jpg) for the exact
                    # failure images - not guessed. (1) AUT-0006.jpg: Gemini boxed a location with
                    # NO SAM3 region there at all (confirmed - only 3 masks existed, none near that
                    # box) - a pure invention, not a hint-following error. (2) AUT-0008.jpg: SAM3
                    # masks #28/#29 landed exactly on two real screws that Gemini discarded - a
                    # real fastener wrongly rejected. (3) AUT-0007.jpg: SAM3 correctly marked small
                    # individual bolt heads ON a latch mechanism (masks #20/#27 etc.), but Gemini's
                    # box expanded to cover the whole mechanism housing instead of the individual
                    # bolt. All three point the same way: SAM3's concept-search is comprehensive in
                    # practice (confirmed - it marked every real fastener in all three images,
                    # including the ones Gemini got wrong), so Gemini's job on these images should
                    # be filtering what's already found, not inventing independently, and box
                    # tightness should follow the highlighted region's own extent, not expand to a
                    # surrounding housing. Also reworded the v16 "reject freely" framing (which
                    # over-indexed toward rejection, plausibly contributing to (2)) into "reason
                    # hard before accepting AND before rejecting" - rejecting a real fastener is now
                    # explicitly stated as equally wrong as inventing a fake one, not a safer
                    # default. Geometry/labels unaffected.
                    # REVERTED to v15's content by explicit instruction, after re-testing v17 on the
                    # same 3 failure images: AUT-0007's box-scope problem was genuinely fixed, but
                    # AUT-0006 (phantom box, no supporting SAM3 region at all) and AUT-0008 (2 real
                    # SAM3-marked screws still wrongly discarded) were UNCHANGED - 1 of 3 targeted
                    # fixes actually landed. PROMPT_VERSION reset to the real "15" rather than bumped
                    # to a new number, since the active text is byte-identical to true v15 - v16/v17
                    # were edited in place and their exact text was NOT kept as separate constants,
                    # so there is nothing distinct for a new number to point to. Standing practice
                    # from this point forward (explicit instruction): every FUTURE distinct prompt
                    # version gets its own preserved SYSTEM_PROMPT_V<N> constant (never edited in
                    # place once superseded), with SYSTEM_PROMPT simply assigned to whichever is
                    # currently active - so any future revert is a real, cheap reassignment, not a
                    # lossy re-edit.

SYSTEM_PROMPT_V1 = """You are labeling images of bolts, nuts, and other threaded fasteners for a
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
  guessing. This rule is about whether something IS a fastener at all - it is NOT a reason to skip
  something you can already tell is a nut/bolt/screw head just because unusual lighting, a
  shadowed recess, an odd viewing angle, or partial occlusion makes the ok/defective/corroded call
  harder. If the shape is clearly fastener hardware, box it and make your best classification call
  (defaulting toward bolt_corroded/bolt_defective over bolt_ok if genuinely torn between two of the
  three, per the "do not default to bolt_ok when unsure" rule below) - do not let classification
  uncertainty suppress a detection you're actually confident about.
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
  or a whole assembly/panel. This applies even when multiple fasteners sit on the SAME bracket,
  clamp, or plate right next to each other (e.g. two bolts threaded into one cable-clamp body) -
  each fastener still gets its own box tightly cropped to just that one head/nut, not a box that
  spans both.
  Concretely, a wire-rope/cable clamp (a cast metal saddle or U-bolt body with a wire rope passing
  through it and one or two threaded studs+nuts holding it closed) is a VERY common shape in this
  dataset - it looks like "one part" at a glance, but each individual nut/stud visible on it is
  its own fastener and needs its own tightly-cropped box. A clamp with 2 visible nuts must produce
  2 boxes, each cropped to just one nut - never 1 box covering the whole clamp body, and never 1
  box spanning from one nut to the other.
- Box tightness: every box's four edges must each touch the visible extent of that fastener (head/
  cap, plus any same-fastener shank/thread per the box scope rule above) on that side - no padding,
  margin, or slack on any edge. A box that is visibly looser than the fastener it's drawn around,
  or that clips off part of the fastener, is wrong even if the label is correct.
- Highlighted regions: some images come with a set of highlighted regions - from a pass that
  searched the image specifically for fastener-like concepts (bolt, screw, nut, fastener, rivet)
  using a text-prompted segmentation model, run before you saw the image. This model has to
  understand what the WORD "bolt" visually means, similar to how you do - so it can still miss the
  same atypical/ambiguous fasteners you might miss, but in practice it finds real fasteners well -
  treat it as a strong signal, not noise to filter past. It is NOT authoritative - a hint to help
  you notice things, not a checklist to trust or reproduce blindly. You must still verify every
  highlighted region independently against every rule above, and you must still detect any genuine
  fastener that has no highlighted region near it at all.
  Regions are normally given as a VISUAL OVERLAY IMAGE, immediately after the raw photo: the same
  photo with each region's outline drawn directly on it (no numbers or labels on the regions
  themselves - just outlines) - use this to actually SEE where each region is, at a glance, like
  you would for anything else in the photo. Judge each highlighted region purely by what it visibly
  contains, never by a number or label - there isn't one. (Rare fallback: if no overlay image was
  available, regions are instead given as a list of raw polygon coordinates - [x, y] point pairs,
  normalized 0-1000, same scale as box_2d - describing the same thing in text form only; treat that
  the same way, just via a different channel.) When you do detect a fastener, your own output
  box_2d must still be a tight bounding box per the box tightness rule above, regardless of whether
  it came from a highlighted region or was found independently - you are not asked to output
  polygons yourself, only to use what you're given as a visual aid.
  MANDATORY ACCOUNTING: every highlighted region could be a bolt - you must judge each one and
  account for it in your output's concept_candidate_dispositions field, one entry per region, in
  the same order the regions appear (reading the image left-to-right then top-to-bottom), no
  omissions even for an image with many regions. For each: if it IS a bolt and became one of your
  output boxes, mark accepted=true. If it is NOT a bolt, mark accepted=false and you MUST give a
  SPECIFIC, concrete reason (not a vague "not a fastener") - e.g. background/shadow with no real
  hardware there, a cable-clamp body excluded by the box-scope rules, the same physical fastener as
  another region already boxed, or too occluded/blurred to classify confidently. You are still free
  to output MORE boxes than there are highlighted regions (detect real fasteners this pass missed
  entirely) - this accounting requirement only means every region this pass DID highlight must be
  explicitly resolved, one way or the other, not silently dropped."""

# Reconstructed from the actual diffs applied earlier this session (previously PROMPT_VERSION 16 -
# "boxes in the air" + never-merge-by-proximity + bolt_defective visibility). Real regression found
# via visual QC afterward: this framing over-indexed toward rejection ("do it freely"), plausibly
# contributing to real fasteners being wrongly discarded - see V3's docstring for the fix that was
# tried, and SYSTEM_PROMPT's assignment below for which of these three is currently active.
SYSTEM_PROMPT_V2 = SYSTEM_PROMPT_V1.replace(
    """- bolt_corroded: visible rust, pitting, or corrosion discoloration on the fastener's surface, EVEN
  IF the fastener also looks loose/damaged (corrosion is frequently the root cause of the
  mechanical symptom, and correct remediation differs - a corroded bolt should not simply be
  tightened, it likely needs replacement).

Rules:""",
    """- bolt_corroded: visible rust, pitting, or corrosion discoloration on the fastener's surface, EVEN
  IF the fastener also looks loose/damaged (corrosion is frequently the root cause of the
  mechanical symptom, and correct remediation differs - a corroded bolt should not simply be
  tightened, it likely needs replacement).

bolt_defective is a real, expected, independent category - not a rare edge case, and not something
that only applies when corrosion is absent. For EVERY fastener, evaluate its mechanical condition
(is it rotated, protruding, backed-out, gapped from the mating surface, bent, sheared, cracked, or
stripped?) on its own merits, the exact same way you already evaluate it for corrosion - do this
check BEFORE the priority tie-break below, not instead of it. Only after you've genuinely checked
both does the tie-break decide which single label wins when a fastener happens to show both.

Rules:""",
).replace(
    """  box spanning from one nut to the other.
- Box tightness:""",
    """  box spanning from one nut to the other.
  Physical CLOSENESS between two fasteners is NEVER, by itself, a reason to merge them into one
  box. However tightly two fasteners are packed together, however small the gap between them - if
  they are two separate pieces of hardware, they get two separate tight boxes. Distance to a
  neighboring fastener has no bearing on this rule at all.
- Box tightness:""",
).replace(
    """  fastener that has no highlighted region near it at all.
  Regions are normally given as a VISUAL OVERLAY IMAGE,""",
    """  fastener that has no highlighted region near it at all.
  A region being highlighted is a suggestion to LOOK there - it is never, by itself, evidence that
  a fastener is actually present. Do not box a highlighted region just because it was highlighted:
  look closely and deeply at what that region actually contains before deciding, exactly as
  carefully as you would look at any other part of the photo, and box it only if you independently
  confirm real, visible fastener hardware there. Rejecting a highlighted region (accepted=false) is
  a normal, expected, and frequent outcome, not a failure - do it freely whenever your own
  independent look doesn't confirm real hardware, as long as you give the specific reason the
  accounting rule below requires.
  Regions are normally given as a VISUAL OVERLAY IMAGE,""",
)

# Reconstructed from the actual diffs applied earlier this session (previously PROMPT_VERSION 17 -
# real regressions found via visual QC on V2, cross-checked against the actual SAM3 masks for the
# exact failure images: AUT-0006.jpg had a box with NO supporting SAM3 region at all (pure
# invention); AUT-0008.jpg had 2 real SAM3-marked screws wrongly discarded; AUT-0007.jpg had a box
# that expanded to cover a whole latch mechanism instead of the individual SAM3-marked bolt heads
# on it. Reworded "Highlighted regions" to say SAM3 is comprehensive in practice (filter, don't
# invent) and to require reasoning hard in BOTH directions, not just toward rejection. Re-tested
# against the same 3 images: the box-scope fix (AUT-0007) genuinely worked, but the other two
# (AUT-0006 invention, AUT-0008 wrongful rejection) were UNCHANGED - only 1 of 3 targeted problems
# actually landed, which is why V1 was reselected as active below rather than this one.
SYSTEM_PROMPT_V3 = SYSTEM_PROMPT_V2.replace(
    """  same atypical/ambiguous fasteners you might miss, but in practice it finds real fasteners well -
  treat it as a strong signal, not noise to filter past. It is NOT authoritative - a hint to help
  you notice things, not a checklist to trust or reproduce blindly. You must still verify every
  highlighted region independently against every rule above, and you must still detect any genuine
  fastener that has no highlighted region near it at all.
  A region being highlighted is a suggestion to LOOK there - it is never, by itself, evidence that
  a fastener is actually present. Do not box a highlighted region just because it was highlighted:
  look closely and deeply at what that region actually contains before deciding, exactly as
  carefully as you would look at any other part of the photo, and box it only if you independently
  confirm real, visible fastener hardware there. Rejecting a highlighted region (accepted=false) is
  a normal, expected, and frequent outcome, not a failure - do it freely whenever your own
  independent look doesn't confirm real hardware, as long as you give the specific reason the
  accounting rule below requires.""",
    """  same atypical/ambiguous fasteners you might miss, but IN PRACTICE it is comprehensive: it
  reliably finds and highlights essentially every real fastener in the image, including small or
  partially-occluded ones. Given that, your main job on a highlighted-region image is to CAREFULLY
  FILTER what's already been found, not to independently invent new detections - you should only
  add a box with NO highlighted region near it at all when you are genuinely confident real
  hardware is there, since a real fastener with no highlighted region anywhere near it should be a
  rare exception, not routine. It is still NOT authoritative - you must verify every highlighted
  region independently against every rule above - just don't treat "detect independently" as your
  default mode of operation on these images.
  A region being highlighted is a suggestion to LOOK there - it is never, by itself, evidence that
  a fastener is actually present, and it is never, by itself, evidence that one is ABSENT either.
  Reason hard in BOTH directions before deciding, not just one: before you ACCEPT a region (box it),
  confirm you can actually see real fastener hardware there, not just a plausible-looking shape.
  Before you REJECT a region (accepted=false), also confirm you've actually looked closely enough
  to be sure it ISN'T a fastener - do not reject a region just to be cautious, and do not accept one
  just because it was highlighted. If a highlighted region genuinely does show a real fastener, you
  MUST box it, even if it's small, awkwardly lit, or oddly angled - rejecting a real fastener is
  exactly as wrong as inventing a fake one, not a "safer" default. When you do accept a highlighted
  region, your box should closely match that specific region's own extent (the fastener head/nut it
  outlines) - do not let the box expand to cover a larger surrounding bracket, housing, or mechanism
  just because the highlighted region touches or overlaps it; box only the individual fastener,
  per the box scope and box tightness rules above.""",
)

# Currently active version - reassign this to switch versions instead of editing prompt text in
# place (standing practice from here on: each future distinct version gets its own preserved
# SYSTEM_PROMPT_V<N> constant above, never edited after being superseded, so a revert is always a
# cheap, honest reassignment here, not a lossy re-edit). Currently V3 (explicit instruction) - see
# V3's docstring above: only 1 of 3 targeted problems from that version's testing actually landed
# (the box-scope/AUT-0007 fix), the other two (AUT-0006 invention, AUT-0008 wrongful rejection)
# were unchanged versus V2.
SYSTEM_PROMPT = SYSTEM_PROMPT_V2


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
    concept_overlay_dir: Optional[Path] = None,
) -> Tuple[Dict[str, BoltAnnotation], int]:
    """One API call annotates ALL of image_paths together. Returns ({filename: BoltAnnotation},
    total_tokens_used - 0 if the call never went through). Any filename the model doesn't return
    an entry for is simply absent from the dict; the caller treats that as a failure for that
    image (same as the old per-image None-return convention, just resolved per-file afterward
    instead of per-call).

    concept_hints: {filename: [polygon, ...]} read from cache/sam_regions_concept/, or None if no
    cache file existed for ANY pending image in this batch (treated as that source being disabled
    entirely). None means nothing is sent at all; an empty list for a given filename (that image
    has a cache file, but the pass found nothing) is distinguished from "no cache file for this
    image" so the model isn't misled either way.

    concept_overlay_dir: the numbered mask overlay 03_propose_regions_concept.py already draws for
    QC (qc/sam_proposals_concept/<filename>) - when it exists, THIS is what actually gets sent as
    the candidate list (an image with each region's outline drawn directly on the photo, NO index
    numbers - see draw_numbers=False in 03_propose_regions_concept.py), not the raw polygon
    coordinates. Real finding this session: asking a model to mentally re-project dozens of raw
    [[x,y],...] numbers back onto the photo is a much harder, more error-prone channel than just
    letting it SEE where the regions are - plausibly a real factor in a severe hallucination
    regression observed on one model under the raw-text-only version of this prompt. Numbers were
    then dropped from the overlay image ITSELF too (an earlier version burned index numbers onto
    it) on a further real concern: literal numbers on the photo may bias the model toward treating
    each number as something to draw a box around, rather than judging the highlighted region on
    its own visual merits. Falls back to sending the raw polygon list as text (still unnumbered)
    only if the overlay file doesn't exist for some reason (cache present but QC image missing) -
    see SYSTEM_PROMPT's "Candidate regions" rule for how this is described to the model."""
    contents: list = []
    for p in image_paths:
        contents.append(f"Image: {p.name}")
        contents.append(types.Part.from_bytes(data=p.read_bytes(), mime_type=_mime_for(p)))
        if concept_hints is not None:
            polygons = concept_hints.get(p.name, [])
            if polygons:
                overlay_path = concept_overlay_dir / p.name if concept_overlay_dir is not None else None
                if overlay_path is not None and overlay_path.exists():
                    # The overlay image has each region's outline drawn on the photo (no numbers -
                    # see draw_mask_overlay's draw_numbers=False), so that alone is the region list
                    # now - no redundant coordinate text, and nothing for the model to count/number.
                    contents.append(
                        f"Highlighted candidate regions for {p.name}: {len(polygons)} region(s) "
                        f"from a text-prompted segmentation pass that searched specifically for "
                        f"fastener-like concepts, outlined directly on the following overlay "
                        f"image - hints only, not authoritative. You MUST return exactly "
                        f"{len(polygons)} concept_candidate_dispositions entries for {p.name}, "
                        f"one per highlighted region, in the order the regions appear reading the "
                        f"image left-to-right then top-to-bottom."
                    )
                    contents.append(types.Part.from_bytes(
                        data=overlay_path.read_bytes(), mime_type=_mime_for(overlay_path)
                    ))
                else:
                    # Fallback for the rare/abnormal case where a candidate cache exists but its
                    # QC overlay image doesn't - falls back to raw coordinate text (still no
                    # per-region numbers) so the model still gets SOME region information.
                    raw = "; ".join(str(poly) for poly in polygons)
                    contents.append(
                        f"Candidate regions for {p.name} (no overlay image available this time, "
                        f"raw coordinates only), from a text-prompted segmentation pass that "
                        f"searched specifically for fastener-like concepts - each is a polygon "
                        f"outline (list of [x,y] points, 0-1000 normalized, NOT a bounding box), "
                        f"hints only, not authoritative: {raw}. You MUST return exactly "
                        f"{len(polygons)} concept_candidate_dispositions entries for {p.name}, "
                        f"one per region, in the same order given here."
                    )
            else:
                contents.append(f"No concept-targeted candidate regions were found for {p.name} - "
                                 f"return an empty concept_candidate_dispositions list for it.")
    contents.append(BATCH_USER_PROMPT)

    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        response_mime_type="application/json",
        response_schema=BatchAnnotation,
        temperature=0,  # real regression observed this session with the default (non-zero)
        # temperature: re-running the SAME image at the SAME PROMPT_VERSION produced a different,
        # rule-violating result (two adjacent bolts on one clamp bracket merged into a single box,
        # instead of one box per fastener) - with no temperature pinned, a before/after prompt
        # comparison can't tell a real prompt regression apart from ordinary sampling noise. Pinning
        # to 0 for a structured-extraction task like this is standard practice and makes future
        # prompt iteration actually attributable.
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
                    by_file[img.file] = BoltAnnotation(
                        boxes=img.boxes,
                        concept_candidate_dispositions=img.concept_candidate_dispositions,
                    )
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
                         help="Ignore cache/sam_regions_concept/ even if that stage was run - send "
                              "images with no candidate hints at all. Use this to A/B-test hint "
                              "impact without deleting that cache.")
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
    # UNNUMBERED overlay dir (written by 03_propose_regions_concept.py specifically for this stage)
    # - deliberately NOT qc/sam_proposals_concept/ (that copy has index numbers burned on for OUR
    # OWN human QC inspection; sending numbers to Gemini risked biasing it toward treating each
    # number as something to draw a box around - real concern raised this session).
    sam_regions_concept_overlay_dir = out / "qc" / "sam_proposals_concept_for_gemini"
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
        force=True,  # a package imported before this point (google-genai/etc.) may already have
                     # attached its own root-logger handler, which makes a plain basicConfig() a
                     # silent no-op - force=True always (re)configures regardless (see
                     # 03_propose_regions_concept.py for the real crash that surfaced this).
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

    # --- Build hints for `pending` from the upstream SAM3 concept-search cache. See
    # _load_hints_for's docstring for the None-vs-empty-dict-vs-per-image-missing semantics.
    concept_hints: Optional[Dict[str, List[List[List[int]]]]] = None
    if pending and not args.skip_region_hints:
        concept_hints = _load_hints_for(pending, sam_regions_concept_cache_dir,
                                         "SAM concept-proposal (run 03_propose_regions_concept.py)")
        if concept_hints is None:
            log.warning("No SAM concept-proposal cache found for ANY pending image - did you run "
                        "03_propose_regions_concept.py first?")
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
            client, pending, args.model, args.retries, args.backoff, concept_hints,
            sam_regions_concept_overlay_dir,
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
                     "boxes": [b.model_dump() for b in ann.boxes],
                     "concept_candidate_dispositions": [
                         d.model_dump() for d in ann.concept_candidate_dispositions
                     ]},
                    indent=2,
                ),
                encoding="utf-8",
            )
            log.info("OK: %s -> %d box(es)", img_path.name, len(ann.boxes))
            # No index field on the model's response by design (see ConceptCandidateDisposition's
            # docstring) - enumerate() here is OUR OWN positional index for the log line only,
            # never something the model referenced or was asked to produce.
            discarded = [(i, d) for i, d in enumerate(ann.concept_candidate_dispositions) if not d.accepted]
            for i, d in discarded:
                log.info("  DISCARDED region %d for %s: %s", i, img_path.name, d.reason)
            if discarded:
                log.info("  %s: %d/%d region(s) discarded (see reasons above)",
                          img_path.name, len(discarded), len(ann.concept_candidate_dispositions))
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
