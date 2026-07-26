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

BATCHING (auto-chunked - explicit instruction: the caller should never have to hand-pick a size)
--------
Every image that needs annotating in a given run (i.e. not already cached under the current
model+taxonomy) is automatically split into as many chunks as needed via
chunk_images_for_budget() to stay under the REAL confirmed input-token cap for --model (see "REAL
INPUT-TOKEN CEILING" below) - you never need to pick a --limit small enough to fit one request;
pass any --limit (or none) and the chunking happens internally, one generate_content() call per
chunk. Each image within a chunk is sent as an "Image: <filename>" text part immediately followed
by a File-API reference to its bytes (see call_gemini_batch), and the model is asked to return one
result per image tagged with that same filename so results can be matched back up within that
chunk. Per-image caching (cache/raw_gemini/<stem>.json, or cache/raw_gemini_pass{N}/ in multi-pass
mode) is unchanged either way, and is what makes a second run - or a run that failed partway
through several chunks - just pick up wherever the last one stopped.

--limit still caps the total image COUNT considered this run (independent of chunking) - the
resumable cache means a second run with a different --limit just covers more, same as always.

Chunk boundaries are planned from a fast offline estimate (no network calls needed to plan
potentially hundreds of images - see ESTIMATED_TOKENS_PER_IMAGE/ESTIMATED_FIXED_TOKENS_PER_CALL);
each actual chunk still gets a REAL count_tokens() pre-flight check inside call_gemini_batch as
the authoritative confirmation before it's ever sent. See Google's own docs for the OTHER hard
per-request ceilings this script does not itself chunk around (still just logged, never enforced):
100MB inline payload cap (https://ai.google.dev/gemini-api/docs/file-input-methods) and 3,600
images/request (https://ai.google.dev/gemini-api/docs/image-understanding) - in practice the
131,072-input-token ceiling is reached first for this pipeline's typical image sizes.

If the response is missing an entry for some image in a chunk's call (truncated output, or the
model dropping/mis-tagging one), that image is logged and written to failures.txt - it is NOT
retried automatically at a smaller size within that chunk. Just re-run the script and the missing
image gets picked up as still-pending (other successful chunks are untouched, thanks to the
per-image cache).

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
    python 04_annotate_with_gemini.py             # full run over everything in npu_bolt/, auto-chunked
    python 04_annotate_with_gemini.py --passes 5  # 5 independent passes -> 06_vote_consensus.py next

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

REAL INPUT-TOKEN CEILING (confirmed via a real 400 ClientError, cross-checked against
https://ai.google.dev/gemini-api/docs/robotics-overview and the DeepMind model card, both agree):
131,072 input tokens / 65,536 output tokens - NOT the 1,048,576 Google's docs briefly (and
incorrectly) showed for this model. Regression on this session's real run_log.txt data: ~2,240
marginal input tokens per image + ~2,700 fixed tokens/call overhead (system prompt + wrapper text),
so the real safe batch size is roughly (131,072 - 2,700) / 2,240 ~= 55-57 images - see
MODEL_MAX_INPUT_TOKENS below and call_gemini_batch's pre-flight count_tokens() check, which reports
the REAL number for a given batch before it's ever sent.

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
on a large chunk.

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
from google.genai import errors as genai_errors

from annotate_common import (
    CLASSES, IMG_EXTS, BoltAnnotation, BatchAnnotation, draw_visualization, filter_valid_boxes,
    _mime_for,
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
# and compared against on the next run - a mismatch (different model or PROMPT_VERSION) is always
# automatically re-annotated in main(), no flag needed.
DEFAULT_MODEL = "gemini-robotics-er-1.6-preview"

# Prefix tagged onto every File API upload's display_name (see call_gemini_batch) - lets the
# finally: cleanup sweep find and delete ANY file this script ever uploaded via a client.files.list()
# scan, even one whose upload() call was interrupted (e.g. Ctrl+C) before its return value could be
# locally tracked for the normal by-name delete. Confirmed via the SDK source
# (google/genai/files.py Files.upload) that the file object - including display_name - is created
# server-side BEFORE the byte upload begins, so it exists to be found even if the upload never
# locally completed.
UPLOADED_FILE_DISPLAY_NAME_PREFIX = "circe-annotate-"

# Google's documented hard per-request ceilings, promoted from comment-only citations into real
# constants so actual usage can be logged against them every run - reference only, never
# enforced/blocked here (this script has no way to know your account's actual live limits, only
# what Google publishes).
GOOGLE_MAX_PAYLOAD_MB = 100.0               # inline payload cap
                                             # https://ai.google.dev/gemini-api/docs/file-input-methods
GOOGLE_MAX_IMAGES_PER_REQUEST = 3600        # https://ai.google.dev/gemini-api/docs/image-understanding

# Real, per-model INPUT/OUTPUT token caps - the gemini-3.5-flash number previously used as a
# stand-in for every model turned out to be genuinely wrong for gemini-robotics-er-1.6-preview: a
# real 400 ClientError this session ("The input token count exceeds the maximum number of tokens
# allowed 131072") confirmed the ACTUAL cap is 131,072 in / 65,536 out - 8x smaller. Cross-checked
# against https://ai.google.dev/gemini-api/docs/robotics-overview ("Input token limit: 131,072 /
# Output token limit: 65,536") and the DeepMind model card (deepmind.google/models/model-cards/
# gemini-robotics-er-1-6 - "128k context window" / "64K token output", same numbers). Google's own
# docs briefly showed 1,048,576 for this exact model before being corrected - see
# https://discuss.ai.google.dev/t/gemini-robotics-er-1-6-preview-input-token-limit-doesnt-match-documentation/140573
# - so a documented number for ONE model is never a safe stand-in for another.
MODEL_MAX_INPUT_TOKENS = {
    "gemini-robotics-er-1.6-preview": 131_072,
}
MODEL_MAX_OUTPUT_TOKENS = {
    "gemini-robotics-er-1.6-preview": 65_536,
}
# Fallback ONLY for a --model not yet confirmed above - gemini-3.5-flash's own documented number,
# explicitly NOT assumed accurate for any other model (see the mismatch that prompted this fix).
FALLBACK_MAX_INPUT_TOKENS = 1_048_576  # https://ai.google.dev/gemini-api/docs/models/gemini-3.5-flash

# Real regression from this session's actual run_log.txt data (7 real batches, 250 images total:
# 92039/40, 70009/30, 70219/30, 69653/30, 92605/40, 92395/40, 92511/40 prompt-tokens/images) - used
# ONLY by chunk_images_for_budget() below to plan chunk boundaries WITHOUT any network call, so
# planning a chunk split over a large --limit doesn't itself require uploading every candidate
# image first. The REAL, authoritative check still happens per actual chunk inside
# call_gemini_batch via a real count_tokens() call before every generate_content() send - this
# estimate only decides where to draw chunk boundaries, it never gates a send.
ESTIMATED_TOKENS_PER_IMAGE = 2240
ESTIMATED_FIXED_TOKENS_PER_CALL = 2700  # system prompt + BATCH_USER_PROMPT + per-image "Image: x" labels
CHUNK_SAFETY_MARGIN = 0.9  # plan chunks to use at most 90% of the real input-token cap

# Matches whichever SYSTEM_PROMPT_V<N> constant is currently active below (SYSTEM_PROMPT = ...).
# Change this to the same number whenever you switch which one is active.
# Bumped 3 -> 4 for the new SYSTEM_PROMPT_V3 (crooked-fastener bolt_defective rule, explicit
# instruction): every image already cached under raw_gemini/*.json is tagged prompt_version=3 from
# when SYSTEM_PROMPT was pointed at V2's content - if this stayed 3, all of that existing cache
# would be wrongly treated as "already current" for the NEW V3 content and silently keep its old
# label, so the crooked-fastener rule would never actually apply to any already-annotated image.
# Bumping to a fresh, never-before-used number correctly flags all existing cache as stale, which
# is now always automatically re-annotated on the next run (real API cost) - no flag needed.
PROMPT_VERSION = 4
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

# SUPERSEDED - kept only so its exact text isn't lost, never re-activate under the name "V3":
# this was the "filter, don't invent" experiment (SAM3-is-comprehensive framing). Re-tested against
# 3 known-bad images: the box-scope fix (AUT-0007) genuinely worked, but the other two (AUT-0006
# phantom-box invention, AUT-0008 wrongful rejection) were UNCHANGED - only 1 of 3 targeted problems
# actually landed, so it was never made active (SYSTEM_PROMPT stayed on V2). Explicit instruction
# reused the name "V3" below for new, unrelated content built on V2 - see SYSTEM_PROMPT_V3 further
# down for what's actually active.
_SYSTEM_PROMPT_V3_FILTER_DONT_INVENT_SUPERSEDED = SYSTEM_PROMPT_V2.replace(
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

# ACTIVE. Explicit instruction: V2 confirmed as the best base so far (the filter-don't-invent
# experiment above didn't pan out); this bumps V2's bolt_defective definition to also catch a
# fastener that's visibly CROOKED/TILTED/not driven straight - e.g. a bolt or nail-like fastener
# leaning at an angle instead of sitting perpendicular/flush to its mounting surface - as its own,
# standalone sufficient condition for bolt_defective, even with no other symptom (no visible
# rotation/gap/bending of the material itself). Real gap being closed: the pre-existing wording
# ("bent... or otherwise mechanically deformed") describes the FASTENER'S OWN material being bent,
# not the whole fastener sitting crooked in its hole/seat - a crooked-but-otherwise-undamaged
# fastener could previously fall through the cracks toward bolt_ok. Geometry/box rules unaffected.
SYSTEM_PROMPT_V3 = SYSTEM_PROMPT_V2.replace(
    """- bolt_defective: a NON-corrosion mechanical problem is visible - rotation, protrusion, backing-out,
  a gap between the fastener and the mating surface (not fully tightened), OR the head/thread/body
  is visibly bent, sheared, cracked, stripped, or otherwise mechanically deformed. Covers both
  "loose" and "damaged" as one class - do not try to distinguish them further.""",
    """- bolt_defective: a NON-corrosion mechanical problem is visible - rotation, protrusion, backing-out,
  a gap between the fastener and the mating surface (not fully tightened), OR the head/thread/body
  is visibly bent, sheared, cracked, stripped, or otherwise mechanically deformed. Covers both
  "loose" and "damaged" as one class - do not try to distinguish them further.
  CROOKED counts on its own, with no other symptom required: if the fastener is visibly NOT sitting
  straight/plumb relative to the surface it's driven into or mounted on - tilted, leaning, angled
  off-axis, the way a crooked/bent nail sits instead of one driven straight in - that alone makes it
  bolt_defective. Do not require a second symptom (rotation, a visible gap, bent material) before
  applying this; crookedness of the fastener's own mounting angle is sufficient by itself.""",
)

# Currently active version - reassign this to switch versions instead of editing prompt text in
# place (standing practice from here on: each future distinct version gets its own preserved
# SYSTEM_PROMPT_V<N> constant above, never edited after being superseded, so a revert is always a
# cheap, honest reassignment here, not a lossy re-edit). Currently V3 (explicit instruction, see
# V3's comment above) - built on V2, the confirmed-best base, adding the crooked-fastener rule to
# bolt_defective.
SYSTEM_PROMPT = SYSTEM_PROMPT_V3


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


def _build_batch_contents(
    client: genai.Client, image_paths: List[Path],
    concept_hints: Optional[Dict[str, List[List[List[int]]]]],
    concept_overlay_dir: Optional[Path],
) -> Tuple[list, List[str], float, float, int]:
    """Uploads every image (and, where hints exist, its overlay) for one batch via the Gemini File
    API and builds the `contents` list that references them - factored out of call_gemini_batch so
    the same upload+build logic isn't duplicated anywhere it's needed again. Returns (contents,
    uploaded_file_names, original_mb, overlay_mb, overlay_count). Does NOT delete the uploaded
    files - that's the caller's responsibility (call_gemini_batch does it in a finally: block once
    it's done with `contents`).

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
    see SYSTEM_PROMPT's "Candidate regions" rule for how this is described to the model.

    IMAGES ARE UPLOADED VIA THE FILE API (client.files.upload), not sent inline as raw bytes - each
    image/overlay is uploaded ONCE, then referenced by URI (types.Part.from_uri) so the caller can
    reuse the SAME contents for both a pre-flight count_tokens() precheck AND the real
    generate_content() call without the same bytes crossing the network twice."""
    contents: list = []
    original_bytes_total = 0
    overlay_bytes_total = 0
    overlay_count = 0
    uploaded_file_names: List[str] = []
    total = len(image_paths)
    for i, p in enumerate(image_paths, start=1):
        # The only per-image signal without this is the google-genai SDK's own raw httpx request
        # logging (bare "POST .../upload/v1beta/files ... 200 OK" lines, no filename/index/total) -
        # real user confusion this session ("it doesn't show any progress thing due x/300
        # womsthing") on a chunk that can be 40+ images, each a slow (~1-5s) resumable upload.
        log.info("[%d/%d] uploading %s", i, total, p.name)
        contents.append(f"Image: {p.name}")
        original_bytes_total += p.stat().st_size
        uploaded = client.files.upload(
            file=p,
            config=types.UploadFileConfig(
                display_name=f"{UPLOADED_FILE_DISPLAY_NAME_PREFIX}{p.name}"
            ),
        )
        uploaded_file_names.append(uploaded.name)
        contents.append(types.Part.from_uri(file_uri=uploaded.uri, mime_type=uploaded.mime_type))
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
                    overlay_bytes_total += overlay_path.stat().st_size
                    overlay_count += 1
                    uploaded_overlay = client.files.upload(
                        file=overlay_path,
                        config=types.UploadFileConfig(
                            display_name=f"{UPLOADED_FILE_DISPLAY_NAME_PREFIX}overlay-{p.name}"
                        ),
                    )
                    uploaded_file_names.append(uploaded_overlay.name)
                    contents.append(types.Part.from_uri(
                        file_uri=uploaded_overlay.uri, mime_type=uploaded_overlay.mime_type
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
    original_mb = original_bytes_total / (1024 * 1024)
    overlay_mb = overlay_bytes_total / (1024 * 1024)
    return contents, uploaded_file_names, original_mb, overlay_mb, overlay_count


def call_gemini_batch(
    client: genai.Client, image_paths: List[Path], model: str, retries: int, backoff: float,
    concept_hints: Optional[Dict[str, List[List[List[int]]]]] = None,
    concept_overlay_dir: Optional[Path] = None,
    temperature: Optional[float] = 0,
) -> Tuple[Dict[str, BoltAnnotation], int, float, float]:
    """One API call annotates ALL of image_paths together. Returns ({filename: BoltAnnotation},
    total_tokens_used, original_images_mb, overlay_images_mb) - tokens/mb are all 0 if the call
    never went through. original_images_mb/overlay_images_mb are the actual payload size of what
    was SENT this call, always computed regardless of whether the call succeeds, so a failed call
    still reports what it tried to send. Any filename the model doesn't return an entry for is
    simply absent from the dict; the caller treats that as a failure for that
    image (same as the old per-image None-return convention, just resolved per-file afterward
    instead of per-call).

    temperature: 0 (the default) pins deterministic output - required for reproducible prompt A/B
    testing (a real regression this session: re-running the SAME image at temperature=0 gave the
    SAME result both times, proving determinism; the default non-zero temperature did NOT). Pass
    None for multi-pass/voting mode instead - N deterministic passes would trivially agree 5/5
    every time, which defeats the entire point of majority voting; None omits `temperature=` from
    GenerateContentConfig entirely, letting the API's own real default (genuine sampling variance)
    produce passes that can actually disagree, which is what voting needs.

    See _build_batch_contents (called at the top of this function) for the concept_hints/
    concept_overlay_dir/File-API-upload documentation - unchanged, just factored out."""
    contents, uploaded_file_names, original_mb, overlay_mb, overlay_count = _build_batch_contents(
        client, image_paths, concept_hints, concept_overlay_dir
    )
    try:
        total_mb = original_mb + overlay_mb
        total_image_parts = len(image_paths) + overlay_count
        log.info("  batch payload: %d original image(s) = %.2f MB, %d overlay image(s) = %.2f MB, "
                  "%.2f MB total (%.1f%% of Google's documented %.0f MB inline-payload cap; "
                  "%d image part(s) total, %.1f%% of the %d images/request cap)",
                  len(image_paths), original_mb, overlay_count, overlay_mb, total_mb,
                  total_mb / GOOGLE_MAX_PAYLOAD_MB * 100, GOOGLE_MAX_PAYLOAD_MB,
                  total_image_parts, total_image_parts / GOOGLE_MAX_IMAGES_PER_REQUEST * 100,
                  GOOGLE_MAX_IMAGES_PER_REQUEST)

        # PRE-FLIGHT real token count, via count_tokens() - NOT an estimate. Uses the SAME uploaded
        # file references already in `contents` (no re-upload). system_instruction can't be passed
        # to count_tokens on the Gemini Developer API (real error hit this session: "system_instruction
        # parameter is only supported in Gemini Enterprise Agent Platform mode") - worked around by
        # counting SYSTEM_PROMPT as plain content on its own and adding the two counts together.
        # Purely informational (explicit instruction: no auto-blocking on this number) -
        # generate_content() below is always still called regardless of what this says.
        input_cap = MODEL_MAX_INPUT_TOKENS.get(model, FALLBACK_MAX_INPUT_TOKENS)
        is_confirmed_cap = model in MODEL_MAX_INPUT_TOKENS
        system_prompt_tokens = client.models.count_tokens(model=model, contents=SYSTEM_PROMPT).total_tokens
        content_tokens = client.models.count_tokens(model=model, contents=contents).total_tokens
        precomputed_total = system_prompt_tokens + content_tokens
        log.info("  batch PRE-FLIGHT token count (real, via count_tokens, before sending): "
                  "content=%s + system_prompt=%s = %s tokens (%.1f%% of the %s%s input-token cap)",
                  content_tokens, system_prompt_tokens, precomputed_total,
                  precomputed_total / input_cap * 100, f"{input_cap:,}",
                  "" if is_confirmed_cap else " fallback, unconfirmed for this model")

        # temperature=0 (single-pass default): real regression observed this session with the
        # default (non-zero) temperature - re-running the SAME image at the SAME PROMPT_VERSION
        # produced a different, rule-violating result (two adjacent bolts on one clamp bracket
        # merged into a single box, instead of one box per fastener); with no temperature pinned, a
        # before/after prompt comparison can't tell a real prompt regression apart from ordinary
        # sampling noise. Pinning to 0 makes future single-pass prompt iteration attributable.
        # temperature=None (multi-pass/voting mode): the caller passes None specifically so this
        # omits `temperature=` entirely, letting the API's real default (genuine sampling variance)
        # produce passes that can actually disagree - see call_gemini_batch's docstring.
        config_kwargs = dict(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=BatchAnnotation,
        )
        if temperature is not None:
            config_kwargs["temperature"] = temperature
        config = types.GenerateContentConfig(**config_kwargs)

        last_err = None
        for attempt in range(1, retries + 1):
            try:
                response = client.models.generate_content(model=model, contents=contents, config=config)
                usage = getattr(response, "usage_metadata", None)
                tokens = getattr(usage, "total_token_count", 0) or 0
                if response.parsed is not None:
                    if usage is not None:
                        prompt_tok = getattr(usage, "prompt_token_count", 0) or 0
                        output_tok = getattr(usage, "candidates_token_count", 0) or 0
                        output_cap = MODEL_MAX_OUTPUT_TOKENS.get(model, 65_536)
                        log.info("  batch tokens (real, post-response): prompt=%s (%.1f%% of the "
                                  "%s%s input-token cap) output=%s (%.1f%% of the %s output-token "
                                  "cap) total=%s",
                                  prompt_tok, prompt_tok / input_cap * 100, f"{input_cap:,}",
                                  "" if is_confirmed_cap else " fallback",
                                  output_tok, output_tok / output_cap * 100, f"{output_cap:,}", tokens)
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
                    return by_file, tokens, original_mb, overlay_mb
                last_err = f"response.parsed was None; raw text: {response.text[:500]!r}"
                # Log IMMEDIATELY, not just once at the very end - previously last_err was only ever
                # surfaced after ALL retries were exhausted, so a run stuck mid-retry gave no visibility
                # into WHY it was failing until it was too late to act on.
                log.error("Batch call attempt %d/%d FAILED for %d image(s): %s",
                          attempt, retries, len(image_paths), last_err)
            except Exception as e:  # noqa: BLE001 - deliberately broad, this is a best-effort batch job
                # Extract EVERY field google-genai actually gives us on an API error (ClientError/
                # ServerError, both subclasses of APIError - google/genai/errors.py) - the HTTP code,
                # Google's own status string, Google's own message, AND the full raw error details dict
                # (which for a 400 typically names the exact violated constraint, e.g. payload/size
                # limits) - not just str(e)/repr(e), which collapse all of that into one opaque line.
                if isinstance(e, genai_errors.APIError):
                    last_err = (f"{type(e).__name__}: HTTP code={e.code} status={e.status!r} "
                                f"message={e.message!r} details={e.details!r}")
                else:
                    last_err = f"{type(e).__name__}: {e}"
                # exc_info=True (only valid while still inside this except block) attaches the FULL
                # traceback to this same log record, so both the structured API-error fields above AND
                # the raw Python stack trace land together in run_log.txt for this exact attempt.
                log.error("Batch call attempt %d/%d FAILED for %d image(s): %s",
                          attempt, retries, len(image_paths), last_err, exc_info=True)
            if attempt < retries:
                time.sleep(backoff * attempt)
        log.error("Batch call FAILED for %d image(s) after %d attempts: %s",
                  len(image_paths), retries, last_err)
        return {}, 0, original_mb, overlay_mb
    finally:
        # Clean up every uploaded file regardless of success/failure above - they'd auto-expire in
        # 48h anyway, so a failed delete is logged but never raised (not worth failing the whole
        # batch result over housekeeping).
        cleanup_total = len(uploaded_file_names)
        for i, name in enumerate(uploaded_file_names, start=1):
            log.info("  [%d/%d] deleting uploaded file %s", i, cleanup_total, name)
            try:
                client.files.delete(name=name)
            except Exception as e:  # noqa: BLE001 - cleanup best-effort, never masks the real result
                log.warning("  failed to delete uploaded file %s (will auto-expire in 48h): %s",
                            name, e)
        # SWEEP for anything the by-name delete above couldn't catch - specifically an upload()
        # call interrupted (e.g. Ctrl+C) before its return value could be appended to
        # uploaded_file_names locally. The file object is created server-side (with our
        # display_name) BEFORE the byte upload begins (confirmed via the SDK source), so it's
        # findable here by display_name prefix even when never locally tracked. Also incidentally
        # sweeps up any leak from a past crashed run. Cheap (files.list, not a generation call) -
        # runs every batch, not just on interrupt, since that's the simplest correct behavior.
        try:
            for f in client.files.list():
                if (f.display_name and f.display_name.startswith(UPLOADED_FILE_DISPLAY_NAME_PREFIX)
                        and f.name not in uploaded_file_names):
                    try:
                        client.files.delete(name=f.name)
                        log.warning("  swept up an untracked uploaded file %s (display_name=%s) - "
                                    "likely orphaned by an interrupt during its own upload() call",
                                    f.name, f.display_name)
                    except Exception as e:  # noqa: BLE001 - sweep cleanup is best-effort too
                        log.warning("  failed to delete swept-up file %s: %s", f.name, e)
        except Exception as e:  # noqa: BLE001 - the sweep itself is best-effort, never fatal
            log.warning("  file cleanup sweep failed (any leftover files will auto-expire in 48h): %s", e)


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


def chunk_images_for_budget(pending: List[Path], model: str) -> List[List[Path]]:
    """Splits `pending` into sub-batches that should each stay under the real input-token cap for
    `model`, using the ESTIMATED_TOKENS_PER_IMAGE/ESTIMATED_FIXED_TOKENS_PER_CALL regression (no
    network calls - this only needs to plan boundaries, not measure exactly; call_gemini_batch
    still does a REAL count_tokens() check per actual chunk before ever sending). Explicit
    instruction this session: the caller (main()) should never have to hand-pick --limit to avoid
    the token cap - "if i say 300 you must handle internally" - this is that internal handling.

    Greedy: walk `pending` in order, add images to the current chunk while
    ESTIMATED_FIXED_TOKENS_PER_CALL + running_count * ESTIMATED_TOKENS_PER_IMAGE stays under
    input_cap * CHUNK_SAFETY_MARGIN; start a new chunk once the next image would exceed it. A
    single image is always its own chunk-of-one even if that alone estimates over budget (the
    estimate is approximate; a real oversized single image is still attempted and let the real
    count_tokens()/generate_content() calls be the actual judge, same as any other real failure)."""
    input_cap = MODEL_MAX_INPUT_TOKENS.get(model, FALLBACK_MAX_INPUT_TOKENS)
    budget = input_cap * CHUNK_SAFETY_MARGIN
    chunks: List[List[Path]] = []
    current: List[Path] = []
    for img_path in pending:
        projected = ESTIMATED_FIXED_TOKENS_PER_CALL + (len(current) + 1) * ESTIMATED_TOKENS_PER_IMAGE
        if current and projected > budget:
            chunks.append(current)
            current = []
        current.append(img_path)
    if current:
        chunks.append(current)
    return chunks


def run_annotation_pass(
    client: genai.Client, images: List[Path], model: str, retries: int, backoff: float,
    skip_region_hints: bool, raw_dir: Path, gemini_raw_dir: Path,
    sam_regions_concept_cache_dir: Path, sam_regions_concept_overlay_dir: Path,
    temperature: Optional[float], pass_label: str,
) -> Tuple[int, int, List[str], int, float, float]:
    """Runs ONE full annotation pass over `images`: cache resolution -> hint loading -> chunked API
    calls (chunk_images_for_budget - internal auto-chunking, the caller never needs to hand-pick a
    size that fits under the token cap) -> per-image JSON write -> QC visualization. Used both for
    the normal single-pass run (raw_dir=cache/raw_gemini, temperature=0) and for each of N
    independent multi-pass runs (raw_dir=cache/raw_gemini_pass{p}, temperature=None - see
    call_gemini_batch's docstring for why). `pass_label` is only for log-line prefixing (e.g.
    "pass 3/5"), does not affect behavior. Returns (done, cached, failures, tokens, original_mb,
    overlay_mb)."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    gemini_raw_dir.mkdir(parents=True, exist_ok=True)

    failures: List[str] = []
    done, cached, total_tokens = 0, 0, 0
    total_original_mb, total_overlay_mb = 0.0, 0.0

    # --- resolve cache status for every image up front. Images that are already usable
    # (current-model-and-taxonomy cache hit) get their result immediately. Everything else goes
    # into `pending`, chunked and sent to the API below.
    results: Dict[Path, BoltAnnotation] = {}
    pending: List[Path] = []

    for img_path in images:
        raw_path = raw_dir / f"{img_path.stem}.json"
        if raw_path.exists():
            cached_data = json.loads(raw_path.read_text(encoding="utf-8"))
            cached_model = cached_data.get("model", "unknown")
            cached_version = cached_data.get("prompt_version", "unknown")
            is_current = cached_model == model and cached_version == PROMPT_VERSION

            if is_current:
                results[img_path] = BoltAnnotation(boxes=cached_data["boxes"])
                cached += 1
                log.info("%s [%s] - SKIP: already annotated by current model+taxonomy (%s, v%d)",
                          img_path.name, pass_label, model, PROMPT_VERSION)
                continue
            else:
                # Stale cache (different model or PROMPT_VERSION) is ALWAYS re-annotated - no flag
                # gating this (removed by explicit instruction: the old --reannotate-stale flag was
                # judged unnecessary complexity, since keeping stale annotations around silently
                # defeats the point of bumping PROMPT_VERSION in the first place).
                log.info("%s [%s] - RE-ANNOTATE: was model=%s taxonomy=v%s, redoing with model=%s "
                          "taxonomy=v%d (stale cache is always redone)",
                          img_path.name, pass_label, cached_model, cached_version, model, PROMPT_VERSION)
        pending.append(img_path)

    # --- Build hints for `pending` from the upstream SAM3 concept-search cache. See
    # _load_hints_for's docstring for the None-vs-empty-dict-vs-per-image-missing semantics.
    concept_hints: Optional[Dict[str, List[List[List[int]]]]] = None
    if pending and not skip_region_hints:
        concept_hints = _load_hints_for(pending, sam_regions_concept_cache_dir,
                                         "SAM concept-proposal (run 03_propose_regions_concept.py)")
        if concept_hints is None:
            log.warning("[%s] No SAM concept-proposal cache found for ANY pending image - did you "
                        "run 03_propose_regions_concept.py first?", pass_label)
    elif pending:
        log.info("[%s] --skip-region-hints set - sending %d image(s) to %s with no segmentation "
                  "hints", pass_label, len(pending), model)

    # --- Chunked API calls: chunk_images_for_budget splits `pending` internally so the caller
    # never has to hand-pick a --limit that fits under the token cap ("if i say 300 you must
    # handle internally" - explicit instruction). Each chunk still gets its own REAL count_tokens()
    # pre-flight check inside call_gemini_batch as the authoritative confirmation.
    if pending:
        chunks = chunk_images_for_budget(pending, model)
        log.info("[%s] ANNOTATE: %d image(s) split into %d chunk(s) to stay under the real "
                  "input-token cap (sizes: %s)", pass_label, len(pending), len(chunks),
                  ", ".join(str(len(c)) for c in chunks))
        for chunk_i, chunk in enumerate(chunks, 1):
            log.info("[%s] chunk %d/%d (%d image(s)): %s", pass_label, chunk_i, len(chunks),
                      len(chunk), ", ".join(p.name for p in chunk))
            by_file, tokens, original_mb, overlay_mb = call_gemini_batch(
                client, chunk, model, retries, backoff, concept_hints,
                sam_regions_concept_overlay_dir, temperature,
            )
            total_tokens += tokens
            total_original_mb += original_mb
            total_overlay_mb += overlay_mb

            for img_path in chunk:
                ann = by_file.get(img_path.name)
                if ann is None:
                    log.error("[%s] FAILED %s: not present in batch response",
                              pass_label, img_path.name)
                    failures.append(img_path.name)
                    continue
                raw_path = raw_dir / f"{img_path.stem}.json"
                raw_path.write_text(
                    json.dumps(
                        {"model": model, "prompt_version": PROMPT_VERSION,
                         "boxes": [b.model_dump() for b in ann.boxes],
                         "concept_candidate_dispositions": [
                             d.model_dump() for d in ann.concept_candidate_dispositions
                         ]},
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                log.info("[%s] OK: %s -> %d box(es)", pass_label, img_path.name, len(ann.boxes))
                # No index field on the model's response by design (see ConceptCandidateDisposition's
                # docstring) - enumerate() here is OUR OWN positional index for the log line only,
                # never something the model referenced or was asked to produce.
                discarded = [(i, d) for i, d in enumerate(ann.concept_candidate_dispositions) if not d.accepted]
                for i, d in discarded:
                    log.info("  [%s] DISCARDED region %d for %s: %s", pass_label, i, img_path.name, d.reason)
                if discarded:
                    log.info("  [%s] %s: %d/%d region(s) discarded (see reasons above)",
                              pass_label, img_path.name, len(discarded),
                              len(ann.concept_candidate_dispositions))
                results[img_path] = ann
                done += 1
    else:
        log.info("[%s] Nothing to annotate - all images already have usable cached results.",
                  pass_label)

    # --- Regenerate gemini_raw_dir for every image that has a result (cheap, no API calls,
    # applies uniformly to fresh/cached results) - Gemini's boxes exactly as returned (structurally
    # valid ones only), before 05_tighten_boxes.py touches geometry at all.
    for img_path in images:
        result = results.get(img_path)
        if result is None:
            continue
        result = filter_valid_boxes(result, img_path.name)
        draw_visualization(img_path, result, gemini_raw_dir / img_path.name)

    return done, cached, failures, total_tokens, total_original_mb, total_overlay_mb


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path, default=Path(__file__).parent / "circe_datasets" / "npu_bolt" / "working_images")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "circe_datasets" / "npu_bolt")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                         help=f"Gemini model ID to call (default {DEFAULT_MODEL!r}). Switching "
                              "models never needs a code edit - just pass a different --model. "
                              "Images cached under a different model or PROMPT_VERSION are always "
                              "automatically re-annotated (real API cost) - no flag needed.")
    parser.add_argument("--skip-region-hints", action="store_true",
                         help="Ignore cache/sam_regions_concept/ even if that stage was run - send "
                              "images with no candidate hints at all. Use this to A/B-test hint "
                              "impact without deleting that cache.")
    parser.add_argument("--limit", type=int, default=None,
                         help="Only consider the first N images from --src this run. Batch SIZE "
                              "within that is now handled internally (see --passes/chunking below) "
                              "- --limit only caps the total image COUNT considered.")
    parser.add_argument("--retries", type=int, default=3,
                         help="Retries apply to a WHOLE chunk's API call, not per-image - if it "
                              "fails outright it's retried (same request) up to this many times "
                              "before that chunk is treated as failed. This does not make extra "
                              "calls to cover partial misses - it only re-sends an already-failed "
                              "request.")
    parser.add_argument("--backoff", type=float, default=3.0, help="Base seconds for retry backoff")
    parser.add_argument("--passes", type=int, default=1,
                         help="Run this many INDEPENDENT annotation passes over the same image set "
                              "instead of one (explicit instruction: for majority-vote consensus - "
                              "run 06_vote_consensus.py after this to merge them). --passes 1 "
                              "(default) is unchanged existing behavior: temperature=0 "
                              "(deterministic), writes straight to cache/raw_gemini/. --passes > 1 "
                              "uses the API's real non-zero default temperature instead - N "
                              "deterministic passes would trivially agree 5/5 every time, which "
                              "defeats majority voting entirely - and each pass writes to its OWN "
                              "cache/raw_gemini_pass{N}/ + qc/gemini_raw_pass{N}/ (never touches "
                              "cache/raw_gemini/ directly in this mode - that's "
                              "06_vote_consensus.py's job, once all passes are done). Every pass's "
                              "raw output is kept on disk for manual QC, not just the consensus.")
    parser.add_argument("--clean", action="store_true",
                         help="DANGER: a full clean run. With --passes 1 (default): deletes "
                              "qc/gemini_raw/, failures.txt, AND cache/raw_gemini/ before running. "
                              "With --passes > 1: deletes failures.txt AND every "
                              "cache/raw_gemini_pass*/ + qc/gemini_raw_pass*/ folder found on disk "
                              "(glob-matched - also cleans up leftovers from a run made with a "
                              "DIFFERENT --passes count) - does NOT touch cache/raw_gemini/ in this "
                              "mode, since in multi-pass mode that's 06_vote_consensus.py's output, "
                              "not this script's.")
    args = parser.parse_args()

    src = args.src.resolve()
    out = args.out.resolve()
    sam_regions_concept_cache_dir = out / "cache" / "sam_regions_concept"
    # UNNUMBERED overlay dir (written by 03_propose_regions_concept.py specifically for this stage)
    # - deliberately NOT qc/sam_proposals_concept/ (that copy has index numbers burned on for OUR
    # OWN human QC inspection; sending numbers to Gemini risked biasing it toward treating each
    # number as something to draw a box around - real concern raised this session).
    sam_regions_concept_overlay_dir = out / "qc" / "sam_proposals_concept_for_gemini"

    if args.clean:
        (out / "failures.txt").unlink(missing_ok=True)
        if args.passes == 1:
            shutil.rmtree(out / "qc" / "gemini_raw", ignore_errors=True)
            shutil.rmtree(out / "cache" / "raw_gemini", ignore_errors=True)
        else:
            for d in (out / "cache").glob("raw_gemini_pass*"):
                shutil.rmtree(d, ignore_errors=True)
            for d in (out / "qc").glob("gemini_raw_pass*"):
                shutil.rmtree(d, ignore_errors=True)

    (out / "cache").mkdir(parents=True, exist_ok=True)
    (out / "qc").mkdir(parents=True, exist_ok=True)

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
        if args.passes == 1:
            log.info("--clean: wiped qc/gemini_raw/, failures.txt, AND cache/raw_gemini/ - full "
                      "re-annotation ahead for every image in this run.")
        else:
            log.info("--clean: wiped every cache/raw_gemini_pass*/ + qc/gemini_raw_pass*/ + "
                      "failures.txt - full re-annotation ahead for all %d passes this run "
                      "(cache/raw_gemini/ untouched - that's 06_vote_consensus.py's output).",
                      args.passes)

    client = genai.Client()

    images = sorted(f for f in src.iterdir() if f.is_file() and f.suffix.lower() in IMG_EXTS)
    if args.limit:
        images = images[: args.limit]
    log.info("Images found: %d", len(images))

    # Cost/RPD visibility (informational only, no auto-blocking - explicit earlier instruction
    # against fail-fast/pre-flight blocking): an upper-bound estimate BEFORE the per-pass cache is
    # even resolved (a real run may end up cheaper once already-annotated images are skipped), so
    # the real scale of a "300 images x 5 passes" run is visible up front, not discovered mid-run.
    est_chunks_per_pass = len(chunk_images_for_budget(images, args.model)) if images else 0
    est_total_calls = est_chunks_per_pass * args.passes
    log.info("Cost estimate (upper bound, before cache is applied): up to %d chunk(s)/pass x %d "
              "pass(es) = up to %d generate_content call(s) this run. Check your account's real "
              "RPD for %s at https://aistudio.google.com/rate-limit?timeRange=last-28-days.",
              est_chunks_per_pass, args.passes, est_total_calls, args.model)

    if args.passes == 1:
        raw_dir = out / "cache" / "raw_gemini"
        gemini_raw_dir = out / "qc" / "gemini_raw"
        done, cached, failures, total_tokens, total_original_mb, total_overlay_mb = run_annotation_pass(
            client, images, args.model, args.retries, args.backoff, args.skip_region_hints,
            raw_dir, gemini_raw_dir, sam_regions_concept_cache_dir, sam_regions_concept_overlay_dir,
            temperature=0, pass_label="single-pass",
        )
        if failures:
            (out / "failures.txt").write_text("\n".join(failures), encoding="utf-8")

        total_mb_run = total_original_mb + total_overlay_mb
        run_input_cap = MODEL_MAX_INPUT_TOKENS.get(args.model, FALLBACK_MAX_INPUT_TOKENS)
        log.info("Done. %d newly/re-annotated, %d cached (current model+taxonomy), %d failed, "
                  "%d tokens used this run (%.1f%% of the %s input-token cap%s; real API "
                  "calls only - cached images cost nothing), %.2f MB original + %.2f MB overlay = "
                  "%.2f MB images sent this run (%.1f%% of Google's %.0f MB inline-payload cap).",
                  done, cached, len(failures), total_tokens,
                  total_tokens / run_input_cap * 100, f"{run_input_cap:,}",
                  "" if args.model in MODEL_MAX_INPUT_TOKENS else " fallback, unconfirmed for this model",
                  total_original_mb, total_overlay_mb,
                  total_mb_run, total_mb_run / GOOGLE_MAX_PAYLOAD_MB * 100, GOOGLE_MAX_PAYLOAD_MB)
        if failures:
            log.info("Failures logged to %s", out / "failures.txt")
        log.info("Raw results cached at %s. Run 05_tighten_boxes.py next to produce dataset/ + final qc/.",
                  raw_dir)
    else:
        grand_done, grand_cached, grand_failed, grand_tokens = 0, 0, 0, 0
        grand_original_mb, grand_overlay_mb = 0.0, 0.0
        all_failures: List[str] = []
        for p in range(1, args.passes + 1):
            pass_label = f"pass {p}/{args.passes}"
            log.info("=== Starting %s ===", pass_label)
            raw_dir = out / "cache" / f"raw_gemini_pass{p}"
            gemini_raw_dir = out / "qc" / f"gemini_raw_pass{p}"
            done, cached, failures, tokens, original_mb, overlay_mb = run_annotation_pass(
                client, images, args.model, args.retries, args.backoff, args.skip_region_hints,
                raw_dir, gemini_raw_dir, sam_regions_concept_cache_dir,
                sam_regions_concept_overlay_dir, temperature=None, pass_label=pass_label,
            )
            grand_done += done
            grand_cached += cached
            grand_failed += len(failures)
            grand_tokens += tokens
            grand_original_mb += original_mb
            grand_overlay_mb += overlay_mb
            all_failures.extend(f"{pass_label}: {name}" for name in failures)
            log.info("=== %s done: %d newly-annotated, %d cached, %d failed, %d tokens ===",
                      pass_label, done, cached, len(failures), tokens)

        if all_failures:
            (out / "failures.txt").write_text("\n".join(all_failures), encoding="utf-8")

        total_mb_run = grand_original_mb + grand_overlay_mb
        run_input_cap = MODEL_MAX_INPUT_TOKENS.get(args.model, FALLBACK_MAX_INPUT_TOKENS)
        log.info("Done. %d pass(es) complete. %d newly/re-annotated, %d cached, %d failed "
                  "(summed across all passes), %d tokens used this run (real API calls only), "
                  "%.2f MB original + %.2f MB overlay = %.2f MB images sent this run (%.1f%% of "
                  "Google's %.0f MB inline-payload cap, summed across all chunks/passes).",
                  args.passes, grand_done, grand_cached, grand_failed, grand_tokens,
                  grand_original_mb, grand_overlay_mb, total_mb_run,
                  total_mb_run / GOOGLE_MAX_PAYLOAD_MB * 100, GOOGLE_MAX_PAYLOAD_MB)
        if all_failures:
            log.info("Failures logged to %s", out / "failures.txt")
        log.info("Per-pass results cached at %s/cache/raw_gemini_pass{1..%d}/. Run "
                  "06_vote_consensus.py next to majority-vote them into cache/raw_gemini/, then "
                  "05_tighten_boxes.py to produce dataset/ + final qc/.", out, args.passes)


if __name__ == "__main__":
    main()
