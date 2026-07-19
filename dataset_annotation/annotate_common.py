"""Shared constants, pydantic models, and helper functions for the dataset_annotation pipeline,
split across four staged scripts so expensive SAM inference and paid Gemini calls are never
re-run just to test a downstream stage:

  02_propose_regions_dumb.py    - SAM2 promptless/automatic region-proposal pass (local, free)
  03_propose_regions_concept.py - SAM3 text-prompted concept region-proposal pass (local, free)
  04_annotate_with_gemini.py    - the one Gemini API call (costs real quota)
  05_tighten_boxes.py           - SAM3 box-tightening pass + final dataset/qc output (local, free)

Each stage reads an earlier stage's cache/ output and is independently re-runnable without
re-paying for upstream stages: re-running 05 after tweaking tightening/QC-drawing code does not
re-call Gemini or re-run SAM proposals; changing --concepts and re-running 03 does not touch
cache/raw_gemini or trigger a new Gemini call.

Both SAM roles use the OFFICIAL Meta packages directly (facebookresearch/sam3,
facebookresearch/sam2), not the `ultralytics` wrapper this pipeline used earlier this session.
Switched after two real problems hit with the ultralytics path specifically: (1) a cascading CUDA
OOM on a 100-image run; (2) repeated friction fighting ultralytics' own CLI-style argument
whitelist (`points_stride`/`crop_n_layers` rejected by `check_dict_alignment()` - confirmed against
`ultralytics/cfg/default.yaml`), which made SAM2's grid-density tuning unreachable. The official
SAM2 `SAM2AutomaticMaskGenerator` class takes `points_per_side`/`crop_n_layers` etc. as genuine,
direct constructor arguments - no whitelist in the way.

The cascading CUDA OOM (item 1) reproduced on the OFFICIAL SAM2 package too, and this time the real
root cause got fully verified, not just theorized: on a real `--limit 10` run, images 1-7 (all
640x640 test images) succeeded; image 8 (`AUT-0000.jpg`, a real field photo at 2736x3648 - ~24x
more pixels) failed, and every image after it failed identically. SAM2's automatic mode does NOT
downscale its input - it runs the full point-grid pipeline (1024 points at points_per_side=32) and
upsamples every candidate mask back to the ORIGINAL resolution before filtering, directly at
whatever size it's given. At ~10 megapixels on a 6GB card that's a genuine, one-shot allocation
failure - proven (not assumed) by the fact that the very next call, `torch.cuda.empty_cache()`
itself (a pure deallocation call, allocates nothing), threw the identical CUDA error: that only
happens when the CUDA context is already corrupted/poisoned, not when there's merely not enough
free memory. `_clear_cuda_cache()` (below) is real, correct, PyTorch-recommended practice in
general, but it cannot fix a poisoned context - only preventing the original OOM can. Fix: SAM2's
automatic pass now downscales the image to DUMB_MAX_DIM before calling generate() (matching what
SAM3's own official Sam3Processor already does internally regardless of input size - confirmed via
source read that it resizes to a fixed 1008x1008 via its own transform pipeline, which is why SAM3
never hit this specific failure), then resizes returned masks back up to the real original
resolution via the existing _resize_mask_to_shape helper before anything downstream sees them.

Not meant to be run directly - `python annotate_common.py` does nothing useful.
"""
import logging
from pathlib import Path
from typing import Dict, List, Literal, Optional

import cv2
import numpy as np
from pydantic import BaseModel, Field

log = logging.getLogger("annotate")

DEFAULT_CONCEPTS = ["bolt", "screw", "nut", "fastener", "rivet"]  # text prompts for SAM3 concept
                                                                   # search - override via --concepts
DEFAULT_SAM2_MODEL_ID = "facebook/sam2.1-hiera-tiny"  # smallest/fastest SAM2.1 variant on the Hub -
                                                        # confirmed real repo ID this session (not
                                                        # gated, auto-downloads via huggingface_hub)
DEFAULT_POINTS_PER_SIDE = 32  # SAM2AutomaticMaskGenerator's own default grid density - a REAL,
                               # directly-reachable tuning knob here (unlike the ultralytics dead
                               # end) - lower this via --points-per-side if a run is too slow
DUMB_MAX_DIM = 1024  # SAM2's automatic mode does NOT downscale internally (unlike SAM3's
                      # Sam3Processor, which always resizes to a fixed 1008x1008) - real photos at
                      # ~10 megapixels caused a genuine CUDA OOM that also corrupted the CUDA
                      # context (verified: even torch.cuda.empty_cache() itself then failed).
                      # propose_regions_dumb resizes the long edge down to this before calling
                      # generate(), then resizes returned masks back up to the real resolution.
MIN_REGION_AREA_FRAC = 0.0005  # drop SAM2 masks smaller than this fraction of image area -
                                # noise/speckle (the only filter available for the dumb pass - no
                                # semantic signal like the concept pass has)
MAX_REGION_AREA_FRAC = 0.05    # drop masks larger than this fraction of image area - background/
                                # large structural regions, not a single small fastener. Lowered
                                # from 0.15 after real visual QC evidence this session: masks that
                                # size were consistently sky/pole/decking/cable-wrap background,
                                # never real hardware, on the inspected images.
MAX_REGION_HINTS = 30           # cap candidates sent as text per image, keeps the hint compact and
                                 # avoids drowning the model in low-value candidates
MAX_POLYGON_POINTS = 12         # cap per-candidate polygon vertex count via progressive
                                 # simplification (cv2.approxPolyDP) - still a real simplified
                                 # outline, NOT a box conversion, just fewer points on that outline

Label = Literal["bolt_ok", "bolt_defective", "bolt_corroded"]
CLASSES = ["bolt_ok", "bolt_defective", "bolt_corroded"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# "Traffic light" convention for the QC overlay: green=ok, dark red=defective (mechanical),
# rust-brown=corroded (rust literally is this color, easy to eyeball-confirm against the photo).
BOX_COLORS = {
    "bolt_ok": (40, 180, 40),
    "bolt_defective": (220, 20, 20),
    "bolt_corroded": (150, 90, 20),
}

# Deliberately outside BOX_COLORS' traffic-light palette (green/red/brown) so proposals read as
# "not yet classified" at a glance, not a 4th defect class. Two distinct colors so the two SAM
# sources stay visually distinguishable (saved in separate qc/ folders either way).
SAM_CONCEPT_COLOR = (255, 220, 0)   # BGR - bright cyan - text-prompted/concept-targeted (SAM3)
SAM_DUMB_COLOR = (0, 200, 255)      # BGR - amber/orange - promptless/automatic (SAM2)


class BoltBox(BaseModel):
    box_2d: List[int] = Field(description="[ymin, xmin, ymax, xmax], normalized 0-1000")
    label: Label


class ConceptCandidateDisposition(BaseModel):
    """Accountability record for ONE highlighted concept-targeted (SAM3) region: real user feedback
    this session was that Gemini appeared to silently ignore correct SAM3 hints on at least one
    image, with no way to tell whether a given region was considered-and-rejected or never looked
    at. Every highlighted region given for an image must get exactly one of these - no silent
    drops. Deliberately has NO index/number field - an earlier version numbered each region and
    burned that number onto the overlay image sent to Gemini, and real user concern this session
    was that literal numbers on the photo could bias the model toward treating each number as
    something to draw a box around, rather than judging the highlighted region on its own visual
    merits. Ordering alone (same order regions were given) is enough to line dispositions back up
    to regions in code - the model never needs to name or number which region it means."""
    accepted: bool = Field(description="True if this region was judged to contain a genuine "
                                        "fastener and became one of this image's output boxes")
    reason: str = Field(description="If accepted: brief, e.g. 'matched a real fastener'. If NOT "
                                     "accepted (this region is not a bolt): a specific, concrete "
                                     "reason - not a generic phrase like 'not a fastener'. E.g. "
                                     "'background/shadow, no hardware visible here', 'cable clamp "
                                     "body, excluded by rule, not a structural bolt', 'same "
                                     "fastener as another region already boxed', 'too "
                                     "blurred/occluded to classify confidently'.")


class BoltAnnotation(BaseModel):
    boxes: List[BoltBox]
    concept_candidate_dispositions: List[ConceptCandidateDisposition] = Field(default_factory=list)


class ImageBoxes(BaseModel):
    """One image's worth of results within a batched call, tagged with the filename so the
    caller can match it back up to the right Path (Gemini gives no other positional guarantee
    that's safe to rely on)."""
    file: str = Field(description="Exact filename as given in that image's 'Image: <filename>' label")
    boxes: List[BoltBox]
    concept_candidate_dispositions: List[ConceptCandidateDisposition] = Field(
        description="EXACTLY one entry per highlighted concept-targeted region given for this "
                    "image, in the same order those regions were shown - omit nothing, even if "
                    "this image had zero such regions (then this is just an empty list)."
    )


class BatchAnnotation(BaseModel):
    images: List[ImageBoxes]


# Old label name -> current label name, for lossless remaps only (bolt_defective IS the union of
# what used to be bolt_loose+bolt_damaged, so this loses no information for the ok/corroded/
# mechanical-vs-not distinction). What it can't recover: under the old priority order
# (bolt_damaged > bolt_loose > bolt_corroded), a fastener that was BOTH corroded and loose/damaged
# would have been labeled loose/damaged, not corroded - the current v2 priority would call that
# bolt_corroded instead. That handful of double-defect cases stays slightly wrong until re-run with
# --reannotate-stale; not worth an API call to fix automatically without being asked. Safe/no-op to
# apply unconditionally to already-current labels (they just won't match any key here).
LABEL_MIGRATIONS = {"bolt_loose": "bolt_defective", "bolt_damaged": "bolt_defective"}


def migrate_boxes(boxes: list, image_name: str) -> list:
    migrated = []
    for b in boxes:
        label = LABEL_MIGRATIONS.get(b["label"], b["label"])
        if label not in CLASSES:
            log.warning("  dropping one box on %s: old label %r has no mapping to the current "
                        "taxonomy %s", image_name, b["label"], CLASSES)
            continue
        migrated.append({"box_2d": b["box_2d"], "label": label})
    return migrated


def _mime_for(image_path: Path) -> str:
    return "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"


# --- Local SAM helpers (official facebookresearch/sam3 + facebookresearch/sam2 packages) ---
# Both are imported lazily (inside the two _load_* functions, not at module level) so
# 04_annotate_with_gemini.py (the pure-API stage) never needs either installed at all.

_SAM3_CACHE: Dict[str, object] = {}
_SAM2_CACHE: Dict[tuple, object] = {}


def _load_sam3_processor(checkpoint_path: Optional[str] = None, confidence_threshold: float = 0.5):
    """Lazy singleton (one per checkpoint_path+confidence_threshold combination): the official
    SAM3 model + Sam3Processor, loaded once per process. Used by 05_tighten_boxes.py's tighten_box
    (box-prompted tightening) - always called with its own explicit threshold
    (DEFAULT_SAM3_TIGHTEN_CONFIDENCE_THRESHOLD), so 0.5 here is just Sam3Processor's own real
    default, never actually relied on. (03_propose_regions_concept.py builds its own
    Sam3Processor directly now instead of going through this cache - see that script's docstring
    for why: the earlier version routed images through _load_rgb's numpy array into set_image(),
    which silently misreads an HWC array's channel count as its width - real bug, fixed by passing
    a PIL Image straight through instead.)

    checkpoint_path=None (default): build_sam3_image_model() auto-downloads the gated sam3
    checkpoint via huggingface_hub - requires `hf auth login` with access already granted on the
    gated facebook/sam3 Hugging Face repo. checkpoint_path=<local .pt path>: skips the HF
    download/auth entirely and loads that file directly - confirmed via source read of
    sam3/model_builder.py:build_sam3_image_model that passing checkpoint_path explicitly bypasses
    the `if load_from_HF and checkpoint_path is None: ... download_ckpt_from_hf(...)` branch
    completely, going straight to loading the given file. Use this if you already have sam3.pt
    downloaded manually (e.g. via a browser after accepting the gated-access terms) rather than
    through an authenticated huggingface_hub CLI session - having the file on disk does NOT mean
    huggingface_hub has a cached token; those are two independent things.

    confidence_threshold: passed straight to Sam3Processor's own constructor argument of the same
    name - tighten_box always passes DEFAULT_SAM3_TIGHTEN_CONFIDENCE_THRESHOLD explicitly, so this
    function's own default is never actually relied on."""
    key = (checkpoint_path, confidence_threshold)
    if key not in _SAM3_CACHE:
        try:
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
        except ImportError as e:
            raise SystemExit(
                f"Official sam3 package required: git clone https://github.com/facebookresearch/sam3 "
                f"&& cd sam3 && pip install -e . (Python >= 3.12, torch >= 2.10 required - see "
                f"README.md Setup). Also requires `hf auth login` with access already granted to the "
                f"gated facebook/sam3 Hugging Face repo, UNLESS you pass --sam3-checkpoint pointing "
                f"at an already-downloaded local sam3.pt. Import error: {e!r}"
            ) from e
        if checkpoint_path:
            log.info("Loading SAM3 model from local checkpoint %s (skips HF download/auth), "
                      "confidence_threshold=%.2f...", checkpoint_path, confidence_threshold)
        else:
            log.info("Loading SAM3 model (official facebookresearch/sam3, auto-downloads via "
                      "huggingface_hub on first use - requires prior `hf auth login`), "
                      "confidence_threshold=%.2f...", confidence_threshold)
        model = build_sam3_image_model(checkpoint_path=checkpoint_path)
        _SAM3_CACHE[key] = Sam3Processor(model, confidence_threshold=confidence_threshold)
    return _SAM3_CACHE[key]


def _load_sam2_generator(model_id: str, points_per_side: int):
    """Lazy singleton per (model_id, points_per_side): the official SAM2AutomaticMaskGenerator,
    used by 02_propose_regions_dumb.py for the promptless/automatic proposal pass. Built
    explicitly in two steps (build_sam2_hf then the generator constructor directly) rather than
    relying on SAM2AutomaticMaskGenerator.from_pretrained(model_id, **kwargs)'s single kwargs dict
    forwarding to BOTH build_sam2_hf(...) and the generator __init__ - confirmed via source read
    of sam2/build_sam.py that build_sam2_hf's own signature doesn't accept points_per_side, so
    that forwarding is ambiguous/fragile for generator-only kwargs like this one. Not gated - no
    HF access request needed, unlike SAM3."""
    key = (model_id, points_per_side)
    if key not in _SAM2_CACHE:
        try:
            from sam2.build_sam import build_sam2_hf
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        except ImportError as e:
            raise SystemExit(
                f"Official sam2 package required: git clone https://github.com/facebookresearch/sam2 "
                f"&& cd sam2 && pip install -e . (Python >= 3.10, torch >= 2.5.1 required). Import "
                f"error: {e!r}"
            ) from e
        log.info("Loading SAM2 model %s (points_per_side=%d)...", model_id, points_per_side)
        sam_model = build_sam2_hf(model_id)
        _SAM2_CACHE[key] = SAM2AutomaticMaskGenerator(sam_model, points_per_side=points_per_side)
    return _SAM2_CACHE[key]


def _sam3_autocast():
    """Context manager for every SAM3 forward pass (set_image/set_text_prompt/
    add_geometric_prompt). Real, sourced fix - not guessed: confirmed via
    github.com/facebookresearch/sam3/issues/526 that Sam3Processor.set_image does not itself wrap
    the encoder call in torch.autocast(dtype=torch.bfloat16), while the model's own weights are
    genuinely mixed-precision by design (some layers meant to run in bf16 under autocast - other
    SAM3 predictor classes in this same package, e.g. sam3_base_predictor.py/
    sam3_tracking_predictor.py, DO enter this exact autocast context themselves, confirmed via
    source read - Sam3Processor is the one that's missing it). An earlier attempt to fix this by
    forcing the whole model to a single dtype via model.float() was wrong - it fights the model's
    intended mixed-precision design rather than fixing the actual gap (the missing autocast
    context), and per the GitHub issue, `.float()` alone does not fully resolve it anyway.
    device_type="cuda" is hardcoded since this whole pipeline requires a CUDA GPU regardless."""
    import torch
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def _clear_cuda_cache() -> None:
    """Called after every image's SAM call, in all three per-image SAM functions below. Real,
    PyTorch-recommended practice in general (torch.cuda.empty_cache() returns cached-but-unused
    blocks to PyTorch's own allocator pool) - but it cannot fix an already-poisoned CUDA context
    (confirmed this session: a genuine large-image OOM in propose_regions_dumb once left the
    context broken enough that even this call itself raised the same CUDA error - see
    DUMB_MAX_DIM's comment block for the real fix to that root cause). Wrapped in its own
    try/except so a still-poisoned context (from some other future cause) logs and lets the run
    continue - a best-effort cleanup call should never be what crashes the whole batch. Lazy-
    imports torch and no-ops if CUDA isn't available."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    except Exception as e:  # noqa: BLE001 - a cleanup call must never crash the run
        log.warning("  torch.cuda.empty_cache() itself failed: %r - CUDA context may be in a bad "
                    "state; subsequent images may also fail until the process is restarted", e)


def _load_rgb(image_path: Path) -> Optional[np.ndarray]:
    """Loads via PIL, matching the official SAM3/SAM2 README examples exactly (both packages'
    own basic-usage code loads with PIL.Image.open(...).convert("RGB")) - reverted from a cv2-based
    version this session per explicit instruction to match the official path, even though a direct
    byte-for-byte comparison confirmed this session that cv2.imread+cvtColor(BGR2RGB) produces an
    identical array to PIL for a real test image (so this wasn't fixing a real bug - it's about
    matching the documented/expected loading path exactly, not correctness). Returns an HWC uint8
    RGB array, or None if the file can't be opened."""
    try:
        from PIL import Image
        return np.array(Image.open(image_path).convert("RGB"))
    except Exception as e:  # noqa: BLE001 - a bad image file must not abort the whole run
        log.warning("  Could not open %s as an image: %r", image_path.name, e)
        return None


def _resize_mask_to_shape(mask: np.ndarray, target_shape: tuple) -> np.ndarray:
    """SAM/segmentation models frequently run inference at a different internal resolution than
    the input image (sometimes with letterbox padding) - if a mask's own array shape doesn't match
    the original image's actual pixel dimensions, coordinates derived directly from that raw shape
    would be silently WRONG (scaled and/or offset relative to the real image), not just imprecise.
    Resizes (nearest-neighbor - it's a binary mask, no blur wanted) to target_shape=(height, width)
    so every downstream coordinate is guaranteed correct relative to the real image, regardless of
    whatever internal resolution the model actually used. In practice this is a no-op for SAM3
    masks (its own pipeline already interpolates back to the original image size - confirmed via
    source read of Sam3Processor._forward_grounding) but is kept as a defensive guarantee rather
    than assumed, and SAM2's masks do need it (native mask resolution, not pre-resized)."""
    arr = np.asarray(mask)
    th, tw = int(target_shape[0]), int(target_shape[1])
    if arr.shape[:2] == (th, tw):
        return arr.astype(bool)
    resized = cv2.resize(arr.astype(np.uint8), (tw, th), interpolation=cv2.INTER_NEAREST)
    return resized.astype(bool)


def _mask_to_box_2d(mask: np.ndarray, target_shape: tuple) -> Optional[List[int]]:
    """mask: 2D array, truthy where the segmented object is. target_shape: (height, width) of the
    ORIGINAL image this mask corresponds to (see _resize_mask_to_shape - always required now, not
    optional, since skipping the resize is exactly the bug that produced wrong boxes before this
    was added). Returns [ymin,xmin,ymax,xmax] normalized 0-1000, or None if the mask has no true
    pixels (degenerate/empty - can happen if a box prompt lands somewhere SAM can't segment)."""
    arr = _resize_mask_to_shape(mask, target_shape)
    ys, xs = np.where(arr)
    if ys.size == 0:
        return None
    h, w = arr.shape
    ymin, ymax = int(ys.min()), int(ys.max())
    xmin, xmax = int(xs.min()), int(xs.max())
    return [
        round(ymin / h * 1000),
        round(xmin / w * 1000),
        round(min(h, ymax + 1) / h * 1000),
        round(min(w, xmax + 1) / w * 1000),
    ]


def _mask_to_polygon(mask: np.ndarray, target_shape: tuple) -> Optional[List[List[int]]]:
    """mask: 2D array, truthy where the segmented object is. Returns a simplified polygon outline
    as [[x,y], [x,y], ...] normalized 0-1000 (point order - a genuinely different, richer shape
    than box_2d's [ymin,xmin,ymax,xmax], NOT a box converted from a mask - deliberately not done
    here per explicit instruction), or None if the mask is empty or has no extractable contour.
    Resized to target_shape first (see _resize_mask_to_shape) for the same correctness reason as
    _mask_to_box_2d. Simplified via cv2.approxPolyDP down to at most MAX_POLYGON_POINTS points,
    progressively (not truncated - truncating would cut off part of the outline instead of
    representing the whole shape more coarsely) - keeps hint text a reasonable size per candidate
    while still describing the real outline, not a bounding rectangle. This simplification is used
    ONLY for the text hint sent to Gemini - the QC *image* draws the real, unsimplified mask via
    draw_mask_overlay instead (a simplified outline was found to visually distort irregular real
    masks too much to be trusted for QC by eye)."""
    arr = _resize_mask_to_shape(mask, target_shape)
    if not arr.any():
        return None
    h, w = arr.shape
    contours, _ = cv2.findContours(arr.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) <= 0:
        return None
    perimeter = cv2.arcLength(largest, True)
    epsilon = 0.01 * perimeter
    simplified = cv2.approxPolyDP(largest, epsilon, True)
    points = simplified.reshape(-1, 2)
    tries = 0
    while len(points) > MAX_POLYGON_POINTS and tries < 8:
        epsilon *= 1.5
        simplified = cv2.approxPolyDP(largest, epsilon, True)
        points = simplified.reshape(-1, 2)
        tries += 1
    if len(points) > MAX_POLYGON_POINTS:
        # Still too many after aggressive simplification (unusual, e.g. a very jagged mask) -
        # evenly subsample rather than truncate, so kept points still span the whole outline.
        idx = np.linspace(0, len(points) - 1, MAX_POLYGON_POINTS).round().astype(int)
        points = points[idx]
    if len(points) < 3:
        return None
    return [[round(x / w * 1000), round(y / h * 1000)] for x, y in points]


def propose_regions_dumb(
    image_path: Path, model_id: str, points_per_side: int
) -> "tuple[List[List[List[int]]], List[np.ndarray]]":
    """Runs SAM2's official promptless/automatic "segment everything" mode
    (SAM2AutomaticMaskGenerator) - blind, no text prompt, no concept understanding at all. Filters
    candidates by plausible size (MIN/MAX_REGION_AREA_FRAC - the only filter available here, since
    there's no semantic signal like the concept-search pass has) and caps the count
    (MAX_REGION_HINTS, keeping the largest). Returns (polygons, masks) - polygons for the Gemini
    hint text, masks (the real, unsimplified boolean arrays) for QC drawing via
    draw_mask_overlay. Never raises: a failure here just means no geometric hints for this image.

    Speed is UNVERIFIED on real hardware as of this writing - points_per_side is a genuine,
    directly-reachable tuning knob here (unlike the ultralytics/SAM3 dead end hit earlier this
    session), so if a real test is too slow, lower --points-per-side before assuming SAM2 itself is
    unusably slow for this dataset.

    Downscales to DUMB_MAX_DIM before calling generate() - SAM2's automatic mode does not do this
    itself (unlike SAM3's Sam3Processor), and running it on a real ~10-megapixel field photo caused
    a genuine, verified CUDA OOM that also corrupted the CUDA context (see module docstring).
    Returned masks are resized back up to the real image resolution immediately after, via
    _resize_mask_to_shape, so everything downstream still works in true-resolution coordinates."""
    img = _load_rgb(image_path)
    if img is None:
        return [], []
    target_shape = img.shape[:2]
    small_img = img
    long_edge = max(target_shape)
    if long_edge > DUMB_MAX_DIM:
        scale = DUMB_MAX_DIM / long_edge
        new_w, new_h = round(target_shape[1] * scale), round(target_shape[0] * scale)
        small_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    try:
        generator = _load_sam2_generator(model_id, points_per_side)
        try:
            anns = generator.generate(small_img)
        except Exception as e:  # noqa: BLE001
            log.warning("  SAM2 automatic proposal failed on %s: %r - proceeding without "
                        "geometric hints for this image", image_path.name, e)
            return [], []

        scored: List[tuple] = []
        total_px = target_shape[0] * target_shape[1]
        for ann in anns:
            mask_arr = _resize_mask_to_shape(np.asarray(ann["segmentation"], dtype=bool), target_shape)
            area_frac = float(mask_arr.sum()) / total_px
            if not (MIN_REGION_AREA_FRAC <= area_frac <= MAX_REGION_AREA_FRAC):
                continue
            polygon = _mask_to_polygon(mask_arr, target_shape)
            if polygon is not None:
                scored.append((area_frac, polygon, mask_arr))

        # Ascending, NOT descending: real evidence this session (visual QC inspection of
        # qc/sam_proposals_dumb/*.jpg) showed the largest surviving masks are almost always
        # background/structure (sky patches, whole poles, wood-plank decking, long cable-wrap
        # segments), not individual fasteners - on one image, 10 of 19 candidate slots were large
        # low-value blobs while the genuinely useful bolt-head masks were the smallest, at the end
        # of a descending sort. Small, compact regions are far more likely to be individual
        # hardware for this dataset's subject matter, so prefer keeping those when trimming to
        # MAX_REGION_HINTS.
        scored.sort(key=lambda t: t[0])
        top = scored[:MAX_REGION_HINTS]
        return [p for _, p, _ in top], [m for _, _, m in top]
    finally:
        _clear_cuda_cache()


# Deliberately low: Gemini already found a real box, we're just asking SAM3 to refine it, not
# asking "did you find anything at all" - filtering that refinement by a strict confidence score
# would just make tightening silently no-op more often, falling back to Gemini's original
# (looser) box more than necessary.
DEFAULT_SAM3_TIGHTEN_CONFIDENCE_THRESHOLD = 0.1


def tighten_boxes_for_image(
    image_path: Path, boxes_2d: List[List[int]], sam3_checkpoint: Optional[str] = None,
    confidence_threshold: float = DEFAULT_SAM3_TIGHTEN_CONFIDENCE_THRESHOLD,
) -> List[List[int]]:
    """Runs SAM3's official box-prompted mode (Sam3Processor.add_geometric_prompt) once per box in
    boxes_2d, but encodes the image via set_image() only ONCE for all of them - real, confirmed bug
    fixed this session: an earlier per-box tighten_box() called set_image() (a full backbone
    forward pass, real GPU memory each time) separately for EVERY box, even when many boxes
    belonged to the same image (confirmed: 16 redundant encodes of the same image on a real run).
    Under a heavier annotation batch (more boxes/image) this caused a real, reproducible cascading
    CUDA OOM starting partway through a run, after which even torch.cuda.empty_cache() itself
    failed identically (the same poisoned-context signature established earlier this session) -
    every subsequent image's tightening silently fell back to untightened boxes for the rest of
    the run. Fix mirrors the already-correct reuse pattern in propose_regions_concept: one
    set_image() per image, then cheap reset_all_prompts()+add_geometric_prompt() per box.

    Returns one box_2d per input box_2d, same order - each individually falls back to its own
    original box_2d (never a worse or crashing result) if SAM3 produces an empty/degenerate mask
    for that specific prompt, or if that specific prompt call fails; a single box's tightening
    failure must never discard a valid detection or affect any other box's result. If set_image()
    itself fails (whole-image failure, not per-box), ALL boxes fall back to their originals.

    add_geometric_prompt's box format is [center_x, center_y, width, height] normalized to [0,1]
    (confirmed via source read of sam3/model/sam3_image_processor.py - NOT pixel xyxy like the
    ultralytics convention used earlier this session) - converted from this file's
    box_2d=[ymin,xmin,ymax,xmax]/1000 convention using the same cx/cy/w/h math already used by
    write_yolo_label.

    Loads the image via PIL directly (not _load_rgb's numpy array) and passes the PIL Image
    straight to set_image(). Real, confirmed bug found this session: Sam3Processor.set_image()'s
    numpy/tensor branch reads `height, width = image.shape[-2:]`, which assumes CHW - on an HWC
    array (what _load_rgb / np.array(PIL Image) produces) that reads the channel count (3) as the
    width, corrupting state["original_width"] and every mask/box scale factor derived from it for
    the rest of this call. The isinstance(image, PIL.Image.Image) branch reads image.size
    correctly instead."""
    if not boxes_2d:
        return []
    from PIL import Image

    results: List[List[int]] = list(boxes_2d)
    try:
        processor = _load_sam3_processor(sam3_checkpoint, confidence_threshold)
        image = Image.open(image_path).convert("RGB")
        target_shape = (image.height, image.width)
        try:
            with _sam3_autocast():
                state = processor.set_image(image)
        except Exception as e:  # noqa: BLE001 - a whole-image failure falls back ALL boxes
            log.warning("  SAM3 set_image failed on %s: %r - keeping all %d original box(es)",
                        image_path.name, e, len(boxes_2d))
            return results

        for i, box_2d in enumerate(boxes_2d):
            try:
                ymin, xmin, ymax, xmax = box_2d
                cx, cy = (xmin + xmax) / 2000.0, (ymin + ymax) / 2000.0
                w, h = (xmax - xmin) / 1000.0, (ymax - ymin) / 1000.0
                with _sam3_autocast():
                    processor.reset_all_prompts(state)
                    box_state = processor.add_geometric_prompt(
                        box=[cx, cy, w, h], label=True, state=state
                    )
                    # box_state["masks"] is (N, 1, H, W) - _forward_grounding's interpolate() call
                    # adds a channel dim (unsqueeze(1), required by interpolate's NCHW expectation)
                    # that never gets squeezed back out - confirmed via a real crash this session.
                    masks = box_state["masks"].cpu().numpy().squeeze(1)
                    # box_state["scores"] can come out as bfloat16 under the autocast context above
                    # (sigmoid/multiply ops inherit it) - numpy has no bfloat16 dtype at all, so
                    # .numpy() on it raises TypeError('Got unsupported ScalarType BFloat16') -
                    # confirmed via a real crash this session. .float() first is the standard fix
                    # (masks stays bool from the `> 0.5` comparison in _forward_grounding
                    # regardless of autocast, so it doesn't need this).
                    scores = box_state["scores"].float().cpu().numpy()
                if len(masks) == 0:
                    continue
                best = int(np.argmax(scores))
                tight = _mask_to_box_2d(masks[best], target_shape)
                if tight is not None:
                    results[i] = tight
                del box_state, masks, scores
            except Exception as e:  # noqa: BLE001 - one box's failure must not affect the others
                log.warning("  SAM3 box-tightening failed on %s box=%r: %r - keeping the "
                            "original box", image_path.name, box_2d, e)
        del state
    except Exception as e:  # noqa: BLE001 - tightening must never crash or discard a detection
        log.warning("  SAM3 box-tightening failed outright on %s: %r - keeping all original "
                    "box(es)", image_path.name, e)
    finally:
        _clear_cuda_cache()
    return results


def _valid_box(box_2d: list) -> bool:
    """Structural sanity check only (not a tightness/accuracy check - that's a QC-by-eye job via
    visualized/, not something code can judge). Deliberately NOT a pydantic validator on BoltBox
    itself: BoltBox doubles as the response_schema Gemini's output is parsed against, and a raising
    validator there would fail response.parsed for the WHOLE batch over a single bad box in one
    image, burning all --retries and losing every image in the request. This runs after parsing
    succeeds, so one bad box just gets dropped and logged instead."""
    if not isinstance(box_2d, list) or len(box_2d) != 4:
        return False
    ymin, xmin, ymax, xmax = box_2d
    if not all(isinstance(v, (int, float)) for v in box_2d):
        return False
    if not (0 <= ymin < ymax <= 1000 and 0 <= xmin < xmax <= 1000):
        return False
    return True


def filter_valid_boxes(ann: BoltAnnotation, image_name: str) -> BoltAnnotation:
    kept = []
    for box in ann.boxes:
        if _valid_box(box.box_2d):
            kept.append(box)
        else:
            log.warning("  dropping one malformed box on %s: box_2d=%r label=%r (not 4 values, "
                        "non-numeric, out of [0,1000], or ymin>=ymax / xmin>=xmax)",
                        image_name, box.box_2d, box.label)
    return BoltAnnotation(boxes=kept)


def write_yolo_label(ann: BoltAnnotation, out_path: Path) -> None:
    """class_id cx cy w h, all normalized 0-1 (standard Ultralytics/YOLO26 format - verified
    against an existing label in model_training/merged/*/labels/*.txt this session)."""
    lines = []
    for box in ann.boxes:
        ymin, xmin, ymax, xmax = box.box_2d
        cx, cy = (xmin + xmax) / 2000.0, (ymin + ymax) / 2000.0
        w, h = (xmax - xmin) / 1000.0, (ymax - ymin) / 1000.0
        cls_id = CLASSES.index(box.label)
        lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    # Empty file (no lines) is the valid YOLO convention for a background/no-object image.
    out_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _rects_overlap(a: tuple, b: tuple) -> bool:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return not (ax1 < bx0 or ax0 > bx1 or ay1 < by0 or ay0 > by1)


def draw_visualization(image_path: Path, ann: BoltAnnotation, out_path: Path) -> None:
    img = cv2.imread(str(image_path))
    height, width = img.shape[:2]
    # Scale line/font to actual resolution - a fixed width=3 is invisible on a 3648x2736 photo.
    line_width = max(4, round(min(width, height) / 250))
    font_scale = max(1.0, min(width, height) / 900)
    # Text thickness must stay small relative to font_scale, NOT tied to line_width - deriving it
    # from line_width made round letters (e.g. the "o" in bolt_ok) render as a closed blob that
    # reads as "c" (confirmed by eye against raw_gemini/AUT-0001.json, which has the correct text).
    thickness = max(2, round(font_scale))
    font = cv2.FONT_HERSHEY_SIMPLEX

    placed_labels: List[tuple] = []  # label text rects already placed on this image

    for box in ann.boxes:
        ymin, xmin, ymax, xmax = box.box_2d
        x0, y0 = round(xmin / 1000.0 * width), round(ymin / 1000.0 * height)
        x1, y1 = round(xmax / 1000.0 * width), round(ymax / 1000.0 * height)
        color = BOX_COLORS[box.label][::-1]  # RGB -> BGR for cv2

        cv2.rectangle(img, (x0, y0), (x1, y1), color, line_width)

        (text_w, text_h), baseline = cv2.getTextSize(box.label, font, font_scale, thickness)
        label_h = text_h + baseline + 6
        label_top = max(0, y0 - label_h)
        label_rect = (x0, label_top, x0 + text_w + 8, label_top + label_h)
        # Nearby boxes can produce overlapping labels that smear into unreadable text (seen in
        # testing: two close bolt_ok labels overlapped into garbage). Stack downward instead.
        while any(_rects_overlap(label_rect, other) for other in placed_labels):
            label_top += label_h + 2
            label_rect = (x0, label_top, x0 + text_w + 8, label_top + label_h)
        placed_labels.append(label_rect)

        # No filled background - a solid box was covering the actual bolt/hardware underneath it,
        # defeating the point of a QC image. Black outline stroke + colored fill stays readable
        # against any background without hiding image content.
        text_pos = (x0 + 4, label_rect[1] + text_h)
        cv2.putText(img, box.label, text_pos, font, font_scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
        cv2.putText(img, box.label, text_pos, font, font_scale, color, thickness, cv2.LINE_AA)

    cv2.imwrite(str(out_path), img)


def draw_mask_overlay(
    image_path: Path, masks: List[np.ndarray], color: tuple, out_path: Path, alpha: float = 0.45,
    draw_numbers: bool = True,
) -> None:
    """QC image showing every candidate mask actually sent as a hint for this image, drawn as a
    real alpha-blended colored overlay on the REAL mask pixels - not a simplified polygon outline.
    This is the standard way every official Meta SAM demo notebook (SAM/SAM2/SAM3 alike)
    visualizes segmentation results, and it loses zero shape information (a prior version of this
    function drew simplified cv2.approxPolyDP outlines instead, which could visually distort an
    irregular real mask enough to barely read as "the mask" any more - real user feedback this
    session). `color` is BGR (SAM_CONCEPT_COLOR or SAM_DUMB_COLOR).

    draw_numbers: if True (default - unchanged behavior for human-facing QC images), also draws a
    numbered outline on top of each mask so individual candidates stay distinguishable when several
    overlap. Real user concern this session about the version of this image sent TO GEMINI
    specifically (not the human QC copy): burning index numbers onto the photo may bias the model
    toward treating each number as something to draw a box around, rather than treating the
    highlighted region itself as the hint - pass False for that copy so Gemini only sees colored
    region outlines, no numbers at all."""
    img = cv2.imread(str(image_path))
    height, width = img.shape[:2]
    overlay = img.copy()
    color_arr = np.array(color, dtype=np.uint8)

    for mask_arr in masks:
        arr = _resize_mask_to_shape(mask_arr, (height, width))
        overlay[arr] = color_arr

    blended = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

    line_width = max(1, round(min(width, height) / 600))
    font_scale = max(0.6, min(width, height) / 1500)
    thickness = max(1, round(font_scale))
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i, mask_arr in enumerate(masks):
        arr = _resize_mask_to_shape(mask_arr, (height, width))
        contours, _ = cv2.findContours(arr.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        cv2.drawContours(blended, contours, -1, color, line_width)
        if not draw_numbers:
            continue
        largest = max(contours, key=cv2.contourArea)
        x0, y0 = int(largest[0][0][0]), int(largest[0][0][1])
        cv2.putText(blended, str(i), (x0 + 2, max(0, y0 - 3)), font, font_scale, (0, 0, 0),
                    thickness + 2, cv2.LINE_AA)
        cv2.putText(blended, str(i), (x0 + 2, max(0, y0 - 3)), font, font_scale, color,
                    thickness, cv2.LINE_AA)

    cv2.imwrite(str(out_path), blended)


def draw_tightening_debug(image_path: Path, debug_records: List[dict], out_path: Path) -> None:
    """QC image showing, per detection, the box Gemini/ER originally returned (thin gray) versus
    what SAM tightened it to (the normal class color, same as draw_visualization) - so tightening
    can be judged visually instead of just reading before/after numbers in qc/sam_regions_debug/
    JSON. Both drawn on the same image for direct comparison."""
    img = cv2.imread(str(image_path))
    height, width = img.shape[:2]
    line_width = max(3, round(min(width, height) / 300))
    before_color = (170, 170, 170)  # BGR gray - neutral, distinct from any class color

    for rec in debug_records:
        color = BOX_COLORS[rec["label"]][::-1]  # RGB -> BGR
        for box_2d, box_color, width_px in ((rec["before"], before_color, max(1, line_width - 1)),
                                             (rec["after"], color, line_width)):
            ymin, xmin, ymax, xmax = box_2d
            x0, y0 = round(xmin / 1000.0 * width), round(ymin / 1000.0 * height)
            x1, y1 = round(xmax / 1000.0 * width), round(ymax / 1000.0 * height)
            cv2.rectangle(img, (x0, y0), (x1, y1), box_color, width_px)

    cv2.imwrite(str(out_path), img)


def write_data_yaml(out: Path) -> None:
    lines = [
        f"path: {out.as_posix()}",
        "train: images  # unsplit single source - real train/valid/test split happens at merge time",
        "val: images",
        "",
        f"nc: {len(CLASSES)}",
        "names:",
    ] + [f"  {i}: {name}" for i, name in enumerate(CLASSES)]
    (out / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
