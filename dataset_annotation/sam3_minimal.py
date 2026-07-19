"""Minimal, standalone SAM3 inference - as close to the official README's basic usage as possible.
The ONLY addition beyond the README's own example is the torch.autocast(bfloat16) context, which
is not optional - the README's code crashes without it on this hardware/checkpoint (confirmed real
this session, matches github.com/facebookresearch/sam3/issues/526). Everything else is exactly
README-simple: build model once, set_image, set_text_prompt, read masks/boxes/scores.

Prints each detection's raw score and "extent" (mask area / its own bounding-box area) with NO
filtering applied - diagnostic to see what SAM3 actually returns before any of our own
post-processing (confidence threshold, degenerate-rectangle rejection, dedup) touches it. Also
saves a real mask overlay image (reusing annotate_common.draw_mask_overlay - same cv2-based
drawing the real pipeline uses, not reinvented here) to outputs/<image_stem>_<prompt>.jpg so the
raw, unfiltered detections can be judged by eye, not just by number.

Usage: python sam3_minimal.py <image_path> <text_prompt> <checkpoint> <confidence_threshold>
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

from annotate_common import draw_mask_overlay

image_path = sys.argv[1] if len(sys.argv) > 1 else "npu_bolt/AAAA.jpg"
prompt = sys.argv[2] if len(sys.argv) > 2 else "bolt"
checkpoint = sys.argv[3] if len(sys.argv) > 3 else "sam3.pt"
threshold = float(sys.argv[4]) if len(sys.argv) > 4 else 0.2

model = build_sam3_image_model(checkpoint_path=checkpoint)
processor = Sam3Processor(model, confidence_threshold=threshold)

image = Image.open(image_path).convert("RGB")

with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    inference_state = processor.set_image(image)
    output = processor.set_text_prompt(state=inference_state, prompt=prompt)

masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
masks_np = masks.cpu().numpy()
scores_np = scores.float().cpu().numpy()
boxes_np = boxes.float().cpu().numpy()

print(f"image={image_path} prompt={prompt!r} threshold={threshold} -> {len(scores_np)} raw "
      f"detection(s), no filtering beyond Sam3Processor's own confidence_threshold")
print(f"masks.shape={tuple(masks.shape)} (note: (N,1,H,W) if it still has the unsqueezed channel dim)")

flat_masks = []
for i in range(len(scores_np)):
    m = masks_np[i]
    if m.ndim == 3:
        m = m[0]
    flat_masks.append(m)
    ys, xs = np.where(m)
    if ys.size == 0:
        print(f"  [{i}] score={scores_np[i]:.3f} box={boxes_np[i]} EMPTY MASK")
        continue
    box_area = (ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)
    extent = float(m.sum()) / box_area if box_area > 0 else 0.0
    print(f"  [{i}] score={scores_np[i]:.3f} box={boxes_np[i]} mask_pixels={int(m.sum())} "
          f"bbox_area={box_area} extent={extent:.3f}")

out_dir = Path("outputs")
out_dir.mkdir(exist_ok=True)
out_path = out_dir / f"{Path(image_path).stem}_{prompt}.jpg"
draw_mask_overlay(Path(image_path), flat_masks, (255, 220, 0), out_path)
print(f"Saved mask overlay ({len(flat_masks)} raw detections, unfiltered) to {out_path}")
