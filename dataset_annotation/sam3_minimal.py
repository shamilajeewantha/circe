"""Minimal, standalone SAM3 inference - as close to the official README's basic usage as possible.
The ONLY addition beyond the README's own example is the torch.autocast(bfloat16) context, which
is not optional - the README's code crashes without it on this hardware/checkpoint (confirmed real
this session, matches github.com/facebookresearch/sam3/issues/526). Everything else is exactly
README-simple: build model once, set_image, set_text_prompt, read masks/boxes/scores.

Also prints each detection's raw score and "extent" (mask area / its own bounding-box area) with
NO filtering applied - diagnostic to see what SAM3 actually returns before any of our own
post-processing (confidence threshold, degenerate-rectangle rejection, dedup) touches it.

Usage: python sam3_minimal.py <image_path> <text_prompt>
"""
import sys
import numpy as np
import torch
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

image_path = sys.argv[1] if len(sys.argv) > 1 else "npu_bolt/AAAA.jpg"
prompt = sys.argv[2] if len(sys.argv) > 2 else "bolt"
checkpoint = sys.argv[3] if len(sys.argv) > 3 else "sam3.pt"

threshold = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5
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

print(f"image={image_path} prompt={prompt!r} -> {len(scores_np)} raw detection(s) "
      f"(Sam3Processor's own default confidence_threshold=0.5, no other filtering)")
print(f"masks.shape={tuple(masks.shape)} (note: (N,1,H,W) if it still has the unsqueezed channel dim)")
for i in range(len(scores_np)):
    m = masks_np[i]
    if m.ndim == 3:
        m = m[0]
    ys, xs = np.where(m)
    if ys.size == 0:
        print(f"  [{i}] score={scores_np[i]:.3f} box={boxes_np[i]} EMPTY MASK")
        continue
    box_area = (ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)
    extent = float(m.sum()) / box_area if box_area > 0 else 0.0
    print(f"  [{i}] score={scores_np[i]:.3f} box={boxes_np[i]} mask_pixels={int(m.sum())} "
          f"bbox_area={box_area} extent={extent:.3f}")
