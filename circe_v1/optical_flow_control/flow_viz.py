"""
Pure drawing helpers for the Flow Lab — no state, no I/O.

Everything the optical-flow algorithm "sees" and "decides" is drawn onto a
copy of the frame so a human (or a future debugging session) can eyeball it:

  - draw_flow_arrows : sparse grid of per-tile motion arrows (the raw evidence),
                       adapted from OpenCV's samples/python/opt_flow.py draw_flow
  - draw_hsv         : the whole dense flow field as color (hue=direction,
                       value=magnitude), the classic OpenCV flow colorization
  - draw_roi_box     : the central region the stabilizer actually samples
  - draw_drift_vector: one big arrow = this frame's deadbanded flow velocity
                       (vs the small arrows' raw per-tile "evidence")
  - draw_position_marker: persistent home-cross + position-dot = cumulative
                       displacement since origin — what Displacement Hold acts on
  - draw_hud         : text overlay with all the live numbers

The flow field / ROI come from FlowStabilizer at `work_size` resolution; these
helpers take a `scale` = display_w / work_w so they draw correctly on a
larger display frame.
"""

from __future__ import annotations

import cv2
import numpy as np

from flow_stabilizer import FlowCorrection

# BGR colors
_GREEN = (0, 255, 0)
_YELLOW = (0, 255, 255)
_CYAN = (255, 255, 0)
_RED = (0, 0, 255)
_WHITE = (255, 255, 255)
_BLACK = (0, 0, 0)


def draw_flow_arrows(img: np.ndarray, flow: np.ndarray, scale: float,
                     step: int = 16, color=_GREEN) -> np.ndarray:
    """Draw a sparse grid of flow arrows. `flow` is in work_size coords;
    `scale` maps work_size -> img size. `step` is the grid spacing in
    work_size pixels."""
    h, w = flow.shape[:2]
    ys, xs = np.mgrid[step // 2:h:step, step // 2:w:step].reshape(2, -1)
    fx, fy = flow[ys, xs].T
    for x, y, dx, dy in zip(xs, ys, fx, fy):
        x0, y0 = int(x * scale), int(y * scale)
        x1, y1 = int((x + dx) * scale), int((y + dy) * scale)
        cv2.arrowedLine(img, (x0, y0), (x1, y1), color, 1,
                        line_type=cv2.LINE_AA, tipLength=0.35)
        cv2.circle(img, (x0, y0), 1, color, -1)
    return img


def flow_to_hsv(flow: np.ndarray, out_size: tuple[int, int]) -> np.ndarray:
    """Colorize a dense flow field: hue=direction, value=magnitude.
    Returns a BGR image resized to out_size (w, h)."""
    h, w = flow.shape[:2]
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[..., 1] = 255
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = (ang * 180 / np.pi / 2).astype(np.uint8)
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return cv2.resize(bgr, out_size, interpolation=cv2.INTER_NEAREST)


def draw_roi_box(img: np.ndarray, roi, scale: float, color=_YELLOW) -> np.ndarray:
    """Draw the sampled ROI rectangle. `roi` = (x0, y0, w, h) in work_size coords."""
    if roi is None:
        return img
    x0, y0, rw, rh = roi
    p0 = (int(x0 * scale), int(y0 * scale))
    p1 = (int((x0 + rw) * scale), int((y0 + rh) * scale))
    cv2.rectangle(img, p0, p1, color, 1)
    cv2.putText(img, "sampled ROI", (p0[0] + 3, p0[1] + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    return img


def draw_drift_vector(img: np.ndarray, dx: float, dy: float,
                      arrow_gain: float = 12.0, color=_CYAN) -> np.ndarray:
    """One big arrow from image center = this frame's deadbanded flow
    VELOCITY (no smoothing). Decays to ~0 the instant motion stops — this is
    "how fast am I moving right now", not "how far off from home am I". For
    that, see draw_position_marker below."""
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2
    tip = (int(cx + dx * arrow_gain), int(cy + dy * arrow_gain))
    cv2.arrowedLine(img, (cx, cy), tip, color, 3, line_type=cv2.LINE_AA, tipLength=0.3)
    cv2.circle(img, (cx, cy), 4, color, -1)
    return img


def draw_position_marker(img: np.ndarray, cum_dx: float, cum_dy: float,
                         gain: float = 3.0, color=_RED) -> np.ndarray:
    """Persistent marker for cumulative drift (a POSITION, integrated over
    the whole session) — unlike draw_drift_vector, this stays put when the
    scene is still and only moves with actual net displacement since the
    last reset_origin(). A fixed 'home' cross marks the origin; the filled
    dot is the current estimated position; the line between them is the
    total accumulated offset."""
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2
    px = int(np.clip(cx + cum_dx * gain, 0, w - 1))
    py = int(np.clip(cy + cum_dy * gain, 0, h - 1))
    # home cross
    cv2.drawMarker(img, (cx, cy), color, markerType=cv2.MARKER_CROSS,
                   markerSize=14, thickness=2)
    # line from home to current estimated position, then the position dot
    cv2.line(img, (cx, cy), (px, py), color, 1, cv2.LINE_AA)
    cv2.circle(img, (px, py), 6, color, -1)
    cv2.circle(img, (px, py), 6, _WHITE, 1, cv2.LINE_AA)
    return img


def _put(img, text, org, color=_WHITE, scale=0.45):
    # black outline then colored text for readability over any background
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, _BLACK, 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def draw_hud(img: np.ndarray, corr: FlowCorrection, *, source_fps: float,
             engaged: bool, source_name: str, sent: bool) -> np.ndarray:
    """Text overlay with all the live numbers."""
    lines = [
        f"src={source_name} fps={source_fps:4.1f}  compute={corr.compute_ms:4.1f}ms",
        f"raw  dx={corr.raw_dx:+6.2f} dy={corr.raw_dy:+6.2f}",
        f"flow dx={corr.flow_dx:+6.2f} dy={corr.flow_dy:+6.2f}  valid={corr.valid}",
        f"cum (position, since origin)  dx={corr.cum_dx:+7.1f} dy={corr.cum_dy:+7.1f} px"
        + ("" if corr.coherent else "  [NOISE-GATED, not accumulated]"),
    ]
    y = 18
    for ln in lines:
        _put(img, ln, (8, y))
        y += 18

    if engaged:
        badge = "ENGAGED - SENDING" if sent else "ENGAGED (frame stale -> neutral)"
        _put(img, badge, (8, y + 2), color=_RED, scale=0.5)
    else:
        _put(img, "MONITOR ONLY (not sending)", (8, y + 2), color=_GREEN, scale=0.5)
    return img


def render_arrow_view(frame_bgr: np.ndarray, corr: FlowCorrection, *,
                      source_fps: float, engaged: bool, source_name: str,
                      sent: bool) -> np.ndarray:
    """Full annotated 'arrow' view: original frame + arrows + ROI + drift
    vector + HUD. Returns a BGR image. Safe to call on a valid=False /
    no-flow frame (just draws the HUD + ROI)."""
    out = frame_bgr.copy()
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    h, w = out.shape[:2]
    if corr.work_size[0]:
        scale = w / corr.work_size[0]
    else:
        scale = 1.0
    if corr.flow is not None:
        draw_flow_arrows(out, corr.flow, scale)
        draw_drift_vector(out, corr.flow_dx, corr.flow_dy)
    draw_position_marker(out, corr.cum_dx, corr.cum_dy)
    draw_roi_box(out, corr.roi, scale)
    draw_hud(out, corr, source_fps=source_fps, engaged=engaged,
             source_name=source_name, sent=sent)
    return out


def render_hsv_view(frame_bgr: np.ndarray, corr: FlowCorrection) -> np.ndarray:
    """The dense flow field as a color image, sized to the display frame.
    Falls back to a dim placeholder when there's no flow yet."""
    h, w = frame_bgr.shape[:2]
    if corr.flow is None:
        placeholder = np.zeros((h, w, 3), dtype=np.uint8)
        _put(placeholder, "waiting for flow...", (10, h // 2), color=_WHITE, scale=0.6)
        return placeholder
    return flow_to_hsv(corr.flow, (w, h))
