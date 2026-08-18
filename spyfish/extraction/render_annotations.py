"""
Draw and crop ML detection boxes onto video frames.

Pure image functions: numpy arrays in, numpy arrays out. No config reads, no
file I/O, no video seeking, callers own all of that. Everything here is
directly unit-testable against small synthetic frames.

Two coordinate conventions meet in this module and must not be confused:

  * The raw ML CSV stores YOLO ``xywh`` where **x, y are the box CENTRE** and
    the values are absolute pixels in the source video's native resolution.
  * Everything downstream (COCO, cv2.rectangle, cropping) wants **corners**.

``center_to_corners`` is the only place that conversion should happen.

Rotation: ``run_inference`` reads frames straight from cv2 and ignores the
container's rotation flag, so its box coordinates are in *unrotated* pixel
space. ``extract_one_frame_from_cap`` applies the rotation to the pixels. Any
code that draws inference boxes onto an extracted frame must therefore put the
box through ``rotate_bbox`` first, with the same degrees the frame was rotated
by. See ``_ROTATION_MAP`` in ``extract_frames.py``, the two must agree.
"""

import logging

import cv2
import numpy as np

# ── geometry ─────────────────────────────────────────────────────────────────


def center_to_corners(
    cx: float, cy: float, w: float, h: float
) -> tuple[float, float, float, float]:
    """YOLO centre-format box → (x1, y1, x2, y2) corners."""
    return cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0


def rotate_bbox(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    rotation: int,
    src_w: int,
    src_h: int,
) -> tuple[float, float, float, float]:
    """Map a box from unrotated frame space into rotated frame space.

    ``src_w``/``src_h`` are the dimensions of the *unrotated* frame. For 90 and
    270 the rotated frame has those swapped, and the returned coordinates are
    in that swapped space.

    Must mirror ``extract_frames._ROTATION_MAP`` exactly:
      90  → cv2.ROTATE_90_CLOCKWISE
      180 → cv2.ROTATE_180
      270 → cv2.ROTATE_90_COUNTERCLOCKWISE
    """
    rotation %= 360
    if rotation == 0:
        return x1, y1, x2, y2
    if rotation == 90:
        # clockwise: (x, y) → (src_h - y, x)
        return src_h - y2, x1, src_h - y1, x2
    if rotation == 180:
        return src_w - x2, src_h - y2, src_w - x1, src_h - y1
    if rotation == 270:
        # counter-clockwise: (x, y) → (y, src_w - x)
        return y1, src_w - x2, y2, src_w - x1
    raise ValueError(f"Unsupported rotation {rotation}; expected 0, 90, 180 or 270.")


def iou(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> float:
    """Intersection-over-union of two corner-format boxes, 0.0–1.0.

    Used to decide whether two detections are the same animal seen twice
    (high overlap) or two different animals (low overlap). Time proximity
    alone cannot distinguish those.
    """
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_w = min(ax2, bx2) - max(ax1, bx1)
    inter_h = min(ay2, by2) - max(ay1, by1)
    if inter_w <= 0 or inter_h <= 0:
        return 0.0
    inter = inter_w * inter_h
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


# ── drawing ──────────────────────────────────────────────────────────────────


def draw_box(
    frame: np.ndarray,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    colour: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> None:
    """Draw a three-layer rectangle in place: black, colour, black.

    The black hairlines either side of the coloured stroke keep the box visible
    against both bright sand and dark reef without a filled backing that would
    hide the animal being judged. Mutates ``frame``.
    """
    p1, p2 = (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2)))
    black = (0, 0, 0)
    cv2.rectangle(
        frame, (p1[0] - 2, p1[1] - 2), (p2[0] + 2, p2[1] + 2), black, 1, cv2.LINE_AA
    )
    cv2.rectangle(frame, p1, p2, colour, thickness, cv2.LINE_AA)
    cv2.rectangle(
        frame, (p1[0] + 2, p1[1] + 2), (p2[0] - 2, p2[1] - 2), black, 1, cv2.LINE_AA
    )


def draw_label(
    frame: np.ndarray,
    text: str,
    box: tuple[float, float, float, float] | None = None,
    font_scale: float = 0.9,
) -> None:
    """Burn a label onto the frame as white text with a black outline.

    Outlined rather than boxed for the same reason as ``draw_box``: it stays
    readable on any background without covering pixels.

    Positioned above ``box`` when there is room, below it otherwise, and
    clamped inside the frame as a last resort. The font shrinks until the text
    fits the frame width, then truncates with an ellipsis, a long scientific
    name on a narrow crop must not run off the edge.

    Mutates ``frame``.
    """
    if not text:
        return

    font = cv2.FONT_HERSHEY_SIMPLEX
    h, w = frame.shape[:2]
    max_width = max(8, w - 8)
    thickness = max(1, int(font_scale * 2))

    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    while tw > max_width and font_scale > 0.4:
        font_scale *= 0.9
        thickness = max(1, int(font_scale * 2))
        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    if tw > max_width:
        while len(text) > 1:
            text = text[:-1]
            (tw, th), baseline = cv2.getTextSize(
                text + "…", font, font_scale, thickness
            )
            if tw <= max_width:
                text += "…"
                break

    pad = max(10, int(10 * font_scale))
    if box is not None:
        bx1, by1, bx2, by2 = box
        x = int(max(4, min(w - 4 - tw, (bx1 + bx2) / 2 - tw / 2)))
        if by1 - pad - th >= 0:
            y = int(by1 - pad)
        elif by2 + pad + th <= h:
            y = int(by2 + pad + th)
        else:
            y = int(max(th + 4, min(h - 4, by1 - pad)))
    else:
        x = max(4, (w - tw) // 2)
        y = th + baseline + 6

    cv2.putText(
        frame, text, (x, y), font, font_scale, (0, 0, 0), thickness + 3, cv2.LINE_AA
    )
    cv2.putText(
        frame, text, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA
    )


# ── cropping ─────────────────────────────────────────────────────────────────


def crop_window(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    frame_w: int,
    frame_h: int,
    scale: float,
    max_frame_fraction: float,
) -> tuple[int, int, int, int]:
    """Square crop window around a box.

    **The invariant: the animal occupies ``1/scale`` of the crop, always.** At
    ``scale=2.0`` the box fills half the frame whether it was 40 px or 400 px in
    the source. That is what lets a volunteer judge subject after subject
    without recalibrating their eye each time.

    An earlier version had a ``min_window_px`` floor so that tiny boxes kept
    some surrounding context. It broke the invariant badly, at a 256 px floor,
    a 28 px box filled 11% of its crop while a 133 px box filled 50%, so the
    same animal appeared at wildly different apparent sizes depending on how far
    away it happened to be. The floor is gone.

    The cost is magnification: with no floor, a small box is blown up hard (a
    28 px box becomes 14x at an 800 px output, which is mush). That is a
    *selection* problem, not a rendering one, ``min_bbox_px`` in the target
    selector is the correct knob, and it should be set so the worst-case upscale
    stays acceptable: ``min_bbox_px >= output_px / (scale * max_upscale)``.

    One clamp survives: ``max_frame_fraction`` of the frame's shorter side.
    An animal filling the shot cannot be given 2x surround that does not exist,
    so those crops degenerate to a near-full-frame view, which is the right
    thing to show anyway.

    Edge behaviour is **slide, then clip**. A window overhanging the frame is
    first slid back inside, but only as far as the box's own margin allows, so
    the animal never leaves the window; whatever overhang remains is clipped and
    ``pad_to_square`` fills it.

    Pure clipping (the previous behaviour) padded 64% of a real 421-crop set,
    median 13% of the image. Most of that was not "the animal is at the frame
    edge" but "a square window does not fit in a 16:9 frame": a 437 px window
    overhangs whenever the box centre is within ~218 px of the top or bottom,
    which is most fish, since they sit near the substrate. Black bars are wasted
    screen on the phones this is designed for.

    Pure sliding was rejected for the opposite reason. Our fish are often
    genuinely half out of shot, and sliding a window fully inside then shows
    water on the wrong side with the animal jammed against an edge. Capping the
    slide at the box's margin keeps the animal enclosed and roughly central,
    while removing the padding in the common case where it bought nothing.

    Returns integer (left, top, right, bottom), always inside the frame.
    """
    box_w, box_h = x2 - x1, y2 - y1
    side = scale * max(box_w, box_h)
    side = min(side, max_frame_fraction * min(frame_w, frame_h))

    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    half = side / 2.0

    def _slide(lo: float, hi: float, box_lo: float, box_hi: float, limit: float):
        """Shift [lo, hi] back inside [0, limit] without uncovering the box."""
        if lo < 0.0:
            # Room to move right before the window's left edge passes the box's.
            lo, hi = (lambda s: (lo + s, hi + s))(min(-lo, max(0.0, box_lo - lo)))
        elif hi > limit:
            lo, hi = (lambda s: (lo - s, hi - s))(
                min(hi - limit, max(0.0, hi - box_hi))
            )
        return lo, hi

    x_lo, x_hi = _slide(cx - half, cx + half, x1, x2, float(frame_w))
    y_lo, y_hi = _slide(cy - half, cy + half, y1, y2, float(frame_h))

    left = int(round(max(0.0, x_lo)))
    top = int(round(max(0.0, y_lo)))
    right = int(round(min(float(frame_w), x_hi)))
    bottom = int(round(min(float(frame_h), y_hi)))
    return left, top, right, bottom


def pad_to_square(
    crop: np.ndarray, border_bgr: tuple[int, int, int] = (30, 30, 30)
) -> np.ndarray:
    """Pad a clipped crop back to square with a neutral border.

    Keeps every subject the same shape so volunteers do not have to recalibrate
    on apparent size between subjects, and reads honestly as "the frame ends
    here" rather than pretending the missing area was water.
    """
    h, w = crop.shape[:2]
    if h == w:
        return crop
    side = max(h, w)
    top = (side - h) // 2
    bottom = side - h - top
    left = (side - w) // 2
    right = side - w - left
    return cv2.copyMakeBorder(
        crop, top, bottom, left, right, cv2.BORDER_CONSTANT, value=border_bgr
    )


def render_detection_crop(
    frame: np.ndarray,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    label: str,
    scale: float,
    max_frame_fraction: float,
    output_px: int,
    box_colour: tuple[int, int, int] = (0, 255, 0),
    border_bgr: tuple[int, int, int] = (30, 30, 30),
) -> np.ndarray:
    """Crop around one detection, draw its box and label, return a square image.

    Order matters: crop first, then draw, so the box stroke and label are sized
    relative to the *output* image rather than being scaled up with it. Drawing
    before the resize would give a 96 px box a hairline border and a 1865 px box
    a fat one.

    ``x1..y2`` must already be in the coordinate space of ``frame``, apply
    ``rotate_bbox`` before calling if the frame was rotated.
    """
    frame_h, frame_w = frame.shape[:2]
    left, top, right, bottom = crop_window(
        x1, y1, x2, y2, frame_w, frame_h, scale, max_frame_fraction
    )
    crop = frame[top:bottom, left:right].copy()
    if crop.size == 0:
        raise ValueError(
            f"Empty crop for box ({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}) "
            f"in {frame_w}x{frame_h} frame."
        )

    crop = pad_to_square(crop, border_bgr)

    # Scale the box into the padded crop's coordinates.
    pad_x = (crop.shape[1] - (right - left)) // 2
    pad_y = (crop.shape[0] - (bottom - top)) // 2
    ratio = output_px / float(crop.shape[0])
    crop = cv2.resize(
        crop,
        (output_px, output_px),
        interpolation=cv2.INTER_CUBIC if ratio > 1 else cv2.INTER_AREA,
    )

    bx1 = (x1 - left + pad_x) * ratio
    by1 = (y1 - top + pad_y) * ratio
    bx2 = (x2 - left + pad_x) * ratio
    by2 = (y2 - top + pad_y) * ratio

    draw_box(crop, bx1, by1, bx2, by2, colour=box_colour)
    draw_label(crop, label, box=(bx1, by1, bx2, by2))
    return crop


def render_annotated_frame(
    frame: np.ndarray,
    detections: list[dict],
    box_colour: tuple[int, int, int] = (0, 255, 0),
    max_width_px: int = 1280,
) -> np.ndarray:
    """Draw every detection on a full frame, for the "any fish missed?" question.

    ``detections`` are dicts with ``x1, y1, x2, y2`` in this frame's coordinate
    space and an optional ``label``. Downscaled to ``max_width_px`` so the
    subject loads quickly on a phone; boxes are drawn after the resize for the
    same reason as in ``render_detection_crop``.
    """
    out = frame.copy()
    ratio = 1.0
    if out.shape[1] > max_width_px:
        ratio = max_width_px / float(out.shape[1])
        out = cv2.resize(
            out,
            (max_width_px, int(round(out.shape[0] * ratio))),
            interpolation=cv2.INTER_AREA,
        )

    for det in detections:
        bx1, by1 = det["x1"] * ratio, det["y1"] * ratio
        bx2, by2 = det["x2"] * ratio, det["y2"] * ratio
        draw_box(out, bx1, by1, bx2, by2, colour=box_colour)
        if det.get("label"):
            draw_label(out, det["label"], box=(bx1, by1, bx2, by2), font_scale=0.6)
    return out


def warn_if_rotated(rotation: int, drop_id: str) -> None:
    """Announce non-zero container rotation.

    Every BUV video checked so far reports 0, a fixed rig shoots landscape.
    But a rotated video would silently produce crops with the box in the wrong
    place, which looks like a model problem rather than a rendering one. Rare
    *and* silent is the expensive combination, so it gets a warning even though
    ``rotate_bbox`` handles it correctly.
    """
    if rotation:
        logging.warning(
            f"{drop_id}: video reports {rotation}° container rotation. Boxes are "
            "being transformed to match the rotated frame, check the rendered "
            "crops for this drop before uploading."
        )
