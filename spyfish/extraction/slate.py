"""Find the handwritten metadata slate at the start of a BUV video.

Rangers hold a whiteboard (site, date, time, lat/lon, depth) up to the GoPro
before the rig goes in the water. This scans the opening seconds, scores each
sampled frame for "big white board with handwriting on it", and saves the best
full-res frame as ``{drop}_slate.jpg`` so it can be read (by eye now, OCR later)
and cross-checked against the deployment metadata.

Runs at the start of ``--ml`` (``MLRunner.run_inference_loop``), when the video
is already on local disk; ``scripts/wip/find_slate.py`` runs it without ML.
Thresholds live in ``config.yaml`` under ``extraction.slate_detection``.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from spyfish.config.base import get_required
from spyfish.config.wrapper import config
from spyfish.extraction.extract_frames import read_frame_from_cap, read_video_rotation

_SECTION = "extraction.slate_detection"


@dataclass(frozen=True)
class SlateParams:
    scan_width: int
    step_seconds: float
    max_seconds: float
    grace_seconds: float
    paper_value_min: int
    paper_saturation_max: int
    ink_value_max: int
    min_paper_frac: float
    min_solidity: float
    max_aspect: float
    max_edge_frac: float
    min_strokes: int
    stroke_area_px: tuple[int, int]

    @classmethod
    def from_config(cls) -> "SlateParams":
        raw = config.slate_detection
        lo, hi = get_required(raw, "stroke_area_px", _SECTION)
        return cls(
            scan_width=int(get_required(raw, "scan_width", _SECTION)),
            step_seconds=float(get_required(raw, "step_seconds", _SECTION)),
            max_seconds=float(get_required(raw, "max_seconds", _SECTION)),
            grace_seconds=float(get_required(raw, "grace_seconds", _SECTION)),
            paper_value_min=int(get_required(raw, "paper_value_min", _SECTION)),
            paper_saturation_max=int(
                get_required(raw, "paper_saturation_max", _SECTION)
            ),
            ink_value_max=int(get_required(raw, "ink_value_max", _SECTION)),
            min_paper_frac=float(get_required(raw, "min_paper_frac", _SECTION)),
            min_solidity=float(get_required(raw, "min_solidity", _SECTION)),
            max_aspect=float(get_required(raw, "max_aspect", _SECTION)),
            max_edge_frac=float(get_required(raw, "max_edge_frac", _SECTION)),
            min_strokes=int(get_required(raw, "min_strokes", _SECTION)),
            stroke_area_px=(int(lo), int(hi)),
        )


@dataclass
class FrameScore:
    seconds: float
    paper_frac: float
    solidity: float
    aspect: float
    edge_frac: float
    strokes: int
    score: float  # 0 = not a slate

    def as_row(self) -> dict:
        return {
            "slate_seconds": self.seconds,
            "slate_score": round(self.score, 3),
            "paper_frac": round(self.paper_frac, 3),
            "solidity": round(self.solidity, 2),
            "aspect": round(self.aspect, 2),
            "edge_frac": round(self.edge_frac, 3),
            "strokes": self.strokes,
        }


def score_frame(frame, p: SlateParams) -> FrameScore:
    """Score one BGR frame for slate-likeness.

    A slate is the largest bright, unsaturated blob that is (a) big, (b) convex
    and board-shaped, (c) smooth inside (whiteboard, not non-slip deck) and
    (d) carries a handful of small dark strokes (handwriting). Each gate was
    needed: "big white area with some dark pixels" alone picks the boat deck.
    """
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (p.scan_width, max(1, int(h * p.scan_width / w))))
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    val, sat = hsv[..., 2], hsv[..., 1]
    empty = FrameScore(0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0)

    paper = ((val > p.paper_value_min) & (sat < p.paper_saturation_max)).astype(
        np.uint8
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    # close first so ink strokes do not split the board into fragments
    paper = cv2.morphologyEx(paper, cv2.MORPH_CLOSE, kernel)
    paper = cv2.morphologyEx(paper, cv2.MORPH_OPEN, kernel)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(paper, connectivity=8)
    if n <= 1:
        return empty
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    paper_frac = float(stats[biggest, cv2.CC_STAT_AREA] / paper.size)
    mask = (labels == biggest).astype(np.uint8)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    hull_area = cv2.contourArea(cv2.convexHull(contour))
    solidity = float(cv2.contourArea(contour) / hull_area) if hull_area else 0.0
    (_, _), (rw, rh), _ = cv2.minAreaRect(contour)
    aspect = float(max(rw, rh) / max(min(rw, rh), 1.0))

    region = cv2.dilate(mask, kernel)  # include the strokes that CLOSE filled in
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    edge_frac = float((edges[region > 0] > 0).mean())
    ink = ((val < p.ink_value_max) & (region > 0)).astype(np.uint8)
    n_ink, _, ink_stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    areas = ink_stats[1:, cv2.CC_STAT_AREA] if n_ink > 1 else np.array([])
    lo, hi = p.stroke_area_px
    strokes = int(((areas >= lo) & (areas <= hi)).sum())

    ok = (
        paper_frac >= p.min_paper_frac
        and solidity >= p.min_solidity
        and aspect <= p.max_aspect
        and edge_frac <= p.max_edge_frac
        and strokes >= p.min_strokes
    )
    # bigger and smoother is better; the gates above do the real discrimination
    score = paper_frac * solidity * (1.0 - edge_frac) if ok else 0.0
    return FrameScore(0.0, paper_frac, solidity, aspect, edge_frac, strokes, score)


def find_slate_frame(
    video: Path,
    p: Optional[SlateParams] = None,
    max_seconds: Optional[float] = None,
) -> tuple[Optional[FrameScore], Optional[np.ndarray]]:
    """Scan the start of `video`; return (best score, full-res frame) or (None, None).

    Samples one frame per `step_seconds`. Stops `grace_seconds` after the last
    slate-like frame (the board came and went, the best frame is behind us),
    or at the scan limit if none turned up: the smaller of the configured
    `max_seconds`, the video length, and `max_seconds` passed in (the pipeline
    passes `sampling_start`: the rig is on the bottom by then, so the board
    was shown earlier).
    """
    p = p or SlateParams.from_config()
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video {video}")
    try:
        rotation = read_video_rotation(cap)
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
        end = p.max_seconds
        if max_seconds is not None and max_seconds > 0:
            end = min(end, float(max_seconds))
        if (
            fps
        ):  # stay inside the file so a short clip does not log a read error per step
            end = min(end, frames / fps - p.step_seconds)
        best: Optional[FrameScore] = None
        last_hit: Optional[float] = None
        t = 0.0
        while t <= end:
            frame = read_frame_from_cap(cap, t, rotation=rotation)
            if frame is None:
                break
            fs = score_frame(frame, p)
            fs.seconds = t
            if fs.score > 0:
                last_hit = t
            if best is None or fs.score > best.score:
                best = fs
            if last_hit is not None and t - last_hit >= p.grace_seconds:
                break
            t += p.step_seconds

        if best is None or best.score <= 0:
            return None, None
        return best, read_frame_from_cap(cap, best.seconds, rotation=rotation)
    finally:
        cap.release()


def save_slate_frame(
    video_path: Path,
    out_path: Path,
    max_seconds: Optional[float] = None,
    force: bool = False,
) -> Optional[FrameScore]:
    """Scan `video_path` and write the best slate frame to `out_path`.

    Skips the scan when `out_path` already exists unless `force`. Returns the
    winning score, or None when nothing was written (already there, or no
    slate found).
    """
    if out_path.exists() and not force:
        logging.info(f"{video_path.name}: slate frame already at {out_path}, skipping.")
        return None

    best, frame = find_slate_frame(video_path, max_seconds=max_seconds)
    if best is None or frame is None:
        logging.info(
            f"{video_path.name}: no metadata slate found at the start of the video."
        )
        return None

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    logging.info(
        f"{video_path.name}: slate frame at {best.seconds:.0f}s saved to {out_path}."
    )
    return best


def extract_slate_frame(
    drop_id: str,
    video_path: Path,
    max_seconds: Optional[float] = None,
    force: bool = False,
) -> Optional[Path]:
    """Save the drop's slate JPEG next to its other artifacts; skip if already there.

    `max_seconds` caps the scan (the pipeline passes `sampling_start`); `force`
    re-scans over an existing JPEG (the pipeline never does, the wip script can).
    Returns the JPEG path, or None when no slate was found (logged, not an
    error: plenty of deployments start underwater or never show the board).
    """
    config.validate_drop_id(drop_id)
    out_path = config.get_slate_frame_path(drop_id)
    if out_path.exists() and not force:
        logging.info(f"{drop_id}: slate frame already at {out_path}, skipping.")
        return out_path
    best = save_slate_frame(video_path, out_path, max_seconds=max_seconds, force=force)
    return out_path if best is not None else None
