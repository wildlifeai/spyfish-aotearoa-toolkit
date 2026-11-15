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

import csv
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np

try:
    import av  # PyAV: keyframe-only decode for the scan
except ImportError:  # pragma: no cover - NeSI venv has it; a bare env falls back to cv2
    av = None

from spyfish.config.base import get_required
from spyfish.config.wrapper import config
from spyfish.extraction.extract_frames import read_frame_from_cap, read_video_rotation
from spyfish.utils import move_to_trash

_SECTION = "extraction.slate_detection"


@dataclass(frozen=True)
class SlateParams:
    scan_width: int
    step_seconds: float
    max_seconds: float
    grace_seconds: float
    paper_value_min: int
    paper_saturation_max: int
    water_hue_range: tuple[int, int]
    water_saturation_min: int
    water_frame_max: float
    wood_hue_range: tuple[int, int]
    wood_saturation_min: int
    wood_value_min: int
    ink_darker_by: int
    ink_saturation_min: int
    min_paper_frac: float
    min_solidity: float
    min_rect_fill: float
    max_aspect: float
    max_edge_frac: float
    min_strokes: int
    stroke_area_frac: tuple[float, float]
    cut_off_penalty: float
    max_cut_sides: int
    sharpness_tie_band: float
    boards: tuple[str, ...] = (
        "white",
    )  # colour masks to try; wood only where a survey uses it

    @classmethod
    def from_config(cls, drop_id: Optional[str] = None) -> "SlateParams":
        """Thresholds from config; with a drop ID, the board colours its survey uses."""
        raw = config.slate_detection
        lo, hi = get_required(raw, "stroke_area_frac", _SECTION)
        h0, h1 = get_required(raw, "wood_hue_range", _SECTION)
        w0, w1 = get_required(raw, "water_hue_range", _SECTION)
        return cls(
            scan_width=int(get_required(raw, "scan_width", _SECTION)),
            step_seconds=float(get_required(raw, "step_seconds", _SECTION)),
            max_seconds=float(get_required(raw, "max_seconds", _SECTION)),
            grace_seconds=float(get_required(raw, "grace_seconds", _SECTION)),
            paper_value_min=int(get_required(raw, "paper_value_min", _SECTION)),
            paper_saturation_max=int(
                get_required(raw, "paper_saturation_max", _SECTION)
            ),
            water_hue_range=(int(w0), int(w1)),
            water_saturation_min=int(
                get_required(raw, "water_saturation_min", _SECTION)
            ),
            water_frame_max=float(get_required(raw, "water_frame_max", _SECTION)),
            wood_hue_range=(int(h0), int(h1)),
            wood_saturation_min=int(get_required(raw, "wood_saturation_min", _SECTION)),
            wood_value_min=int(get_required(raw, "wood_value_min", _SECTION)),
            ink_darker_by=int(get_required(raw, "ink_darker_by", _SECTION)),
            ink_saturation_min=int(get_required(raw, "ink_saturation_min", _SECTION)),
            min_paper_frac=float(get_required(raw, "min_paper_frac", _SECTION)),
            min_solidity=float(get_required(raw, "min_solidity", _SECTION)),
            min_rect_fill=float(get_required(raw, "min_rect_fill", _SECTION)),
            max_aspect=float(get_required(raw, "max_aspect", _SECTION)),
            max_edge_frac=float(get_required(raw, "max_edge_frac", _SECTION)),
            min_strokes=int(get_required(raw, "min_strokes", _SECTION)),
            stroke_area_frac=(float(lo), float(hi)),
            cut_off_penalty=float(get_required(raw, "cut_off_penalty", _SECTION)),
            max_cut_sides=int(get_required(raw, "max_cut_sides", _SECTION)),
            sharpness_tie_band=float(get_required(raw, "sharpness_tie_band", _SECTION)),
            boards=cls.boards_for(drop_id),
        )

    @staticmethod
    def boards_for(drop_id: Optional[str]) -> tuple[str, ...]:
        """("white",) unless the drop's survey is listed under `wood_surveys`."""
        wood = {
            str(x)
            for x in get_required(config.slate_detection, "wood_surveys", _SECTION)
            or []
        }
        if drop_id is None:
            return ("white",)
        survey = config.get_survey_id_from_drop(config.validate_drop_id(drop_id))
        return ("white", "wood") if survey in wood else ("white",)


@dataclass
class FrameScore:
    seconds: float
    paper_frac: float
    solidity: float
    aspect: float
    rect_fill: (
        float  # blob area / its minimum bounding rectangle: a board fills its box
    )
    edge_frac: float
    strokes: int
    score: float  # 0 = not a slate
    board: str = ""  # which colour mask the blob came from: "white" or "wood"
    sharpness: float = (
        0.0  # Laplacian variance over the board: motion blur pulls it down
    )
    fails: str = (
        ""  # gates this frame failed, comma-joined ("" = passed all); for near-miss reports
    )

    def as_row(self) -> dict:
        return {
            "slate_seconds": self.seconds,
            "slate_score": round(self.score, 3),
            "board": self.board,
            "sharpness": round(self.sharpness, 1),
            "fails": self.fails,
            "paper_frac": round(self.paper_frac, 3),
            "solidity": round(self.solidity, 2),
            "aspect": round(self.aspect, 2),
            "rect_fill": round(self.rect_fill, 2),
            "edge_frac": round(self.edge_frac, 3),
            "strokes": self.strokes,
        }


def score_frame(frame, p: SlateParams) -> FrameScore:
    """Score one BGR frame for slate-likeness.

    A slate is the largest board-coloured blob that is (a) big, (b) convex
    and board-shaped, (c) smooth inside (board, not non-slip deck) and
    (d) carries a handful of small dark strokes (handwriting). Each gate was
    needed: "big pale area with some dark pixels" alone picks the boat deck.

    Two board colours, each its own mask through the same gates, best score
    wins: the white/grey whiteboard most teams use, and the varnished wooden
    clipboard AHE writes on in marker (hue 12-15, saturation 90-180, value
    160-220 in OpenCV HSV; the white mask never saw it, so every AHE drop
    came back "no slate" until 2026-11-03). Which masks run is `p.boards`:
    wood only for the surveys listed in config, because skin and lit deck
    share its colour and it picked those wherever the real board was white.
    """
    gray, hue, sat, val, masks = board_masks(frame, p)
    if water_fraction(hue, sat, p) > p.water_frame_max:
        return FrameScore(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, "white", fails="water")
    best = None
    for board, mask in masks.items():
        fs = _score_board(gray, sat, val, mask.astype(np.uint8), p, board)
        fs.board = board
        if best is None or _closer(fs, best):
            best = fs
    return best if best is not None else _no_blob("white")


def water_fraction(hue, sat, p: SlateParams) -> float:
    """Share of the frame in the water hue band: ~1.0 on the seabed or looking
    at open water, 0.2-0.5 on deck. Shaded white paper and pale seabed are the
    same colour pixel by pixel (blue-tinted, saturation 40-80); the frame as a
    whole tells them apart, so this is a frame gate, not a pixel one."""
    w0, w1 = p.water_hue_range
    return float(((hue >= w0) & (hue <= w1) & (sat > p.water_saturation_min)).mean())


def board_masks(
    frame, p: SlateParams
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """The frame at scan width as (gray, hue, sat, val) and one boolean mask per
    enabled board colour. Public so a debug view can show what the mask caught."""
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (p.scan_width, max(1, int(h * p.scan_width / w))))
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    h0, h1 = p.wood_hue_range
    masks = {
        "white": (val > p.paper_value_min) & (sat < p.paper_saturation_max),
        "wood": (hue >= h0)
        & (hue <= h1)
        & (sat >= p.wood_saturation_min)
        & (val >= p.wood_value_min),
    }
    return gray, hue, sat, val, {b: m for b, m in masks.items() if b in p.boards}


def ink_mask(sat, val, paper, p: SlateParams, board: str) -> np.ndarray:
    """Pixels counted as ink on `paper` (one blob's mask or the whole colour
    mask): darker than the paper's median value, or (white board only)
    saturated like green and blue marker."""
    if not paper.any():
        return np.zeros_like(paper, dtype=bool)
    board_val = float(np.median(val[paper > 0]))
    ink = val < board_val - p.ink_darker_by
    if board == "white":
        ink |= sat > p.ink_saturation_min
    return ink & (paper > 0)


def _closer(a: FrameScore, b: FrameScore) -> bool:
    """Is `a` the better slate (or, when neither passes, the nearer miss) than `b`?"""

    def key(f: FrameScore) -> tuple:
        return (f.score, -len(f.fails.split(",")) if f.fails else 0, f.paper_frac)

    return key(a) > key(b)


def _score_board(gray, sat, val, paper, p: SlateParams, board: str) -> FrameScore:
    """Gate and score every blob of one colour mask that is big enough to be a
    board, best one wins (see `score_frame`).

    Not just the largest: a white hull, gunwale or deck beside the board is
    often the biggest pale blob in the frame, and gating only that one hid
    the written board behind it (SLI_20220228_BUV_SLI_066_01 at 62 s, all of
    TON_20241125_BUV: "no slate", near miss "strokes", board in plain view).
    When no blob is big enough the largest is scored anyway, so the near-miss
    diagnostics still say what the frame came closest to.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    # close first so ink strokes do not split the board into fragments
    paper = cv2.morphologyEx(paper, cv2.MORPH_CLOSE, kernel)
    paper = cv2.morphologyEx(paper, cv2.MORPH_OPEN, kernel)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(paper, connectivity=8)
    if n <= 1:
        return _no_blob(board)
    areas = stats[1:, cv2.CC_STAT_AREA]
    by_size = [1 + int(i) for i in np.argsort(-areas)]
    big = [
        i
        for i in by_size
        if stats[i, cv2.CC_STAT_AREA] / paper.size >= p.min_paper_frac
    ]
    best = None  # any real blob beats "no blob", however many gates it fails: the near miss must describe it
    for idx in big or by_size[:1]:
        fs = _score_blob(gray, sat, val, labels, stats, idx, kernel, p, board)
        if best is None or _closer(fs, best):
            best = fs
    return best


def _no_blob(board: str) -> FrameScore:
    """Nothing of this colour in the frame at all (underwater, or a dark deck)."""
    return FrameScore(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, board, fails="paper_frac")


def _score_blob(
    gray, sat, val, labels, stats, biggest: int, kernel, p: SlateParams, board: str
) -> FrameScore:
    """Gates and score for one connected component of the colour mask."""
    blob_area = int(stats[biggest, cv2.CC_STAT_AREA])
    paper_frac = float(blob_area / labels.size)
    mask = (labels == biggest).astype(np.uint8)
    # how many frame edges the blob's box touches: a board half out of shot
    H, W = labels.shape
    x, y, bw, bh = (
        int(stats[biggest, i])
        for i in (
            cv2.CC_STAT_LEFT,
            cv2.CC_STAT_TOP,
            cv2.CC_STAT_WIDTH,
            cv2.CC_STAT_HEIGHT,
        )
    )
    cut_sides = int(x <= 0) + int(y <= 0) + int(x + bw >= W) + int(y + bh >= H)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    hull_area = cv2.contourArea(cv2.convexHull(contour))
    solidity = float(cv2.contourArea(contour) / hull_area) if hull_area else 0.0
    (_, _), (rw, rh), _ = cv2.minAreaRect(contour)
    aspect = float(max(rw, rh) / max(min(rw, rh), 1.0))
    # Hands are the wood mask's twin (skin has wood's hue, saturation and
    # brightness) and deck patches the white mask's; both are convex enough
    # to pass solidity but fill their bounding box poorly (0.47-0.51 measured)
    # where a board fills it (0.62-0.87).
    rect_fill = float(cv2.contourArea(contour) / (rw * rh)) if rw * rh else 0.0

    region = cv2.dilate(mask, kernel)  # include the strokes that CLOSE filled in
    edges = cv2.Canny(gray, 80, 160)
    edge_frac = float((edges[region > 0] > 0).mean())
    # Not a gate: a tie-break between near-equal frames of the held-steady
    # stretch, so the crispest one is kept for reading (see rank_candidates).
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F)[region > 0].var())
    # Ink relative to THIS board: darker than its median, or (white board
    # only) saturated, which is what green and blue marker look like.
    ink = ink_mask(sat, val, region, p, board).astype(np.uint8)
    n_ink, _, ink_stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    areas = ink_stats[1:, cv2.CC_STAT_AREA] if n_ink > 1 else np.array([])
    lo, hi = (f * blob_area for f in p.stroke_area_frac)  # letters scale with the board
    strokes = int(((areas >= max(lo, 1)) & (areas <= hi)).sum())

    failed = [
        name
        for name, bad in (
            ("paper_frac", paper_frac < p.min_paper_frac),
            ("solidity", solidity < p.min_solidity),
            ("rect_fill", rect_fill < p.min_rect_fill),
            ("aspect", aspect > p.max_aspect),
            ("edge_frac", edge_frac > p.max_edge_frac),
            ("strokes", strokes < p.min_strokes),
            (
                "cut_off",
                cut_sides > p.max_cut_sides,
            ),  # deck, sky and seabed floods run off 3-4 edges; a held board 0-2
        )
        if bad
    ]
    ok = not failed
    # The gates decide "is this a slate"; the score picks the most readable one.
    # Strokes go in the score, not just the gate: ranking on white area alone
    # picked a close-up of the blank lower half over the whole written board
    # (SLI_20260114_BUV_SLI_065_01: 5 strokes, paper_frac 0.58).
    score = paper_frac * solidity * (1.0 - edge_frac) * strokes if ok else 0.0
    score *= (
        1.0 - p.cut_off_penalty * cut_sides / 4.0
    )  # whole board beats the same board half out of shot
    return FrameScore(
        0.0,
        paper_frac,
        solidity,
        aspect,
        rect_fill,
        edge_frac,
        strokes,
        score,
        sharpness=sharpness,
        fails=",".join(failed),
    )


def _scan_end(
    cap: cv2.VideoCapture, p: SlateParams, max_seconds: Optional[float]
) -> float:
    """Where the scan stops: the smallest of config, caller cap and video length."""
    end = p.max_seconds
    if max_seconds is not None and max_seconds > 0:
        end = min(end, float(max_seconds))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    if fps:  # stay inside the file so a short clip does not log a read error per step
        end = min(end, frames / fps - p.step_seconds)
    return end


def _keyframes_av(
    video: Path, p: SlateParams, end: float
) -> Iterator[tuple[float, np.ndarray]]:
    """(seconds, small BGR frame) for the keyframe at or before each `step_seconds`.

    One index seek per step, then ONE decoded frame: the keyframe the seek
    landed on (`skip_frame = NONKEY` keeps the decoder from touching anything
    else). GoPro files have a keyframe every 0.5-2 s, so this reads about one
    packet per step instead of every byte of the opening minutes. That is
    what matters on NeSI's nobackup, where the sequential pass was I/O-bound
    (40-95 s per video; the decode itself is a few seconds) and cv2's
    time-seeks were worse still (each re-decodes from the previous keyframe:
    2-3 min per video with stream timeouts).

    The frame is scaled to `scan_width` by the decoder's own reformatter and
    not rotated: the score is insensitive to a 90/180 turn (blob size,
    solidity, aspect as long/short, stroke count), and the saved full-res
    frame goes through cv2 with the rotation applied as before.
    """
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        stream.codec_context.skip_frame = "NONKEY"
        w, h = stream.codec_context.width, stream.codec_context.height
        small_h = max(2, int(h * p.scan_width / w)) // 2 * 2
        last_pts = None
        t = 0.0
        while t <= end:
            container.seek(
                int(t / stream.time_base), stream=stream, backward=True, any_frame=False
            )
            frame = next(container.decode(stream), None)
            if frame is None:
                return
            if (
                frame.pts != last_pts
            ):  # step shorter than the keyframe interval: same frame again
                last_pts = frame.pts
                secs = (
                    float(frame.pts * stream.time_base) if frame.pts is not None else t
                )
                img = frame.reformat(
                    width=p.scan_width, height=small_h, format="bgr24"
                ).to_ndarray()
                yield secs, img
            t += p.step_seconds


def _frames_cv2(
    cap: cv2.VideoCapture, p: SlateParams, end: float, rotation: int
) -> Iterator[tuple[float, np.ndarray]]:
    """Fallback without PyAV: seek by time once per `step_seconds`."""
    t = 0.0
    while t <= end:
        frame = read_frame_from_cap(cap, t, rotation=rotation)
        if frame is None:
            return
        yield t, frame
        t += p.step_seconds


def scan_frames(
    video: Path, p: SlateParams, max_seconds: Optional[float] = None
) -> Iterator[tuple[float, np.ndarray]]:
    """Every sampled (seconds, frame) of the slate scan, PyAV when available."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video {video}")
    try:
        end = _scan_end(cap, p, max_seconds)
        rotation = read_video_rotation(cap)
        if av is None:
            # cv2 path keeps the capture open for the frame walk below.
            yield from _frames_cv2(cap, p, end, rotation)
            return
    finally:
        # One release for every exit, including an exception from _scan_end
        # or read_video_rotation on the cv2 path (a handle on a ~10 GB file
        # otherwise lives until GC).
        cap.release()
    yield from _keyframes_av(video, p, end)


def rank_candidates(scored: list[FrameScore], tie_band: float) -> list[FrameScore]:
    """Best first: by score, except that frames within `tie_band` of the top
    score are ordered by sharpness. A board held steady for a few seconds
    gives several near-equal frames; the score cannot tell a crisp one from
    one smeared by a hand moving, the Laplacian variance can.
    """
    by_score = sorted(scored, key=lambda f: f.score, reverse=True)
    if not by_score:
        return by_score
    floor = by_score[0].score * (1.0 - tie_band)
    head = [f for f in by_score if f.score >= floor]
    tail = [f for f in by_score if f.score < floor]
    return sorted(head, key=lambda f: f.sharpness, reverse=True) + tail


def find_slate_candidates(
    video: Path,
    p: Optional[SlateParams] = None,
    max_seconds: Optional[float] = None,
    k: int = 1,
    diag: Optional[dict] = None,
    exclude: tuple[float, ...] = (),
    distinct_seconds: float = 0.0,
) -> list[tuple[FrameScore, np.ndarray]]:
    """Scan the start of `video`; the best `k` slate frames, best first, full-res.

    `diag`, when given, receives ``near_miss``: the scanned frame that failed
    the fewest gates (ties: the bigger blob), with the gate names in
    ``fails``. For a video that yields nothing it says whether a board was
    there and what blocked it, or whether nothing board-like ever appeared.

    Samples about one frame per `step_seconds` (see `scan_frames`). Stops
    `grace_seconds` after the last slate-like frame (the board came and went,
    the best frame is behind us), or at the scan limit if none turned up: the
    smaller of the configured `max_seconds`, the video length, and
    `max_seconds` passed in (the pipeline passes `sampling_start`: the rig is
    on the bottom by then, so the board was shown earlier). Empty when no
    frame passed the gates. `k > 1` is for review: when the winner is wrong,
    the right frame is usually a runner-up.
    """
    p = p or SlateParams.from_config()
    scored: list[FrameScore] = []
    seen: list[FrameScore] = []
    last_hit: Optional[float] = None
    near: Optional[FrameScore] = None
    for t, frame in scan_frames(video, p, max_seconds):
        # A moment a person already rejected (as the slate or as a suggestion
        # beside it) is never offered again, so each review round shows new frames.
        if any(abs(t - e) < EXCLUDE_TOLERANCE for e in exclude):
            continue
        fs = score_frame(frame, p)
        fs.seconds = t
        seen.append(fs)
        if near is None or _closer(fs, near):
            near = fs
        if fs.score > 0:
            last_hit = t
            scored.append(fs)
        # Review scans (distinct_seconds > 0) run the whole window: a cluster of
        # early false hits must not end the scan before a later real board.
        if (
            not distinct_seconds
            and last_hit is not None
            and t - last_hit >= p.grace_seconds
        ):
            break
    if diag is not None:
        diag["near_miss"] = near
        diag["near_misses"] = rank_near_misses(seen, k)
        diag["seen"] = seen
    if not scored:
        return []
    ranked = rank_candidates(scored, p.sharpness_tie_band)
    if distinct_seconds:
        ranked = distinct_moments(ranked, distinct_seconds)
    return read_full_frames(video, ranked[:k])


def distinct_moments(ranked: list[FrameScore], gap: float) -> list[FrameScore]:
    """Keep the ranking order but drop frames within `gap` seconds of a better
    one: three runner-ups from one held-up moment are one choice, not three."""
    out: list[FrameScore] = []
    for f in ranked:
        if all(abs(f.seconds - g.seconds) >= gap for g in out):
            out.append(f)
    return out


def rank_near_misses(
    seen: list[FrameScore], k: int, min_gap: float = 2.0
) -> list[FrameScore]:
    """The `k` frames that came closest to passing, for a person to choose from
    when none did: fewest failed gates first, then the most board and writing.
    Underwater frames and frames with no pale blob at all are never offered,
    and picks are `min_gap` seconds apart so they are not the same moment three times.
    """
    pool = [f for f in seen if f.paper_frac > 0 and "water" not in f.fails]
    pool.sort(
        key=lambda f: (
            len(f.fails.split(",")) if f.fails else 0,
            -f.paper_frac * min(f.strokes, 40),
        )
    )
    picked: list[FrameScore] = []
    for f in pool:
        if all(abs(f.seconds - g.seconds) >= min_gap for g in picked):
            picked.append(f)
        if len(picked) == k:
            break
    return picked


EXCLUDE_TOLERANCE = (
    0.5  # seconds: the same keyframe reads as 6.006 one run and 6.01 the next
)


def spread_frames(
    seen: list[FrameScore],
    n: int,
    taken: list[float],
    min_gap: float = 2.0,
    rng: Optional[random.Random] = None,
) -> list[FrameScore]:
    """`n` frames spread over the above-water part of the scan, away from the
    `taken` seconds: something different to pick when every scored choice is
    wrong. Above water only; a frame of the seabed is never a slate.

    The above-water stretch is cut into `n` equal parts and one frame is taken
    from each: at random with `rng` (so repeated review rounds sample different
    moments), from the middle without it (tests, reproducible runs)."""
    pool = [f for f in seen if "water" not in f.fails]
    if n <= 0 or not pool:
        return []
    out: list[FrameScore] = []
    for i in range(n):
        lo = int(i * len(pool) / n)
        hi = max(int((i + 1) * len(pool) / n) - 1, lo)
        slot = (
            rng.randint(lo, hi)
            if rng
            else min(len(pool) - 1, int((i + 0.5) * len(pool) / n))
        )
        # the free frame nearest this slot: a slot on an offered moment moves, it is not lost
        for j in sorted(range(len(pool)), key=lambda j: abs(j - slot)):
            f = pool[j]
            if all(
                abs(f.seconds - s) >= min_gap for s in taken + [g.seconds for g in out]
            ):
                out.append(f)
                break
    return sorted(out, key=lambda f: f.seconds)


def read_full_frames(
    video: Path, frames: list[FrameScore]
) -> list[tuple[FrameScore, np.ndarray]]:
    """Re-read each scored frame from `video` at full resolution (the scan works at scan_width)."""
    if not frames:
        return []
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video {video}")
    try:
        rotation = read_video_rotation(cap)
        out = []
        for fs in frames:
            frame = read_frame_from_cap(cap, fs.seconds, rotation=rotation)
            if frame is not None:
                out.append((fs, frame))
        return out
    finally:
        cap.release()


def find_slate_frame(
    video: Path,
    p: Optional[SlateParams] = None,
    max_seconds: Optional[float] = None,
) -> tuple[Optional[FrameScore], Optional[np.ndarray]]:
    """The single best slate frame of `video`: (score, full-res frame) or (None, None)."""
    found = find_slate_candidates(video, p, max_seconds, k=1)
    return found[0] if found else (None, None)


CANDIDATES_SUFFIX = "_candidates"  # {drop}_slate_candidates/ next to {drop}_slate.jpg
CANDIDATE_WIDTH = 1280  # review thumbnails


def candidates_dir_for(slate_path: Path) -> Path:
    return slate_path.with_name(f"{slate_path.stem}{CANDIDATES_SUFFIX}")


def _write_slate(path: Path, frame: np.ndarray) -> None:
    """Save `frame` as the slate JPEG, no wider than `slate_detection.save_width`."""
    width = int(get_required(config.slate_detection, "save_width", _SECTION))
    h, w = frame.shape[:2]
    if w > width:
        frame = cv2.resize(
            frame, (width, round(h * width / w)), interpolation=cv2.INTER_AREA
        )
    cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])


def save_slate_frame(
    video_path: Path,
    out_path: Path,
    max_seconds: Optional[float] = None,
    force: bool = False,
    candidates: int = 1,
    p: Optional[SlateParams] = None,
    diag: Optional[dict] = None,
    spread: int = 0,
    exclude: tuple[float, ...] = (),
    rng: Optional[random.Random] = None,
    distinct_seconds: float = 0.0,
) -> Optional[FrameScore]:
    """Scan `video_path` and write the best slate frame to `out_path`.

    Skips the scan when `out_path` already exists unless `force`. Returns the
    winning score, or None when nothing was written (already there, or no
    slate found). With `candidates > 1` the runners-up go to
    ``{stem}_candidates/{rank}_{seconds}s.jpg`` as review thumbnails, so a
    wrong winner can be swapped by eye without rescanning the video.
    `spread` more frames, over the above-water part of the scan (random
    within their stretch when `rng` is given), go with them (scores.csv
    ``pick`` = spread): a choice when every scored frame is wrong. Seconds
    in `exclude` (rejected in an earlier review) are never offered.
    """
    if out_path.exists() and not force:
        logging.info(f"{video_path.name}: slate frame already at {out_path}, skipping.")
        return None

    diag = {} if diag is None else diag
    found = find_slate_candidates(
        video_path,
        p,
        max_seconds=max_seconds,
        k=max(1, candidates),
        diag=diag,
        exclude=exclude,
        distinct_seconds=distinct_seconds if candidates > 1 else 0.0,
    )

    def with_spread(picks: list[tuple[FrameScore, np.ndarray]], kinds: list[str]):
        extra = spread_frames(
            diag.get("seen") or [], spread, [fs.seconds for fs, _ in picks], rng=rng
        )
        return picks + read_full_frames(video_path, extra), kinds + ["spread"] * len(
            extra
        )

    if not found:
        if (
            candidates > 1
        ):  # nothing passed: still give the reviewer the closest frames to pick from
            near = read_full_frames(video_path, diag.get("near_misses") or [])
            _write_candidates(out_path, *with_spread(near, ["near miss"] * len(near)))
        near = diag.get("near_miss")
        why = (
            f" Nearest miss at {near.seconds:.0f}s ({near.board}) failed {near.fails}: "
            f"paper={near.paper_frac:.2f} solid={near.solidity:.2f} rect={near.rect_fill:.2f} "
            f"aspect={near.aspect:.2f} edge={near.edge_frac:.3f} strokes={near.strokes}."
            if near is not None
            else ""
        )
        logging.info(
            f"{video_path.name}: no metadata slate found at the start of the video.{why}"
        )
        return None
    best, frame = found[0]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    _write_slate(out_path, frame)
    logging.info(
        f"{video_path.name}: slate frame at {best.seconds:.0f}s saved to {out_path}."
    )
    if candidates > 1:
        _write_candidates(
            out_path, *with_spread(found, ["slate"] + ["runner-up"] * (len(found) - 1))
        )
    return best


def _write_candidates(
    slate_path: Path,
    found: list[tuple[FrameScore, np.ndarray]],
    kinds: Optional[list[str]] = None,
) -> None:
    """Review thumbnails ``{rank}_{seconds}s.jpg`` plus ``scores.csv`` in the
    drop's candidates folder. Passing frames when there is a slate; when there
    is none, the near misses (their ``fails`` column says which gate stopped
    them), so `promote_candidate` can still turn the right one into the slate."""
    if not found:
        return
    cdir = candidates_dir_for(slate_path)
    cdir.mkdir(parents=True, exist_ok=True)
    for old in cdir.glob("*.jpg"):
        move_to_trash(old)  # a re-scan replaces the previous set; nothing is deleted
    rows = []
    for rank, (fs, img) in enumerate(found, start=1):
        h, w = img.shape[:2]
        small = cv2.resize(img, (CANDIDATE_WIDTH, int(h * CANDIDATE_WIDTH / w)))
        cv2.imwrite(
            str(cdir / f"{rank}_{fs.seconds:07.2f}s.jpg"),
            small,
            [cv2.IMWRITE_JPEG_QUALITY, 85],
        )
        rows.append(
            {
                "rank": rank,
                "pick": (kinds or [])[rank - 1] if kinds and rank <= len(kinds) else "",
                **fs.as_row(),
            }
        )
    # the numbers behind the ranking, so a reviewer's "rank 2 was right"
    # can be compared with what the score saw
    with open(cdir / "scores.csv", "w", newline="") as f:
        w_ = csv.DictWriter(f, fieldnames=list(rows[0]))
        w_.writeheader()
        w_.writerows(rows)


def candidate_seconds(candidate: Path) -> float:
    """``2_0014.01s.jpg`` -> 14.01: the second a candidate thumbnail was taken at."""
    return float(candidate.stem.split("_", 1)[1].rstrip("s"))


def promote_candidate(video_path: Path, slate_path: Path, rank: int) -> Optional[float]:
    """Make candidate `rank` the slate: re-read its frame from the video and overwrite the JPEG.

    The thumbnails are review-sized, so the frame is taken from the video
    again at the candidate's second. Returns that second, or None when the
    candidate or the video is missing (logged).
    """
    cdir = candidates_dir_for(slate_path)
    matches = sorted(cdir.glob(f"{rank}_*.jpg")) if cdir.is_dir() else []
    if not matches:
        logging.warning(f"{slate_path.stem}: no candidate {rank} in {cdir}.")
        return None
    if not video_path.exists():
        logging.warning(
            f"{slate_path.stem}: video not at {video_path}, cannot promote."
        )
        return None
    secs = candidate_seconds(matches[0])
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video {video_path}")
    try:
        frame = read_frame_from_cap(cap, secs, rotation=read_video_rotation(cap))
    finally:
        cap.release()
    if frame is None:
        logging.warning(f"{slate_path.stem}: no frame at {secs:.2f}s, cannot promote.")
        return None
    _write_slate(slate_path, frame)
    logging.info(f"{slate_path.stem}: candidate {rank} ({secs:.2f}s) is now the slate.")
    return secs


def save_slate_at(video_path: Path, slate_path: Path, seconds: float) -> bool:
    """Write the frame at `seconds` as the slate: for a board
    a person found by eye (a time strip) where the detector keeps picking the
    wrong frame. Overwrites; the caller moves any old slate aside first."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video {video_path}")
    try:
        frame = read_frame_from_cap(cap, seconds, rotation=read_video_rotation(cap))
    finally:
        cap.release()
    if frame is None:
        logging.warning(
            f"{slate_path.stem}: no frame at {seconds:.2f}s in {video_path.name}."
        )
        return False
    slate_path.parent.mkdir(parents=True, exist_ok=True)
    _write_slate(slate_path, frame)
    logging.info(
        f"{slate_path.stem}: frame at {seconds:.2f}s set as the slate by hand."
    )
    return True


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
    best = save_slate_frame(
        video_path,
        out_path,
        max_seconds=max_seconds,
        force=force,
        candidates=int(
            get_required(config.slate_detection, "review_candidates", _SECTION)
        ),
        spread=int(
            get_required(config.slate_detection, "review_spread_frames", _SECTION)
        ),
        distinct_seconds=float(
            get_required(config.slate_detection, "review_distinct_seconds", _SECTION)
        ),
        p=SlateParams.from_config(drop_id),
    )
    return out_path if best is not None else None
