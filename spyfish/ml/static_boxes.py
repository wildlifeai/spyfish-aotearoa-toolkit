"""Boxes that never move: sticks, sea stars, rocks, the bait canister, rig.

YOLO calls such an object a fish in every frame it sees it, and BioCLIP renames
it a species (a HOR stick became rock lobster at 0.84), so it sits in the MaxN
as a ghost for as long as it is in view. A track here is a run of boxes linked
over time; it is STATIC when the spot it sits on is tight (every box the drop
has around it within ``max_spread_px`` and ``max_size_spread``) for at least
``min_span_seconds``. Knobs: ``training.classifier.static_boxes``.

Static is not the same as junk: a resting blue cod, a tarakihi holding station
or an eel in its hole is static too. The fish score (YOLO's mean confidence
plus how rarely YOLO fires on the box, each standardised on the first review)
separates them: in the 2026-11-12 review of 240 tracks over 14 surveys, every
track scoring below 0 was junk (111 of 111). ``flag_static`` relabels those
tracks' boxes ``static_label`` ("Not a fish?") in the raw CSV: left out of the
ML MaxN (``maxn_exclude_classes``) and of frame choice, still uploaded to BIIGLE
under that label for the expert to delete or rename. The original name stays
in ``static_class``. Offline review tool: ``scripts/two_stage/static_boxes.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from spyfish.config.wrapper import config
from spyfish.extraction.render_annotations import iou

STATIC_TRACK_COLUMN = "static_track"
STATIC_CLASS_COLUMN = "static_class"


@dataclass
class Track:
    rows: list = field(default_factory=list)  # raw CSV row indices
    boxes: list = field(default_factory=list)  # corner boxes
    times: list = field(default_factory=list)

    def mean_box(self) -> tuple:
        return tuple(np.mean(self.boxes, axis=0))

    def add(self, idx: int, box: tuple, t: float) -> None:
        self.rows.append(idx)
        self.boxes.append(box)
        self.times.append(t)


def _corners(r) -> tuple:
    return (r.x - r.w / 2, r.y - r.h / 2, r.x + r.w / 2, r.y + r.h / 2)


def _centre_diag(box: tuple) -> tuple[float, float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2, (y1 + y2) / 2, float(np.hypot(x2 - x1, y2 - y1))


def link_tracks(df: pd.DataFrame, sb) -> list[Track]:
    """Greedy frame-by-frame linking against each open track's MEAN box.

    By centre distance, not overlap: camera sway shifts every object by the
    same pixels, so overlap drops fast for a small box (an 80 px stick was lost
    at +/-15 px) while a distance floor (``link_distance_px``) does not. The
    reach is ``link_distance_box`` x the box diagonal or the floor, whichever
    is larger, and the diagonals must agree within ``link_size_ratio``.
    """
    open_tracks: list[Track] = []
    done: list[Track] = []
    for t, frame_rows in df.groupby("time_seconds", sort=True):
        # Retire tracks that have been silent too long.
        still = []
        for tr in open_tracks:
            (done if t - tr.times[-1] > sb.max_gap_seconds else still).append(tr)
        open_tracks = still
        means = [_centre_diag(tr.mean_box()) for tr in open_tracks]
        # Every allowed (box, track) pair, best first, assigned greedily: a
        # fish crossing the spot in the same frame cannot take the static
        # object's track and push the object into a second one. Cost = distance
        # as a share of the reach + how far the sizes differ (log ratio).
        boxes = [(idx, _corners(r)) for idx, r in frame_rows.iterrows()]
        pairs = []
        for bi, (_, box) in enumerate(boxes):
            cx, cy, diag = _centre_diag(box)
            for j, (mx, my, mdiag) in enumerate(means):
                ratio = diag / max(mdiag, 1e-6)
                if not 1 / sb.link_size_ratio <= ratio <= sb.link_size_ratio:
                    continue
                reach = max(sb.link_distance_box * mdiag, sb.link_distance_px)
                d = float(np.hypot(cx - mx, cy - my))
                if d <= reach:
                    pairs.append((d / reach + abs(np.log(ratio)), bi, j))
        used_boxes, used_tracks = set(), set()
        for _, bi, j in sorted(pairs):
            if bi in used_boxes or j in used_tracks:
                continue
            idx, box = boxes[bi]
            open_tracks[j].add(idx, box, t)
            used_boxes.add(bi)
            used_tracks.add(j)
        for bi, (idx, box) in enumerate(boxes):
            if bi not in used_boxes:
                tr = Track()
                tr.add(idx, box, t)
                open_tracks.append(tr)
    return done + open_tracks


def static_tracks(
    df: pd.DataFrame, drop_id: str, members: dict | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(one row per static track, ``crops_per_track`` raw rows per track).

    ``members``, when given, is filled with ``{track id: [df index, ...]}``:
    every box of each static track, which the pipeline relabels."""
    sb = config.classifier_static_boxes
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()
    times = np.sort(df["time_seconds"].unique())
    # The sampling step inference used: the commonest gap between detection
    # times. The raw CSV lists only frames WITH a detection, so counting its
    # frames would call any track in an otherwise empty drop fully covered.
    gaps = np.round(np.diff(times), 3)
    step = float(pd.Series(gaps[gaps > 0]).mode().iloc[0]) if (gaps > 0).any() else 1.0
    tracks = link_tracks(df, sb)
    cx_all, cy_all = df["x"].to_numpy(), df["y"].to_numpy()
    diag_all = np.hypot(df["w"].to_numpy(), df["h"].to_numpy())
    summary, picks = [], []
    for k, tr in enumerate(tracks):
        t0, t1 = tr.times[0], tr.times[-1]
        span = t1 - t0
        if span < sb.min_span_seconds:
            continue
        if len(tr.rows) < sb.min_boxes:
            continue
        # Judge the SPOT, not the track: every box the drop has around the
        # track's centre. Linking only accepts near, same-size boxes, so a
        # track carved out of the crowd at the bait looks tight by
        # construction; the crowd around it does not. Around a stick nearly
        # every nearby box is the stick.
        tc = np.median(
            [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in tr.boxes], axis=0
        )
        td = float(np.median([np.hypot(b[2] - b[0], b[3] - b[1]) for b in tr.boxes]))
        reach = max(sb.link_distance_box * td, sb.link_distance_px)
        local = np.hypot(cx_all - tc[0], cy_all - tc[1]) <= reach
        lc = np.column_stack([cx_all[local], cy_all[local]])
        spread = float(np.median(np.hypot(*(lc - np.median(lc, axis=0)).T)))
        size_spread = float(
            np.median(np.abs(diag_all[local] / np.median(diag_all[local]) - 1))
        )
        if spread > sb.max_spread_px or size_spread > sb.max_size_spread:
            continue
        coverage = len(set(tr.times)) / (span / step + 1)
        rows = df.loc[tr.rows]
        mean = tr.mean_box()
        track_id = f"{drop_id}|{k}"
        if members is not None:
            members[track_id] = list(tr.rows)
        names = rows["clf_class"] if "clf_class" in rows else rows["class"]
        summary.append(
            {
                "DropID": drop_id,
                "track": track_id,
                "start_s": round(t0, 1),
                "end_s": round(t1, 1),
                "span_min": round(span / 60, 1),
                "boxes": len(rows),
                "coverage": round(coverage, 2),
                "spread_px": round(spread, 1),
                "size_spread": round(size_spread, 2),
                "mean_w": round(mean[2] - mean[0]),
                "mean_h": round(mean[3] - mean[1]),
                "yolo_top": (
                    rows["yolo_class"] if "yolo_class" in rows else rows["class"]
                )
                .mode()
                .iloc[0],
                "bioclip_top": (
                    names.dropna().mode().iloc[0] if names.notna().any() else ""
                ),
                "mean_conf": round(rows["confidence"].mean(), 2),
                # Junk tracks ran 0.28-0.44 YOLO, the two resting fish 0.56-0.59
                # (HOR, 14 tracks): a second signal next to tightness.
                "bioclip_conf": (
                    round(
                        pd.to_numeric(rows["clf_confidence"], errors="coerce").mean(), 2
                    )
                    if "clf_confidence" in rows
                    else None
                ),
            }
        )
        # Crops only from boxes that match the track's typical box: the static
        # object always has the same box, a fish lying across the spot gets a
        # different one (HOR_012, SLI_013: 1-2 of 5 evenly spread crops were
        # passing fish). Then spread those over the span.
        median_box = tuple(np.median(tr.boxes, axis=0))
        overlap = [iou(b, median_box) for b in tr.boxes]
        typical = [i for i, o in enumerate(overlap) if o >= sb.crop_min_iou]
        if len(typical) < sb.crops_per_track:
            # Too few boxes match the median box: it is then a box nobody drew
            # (the track's boxes wander). Fill with the closest ones, in time
            # order, rather than drawing an empty row on the sheet (2026-11-12:
            # SLI_013 and AHE_149 rows came out blank).
            closest = sorted(np.argsort(overlap)[::-1][: sb.crops_per_track])
            typical = sorted(set(typical) | {int(i) for i in closest})
        spread = np.linspace(0, len(typical) - 1, min(sb.crops_per_track, len(typical)))
        pick = rows.iloc[
            [typical[k] for k in np.unique(spread.round().astype(int))]
        ].copy()
        pick["DropID"], pick["track"] = drop_id, track_id
        picks.append(pick)
    return pd.DataFrame(summary), (pd.concat(picks) if picks else pd.DataFrame())


def fish_score(tracks: pd.DataFrame) -> pd.Series:
    """YOLO mean confidence + rarity of detection (-log boxes per minute),
    each standardised on the first review's 136 tracks. Below 0 = junk in
    every reviewed case; higher = more fish-like."""
    sb = config.classifier_static_boxes
    rate = -np.log(tracks["boxes"] / tracks["span_min"])
    return (tracks["mean_conf"] - sb.score_conf_mean) / sb.score_conf_std + (
        rate - sb.score_rate_mean
    ) / sb.score_rate_std


def flag_static(df: pd.DataFrame, drop_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``df`` with every box of a static track scoring below ``flag_below``
    relabelled ``static_label``, and the flagged tracks.

    Idempotent: boxes already relabelled get their ``static_class`` back before
    tracking, so a rerun of ``--ml-post`` flags the same tracks again rather
    than nesting labels. Tracking sees every box above the confidence
    threshold, as the review did.
    """
    sb = config.classifier_static_boxes
    label = config.static_box_label
    out = df.copy()
    if STATIC_CLASS_COLUMN in out.columns:
        was = out[STATIC_CLASS_COLUMN].notna() & (out["class"] == label)
        out.loc[was, "class"] = out.loc[was, STATIC_CLASS_COLUMN]
    out[STATIC_TRACK_COLUMN] = None
    out[STATIC_CLASS_COLUMN] = None
    seen = out[out["confidence"] >= config.confidence_threshold]
    members: dict = {}
    tracks, _ = static_tracks(seen, drop_id, members)
    if tracks.empty:
        return out, tracks
    tracks = tracks.assign(fish_score=fish_score(tracks).round(2))
    flagged = tracks[tracks["fish_score"] < sb.flag_below]
    for track_id in flagged["track"]:
        idx = members[track_id]
        out.loc[idx, STATIC_CLASS_COLUMN] = out.loc[idx, "class"]
        out.loc[idx, "class"] = label
        out.loc[idx, STATIC_TRACK_COLUMN] = track_id
    return out, flagged
