"""Legacy expert MaxN moments as pipeline frames.

Legacy workbooks (external_id='legacy' in the annotations DB) hold one row per
30 s interval per species, each with the exact second the analyst noted. Until
now nothing put those moments back in front of the footage: the Streamlit
Legacy MaxN Check page seeks them live from S3, one deployment at a time, and
throws the frames away. This writes them to disk once, as part of the post-ML
QA step, in two forms:

* ``frames/`` gets the CLEAN frame, named by video second like every other
  pipeline frame. It is model-independent and shared with the BIIGLE and
  Zooniverse upload paths, so it is written once and never overwritten.
* ``qa_frames/legacy/<model>/`` gets the same frame with the model's boxes and
  a banner carrying the expert's count. One folder per model, so re-running
  --ml with a new model adds a sibling folder instead of replacing the old
  one, and the two can be compared without re-extracting anything.

Which moments: every species' peak interval (highest count, earliest on ties,
the rule the workbooks' own MaxCount sheet follows), plus a few other scored
intervals per species chosen at random but seeded by DropID, so the check
covers ordinary moments as well as the peak and the same frames come back on
every run.

Verified against the footage 2026-10-09 (SLI 2024): legacy seconds are video
time from frame 0, no lowering offset, so they can be seeked directly.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import cv2
import pandas as pd

from spyfish.config.base import NULL_DEPLOYMENT
from spyfish.config.wrapper import config
from spyfish.database.annotation_manager import LEGACY_EXTERNAL_ID
from spyfish.extraction.extract_frames import (
    extract_one_frame_from_cap,
    nearest_raw_frame,
    read_video_rotation,
)
from spyfish.ml.draw_frames import (
    Labeller,
    common_name_labeller,
    count_summary,
    draw_banner,
    draw_boxes,
)
from spyfish.utils import generate_frame_filename, seconds_to_time

KIND_PEAK = "legacy_maxn"
KIND_INTERVAL = "legacy_interval"


def legacy_rows_for_drop(ann_db, drop_id: str) -> pd.DataFrame:
    """Legacy expert rows with a species and a usable second, as a frame."""
    rows = ann_db.get_annotations_for_drop(drop_id, annotated_by="expert")
    df = pd.DataFrame(rows)
    if df.empty or "external_id" not in df.columns:
        return pd.DataFrame(
            columns=["scientific_name", "max_interval", "time_of_max_seconds"]
        )
    df = df[df["external_id"] == LEGACY_EXTERNAL_ID]
    df = df[df["scientific_name"].notna() & (df["scientific_name"] != NULL_DEPLOYMENT)]
    df = df[df["time_of_max_seconds"].notna()]
    return df[["scientific_name", "max_interval", "time_of_max_seconds"]].reset_index(
        drop=True
    )


def legacy_review_moments(rows: pd.DataFrame, n_random: int, seed: str) -> pd.DataFrame:
    """The moments worth a frame: each species' peak plus ``n_random`` others.

    Pure: takes the legacy rows (``scientific_name``, ``max_interval``,
    ``time_of_max_seconds``) and returns one row per moment with a ``kind``
    column, ``legacy_maxn`` for the peak and ``legacy_interval`` for the rest.
    Ties at the peak go to the EARLIEST interval, which is what the workbooks'
    MaxCount sheet records (checked on 98 of 99 tied cases, Tapuae 2022/2024).
    Random picks are seeded by ``seed`` (the DropID) so a re-run returns the
    same frames and two models are compared on identical moments.
    """
    cols = ["scientific_name", "max_interval", "time_of_max_seconds", "kind"]
    if rows.empty:
        return pd.DataFrame(columns=cols)
    rows = rows.sort_values(
        ["scientific_name", "max_interval", "time_of_max_seconds"],
        ascending=[True, False, True],
    )
    peaks = rows.groupby("scientific_name", sort=False).head(1).assign(kind=KIND_PEAK)
    out = [peaks]
    if n_random > 0:
        rest = rows.drop(peaks.index)
        picks = []
        for species, group in rest.groupby("scientific_name", sort=False):
            # Per-species seed, so adding a species does not reshuffle the others.
            digest = hashlib.sha256(f"{seed}|{species}".encode()).hexdigest()
            state = int(digest[:8], 16)
            k = min(n_random, len(group))
            if k:
                picks.append(group.sample(n=k, random_state=state))
        if picks:
            out.append(pd.concat(picks).assign(kind=KIND_INTERVAL))
    moments = pd.concat(out, ignore_index=True)[cols]
    return moments.sort_values(
        ["scientific_name", "kind", "time_of_max_seconds"]
    ).reset_index(drop=True)


def _draw_qa_copy(
    clean_path: Path,
    out_path: Path,
    boxes: pd.DataFrame,
    base_conf: float,
    banner: str,
    label: Labeller = str,
) -> None:
    img = cv2.imread(str(clean_path))
    if img is None:
        raise RuntimeError(f"could not read {clean_path}")
    draw_boxes(img, boxes, base_conf, label)
    draw_banner(img, banner)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)


def write_legacy_review_frames(
    drop_id: str,
    video_path: Path,
    raw_df: pd.DataFrame,
    model_name: str,
    ann_db,
    base_conf: float,
) -> list[Path]:
    """Write clean + QA frames for every legacy review moment of one drop.

    Returns the QA frame paths written. A drop with no legacy rows returns an
    empty list without opening the video.
    """
    rows = legacy_rows_for_drop(ann_db, drop_id)
    if rows.empty:
        return []
    moments = legacy_review_moments(
        rows, config.legacy_review_random_intervals, seed=drop_id
    )
    if not video_path.exists():
        logging.warning(
            f"{drop_id}: video not at {video_path}, skipping legacy review frames."
        )
        return []

    frames_dir = config.get_frames_dir(drop_id)
    qa_dir = config.get_drop_dir(drop_id) / "qa_frames" / "legacy" / model_name
    frames_dir.mkdir(parents=True, exist_ok=True)
    max_delta = config.legacy_review_max_time_delta_seconds

    label = common_name_labeller()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"{drop_id}: cv2 could not open {video_path}")
    rotation = read_video_rotation(cap)
    written: list[Path] = []
    try:
        for m in moments.itertuples(index=False):
            legacy_s = float(m.time_of_max_seconds)
            # The raw frame rather than a seek by time, so the boxes drawn on
            # the QA copy were computed on exactly these pixels.
            frame_index, frame_s = nearest_raw_frame(raw_df, legacy_s, max_delta)
            at = frame_s if frame_s is not None else legacy_s
            clean_path = frames_dir / generate_frame_filename(drop_id, at)
            if not clean_path.exists():
                ok = extract_one_frame_from_cap(
                    cap, at, clean_path, frame_index=frame_index, rotation=rotation
                )
                if not ok:
                    logging.warning(f"{drop_id}: no frame at {at:.3f}s, skipped.")
                    continue
            boxes = (
                raw_df[raw_df["frame"] == frame_index]
                if frame_index is not None
                else raw_df.iloc[0:0]
            )
            model_txt = count_summary(boxes, base_conf, label)
            # Common names for the reader, here as everywhere a person looks;
            # the file name below keeps the scientific name as the stable key.
            banner = (
                f"LEGACY {label(m.scientific_name)} n={int(m.max_interval)} "
                f"@ {seconds_to_time(legacy_s)} [{m.kind}]  |  {model_name}: {model_txt}"
            )
            # The clean frame keeps the canonical by-second name (one frame can
            # be the peak for two species); the QA copy says which species and
            # count it is evidence for, so a folder listing reads on its own.
            species_slug = str(m.scientific_name).replace(" ", "_")
            qa_path = qa_dir / (
                f"{clean_path.stem}__{species_slug}__n{int(m.max_interval)}"
                f"__{m.kind}.jpg"
            )
            _draw_qa_copy(clean_path, qa_path, boxes, base_conf, banner, label)
            written.append(qa_path)
    finally:
        cap.release()
    logging.info(
        f"{drop_id}: {len(written)} legacy review frames "
        f"({(moments['kind'] == KIND_PEAK).sum()} peaks) -> {qa_dir}"
    )
    return written
