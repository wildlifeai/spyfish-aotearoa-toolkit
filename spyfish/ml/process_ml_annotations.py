"""
Post-ML annotation processing: MaxN extraction and optional frame drawing for QA review.
This module is the single entry point called by the pipeline runner after YOLO inference.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from spyfish.config.wrapper import config
from spyfish.database.annotation_manager import (
    AnnotationDatabaseManager,
    null_deployment_row,
)
from spyfish.database.manager import DatabaseManager
from spyfish.ml.draw_frames import draw_boxes_on_video_frames
from spyfish.utils import seconds_to_time
from spyfish.visualisations.maxn_visualisation import plot_maxn_timeline


def _build_sampled_grid(raw_df: pd.DataFrame):
    """Rebuild the inference sampling grid from a raw CSV's frame indices.

    The raw CSV only holds frames WITH detections, but the persistence filter
    needs the frames without them (they are the zeros that expose a blip).
    The sampling stride is recovered as the modal difference between
    consecutive detected frame indices — robust to gaps, unlike a gcd, because
    a majority of consecutive detections sit one stride apart on real drops.

    Returns:
        grid_times: absolute seconds per grid slot (synthetic slots interpolated).
        grid_frame_ids: video frame index per slot, -1 for synthetic (no-detection) slots.
        frame_to_idx: {video frame index -> grid slot} for the detected frames.
        dt_grid: seconds between grid slots, or None when the CSV holds a single
            frame and no spacing can be estimated.
    """
    per_frame_times = raw_df.groupby("frame")["time_seconds"].first().sort_index()
    frames = per_frame_times.index.to_numpy(dtype=np.int64)
    times = per_frame_times.to_numpy(dtype=float)

    if len(frames) < 2:
        return times, frames.copy(), {int(frames[0]): 0}, None

    diffs = np.diff(frames)
    stride = int(pd.Series(diffs).mode().iloc[0])
    dt_grid = float(np.median(np.diff(times) / diffs)) * stride

    grid_times: list = []
    grid_frame_ids: list = []
    frame_to_idx: dict = {}
    for k in range(len(frames)):
        frame_to_idx[int(frames[k])] = len(grid_times)
        grid_times.append(times[k])
        grid_frame_ids.append(int(frames[k]))
        if k + 1 < len(frames):
            n_slots = max(1, int(round(diffs[k] / stride)))
            for s in range(1, n_slots):
                grid_times.append(times[k] + (times[k + 1] - times[k]) * s / n_slots)
                grid_frame_ids.append(-1)
    return (
        np.asarray(grid_times),
        np.asarray(grid_frame_ids),
        frame_to_idx,
        dt_grid,
    )


def _gap_fill(counts: np.ndarray, max_gap: int) -> np.ndarray:
    """Morphological closing on a count series: zero-runs up to `max_gap` slots
    flanked by detections on BOTH sides take min(neighbours). Never invents a
    detection — an isolated blip has zeros on both sides and stays exposed."""
    if max_gap <= 0:
        return counts
    filled = counts.copy()
    n = len(filled)
    i = 0
    while i < n:
        if filled[i] != 0:
            i += 1
            continue
        j = i
        while j < n and filled[j] == 0:
            j += 1
        if 0 < i and j < n and (j - i) <= max_gap:
            filled[i:j] = min(filled[i - 1], filled[j])
        i = j
    return filled


def _centered_rolling_min(counts: np.ndarray, window: int) -> np.ndarray:
    """Rolling min over `window` slots, value assigned to the window's middle
    slot. Slots beyond the series edges count as zero, so a visit shorter than
    the window is suppressed even when it touches the first or last detection."""
    if window <= 1:
        return counts
    left = window // 2
    right = window - 1 - left
    padded = np.concatenate([np.zeros(left), counts, np.zeros(right)])
    return np.lib.stride_tricks.sliding_window_view(padded, window).min(axis=1)


def process_maxn(
    raw_df,
    output_csv_path,
    drop_id,
    interval_seconds,
    confidence_threshold,
    model_name,
    persistence_seconds: float = 0.0,
    gap_fill_seconds: float = 0.0,
    exclude_classes: tuple = (),
):
    """
    Processes raw ML detections into MaxN intervals, with persistence filtering.

    A count only sets MaxN if it is sustained across a rolling window of
    consecutive sampled frames (`persistence_seconds`, converted to frames from
    the CSV's own spacing), computed on the FULL timeline before interval
    binning so a visit straddling an interval boundary is not undercounted.
    Single zero frames between detections are gap-filled first
    (`gap_fill_seconds`) so detector flicker on a present animal is forgiven.

    A suppressed spike does not disappear: its row keeps the persistent value
    in MaxInterval (0 allowed) and records the unfiltered single-frame max in
    RawMaxInterval with SpikeFlag/SpikeTimeSeconds, so review selection can
    still surface the model's own confident mistakes while reporting ignores
    them. Rows with MaxInterval 0 are kept in the CSV but not ingested to the
    annotations DB.

    With the defaults (persistence 0, no gap fill, no exclusions) the output
    matches the pre-filter behaviour exactly: single-frame max per
    (interval × class), ties broken by mean confidence.

    Args:
        raw_df: DataFrame of raw ML detections (avoids double CSV read).
        output_csv_path: Path to save MaxN CSV results.
        drop_id: Deployment identifier.
        interval_seconds: Width of each MaxN time window.
        confidence_threshold: Minimum confidence for counting fish.
        model_name: Name of the model used for annotation.
        persistence_seconds: Rolling-min window; 0 disables (single-frame MaxN).
        gap_fill_seconds: Max zero-gap between detections to close; 0 disables.
        exclude_classes: Classes that never count toward MaxN and get no row.

    Returns:
        DataFrame with MaxN results.
    """
    logging.debug(
        f"Processing MaxN for {drop_id} (Interval: {interval_seconds}s, MaxN Threshold: {confidence_threshold})"
    )

    if raw_df.empty:
        logging.warning(f"Empty CSV for {drop_id}")
        return pd.DataFrame()

    df_filtered = raw_df[raw_df["confidence"] >= confidence_threshold]
    if exclude_classes:
        df_filtered = df_filtered[~df_filtered["class"].isin(set(exclude_classes))]

    if df_filtered.empty:
        logging.warning(
            f"No detections above threshold {confidence_threshold} for {drop_id}"
        )
        return pd.DataFrame()

    # Grid from the FULL raw CSV (all classes, base confidence threshold):
    # denser data gives a better stride estimate, and a frame whose only
    # detections are sub-threshold is still a genuinely sampled frame.
    grid_times, grid_frame_ids, frame_to_idx, dt_grid = _build_sampled_grid(raw_df)
    if dt_grid and dt_grid > 0:
        window = (
            max(1, round(persistence_seconds / dt_grid))
            if persistence_seconds > 0
            else 1
        )
        max_gap = round(gap_fill_seconds / dt_grid) if gap_fill_seconds > 0 else 0
    else:
        # Single sampled frame in the whole CSV: nothing can be sustained.
        window = 2 if persistence_seconds > 0 else 1
        max_gap = 0
    grid_intervals = (grid_times // interval_seconds) * interval_seconds

    def _series_rows(sub_df: pd.DataFrame, class_name: str) -> list:
        """MaxN rows for one class's count series."""
        per_frame = sub_df.groupby("frame").agg(
            count=("confidence", "size"),
            mean_conf=("confidence", "mean"),
            time=("time_seconds", "first"),
        )
        counts = np.zeros(len(grid_times))
        for f, prow in per_frame.iterrows():
            counts[frame_to_idx[int(f)]] = prow["count"]
        rollmin = _centered_rolling_min(_gap_fill(counts, max_gap), window)

        def _best_frame(candidate_frames: list) -> tuple:
            """(time, mean_conf) of the candidate with the highest mean
            confidence; earliest frame wins ties (matches the historical
            tiebreak order)."""
            best_conf, best_time = -1.0, 0.0
            for f in sorted(candidate_frames):
                prow = per_frame.loc[f]
                if prow["mean_conf"] > best_conf:
                    best_conf = float(prow["mean_conf"])
                    best_time = float(prow["time"])
            return best_time, best_conf

        rows = []
        per_frame_iv = (per_frame["time"] // interval_seconds) * interval_seconds
        for interval in sorted(per_frame_iv.unique()):
            in_iv = per_frame[per_frame_iv == interval]
            raw_max = int(in_iv["count"].max())
            raw_time, raw_conf = _best_frame(
                in_iv.index[in_iv["count"] == raw_max].tolist()
            )

            iv_mask = grid_intervals == interval
            persistent = int(rollmin[iv_mask].max())

            if persistent > 0:
                # Prefer a real detected frame whose centered window sustains
                # the persistent value; gap-filled synthetic slots can carry it
                # too, but have no boxes to point an expert at.
                cand = [
                    int(grid_frame_ids[i])
                    for i in np.nonzero(iv_mask & (rollmin == persistent))[0]
                    if grid_frame_ids[i] >= 0
                    and int(grid_frame_ids[i]) in per_frame.index
                ]
                best_time, best_conf = (
                    _best_frame(cand) if cand else (raw_time, raw_conf)
                )
            else:
                # Fully suppressed: the raw peak is the only meaningful moment.
                best_time, best_conf = raw_time, raw_conf

            spike = raw_max > persistent
            rows.append(
                {
                    config.drop_id_column: drop_id,
                    config.csv_scientific_name_column: class_name,
                    config.csv_maxn_time_column: seconds_to_time(best_time),
                    config.csv_max_interval_column: persistent,
                    config.csv_annotated_by_column: model_name,
                    config.csv_interval_annotation_column: interval_seconds,
                    config.csv_confidence_agreement_column: round(best_conf, 4),
                    config.csv_maxn_time_seconds_column: best_time,
                    config.csv_raw_max_interval_column: raw_max,
                    config.csv_spike_flag_column: spike,
                    config.csv_spike_time_seconds_column: raw_time if spike else np.nan,
                }
            )
        return rows

    maxn_results = []
    for cls in df_filtered["class"].unique():
        maxn_results.extend(_series_rows(df_filtered[df_filtered["class"] == cls], cls))

    maxn_df = pd.DataFrame(maxn_results)

    if not maxn_df.empty:
        maxn_df = maxn_df.sort_values(
            [
                config.csv_maxn_time_seconds_column,
                config.csv_scientific_name_column,
            ]
        )

    n_spikes = (
        int(maxn_df[config.csv_spike_flag_column].sum()) if not maxn_df.empty else 0
    )
    Path(output_csv_path).parent.mkdir(parents=True, exist_ok=True)
    maxn_df.to_csv(output_csv_path, index=False)
    logging.info(
        f" Saved MaxN {len(maxn_df)} rows for {drop_id} to {output_csv_path}"
        + (f" ({n_spikes} persistence-reduced)" if n_spikes else "")
    )

    return maxn_df


def _ingest_ml_annotations(
    ann_db: AnnotationDatabaseManager,
    drop_id: str,
    maxn_df: pd.DataFrame,
    model_name: str,
):
    """Extracts ingestion logic into a helper method."""
    # Persistence-suppressed rows (MaxInterval 0) stay CSV-only: they exist so
    # review selection can surface the spike, but "reviewed, nothing sustained"
    # must not become a named species observation in the annotations DB. A
    # spike-only deployment therefore falls through to the null-deployment row
    # below — a real zero, which is correct.
    if not maxn_df.empty:
        maxn_df = maxn_df[maxn_df[config.csv_max_interval_column] > 0]
    annotations_to_add = []
    for _, row in maxn_df.iterrows():
        annotations_to_add.append(
            {
                "drop_id": drop_id,
                "scientific_name": row[config.csv_scientific_name_column],
                "time_of_max": row[config.csv_maxn_time_column],
                "time_of_max_seconds": row.get(config.csv_maxn_time_seconds_column),
                "max_interval": row[config.csv_max_interval_column],
                "annotated_by": "ml",
                "interval_annotation": "",
                "confidence_agreement": row[config.csv_confidence_agreement_column],
                "external_id": model_name,
            }
        )
    # Clear only this model's prior rows before re-writing. Scoping by
    # external_id (the model name) means re-running one model leaves any
    # other model's outputs intact in the DB, supports running both
    # binary and species pipelines on the same drop and comparing them
    # side-by-side via the dashboard's Provenance column.
    # The clear must still run when annotations_to_add is empty so a zero-
    # detection re-run wipes stale rows from a prior run of THIS model.
    ann_db.clear_annotations(drop_id, "ml", external_id=model_name)
    if not annotations_to_add:
        # Zero detections is still a result. The null-deployment row records
        # "reviewed, nothing seen" — the same convention the other sources use
        # — so downstream consumers can tell it apart from "ML never ran" and
        # detection-rate denominators include this deployment as a real zero.
        annotations_to_add.append(
            null_deployment_row(drop_id, "ml", external_id=model_name)
        )
    ann_db.add_annotations(annotations_to_add)
    logging.debug(
        f"Ingested {len(annotations_to_add)} ML annotations into detailed database for {drop_id}"
    )


def _run_qa_visualizations(
    raw_df: pd.DataFrame,
    maxn_df: pd.DataFrame,
    drop_id: str,
    video_dir: Path,
    base_conf: float,
    maxn_conf: float,
    interval: float,
    raw_csv_path: Optional[str] = None,
):
    """Draw MaxN timeline plot and lowest-confidence annotated frames for human QA review."""
    # Save MaxN timeline visualisation
    plot_maxn_timeline(
        raw_df=raw_df,
        maxn_df=maxn_df,
        drop_id=drop_id,
        output_dir=config.get_drop_dir(drop_id),
        base_conf=base_conf,
        maxn_conf=maxn_conf,
        interval_seconds=interval,  # type: ignore
    )

    # Draw lowest-confidence frames for QA review
    if raw_df.empty or maxn_df.empty:
        logging.debug(f"Skipping QA frame drawing for {drop_id}: no raw detections.")
        return

    top_maxn = maxn_df.nlargest(4, config.csv_max_interval_column)
    low_conf = maxn_df.nsmallest(4, config.csv_confidence_agreement_column)
    review_df = pd.concat([top_maxn, low_conf]).drop_duplicates()

    # Map TimeOfMaxAbsSeconds back to raw CSV frame numbers
    frame_indices = []
    for t_sec in review_df[config.csv_maxn_time_seconds_column]:
        closest = raw_df.iloc[(raw_df["time_seconds"] - t_sec).abs().argsort()[:1]]
        if closest.empty:
            continue
        frame_indices.append(int(closest["frame"].iloc[0]))

    # 4 evenly-spaced random frames across the detected range for general coverage
    t_min, t_max = raw_df["time_seconds"].min(), raw_df["time_seconds"].max()
    if t_max > t_min:
        boundaries = np.linspace(t_min, t_max, 5)  # 4 equal bands
        for i in range(4):
            band = raw_df[
                (raw_df["time_seconds"] >= boundaries[i])
                & (raw_df["time_seconds"] < boundaries[i + 1])
            ]
            if not band.empty:
                frame_indices.append(int(band.sample(1)["frame"].iloc[0]))
    # First and last detected frames, quick visual check on detection coverage
    frame_indices.append(int(raw_df.loc[raw_df["time_seconds"].idxmin(), "frame"]))
    frame_indices.append(int(raw_df.loc[raw_df["time_seconds"].idxmax(), "frame"]))

    # Find video file
    video_path = video_dir / f"{drop_id}.mp4"
    if not video_path.exists():
        logging.warning(f"Video not found at {video_path}, skipping QA frame drawing")
        return

    frames_dir = str(config.get_drop_dir(drop_id) / "qa_frames")
    draw_boxes_on_video_frames(
        video_path=video_path,
        raw_csv_path=raw_csv_path,
        output_dir=frames_dir,
        frame_list=frame_indices,
        confidence_threshold=base_conf,
        drop_id=drop_id,
    )


def process_one_drop(
    drop_id: str,
    video_dir: Path,
    ann_db: AnnotationDatabaseManager,
    model_name: str,
    interval: float,
    base_conf: float,
    maxn_conf: float,
    draw_images: bool = True,
) -> None:
    """Post-inference processing for a single drop: MaxN + annotation ingest + QA viz.

    QA viz failures are swallowed and logged, they're diagnostic only and must
    not block the caller from marking the drop complete. MaxN extraction and
    annotation ingest failures propagate so the caller can mark the drop as
    errored.
    """
    drop_annotations_dir = config.get_drop_annotations_dir(drop_id)
    drop_annotations_dir.mkdir(parents=True, exist_ok=True)
    raw_csv = str(drop_annotations_dir / f"{drop_id}_{model_name}_raw.csv")
    maxn_csv = str(config.get_maxn_csv_path(drop_id, model_name))

    if not Path(raw_csv).exists():
        raise FileNotFoundError(f"Raw CSV not found for {drop_id} at {raw_csv}")

    raw_df = pd.read_csv(raw_csv)

    # Circuit-breaker: refuse degenerate inference before it propagates downstream.
    # A healthy BUV frame has a handful of detections; hundreds/frame means the model
    # is saturating its max_det cap (out-of-distribution footage, or a mis-set
    # confidence_threshold). Fail loudly here so the drop lands in ml_error rather
    # than building a multi-million-box COCO that later crashes the BIIGLE upload.
    if not raw_df.empty:
        n_frames = max(1, raw_df["time_seconds"].nunique())
        boxes_per_frame = len(raw_df) / n_frames
        if boxes_per_frame > config.ml_max_boxes_per_frame:
            raise ValueError(
                f"{drop_id}: degenerate ML inference, {boxes_per_frame:.0f} "
                f"detections/frame across {n_frames} frames ({len(raw_df):,} total), "
                f"exceeding max_boxes_per_frame={config.ml_max_boxes_per_frame}. The "
                "model is saturating its max_det cap; check confidence_threshold "
                "(0 disables filtering) and whether the footage is out-of-distribution "
                "(turbid/low-visibility). Review the video or exclude this drop."
            )

    maxn_df = process_maxn(
        raw_df,
        maxn_csv,
        drop_id,
        interval_seconds=interval,
        confidence_threshold=maxn_conf,
        model_name=model_name,
        persistence_seconds=config.maxn_persistence_seconds,
        gap_fill_seconds=config.maxn_gap_fill_seconds,
        exclude_classes=tuple(config.maxn_exclude_classes),
    )

    if maxn_df.empty:
        logging.warning(
            f"No MaxN results for {drop_id}. Saving empty CSV for health checks."
        )
        maxn_df = pd.DataFrame(
            columns=[
                config.drop_id_column,
                config.csv_scientific_name_column,
                config.csv_maxn_time_column,
                config.csv_max_interval_column,
                config.csv_annotated_by_column,
                config.csv_interval_annotation_column,
                config.csv_confidence_agreement_column,
                config.csv_maxn_time_seconds_column,
                config.csv_raw_max_interval_column,
                config.csv_spike_flag_column,
                config.csv_spike_time_seconds_column,
            ]
        )
        maxn_df.to_csv(maxn_csv, index=False)

    _ingest_ml_annotations(ann_db, drop_id, maxn_df, model_name)

    if draw_images:
        try:
            _run_qa_visualizations(
                raw_df=raw_df,
                maxn_df=maxn_df,
                drop_id=drop_id,
                video_dir=video_dir,
                base_conf=base_conf,
                maxn_conf=maxn_conf,
                interval=interval,
                raw_csv_path=raw_csv,
            )
        except Exception as e:
            logging.error(
                f"QA visualisation failed for {drop_id} (non-fatal): {e}",
                exc_info=True,
            )


def run_post_ml(
    drop_ids: list,
    video_dir: str,
    draw_images: bool = True,
):
    """
    Batch entry point, kept for REPL/notebook use. The pipeline now runs the
    per-drop work inline inside MLRunner.run_inference_loop so that artifacts
    are written before the drop is marked complete.

    For each drop: extracts MaxN, ingests annotations, optionally draws QA frames.
    Then syncs annotation counts to the main pipeline DB once at the end.
    """
    model_name = Path(config.pipeline_model_path).stem
    db = DatabaseManager()
    ann_db = AnnotationDatabaseManager()
    interval = config.interval_seconds
    base_conf = config.confidence_threshold
    maxn_conf = config.maxn_confidence_threshold

    for drop_id in drop_ids:
        logging.debug(f"Post-ML processing: {drop_id}")
        try:
            process_one_drop(
                drop_id=drop_id,
                video_dir=Path(video_dir),
                ann_db=ann_db,
                model_name=model_name,
                interval=interval,
                base_conf=base_conf,
                maxn_conf=maxn_conf,
                draw_images=draw_images,
            )
            logging.info(f"  → Post-ML processing complete for: {drop_id}")
        except Exception as e:
            logging.error(
                f"Post-ML processing failed for {drop_id}: {e}", exc_info=True
            )

    if drop_ids:
        db.sync_annotation_counts(drop_ids)
        logging.info(
            f"Synchronized annotation counts for {len(drop_ids)} drops to main pipeline DB"
        )
