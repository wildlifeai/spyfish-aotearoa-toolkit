"""
Strategy-based frame selection from a raw ML CSV.

Groups raw detections by frame to compute per-frame count and mean confidence,
then applies the binary/multiclass export strategy with an optional multiplier
to scale counts up and tighten temporal spacing for denser frame coverage.

TODO — frame selection pipeline position is unresolved (in progress):
  The current implementation selects frames directly from the raw ML CSV and
  does NOT use volunteer clip classifications from Zooniverse at all. That
  means the `zooniverse-images` stage (citsci_status: clips_uploaded →
  frames_uploaded) is only ordered after `zooniverse-clips` by convention —
  there is no data dependency on clip classifications.

  Open design question: where should frame selection actually live?
    1. Remove the stage entirely — frames come from the biigle-upload path only.
    2. Move it to run right after `--ml` (no citsci ordering at all).
    3. Keep it after clip classifications but actually consume volunteer clip
       MaxN data to narrow the frame candidates (requires real data dependency).

  Revisit this when the Zooniverse retirement gate (CLIPS_DONE / FRAMES_DONE)
  is wired — at that point the choice between (1), (2), (3) will be easier to
  make. See CitSciStatus in spyfish/config/base.py for the retirement-gate TODO.
"""

import bisect
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from spyfish.config.wrapper import config
from spyfish.database.manager import DatabaseManager
from spyfish.extraction.select_clips import _sample


def _select_frames_with_strategy(
    frame_df: pd.DataFrame,
    drop_id: str,
    sampling_start: float,
    strategy_params: dict,
    is_multiclass: bool,
    video_start_threshold: int,
    frame_cap: int,
) -> pd.DataFrame:
    """
    Select individual frames using temporal spacing only — no clip bucketing.

    Each row in frame_df represents one video frame with its total detection
    count and mean confidence. Selections are deduplicated by
    temporal_spacing_seconds, allowing multiple frames within the same 10s
    clip window.

    Parallel implementation: select_clips.select_clips_with_strategy() applies the
    same MaxN/confusing/start strategy to clip buckets rather than individual frames.
    Key intentional divergences:
      - This function deduplicates by float spacing; clips use 10s bucket keys.
      - No "empty" bucket here — raw CSVs only contain detected frames, so empty
        intervals cannot be sampled at this stage.
      - Cap/priority logic is simpler (no full-video health-check weighting).
    If you change the core strategy logic here, check whether select_clips.py needs
    the same update.
    """
    columns = [
        config.drop_id_column,
        config.csv_sampling_start_column,
        config.csv_clip_max_time_column,
        config.csv_scientific_name_column,
        "SelectionReason",
        config.csv_max_interval_column,
        config.csv_confidence_agreement_column,
    ]

    if frame_df.empty:
        return pd.DataFrame(columns=columns)

    time_col = config.csv_time_seconds_column
    frame_df = frame_df.copy()
    frame_df[config.csv_confidence_agreement_column] = frame_df[
        config.csv_confidence_agreement_column
    ].replace(0, 0.001)
    frame_df[config.csv_confusion_score_column] = (
        frame_df[config.csv_max_interval_column]
        / frame_df[config.csv_confidence_agreement_column]
    )

    # Kept in sorted order so the spacing check is O(log n) via bisect instead
    # of O(n) by scanning every previously-selected time. Frame selection can
    # consider thousands of candidates on dense videos, so the quadratic form
    # was the hot path.
    selected_times_sorted: list[float] = []
    rows: list[dict] = []

    def _spaced(t: float, spacing: float) -> bool:
        """True iff t is at least `spacing` seconds from every selected time.

        Uses bisect to find the insertion point and checks only the two
        immediate neighbors (left and right). Sorted order means no other
        time can be closer than those neighbors.
        """
        idx = bisect.bisect_left(selected_times_sorted, t)
        if idx > 0 and t - selected_times_sorted[idx - 1] < spacing:
            return False
        if (
            idx < len(selected_times_sorted)
            and selected_times_sorted[idx] - t < spacing
        ):
            return False
        return True

    def _add(row, reason: str):
        t = float(row[time_col])
        bisect.insort(selected_times_sorted, t)
        rows.append(
            {
                config.drop_id_column: drop_id,
                config.csv_sampling_start_column: sampling_start,
                config.csv_clip_max_time_column: t,
                config.csv_scientific_name_column: row[
                    config.csv_scientific_name_column
                ],
                "SelectionReason": reason,
                config.csv_max_interval_column: row[config.csv_max_interval_column],
                config.csv_confidence_agreement_column: row[
                    config.csv_confidence_agreement_column
                ],
            }
        )

    if not is_multiclass:
        spacing = strategy_params["temporal_spacing_seconds"]
        n_maxn = strategy_params["maxn_export"]
        n_confusing = strategy_params["confusing_export"]
        n_start = strategy_params["start_export"]

        added_maxn = 0
        for _, row in frame_df.nlargest(
            n_maxn * 3, config.csv_max_interval_column
        ).iterrows():
            if added_maxn >= n_maxn:
                break
            if _spaced(float(row[time_col]), spacing):
                _add(row, "Absolute MaxN")
                added_maxn += 1

        added_confusing = 0
        for _, row in frame_df.sort_values(
            config.csv_confusion_score_column, ascending=False
        ).iterrows():
            if added_confusing >= n_confusing:
                break
            if _spaced(float(row[time_col]), spacing):
                _add(row, "Confusing (High count, low conf)")
                added_confusing += 1

        added_start = 0
        for _, row in _sample(
            frame_df[frame_df[time_col] < video_start_threshold], n_start
        ).iterrows():
            if added_start >= n_start:
                break
            if _spaced(float(row[time_col]), spacing):
                _add(row, "Video Start")
                added_start += 1
    else:
        spacing = strategy_params["temporal_spacing_seconds"]
        n_maxn_per_sp = strategy_params["per_species_maxn_export"]
        n_confusing_per_sp = strategy_params["per_species_confusing_export"]
        n_start = strategy_params["per_video_start_export"]

        for species in frame_df[config.csv_scientific_name_column].unique():
            sp_df = frame_df[frame_df[config.csv_scientific_name_column] == species]

            added_maxn = 0
            for _, row in sp_df.nlargest(
                n_maxn_per_sp * 3, config.csv_max_interval_column
            ).iterrows():
                if added_maxn >= n_maxn_per_sp:
                    break
                if _spaced(float(row[time_col]), spacing):
                    _add(row, f"MaxN ({species})")
                    added_maxn += 1

            added_conf = 0
            for _, row in sp_df.sort_values(
                config.csv_confusion_score_column, ascending=False
            ).iterrows():
                if added_conf >= n_confusing_per_sp:
                    break
                if _spaced(float(row[time_col]), spacing):
                    _add(row, f"Confusing ({species})")
                    added_conf += 1

        added_start = 0
        for _, row in _sample(
            frame_df[frame_df[time_col] < video_start_threshold], n_start
        ).iterrows():
            if added_start >= n_start:
                break
            if _spaced(float(row[time_col]), spacing):
                _add(row, "Video Start")
                added_start += 1

    result = pd.DataFrame(rows, columns=columns)

    if frame_cap and len(result) > frame_cap:
        priority = result[result["SelectionReason"].str.contains("MaxN|Video Start")]
        other = result[~result["SelectionReason"].str.contains("MaxN|Video Start")]
        if len(priority) >= frame_cap:
            result = priority.iloc[:frame_cap]
        else:
            n_needed = min(frame_cap - len(priority), len(other))
            result = pd.concat([priority, other.sample(n_needed, random_state=42)])

    return result.sort_values(config.csv_clip_max_time_column).reset_index(drop=True)


def _run_selection(
    frame_df: pd.DataFrame,
    drop_id: str,
    output_selections_path: str,
    source_label: str,
) -> pd.DataFrame:
    """Apply strategy, write selections CSV, return result.

    Shared by select_frames (ML) and select_frames_from_zooniverse (citsci).
    Callers are responsible for building and validating frame_df; this
    function owns the strategy selection, _select_frames_with_strategy call,
    and file write.
    """
    deployment = DatabaseManager().get_deployment(drop_id)
    if not deployment or deployment.get("sampling_start") is None:
        raise ValueError(f"Missing sampling_start for {drop_id}, cannot select frames.")
    sampling_start = float(deployment["sampling_start"])

    unique_species = frame_df[config.csv_scientific_name_column].unique()
    is_binary = len(unique_species) <= 1 or config.force_binary_strategy

    strategy = config.binary_strategy if is_binary else config.multiclass_strategy

    selections_df = _select_frames_with_strategy(
        frame_df=frame_df,
        drop_id=drop_id,
        sampling_start=sampling_start,
        strategy_params=strategy,
        is_multiclass=not is_binary,
        video_start_threshold=config.video_start_threshold,
        frame_cap=config.clip_cap,
    )

    Path(output_selections_path).parent.mkdir(parents=True, exist_ok=True)
    selections_df.to_csv(output_selections_path, index=False)
    logging.info(
        f"{len(selections_df)} frame selections for {drop_id} "
        f"({'binary' if is_binary else 'multiclass'} strategy, "
        f"{len(frame_df)} candidates from {source_label})."
    )
    return selections_df


def _ml_peak_selections(
    raw_csv_path: str,
    drop_id: str,
    sampling_start: float,
) -> pd.DataFrame:
    """Top-K per-species frame-level peaks from a raw ML CSV.

    Citsci-driven frame selection is locked to integer-second precision
    (Zooniverse only captures volunteer clicks at whole-second resolution),
    so it can miss multi-fish moments that exist for less than a second.
    This helper builds extra selection rows directly from the YOLO raw CSV
    — preserving the inference frame's sub-second timestamp.

    Counts detections at conf >= ``config.ml_peak_min_confidence`` (lower
    than the MaxN counting threshold of 0.5 on purpose), groups by frame
    and species, keeps top-K per species by count with mean confidence as
    a tiebreaker. Returns rows in the same schema as ``_run_selection``'s
    output so the caller can concat them with citsci-derived rows.
    """
    columns = [
        config.drop_id_column,
        config.csv_sampling_start_column,
        config.csv_clip_max_time_column,
        config.csv_scientific_name_column,
        "SelectionReason",
        config.csv_max_interval_column,
        config.csv_confidence_agreement_column,
    ]
    if not Path(raw_csv_path).exists():
        return pd.DataFrame(columns=columns)

    raw = pd.read_csv(raw_csv_path)
    min_conf = config.ml_peak_min_confidence
    raw = raw[raw["confidence"] >= min_conf]
    if raw.empty:
        return pd.DataFrame(columns=columns)

    per_frame_species = (
        raw.groupby(["time_seconds", "class"])
        .agg(count=("confidence", "count"), mean_conf=("confidence", "mean"))
        .reset_index()
    )
    top_k = (
        per_frame_species.sort_values(["count", "mean_conf"], ascending=[False, False])
        .groupby("class")
        .head(config.ml_peak_top_k_per_species)
    )

    rows = []
    for _, r in top_k.iterrows():
        rows.append(
            {
                config.drop_id_column: drop_id,
                config.csv_sampling_start_column: sampling_start,
                config.csv_clip_max_time_column: float(r["time_seconds"]),
                config.csv_scientific_name_column: str(r["class"]),
                "SelectionReason": (
                    f"ML peak (conf>={min_conf}, count={int(r['count'])})"
                ),
                config.csv_max_interval_column: int(r["count"]),
                config.csv_confidence_agreement_column: round(float(r["mean_conf"]), 4),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def select_frames_from_zooniverse(
    maxn_csv_path: str,
    output_selections_path: str,
    drop_id: str,
    ml_raw_csv_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Select frames for BIIGLE upload from a Zooniverse volunteer MaxN CSV.

    One row per (clip, species) already — no grouping needed. MaxInterval is
    the volunteer mode count; ConfidenceAgreement is agreement_pct / 100.
    The confusion score (high count / low agreement) surfaces suspicious clips
    without needing the separate suspicious_minority_find flag.

    When ``ml_raw_csv_path`` is provided and points to an existing raw YOLO
    CSV, the selection is augmented with top-K per-species ML peak frames
    (see ``_ml_peak_selections``). Volunteer-clicked timestamps are integer
    seconds, but a transient multi-fish moment can live in a sub-second
    window — ML peaks surface those frames the volunteers couldn't pin.
    ML peaks within ``config.ml_peak_citsci_dedupe_tolerance_seconds`` of
    any citsci-selected frame are dropped to avoid near-duplicate uploads.
    """
    if not Path(maxn_csv_path).exists():
        raise FileNotFoundError(f"Zooniverse MaxN CSV not found: {maxn_csv_path}")

    frame_df = pd.read_csv(maxn_csv_path)
    if frame_df.empty:
        raise ValueError(
            f"Empty Zooniverse MaxN CSV for {drop_id} — no volunteer consensus to select from."
        )

    # _run_selection expects config.csv_time_seconds_column ("TimeAbsoluteSeconds");
    # the MaxN CSV uses config.csv_maxn_time_seconds_column ("TimeOfMaxAbsoluteSeconds").
    frame_df = frame_df.rename(
        columns={config.csv_maxn_time_seconds_column: config.csv_time_seconds_column}
    )

    citsci_selections = _run_selection(
        frame_df, drop_id, output_selections_path, "Zooniverse MaxN CSV"
    )

    if not ml_raw_csv_path or not Path(ml_raw_csv_path).exists():
        return citsci_selections

    sampling_start = float(DatabaseManager().get_deployment(drop_id)["sampling_start"])
    ml_rows = _ml_peak_selections(ml_raw_csv_path, drop_id, sampling_start)
    if ml_rows.empty:
        return citsci_selections

    tolerance = config.ml_peak_citsci_dedupe_tolerance_seconds
    citsci_times = citsci_selections[config.csv_clip_max_time_column].to_numpy()
    if citsci_times.size:
        keep_mask = ml_rows[config.csv_clip_max_time_column].apply(
            lambda t: bool(abs(citsci_times - t).min() > tolerance)
        )
        ml_extra = ml_rows[keep_mask]
    else:
        ml_extra = ml_rows

    if ml_extra.empty:
        logging.info(
            f"{drop_id}: ML peak augmentation found {len(ml_rows)} candidate(s), "
            f"all within {tolerance}s of a citsci selection — no extra frames added."
        )
        return citsci_selections

    combined = pd.concat([citsci_selections, ml_extra], ignore_index=True)
    cap = config.clip_cap
    if cap and len(combined) > cap:
        # Cap by trimming the tail (ML-peak rows are appended last, so citsci
        # selections are preserved first). Keeps volunteer intent the priority.
        combined = combined.iloc[:cap]
    combined = combined.sort_values(config.csv_clip_max_time_column).reset_index(
        drop=True
    )
    Path(output_selections_path).parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_selections_path, index=False)
    logging.info(
        f"{drop_id}: augmented selection — {len(citsci_selections)} citsci "
        f"+ {len(ml_extra)} ML peak frame(s) (capped at {cap}) → "
        f"{len(combined)} total."
    )
    return combined


def select_frames(
    raw_csv_path: str,
    output_selections_path: str,
    drop_id: str,
) -> pd.DataFrame:
    """
    Select frames from the raw ML CSV using the export strategy.

    Groups raw detections by frame to get per-frame count and mean confidence,
    then runs the strategy to pick MaxN and confusing frames. The empty bucket
    is not used — raw CSVs only contain detected frames.

    Args:
        raw_csv_path: Path to the raw YOLO CSV produced by ML inference.
        output_selections_path: Path to write the selections CSV.
        drop_id: Deployment identifier.

    Returns:
        DataFrame of selected frame moments.
    """
    if not Path(raw_csv_path).exists():
        raise FileNotFoundError(f"Raw CSV not found: {raw_csv_path}")

    raw_df = pd.read_csv(raw_csv_path)
    if raw_df.empty:
        raise ValueError(
            f"Empty raw CSV for {drop_id} — no detections available for frame selection."
        )

    # Filter by confidence threshold, then group by frame.
    # Each row in the raw CSV is one bounding box; grouping gives per-frame
    # count (MaxInterval) and mean confidence (ConfidenceAgreement).
    raw_df = raw_df[raw_df["confidence"] >= config.confidence_threshold]
    if raw_df.empty:
        raise ValueError(f"No detections above confidence threshold for {drop_id}.")

    frame_df = (
        raw_df.groupby(["frame", "class"])
        .agg(
            **{
                config.csv_time_seconds_column: ("time_seconds", "first"),
                config.csv_max_interval_column: ("confidence", "count"),
                config.csv_confidence_agreement_column: ("confidence", "mean"),
            }
        )
        .reset_index()
        .rename(columns={"class": config.csv_scientific_name_column})
        .drop(columns=["frame"])
        .reset_index(drop=True)
    )
    frame_df[config.csv_confidence_agreement_column] = frame_df[
        config.csv_confidence_agreement_column
    ].round(4)

    return _run_selection(frame_df, drop_id, output_selections_path, "ML raw CSV")
