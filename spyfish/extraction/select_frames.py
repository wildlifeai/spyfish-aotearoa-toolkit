"""
Strategy-based frame selection from a raw ML CSV.

Groups raw detections by frame to compute per-frame count and mean confidence,
then applies the binary/multiclass export strategy with an optional multiplier
to scale counts up and tighten temporal spacing for denser frame coverage.
"""

import logging
from pathlib import Path

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

    selected_times: list[float] = []
    rows: list[dict] = []

    def _spaced(t: float, spacing: float) -> bool:
        return all(abs(t - s) >= spacing for s in selected_times)

    def _add(row, reason: str):
        t = float(row[time_col])
        selected_times.append(t)
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
    multiplier = config.frame_multiplier
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
        raw_df.groupby("frame")
        .agg(
            **{
                config.csv_time_seconds_column: ("time_seconds", "first"),
                config.csv_max_interval_column: ("confidence", "count"),
                config.csv_confidence_agreement_column: ("confidence", "mean"),
                config.csv_scientific_name_column: ("class", "first"),
            }
        )
        .reset_index(drop=True)
    )
    frame_df[config.csv_confidence_agreement_column] = frame_df[
        config.csv_confidence_agreement_column
    ].round(4)

    deployment = DatabaseManager().get_deployment(drop_id)
    if not deployment or deployment.get("sampling_start") is None:
        raise ValueError(f"Missing sampling_start for {drop_id}, cannot select frames.")
    sampling_start = float(deployment["sampling_start"])

    if multiplier <= 0:
        logging.error(
            f"multiplier is {multiplier} — must be positive. Defaulting to 1."
        )
    safe_multiplier = multiplier if multiplier > 0 else 1

    unique_species = frame_df[config.csv_scientific_name_column].unique()
    is_binary = len(unique_species) <= 1 or config.force_binary_strategy

    if is_binary:
        base = config.binary_strategy
        scaled_strategy = {
            "maxn_export": round(base["maxn_export"] * safe_multiplier),
            "confusing_export": round(base["confusing_export"] * safe_multiplier),
            "start_export": round(base["start_export"] * safe_multiplier),
            "temporal_spacing_seconds": base["temporal_spacing_seconds"]
            / safe_multiplier,
        }
    else:
        base = config.multiclass_strategy
        scaled_strategy = {
            "per_species_maxn_export": round(
                base["per_species_maxn_export"] * safe_multiplier
            ),
            "per_species_confusing_export": round(
                base["per_species_confusing_export"] * safe_multiplier
            ),
            "per_video_start_export": round(
                base["per_video_start_export"] * safe_multiplier
            ),
            "temporal_spacing_seconds": base["temporal_spacing_seconds"]
            / safe_multiplier,
        }

    selections_df = _select_frames_with_strategy(
        frame_df=frame_df,
        drop_id=drop_id,
        sampling_start=sampling_start,
        strategy_params=scaled_strategy,
        is_multiclass=not is_binary,
        video_start_threshold=config.video_start_threshold,
        frame_cap=round(config.clip_cap * safe_multiplier),
    )

    Path(output_selections_path).parent.mkdir(parents=True, exist_ok=True)
    selections_df.to_csv(output_selections_path, index=False)
    logging.info(
        f"{len(selections_df)} frame selections for {drop_id} "
        f"(multiplier={multiplier}, {'binary' if is_binary else 'multiclass'} strategy, "
        f"{len(frame_df)} candidate frames from raw CSV)."
    )
    return selections_df
