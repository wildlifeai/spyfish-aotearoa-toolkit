"""
Strategy-based frame selection from a MaxN CSV.

Applies the binary/multiclass export strategy with an optional multiplier
to scale counts up and tighten temporal spacing for denser frame coverage.
"""

import logging
from pathlib import Path

import pandas as pd

from spyfish.config.wrapper import config
from spyfish.database.manager import DatabaseManager
from spyfish.extraction.select_clips import select_clips_with_strategy


def select_frames(
    maxn_csv_path: str,
    output_selections_path: str,
    drop_id: str,
) -> pd.DataFrame:
    """
    Select frames from a MaxN CSV using the export strategy.

    Args:
        maxn_csv_path: Path to the MaxN CSV produced by post-ML processing.
        output_selections_path: Path to write the selections CSV.
        drop_id: Deployment identifier.

    Returns:
        DataFrame of selected frame moments.
    """
    multiplier = config.frame_multiplier
    if not Path(maxn_csv_path).exists():
        raise FileNotFoundError(f"MaxN CSV not found: {maxn_csv_path}")

    maxn_df = pd.read_csv(maxn_csv_path)
    if maxn_df.empty:
        raise ValueError(
            f"Empty MaxN CSV for {drop_id} — no detections available for frame selection."
        )

    # Add TimeSeconds column expected by select_clips_with_strategy (convert ms → seconds)
    maxn_df[config.csv_time_seconds_column] = (
        maxn_df[config.csv_maxn_time_ms_column] / 1000.0
    )

    # Get sampling_start from DB
    deployment = DatabaseManager().get_deployment(drop_id)
    if not deployment or deployment.get("sampling_start") is None:
        raise ValueError(f"Missing sampling_start for {drop_id}, cannot select frames.")
    sampling_start = float(deployment["sampling_start"])

    if multiplier <= 0:
        logging.error(
            f"multiplier is {multiplier} — must be positive. Defaulting to 1."
        )
    safe_multiplier = multiplier if multiplier > 0 else 1

    unique_species = maxn_df[config.csv_scientific_name_column].unique()
    is_binary = len(unique_species) <= 1 or config.force_binary_strategy

    if is_binary:
        base = config.binary_strategy
        scaled_strategy = {
            "maxn_export": round(base["maxn_export"] * safe_multiplier),
            "confusing_export": round(base["confusing_export"] * safe_multiplier),
            "empty_export": round(base["empty_export"] * safe_multiplier),
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
            "per_video_empty_export": round(
                base["per_video_empty_export"] * safe_multiplier
            ),
            "per_video_start_export": round(
                base["per_video_start_export"] * safe_multiplier
            ),
            "temporal_spacing_seconds": base["temporal_spacing_seconds"]
            / safe_multiplier,
        }

    selections_df = select_clips_with_strategy(
        df=maxn_df,
        drop_id=drop_id,
        sampling_start=sampling_start,
        clip_length=config.clip_length,
        strategy_params=scaled_strategy,
        is_multiclass=not is_binary,
        video_start_threshold=config.video_start_threshold,
        clip_cap=round(config.clip_cap * safe_multiplier),
    )

    Path(output_selections_path).parent.mkdir(parents=True, exist_ok=True)
    selections_df.to_csv(output_selections_path, index=False)
    logging.info(
        f"{len(selections_df)} frame selections for {drop_id} "
        f"(multiplier={multiplier}, {'binary' if is_binary else 'multiclass'} strategy)."
    )
    return selections_df
