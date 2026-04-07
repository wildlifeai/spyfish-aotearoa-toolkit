"""
Select frames for Biigle upload using the same strategy as Zooniverse clip selection,
scaled by biigle_multiplier for denser expert review coverage.
"""

import logging
from pathlib import Path

import pandas as pd

from spyfish.config.wrapper import config
from spyfish.database.manager import DatabaseManager
from spyfish.extraction.select_clips import select_clips_with_strategy


def select_frames_for_biigle(
    maxn_csv_path: str,
    output_selections_path: str,
    drop_id: str,
) -> pd.DataFrame:
    """
    Select frames from the MaxN CSV for Biigle upload using a scaled export strategy.

    Applies the same binary/multiclass strategy as Zooniverse clip selection, but
    with counts multiplied and temporal spacing divided by config.biigle_multiplier
    for denser expert review coverage.

    Args:
        maxn_csv_path: Path to the MaxN CSV produced by post-ML processing.
        output_selections_path: Path to write the selections CSV.
        drop_id: Deployment identifier.

    Returns:
        DataFrame of selected frame moments with a row per frame.
    """
    if not Path(maxn_csv_path).exists():
        raise FileNotFoundError(f"MaxN CSV not found: {maxn_csv_path}")

    maxn_df = pd.read_csv(maxn_csv_path)
    if maxn_df.empty:
        raise ValueError(
            f"Empty MaxN CSV for {drop_id} — expected frames from strategy but none available."
        )

    # Add TimeSeconds column expected by select_clips_with_strategy
    maxn_df[config.csv_time_seconds_column] = maxn_df[config.csv_maxn_time_ms_column]

    # Get sampling_start from DB
    db = DatabaseManager()
    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT sampling_start FROM deployments WHERE drop_id = ?", (drop_id,)
        )
        row = cursor.fetchone()
    if not row or row["sampling_start"] is None:
        raise ValueError(f"Missing sampling_start for {drop_id}, cannot run Biigle strategy.")
    sampling_start = float(row["sampling_start"])

    # Scale strategy params by biigle_multiplier
    multiplier = config.biigle_multiplier
    unique_species = maxn_df[config.csv_scientific_name_column].unique()
    is_binary = len(unique_species) <= 1 or config.force_binary_strategy

    if is_binary:
        base = config.binary_strategy
        scaled_strategy = {
            "maxn_export": int(base["maxn_export"] * multiplier),
            "confusing_export": int(base["confusing_export"] * multiplier),
            "empty_export": int(base["empty_export"] * multiplier),
            "start_export": int(base["start_export"] * multiplier),
            "temporal_spacing_seconds": base["temporal_spacing_seconds"] / multiplier,
        }
    else:
        base = config.multiclass_strategy
        scaled_strategy = {
            "per_species_maxn_export": int(base["per_species_maxn_export"] * multiplier),
            "per_species_confusing_export": int(base["per_species_confusing_export"] * multiplier),
            "per_video_empty_export": int(base["per_video_empty_export"] * multiplier),
            "per_video_start_export": int(base["per_video_start_export"] * multiplier),
            "temporal_spacing_seconds": base["temporal_spacing_seconds"] / multiplier,
        }

    selections_df = select_clips_with_strategy(
        df=maxn_df,
        drop_id=drop_id,
        sampling_start=sampling_start,
        clip_length=config.clip_length,
        strategy_params=scaled_strategy,
        is_multiclass=not is_binary,
        video_start_threshold=config.video_start_threshold,
        clip_cap=int(config.clip_cap * multiplier),
    )

    Path(output_selections_path).parent.mkdir(parents=True, exist_ok=True)
    selections_df.to_csv(output_selections_path, index=False)
    logging.info(
        f"Biigle-direct: {len(selections_df)} frame selections for {drop_id} "
        f"(multiplier={multiplier}, {'binary' if is_binary else 'multiclass'} strategy)."
    )
    return selections_df
