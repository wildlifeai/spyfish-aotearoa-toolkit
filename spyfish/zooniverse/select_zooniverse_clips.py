import logging
import os
from pathlib import Path

import pandas as pd

from spyfish.config.wrapper import config
from spyfish.database.manager import DatabaseManager
from spyfish.extraction.select_clips import ClipSelector, select_clips_with_strategy
from spyfish.utils import time_to_seconds


def process_zooniverse_clips(maxn_csv_path, output_selections_path, drop_id, config):
    """
    Selects n-second intervals from the MaxN CSV to send to Zooniverse.
    Uses the generic selection strategy with Zooniverse-specific overrides.
    """
    logging.info(f"Selecting Zooniverse intervals for {drop_id}")

    if not os.path.exists(maxn_csv_path):
        raise FileNotFoundError(f"MaxN CSV not found: {maxn_csv_path}")

    db = DatabaseManager()
    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT sampling_start, sampling_end FROM deployments WHERE drop_id = ?",
            (drop_id,),
        )
        dep_row = cursor.fetchone()

    if (
        not dep_row
        or dep_row["sampling_start"] is None
        or dep_row["sampling_end"] is None
    ):
        raise ValueError(
            f"Missing mandatory sampling metadata (SamplingStart/End) for {drop_id}."
        )

    sampling_start = dep_row["sampling_start"]
    sampling_end = dep_row["sampling_end"]

    df = pd.read_csv(maxn_csv_path)

    # Health Check Handling (Zooniverse specific)
    if df.empty:
        logging.info(f"Empty MaxN CSV for {drop_id}. Generating health check clips.")
        selector = ClipSelector(drop_id, sampling_start, config.clip_length)
        duration = sampling_end - sampling_start
        if duration > 0:
            interval_step = duration / config.health_check_count
            for i in range(1, config.health_check_count):
                t = sampling_start + (interval_step * i)
                selector.add_interval(
                    {
                        config.csv_time_seconds_column: t,
                        config.csv_max_interval_column: 0,
                        config.csv_confidence_agreement_column: 1.0,
                    },
                    reason="Health Check (Empty Video)",
                )
        selections_df = selector.finalize_df()
    else:
        # Convert human-readable time to seconds
        df[config.csv_time_seconds_column] = df[config.csv_maxn_time_column].apply(
            time_to_seconds
        )

        unique_species = df[config.csv_scientific_name_column].unique()
        is_binary = len(unique_species) <= 1 or config.force_binary_strategy

        if is_binary:
            strategy_params = config.binary_strategy
        else:
            strategy_params = config.multiclass_strategy

        selections_df = select_clips_with_strategy(
            df=df,
            drop_id=drop_id,
            sampling_start=sampling_start,
            clip_length=config.clip_length,
            strategy_params=strategy_params,
            is_multiclass=not is_binary,
            video_start_threshold=config.video_start_threshold,
            clip_cap=config.clip_cap,
        )

    Path(output_selections_path).parent.mkdir(parents=True, exist_ok=True)
    selections_df.to_csv(output_selections_path, index=False)
    logging.info(
        f"Zooniverse Extractions computed: {len(selections_df)} unique clips selected."
    )
    return selections_df


def main(drop_id):
    logging.info(f"Running Zooniverse extraction for Drop ID: {drop_id}")
    model_name = Path(config.pipeline_model_path).stem

    input_maxn = str(config.get_maxn_csv_path(drop_id, model_name))
    output_selections = str(config.get_selections_csv_path(drop_id))

    process_zooniverse_clips(input_maxn, output_selections, drop_id, config)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Select Zooniverse clips.")
    parser.add_argument("drop_id", type=str, help="The Drop ID to process.")
    args = parser.parse_args()

    main(args.drop_id)
