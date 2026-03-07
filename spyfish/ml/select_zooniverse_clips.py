import pandas as pd
import numpy as np
from pathlib import Path
import os
import time
import logging
from spyfish.config import config
from spyfish.utils import time_to_seconds, seconds_to_time

class ZooniverseSelector:
    """Manages selection state and logic to avoid internal functions."""
    def __init__(self, drop_id, sampling_start, config):
        self.drop_id = drop_id
        self.sampling_start = sampling_start
        self.config = config
        self.clip_length = config.zooniverse_clip_length
        self.selected_intervals = set()
        self.selections_rows = []

    def add_interval(self, row, reason, species="All"):
        interval_start = row['TimeSeconds']
        clip_start_sec = (interval_start // self.clip_length) * self.clip_length

        if clip_start_sec in self.selected_intervals:
            return False

        self.selected_intervals.add(clip_start_sec)
        self.selections_rows.append({
            self.config.drop_id_column: self.drop_id,
            self.config.csv_mapping.get('sampling_start_column', 'SamplingStart'): self.sampling_start,
            self.config.csv_clip_start_column: clip_start_sec,
            self.config.csv_clip_end_column: clip_start_sec + self.clip_length,
            self.config.csv_clip_max_time_column: row['TimeSeconds'],
            self.config.csv_scientific_name_column: species,
            'SelectionReason': reason,
            self.config.csv_max_interval_column: row[self.config.csv_max_interval_column] if self.config.csv_max_interval_column in row else row.get('MaxInterval', 0),
            self.config.csv_confidence_agreement_column: row[self.config.csv_confidence_agreement_column],
        })
        return True


    def check_temporal_spacing(self, candidate_sec, spacing_seconds):
        candidate_clip_start = (candidate_sec // self.clip_length) * self.clip_length
        for s in self.selected_intervals:
            if abs(candidate_clip_start - s) < spacing_seconds:
                return False
        return True

    def finalize_df(self):
        if not self.selections_rows:
            cols = [
                self.config.drop_id_column, self.config.csv_sampling_start_column,
                self.config.csv_clip_start_column, self.config.csv_clip_end_column,
                self.config.csv_clip_max_time_column, self.config.csv_scientific_name_column,
                'SelectionReason', self.config.csv_max_interval_column,
                self.config.csv_confidence_agreement_column
            ]
            return pd.DataFrame(columns=cols)

        df = pd.DataFrame(self.selections_rows)
        df = df.sort_values(self.config.csv_clip_start_column)
        return df

def process_zooniverse_clips(maxn_csv_path, output_selections_path, drop_id, config):
    """
    Selects 10-second intervals from the MaxN CSV to send to Zooniverse.
    Implements rule-based active learning selection (Binary vs Multi-class).
    """
    logging.info(f"Selecting Zooniverse intervals for {drop_id}")

    if not os.path.exists(maxn_csv_path):
        raise FileNotFoundError(f"MaxN CSV not found: {maxn_csv_path}")

    from spyfish.database.manager import DatabaseManager
    db = DatabaseManager()
    with db.get_connection() as conn:
        cursor = conn.cursor()
        # Fetch sampling metadata
        cursor.execute("SELECT sampling_start, sampling_end FROM deployments WHERE drop_id = ?", (drop_id,))
        dep_row = cursor.fetchone()

    if not dep_row or dep_row['sampling_start'] is None or dep_row['sampling_end'] is None:
        logging.warning(f"Missing sampling metadata for {drop_id}. Defaulting sampling_start=0.")
        # TODO it shouldn't default to 0
        sampling_start = 0
        sampling_end = 0
    else:
        sampling_start = dep_row['sampling_start']
        sampling_end = dep_row['sampling_end']

    selector = ZooniverseSelector(drop_id, sampling_start, config)
    df = pd.read_csv(maxn_csv_path)

    # Empty MaxN Handling (Health Checks)
    if df.empty:
        logging.info(f"Empty MaxN CSV for {drop_id}. Generating health check clips.")

        duration = sampling_end - sampling_start
        if duration <= 0:
            logging.warning(f"Invalid duration ({duration}s) for {drop_id}. Skipping health checks.")
        else:
            interval_step = duration / 6
            for i in range(1, 6):
                t = sampling_start + (interval_step * i)
                selector.add_interval({'TimeSeconds': t, self.config.csv_max_interval_column: 0, self.config.csv_confidence_agreement_column: 1.0},
                                     reason='Health Check (Empty Video)')
    else:
        # Normal Processing
        df['TimeSeconds'] = df['TimeOfMax'].apply(time_to_seconds)

    # Pre-calculate the confusion score
    # High count / Low confidence = High ConfusionScore
    # Avoid division by zero by replacing 0 confidence with a tiny float
        df[config.csv_confidence_agreement_column] = df[config.csv_confidence_agreement_column].replace(0, 0.001)
        df[config.csv_confusion_score_column] = df[config.csv_max_interval_column] / df[config.csv_confidence_agreement_column]

        unique_species = df[config.csv_scientific_name_column].unique()
        is_binary = len(unique_species) <= 1
        logging.info(f"Detected {len(unique_species)} species. Using {'Binary' if is_binary else 'Multi-class'} strategy.")

        zoo_config = config.zooniverse_extraction

        if is_binary:
            strat = zoo_config.get('binary_strategy', {})
            n_maxn = strat.get('maxn_clips', 10)
            n_confusing = strat.get('confusing_clips', 20)
            n_empty = strat.get('empty_clips', 5)
            n_start = strat.get('start_clips', 2)
            spacing = strat.get('temporal_spacing_seconds', selector.clip_length)

        # 1. Absolute MaxN
            top_maxn = df.nlargest(n_maxn, config.csv_max_interval_column)
            for _, row in top_maxn.iterrows():
                selector.add_interval(row, reason="Absolute MaxN", species=row[config.csv_scientific_name_column])

        # 2. Confusing
            confusing_df = df.sort_values(config.csv_confusion_score_column, ascending=False)
            added_confusing = 0
            for _, row in confusing_df.iterrows():
                if added_confusing >= n_confusing: break
                # TODO do we need temporal spacing
                if selector.check_temporal_spacing(row['TimeSeconds'], spacing):
                    if selector.add_interval(row, reason="Confusing (High count, low conf)", species=row[config.csv_scientific_name_column]):
                        added_confusing += 1

        # 3. Empty (0 fish)
            empty_df = df[df[config.csv_max_interval_column] == 0]
            if not empty_df.empty:
                for _, row in empty_df.sample(min(n_empty, len(empty_df))).iterrows():
                    selector.add_interval(row, reason="Empty (False Negative Check)", species=row[config.csv_scientific_name_column])

            # 4. Start
            start_df = df[df['TimeSeconds'] < 60] # First minute
            if not start_df.empty:
                for _, row in start_df.sample(min(n_start, len(start_df))).iterrows():
                    selector.add_interval(row, reason="Video Start", species=row[config.csv_scientific_name_column])
        else:
            # Multi-class Strategy
            strat = z_config.get('multiclass_strategy', {})
            n_maxn_per_sp = strat.get('per_species_maxn_clips', 5)
            n_confusing_per_sp = strat.get('per_species_confusing_clips', 10)
            n_empty = strat.get('per_video_empty_clips', 3)
            n_start = strat.get('per_video_start_clips', 2)
            spacing = strat.get('temporal_spacing_seconds', selector.clip_length)

        # Iterate per species
            for species in unique_species:
                sp_df = df[df[config.csv_scientific_name_column] == species]
            # 1. PER SPECIES Absolute MaxN

                for _, row in sp_df.nlargest(n_maxn_per_sp, config.csv_max_interval_column).iterrows():
                    selector.add_interval(row, reason=f"MaxN ({species})", species=species)

            # 2. PER SPECIES Confusing
                conf_sp_df = sp_df.sort_values(config.csv_confusion_score_column, ascending=False)
                added_conf = 0
                for _, row in conf_sp_df.iterrows():
                    if added_conf >= n_confusing_per_sp: break
                    if selector.check_temporal_spacing(row['TimeSeconds'], spacing):
                        if selector.add_interval(row, reason=f"Confusing ({species})", species=species):
                            added_conf += 1

            # 3. Global Empty (across all species)
        # PER VIDEO Global Checks
        # 3. Empty (0 total annotations across all species in that second)
        # To find global empty frames, we find intervals where the max count across all species is 0.
        # But this implies the df must represent all intervals for all species.
        # A simpler way is to find intervals where the overall count of boxes is 0.
        # If the input interval is literally represented by 0...
            empty_df = df.groupby('TimeSeconds')[config.csv_max_interval_column].sum().reset_index()
            true_empty_times = empty_df[empty_df[config.csv_max_interval_column] == 0]['TimeSeconds']
            if not true_empty_times.empty:
                for t in true_empty_times.sample(min(n_empty, len(true_empty_times))):
                    selector.add_interval({'TimeSeconds': t, config.csv_max_interval_column: 0, self.config.csv_confidence_agreement_column: 1.0},
                                         reason="Global Empty", species="All")

        # 4. Start (Global)
            all_times = df['TimeSeconds'].unique()
            start_times = [t for t in all_times if t < 60]
            if start_times:
            # We sample from pandas series easily
                start_times_series = pd.Series(start_times)
                for t in start_times_series.sample(min(n_start, len(start_times_series))):
                    selector.add_interval({'TimeSeconds': t, config.csv_max_interval_column: -1, self.config.csv_confidence_agreement_column: -1.0},
                                         reason="Global Video Start", species="All")

    selections_df = selector.finalize_df()
    Path(output_selections_path).parent.mkdir(parents=True, exist_ok=True)
    selections_df.to_csv(output_selections_path, index=False)
    logging.info(f"Zooniverse Extractions computed: {len(selections_df)} unique clips extracted.")
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
