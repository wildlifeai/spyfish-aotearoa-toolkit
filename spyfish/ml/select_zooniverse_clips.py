import pandas as pd
import numpy as np
from pathlib import Path
import os
import time
import logging
from spyfish.config import config

def process_zooniverse_clips(maxn_csv_path, output_selections_path, drop_id, config):
    """
    Selects 10-second intervals from the MaxN CSV to send to Zooniverse.
    Implements rule-based active learning selection (Binary vs Multi-class).
    """
    logging.info(f"Selecting Zooniverse intervals for {drop_id}")

    if not os.path.exists(maxn_csv_path):
        raise FileNotFoundError(f"MaxN CSV not found: {maxn_csv_path}")

    df = pd.read_csv(maxn_csv_path)
    if df.empty:
        logging.warning(f"Empty MaxN CSV for {drop_id}, no clips to extract.")
        return

    # Helper function to convert HH:MM:SS to seconds
    def time_to_seconds(time_str):
        if pd.isna(time_str):
            return 0
        h, m, s = map(int, time_str.split(':'))
        return h * 3600 + m * 60 + s

    # Helper function to convert seconds to HH:MM:SS
    def seconds_to_time(seconds):
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        return f"{int(hours):02d}:{int(minutes):02d}:{int(secs):02d}"

    # We need the time in seconds to do temporal spacing checks
    df['TimeSeconds'] = df['TimeOfMax'].apply(time_to_seconds)

    # Pre-calculate the confusion score
    # High count / Low confidence = High ConfusionScore
    # Avoid division by zero by replacing 0 confidence with a tiny float
    df['ConfidenceAgreement'] = df['ConfidenceAgreement'].replace(0, 0.001)
    df['ConfusionScore'] = df['MaxInterval'] / df['ConfidenceAgreement']

    selected_intervals = set()
    selections_rows = []

    def add_interval(row, reason, species="All"):
        interval_start = row['TimeSeconds']

        # In Spyfish, intervals are generally bucketed to the 10s mark.
        # But if TimeOfMax is exactly 00:00:15, we want the bucket that contains it (00:00:10 to 00:00:20)
        # So we round down to the nearest 10s bucket to get the start of the clip.
        clip_start_sec = (interval_start // 10) * 10

        if clip_start_sec in selected_intervals:
            return False # Already selected this 10s clip for another reason

        selected_intervals.add(clip_start_sec)
        selections_rows.append({
            'DropID': drop_id,
            'StartTime': seconds_to_time(clip_start_sec),
            'EndTime': seconds_to_time(clip_start_sec + 10),
            'TargetSpecies': species,
            'SelectionReason': reason,
            'MaxCount': row['MaxInterval'],
            'Confidence': row['ConfidenceAgreement']
        })
        return True

    def check_temporal_spacing(candidate_sec, spacing_seconds, current_selections):
        candidate_clip_start = (candidate_sec // 10) * 10
        for s in current_selections:
            if abs(candidate_clip_start - s) < spacing_seconds:
                return False
        return True

    # Check mapping
    unique_species = df['ScientificName'].unique()
    is_binary = len(unique_species) <= 1

    logging.info(f"Detected {len(unique_species)} species. Using {'Binary' if is_binary else 'Multi-class'} strategy.")

    # Load constraints from config or fall back to defaults
    z_config = config.get('zooniverse_extraction', {})

    if is_binary:
        strat = z_config.get('binary_strategy', {})
        n_maxn = strat.get('maxn_clips', 10)
        n_confusing = strat.get('confusing_clips', 20)
        n_empty = strat.get('empty_clips', 5)
        n_start = strat.get('start_clips', 2)
        spacing = strat.get('temporal_spacing_seconds', 30)

        # 1. Absolute MaxN
        top_maxn = df.nlargest(n_maxn, 'MaxInterval')
        for _, row in top_maxn.iterrows():
            add_interval(row, reason="Absolute MaxN", species=row['ScientificName'])

        # 2. Confusing
        confusing_df = df.sort_values('ConfusionScore', ascending=False)
        added_confusing = 0
        for _, row in confusing_df.iterrows():
            if added_confusing >= n_confusing:
                break
            # Must satisfy temporal spacing exclusively against other confusing clips
            # (It's OK if a confusing clip is near a MaxN clip)
            # Actually, per prompt: "spaced at least 30s apart, the ones that are not in MaxN"
            # We check the master `selected_intervals` so it doesn't overlap at all.
            if check_temporal_spacing(row['TimeSeconds'], spacing, selected_intervals):
                if add_interval(row, reason="Confusing (High count, low conf)", species=row['ScientificName']):
                    added_confusing += 1

        # 3. Empty (0 fish)
        empty_df = df[df['MaxInterval'] == 0]
        if not empty_df.empty:
            samples = min(n_empty, len(empty_df))
            empty_samples = empty_df.sample(samples)
            for _, row in empty_samples.iterrows():
                add_interval(row, reason="Empty (False Negative Check)", species=row['ScientificName'])

        # 4. Start
        start_df = df[df['TimeSeconds'] < 60] # First minute
        if not start_df.empty:
            samples = min(n_start, len(start_df))
            start_samples = start_df.sample(samples)
            for _, row in start_samples.iterrows():
                add_interval(row, reason="Video Start", species=row['ScientificName'])

    else:
        # Multi-class Strategy
        strat = z_config.get('multiclass_strategy', {})
        n_maxn_per_sp = strat.get('per_species_maxn_clips', 5)
        n_confusing_per_sp = strat.get('per_species_confusing_clips', 10)
        n_empty = strat.get('per_video_empty_clips', 3)
        n_start = strat.get('per_video_start_clips', 2)
        spacing = strat.get('temporal_spacing_seconds', 30)

        # Iterate per species
        for species in unique_species:
            sp_df = df[df['ScientificName'] == species]

            # 1. PER SPECIES Absolute MaxN
            top_maxn = sp_df.nlargest(n_maxn_per_sp, 'MaxInterval')
            for _, row in top_maxn.iterrows():
                add_interval(row, reason=f"MaxN ({species})", species=species)

            # 2. PER SPECIES Confusing
            conf_sp_df = sp_df.sort_values('ConfusionScore', ascending=False)
            added_conf = 0
            for _, row in conf_sp_df.iterrows():
                if added_conf >= n_confusing_per_sp:
                    break
                if check_temporal_spacing(row['TimeSeconds'], spacing, selected_intervals):
                    if add_interval(row, reason=f"Confusing ({species})", species=species):
                        added_conf += 1

        # PER VIDEO Global Checks
        # 3. Empty (0 total annotations across all species in that second)
        # To find global empty frames, we find intervals where the max count across all species is 0.
        # But this implies the df must represent all intervals for all species.
        # A simpler way is to find intervals where the overall count of boxes is 0.
        # If the input interval is literally represented by 0...
        empty_df = df.groupby('TimeSeconds')['MaxInterval'].sum().reset_index()
        true_empty_times = empty_df[empty_df['MaxInterval'] == 0]['TimeSeconds']

        if not true_empty_times.empty:
            samples = min(n_empty, len(true_empty_times))
            empty_samples = true_empty_times.sample(samples)
            for t in empty_samples:
                # Mock a row dict
                row = {'TimeSeconds': t, 'MaxInterval': 0, 'ConfidenceAgreement': 1.0}
                add_interval(row, reason="Global Empty", species="All")

        # 4. Start (Global)
        all_times = df['TimeSeconds'].unique()
        start_times = [t for t in all_times if t < 60]
        if start_times:
            # We sample from pandas series easily
            start_times_series = pd.Series(start_times)
            samples = min(n_start, len(start_times_series))
            start_samples = start_times_series.sample(samples)
            for t in start_samples:
                row = {'TimeSeconds': t, 'MaxInterval': -1, 'ConfidenceAgreement': -1.0}
                add_interval(row, reason="Global Video Start", species="All")

    # Finalize
    selections_df = pd.DataFrame(selections_rows)
    # Sort chronologically
    if not selections_df.empty:
        selections_df['sort_helper'] = selections_df['StartTime'].apply(time_to_seconds)
        selections_df = selections_df.sort_values('sort_helper').drop(columns=['sort_helper'])
    else:
        selections_df = pd.DataFrame(columns=['DropID', 'StartTime', 'EndTime', 'TargetSpecies', 'SelectionReason', 'MaxCount', 'Confidence'])

    Path(output_selections_path).parent.mkdir(parents=True, exist_ok=True)
    selections_df.to_csv(output_selections_path, index=False)

    logging.info(f"Zooniverse Extractions computed: {len(selections_df)} unique clips extracted.")
    return selections_df

def main():
    if "snakemake" in globals():
        pass
    else:
        logging.info("Running Zooniverse extraction in standalone test mode.")
        drop_id = config.test_drops[0][0]
        model_name = Path(config.model_path or config.mock_model_path).stem

        annotations_dir = repo_root / config.local_manifest_dir_path

        input_maxn = str(annotations_dir / f"{drop_id}_{model_name}_maxn.csv")
        output_selections = str(annotations_dir / f"{drop_id}_zooniverse_selections.csv")

        process_zooniverse_clips(input_maxn, output_selections, drop_id, config)

if __name__ == "__main__":
    main()
