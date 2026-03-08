import pandas as pd
import numpy as np
from pathlib import Path
import os
import logging
from spyfish.config import config
from spyfish.utils import time_to_seconds

class ClipSelector:
    """Manages selection state and logic for generic clip selection."""
    def __init__(self, drop_id, sampling_start, clip_length):
        self.drop_id = drop_id
        self.sampling_start = sampling_start
        self.clip_length = clip_length
        self.selected_intervals = set()
        self.selections_rows = []

    def add_interval(self, row, reason, species="All"):
        """Adds an interval to the selection list if it's not already covered by a clip."""
        # Use the specific column names from config
        time_col = config.csv_time_seconds_column

        interval_start = row[time_col]
        clip_start_sec = (interval_start // self.clip_length) * self.clip_length

        if clip_start_sec in self.selected_intervals:
            return False

        self.selected_intervals.add(clip_start_sec)
        self.selections_rows.append({
            config.drop_id_column: self.drop_id,
            config.csv_sampling_start_column: self.sampling_start,
            config.csv_clip_start_column: clip_start_sec,
            config.csv_clip_end_column: clip_start_sec + self.clip_length,
            config.csv_clip_max_time_column: row[time_col],
            config.csv_scientific_name_column: species,
            'SelectionReason': reason,
            config.csv_max_interval_column: row.get(config.csv_max_interval_column, 0),
            config.csv_confidence_agreement_column: row.get(config.csv_confidence_agreement_column, 1.0),
        })
        return True

    def check_temporal_spacing(self, candidate_sec, spacing_seconds):
        """Returns True if the candidate second is far enough from already selected clips."""
        if spacing_seconds <= 0:
            return True
        candidate_clip_start = (candidate_sec // self.clip_length) * self.clip_length
        for s in self.selected_intervals:
            if abs(candidate_clip_start - s) < spacing_seconds:
                return False
        return True

    def finalize_df(self):
        """Returns a sorted DataFrame of all selected clips."""
        if not self.selections_rows:
            cols = [
                config.drop_id_column, config.csv_sampling_start_column,
                config.csv_clip_start_column, config.csv_clip_end_column,
                config.csv_clip_max_time_column, config.csv_scientific_name_column,
                'SelectionReason', config.csv_max_interval_column,
                config.csv_confidence_agreement_column
            ]
            return pd.DataFrame(columns=cols)

        df = pd.DataFrame(self.selections_rows)
        df = df.sort_values(config.csv_clip_start_column)
        return df

def select_clips_with_strategy(
    df: pd.DataFrame,
    drop_id: str,
    sampling_start: float,
    clip_length: int,
    strategy_params: dict,
    is_multiclass: bool,
    video_start_threshold: int,
    clip_cap: int
) -> pd.DataFrame:
    """
    Core logic for selecting valuable clips from a detection DataFrame.
    """
    selector = ClipSelector(drop_id, sampling_start, clip_length)

    if df.empty:
        return selector.finalize_df()

    # Pre-calculate confusion score
    # High count / Low confidence = High ConfusionScore
    df[config.csv_confidence_agreement_column] = df[config.csv_confidence_agreement_column].replace(0, 0.001)
    df[config.csv_confusion_score_column] = df[config.csv_max_interval_column] / df[config.csv_confidence_agreement_column]

    if not is_multiclass:
        # Binary Strategy
        n_maxn = strategy_params.get('maxn_clips')
        n_confusing = strategy_params.get('confusing_clips')
        n_empty = strategy_params.get('empty_clips')
        n_start = strategy_params.get('start_clips')
        spacing = strategy_params.get('temporal_spacing_seconds')

        # 1. Absolute MaxN
        top_maxn = df.nlargest(n_maxn, config.csv_max_interval_column)
        for _, row in top_maxn.iterrows():
            selector.add_interval(row, reason="Absolute MaxN", species=row[config.csv_scientific_name_column])

        # 2. Confusing
        confusing_df = df.sort_values(config.csv_confusion_score_column, ascending=False)
        added_confusing = 0
        for _, row in confusing_df.iterrows():
            if added_confusing >= n_confusing: break
            if selector.check_temporal_spacing(row[config.csv_time_seconds_column], spacing):
                if selector.add_interval(row, reason="Confusing (High count, low conf)", species=row[config.csv_scientific_name_column]):
                    added_confusing += 1

        # 3. Empty (0 fish)
        empty_df = df[df[config.csv_max_interval_column] == 0]
        if not empty_df.empty:
            for _, row in empty_df.sample(min(n_empty, len(empty_df))).iterrows():
                selector.add_interval(row, reason="Empty (False Negative Check)", species=row[config.csv_scientific_name_column])

        # 4. Start
        start_df = df[df[config.csv_time_seconds_column] < video_start_threshold]
        if not start_df.empty:
            for _, row in start_df.sample(min(n_start, len(start_df))).iterrows():
                selector.add_interval(row, reason="Video Start", species=row[config.csv_scientific_name_column])
    else:
        # Multi-class Strategy
        n_maxn_per_sp = strategy_params.get('per_species_maxn_clips')
        n_confusing_per_sp = strategy_params.get('per_species_confusing_clips')
        n_empty = strategy_params.get('per_video_empty_clips')
        n_start = strategy_params.get('per_video_start_clips')
        spacing = strategy_params.get('temporal_spacing_seconds')

        unique_species = df[config.csv_scientific_name_column].unique()
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
                if selector.check_temporal_spacing(row[config.csv_time_seconds_column], spacing):
                    if selector.add_interval(row, reason=f"Confusing ({species})", species=species):
                        added_conf += 1

        # 3. Global Empty
        empty_df = df.groupby(config.csv_time_seconds_column)[config.csv_max_interval_column].sum().reset_index()
        true_empty_times = empty_df[empty_df[config.csv_max_interval_column] == 0][config.csv_time_seconds_column]
        if not true_empty_times.empty:
            for t in true_empty_times.sample(min(n_empty, len(true_empty_times))):
                selector.add_interval({config.csv_time_seconds_column: t, config.csv_max_interval_column: 0, config.csv_confidence_agreement_column: 1.0},
                                     reason="Global Empty", species="All")

        # 4. Start (Global)
        all_times = df[config.csv_time_seconds_column].unique()
        start_times = [t for t in all_times if t < video_start_threshold]
        if start_times:
            start_times_series = pd.Series(start_times)
            for t in start_times_series.sample(min(n_start, len(start_times_series))):
                selector.add_interval({config.csv_time_seconds_column: t, config.csv_max_interval_column: -1, config.csv_confidence_agreement_column: -1.0},
                                     reason="Global Video Start", species="All")

    selections_df = selector.finalize_df()

    # Subsampling (Clip Cap)
    if clip_cap and len(selections_df) > clip_cap:
        # Priority order: MaxN, Start, Empty, then Confusing
        priority_reasons = ["Absolute MaxN", "Video Start", "Global Video Start"]
        priority_reasons += [r for r in selections_df['SelectionReason'].unique() if "MaxN" in r]

        priority_df = selections_df[selections_df['SelectionReason'].isin(priority_reasons)]
        other_df = selections_df[~selections_df['SelectionReason'].isin(priority_reasons)]

        if len(priority_df) >= clip_cap:
            return priority_df.iloc[:clip_cap]
        else:
            n_needed = clip_cap - len(priority_df)
            sampled_others = other_df.sample(n_needed)
            return pd.concat([priority_df, sampled_others]).sort_values(config.csv_clip_start_column)

    return selections_df
