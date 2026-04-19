import pandas as pd

from spyfish.config.wrapper import config


class ClipSelector:
    """Manages selection state and logic for generic clip selection."""

    def __init__(self, drop_id, sampling_start, clip_length):
        self.drop_id = drop_id
        self.sampling_start = sampling_start
        self.clip_length = clip_length
        self.selected_intervals = set()
        self.selections_rows = []

    def _bucket(self, t: float) -> float:
        """Map a timestamp to its clip-bucket start, aligned from sampling_start."""
        return (
            (t - self.sampling_start) // self.clip_length
        ) * self.clip_length + self.sampling_start

    def add_interval(self, row, reason, species="All"):
        """Adds an interval to the selection list if not already covered."""
        time_col = config.csv_time_seconds_column
        interval_start = row[time_col]

        if interval_start < self.sampling_start:
            return False

        clip_start_absolute = self._bucket(interval_start)

        if clip_start_absolute in self.selected_intervals:
            return False

        self.selected_intervals.add(clip_start_absolute)
        self.selections_rows.append(
            {
                config.drop_id_column: self.drop_id,
                config.csv_sampling_start_column: self.sampling_start,
                config.csv_clip_start_absolute_column: clip_start_absolute,
                config.csv_clip_end_absolute_column: clip_start_absolute
                + self.clip_length,
                config.csv_clip_max_time_column: row[time_col],
                config.csv_scientific_name_column: species,
                "SelectionReason": reason,
                config.csv_max_interval_column: row.get(
                    config.csv_max_interval_column, 0
                ),
                config.csv_confidence_agreement_column: row.get(
                    config.csv_confidence_agreement_column, 1.0
                ),
            }
        )
        return True

    def check_temporal_spacing(self, candidate_sec, spacing_seconds):
        """True if the candidate is far enough from every already-selected clip."""
        if spacing_seconds <= 0:
            return True
        candidate_clip_start = self._bucket(candidate_sec)
        for s in self.selected_intervals:
            if abs(candidate_clip_start - s) < spacing_seconds:
                return False
        return True

    @property
    def _columns(self):
        return [
            config.drop_id_column,
            config.csv_sampling_start_column,
            config.csv_clip_start_absolute_column,
            config.csv_clip_end_absolute_column,
            config.csv_clip_max_time_column,
            config.csv_scientific_name_column,
            "SelectionReason",
            config.csv_max_interval_column,
            config.csv_confidence_agreement_column,
        ]

    def finalize_df(self):
        if not self.selections_rows:
            return pd.DataFrame(columns=self._columns)
        df = pd.DataFrame(self.selections_rows)
        df = df.sort_values(config.csv_clip_start_absolute_column)
        return df


def _sample(df_or_series, n, random_state=42):
    """Sample up to n rows, capped at the available length."""
    return df_or_series.sample(min(n, len(df_or_series)), random_state=random_state)


def select_clips_with_strategy(
    df: pd.DataFrame,
    drop_id: str,
    sampling_start: float,
    clip_length: int,
    strategy_params: dict,
    is_multiclass: bool,
    video_start_threshold: int,
    clip_cap: int,
) -> pd.DataFrame:
    """
    Core logic for selecting valuable clips from a detection DataFrame.

    Parallel implementation: select_frames._select_frames_with_strategy() applies the
    same MaxN/confusing/start strategy to individual frames rather than clip buckets.
    Key intentional divergences:
      - This function deduplicates by clip bucket (10s window); frames use float spacing.
      - This function includes an "empty" bucket (false-negative check); frames do not
        (raw CSV only contains detected frames, so "empty" is not applicable).
      - Cap/priority logic differs to reflect clip vs frame use cases.
    If you change the core strategy logic here, check whether select_frames.py needs
    the same update.
    """
    selector = ClipSelector(drop_id, sampling_start, clip_length)

    if df.empty:
        return selector.finalize_df()

    # Pre-calculate confusion score
    # High count / Low confidence = High ConfusionScore
    df[config.csv_confidence_agreement_column] = df[
        config.csv_confidence_agreement_column
    ].replace(0, 0.001)
    df[config.csv_confusion_score_column] = (
        df[config.csv_max_interval_column] / df[config.csv_confidence_agreement_column]
    )

    if not is_multiclass:
        # Binary Strategy
        n_maxn = strategy_params.get("maxn_export")
        n_confusing = strategy_params.get("confusing_export")
        n_empty = strategy_params.get("empty_export")
        n_start = strategy_params.get("start_export")
        spacing = strategy_params.get("temporal_spacing_seconds")

        # 1. Absolute MaxN — oversample to account for spacing/dedup rejects
        top_maxn = df.nlargest(n_maxn * 3, config.csv_max_interval_column)
        added_maxn = 0
        for _, row in top_maxn.iterrows():
            if added_maxn >= n_maxn:
                break
            if selector.check_temporal_spacing(
                row[config.csv_time_seconds_column], spacing
            ):
                if selector.add_interval(
                    row,
                    reason="Absolute MaxN",
                    species=row[config.csv_scientific_name_column],
                ):
                    added_maxn += 1

        # 2. Confusing
        confusing_df = df.sort_values(
            config.csv_confusion_score_column, ascending=False
        )
        added_confusing = 0
        for _, row in confusing_df.iterrows():
            if added_confusing >= n_confusing:  # type: ignore
                break
            if selector.check_temporal_spacing(
                row[config.csv_time_seconds_column], spacing
            ):
                if selector.add_interval(
                    row,
                    reason="Confusing (High count, low conf)",
                    species=row[config.csv_scientific_name_column],
                ):
                    added_confusing += 1

        # 3. Empty (0 fish)
        empty_df = df[df[config.csv_max_interval_column] == 0]
        if not empty_df.empty:
            for _, row in _sample(empty_df, n_empty).iterrows():  # type: ignore
                selector.add_interval(
                    row,
                    reason="Empty (False Negative Check)",
                    species=row[config.csv_scientific_name_column],
                )

        # 4. Start
        start_df = df[df[config.csv_time_seconds_column] < video_start_threshold]
        if not start_df.empty:
            for _, row in _sample(start_df, n_start).iterrows():  # type: ignore
                selector.add_interval(
                    row,
                    reason="Video Start",
                    species=row[config.csv_scientific_name_column],
                )
    else:
        # Multi-class Strategy
        n_maxn_per_sp = strategy_params.get("per_species_maxn_export")
        n_confusing_per_sp = strategy_params.get("per_species_confusing_export")
        n_empty = strategy_params.get("per_video_empty_export")
        n_start = strategy_params.get("per_video_start_export")
        spacing = strategy_params.get("temporal_spacing_seconds")

        unique_species = df[config.csv_scientific_name_column].unique()
        for species in unique_species:
            sp_df = df[df[config.csv_scientific_name_column] == species]

            # 1. PER SPECIES Absolute MaxN
            for _, row in sp_df.nlargest(
                n_maxn_per_sp, config.csv_max_interval_column
            ).iterrows():
                selector.add_interval(row, reason=f"MaxN ({species})", species=species)

            # 2. PER SPECIES Confusing
            conf_sp_df = sp_df.sort_values(
                config.csv_confusion_score_column, ascending=False
            )
            added_conf = 0
            for _, row in conf_sp_df.iterrows():
                if added_conf >= n_confusing_per_sp:  # type: ignore
                    break
                if selector.check_temporal_spacing(
                    row[config.csv_time_seconds_column], spacing
                ):
                    if selector.add_interval(
                        row, reason=f"Confusing ({species})", species=species
                    ):
                        added_conf += 1

        # 3. Global Empty
        empty_df = (
            df.groupby(config.csv_time_seconds_column)[config.csv_max_interval_column]
            .sum()
            .reset_index()
        )
        true_empty_times = empty_df[empty_df[config.csv_max_interval_column] == 0][
            config.csv_time_seconds_column
        ]
        if not true_empty_times.empty:
            for t in _sample(true_empty_times, n_empty):  # type: ignore
                selector.add_interval(
                    {
                        config.csv_time_seconds_column: t,
                        config.csv_max_interval_column: 0,
                        config.csv_confidence_agreement_column: 1.0,
                    },
                    reason="Global Empty",
                    species="All",
                )

        # 4. Start (Global)
        all_times = df[config.csv_time_seconds_column].unique()
        start_times = [t for t in all_times if t < video_start_threshold]
        if start_times:
            start_times_series = pd.Series(start_times)
            for t in _sample(start_times_series, n_start):  # type: ignore
                selector.add_interval(
                    {
                        config.csv_time_seconds_column: t,
                        config.csv_max_interval_column: -1,
                        config.csv_confidence_agreement_column: -1.0,
                    },
                    reason="Global Video Start",
                    species="All",
                )

    selections_df = selector.finalize_df()

    # Subsampling (Clip Cap)
    if clip_cap and len(selections_df) > clip_cap:
        # Priority order: MaxN, Start, Empty, then Confusing
        priority_reasons = ["Absolute MaxN", "Video Start", "Global Video Start"]
        priority_reasons += [
            r for r in selections_df["SelectionReason"].unique() if "MaxN" in r
        ]

        priority_df = selections_df[
            selections_df["SelectionReason"].isin(priority_reasons)
        ]
        other_df = selections_df[
            ~selections_df["SelectionReason"].isin(priority_reasons)
        ]

        if len(priority_df) >= clip_cap:
            return priority_df.iloc[:clip_cap]
        else:
            n_needed = min(clip_cap - len(priority_df), len(other_df))
            sampled_others = other_df.sample(n_needed, random_state=42)
            return pd.concat([priority_df, sampled_others]).sort_values(
                config.csv_clip_start_absolute_column
            )

    return selections_df
