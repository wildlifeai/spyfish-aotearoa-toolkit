"""
Zooniverse clip selection — picks the most valuable 10-second intervals from the MaxN output.

Applies rule-based active learning to select:
  - Absolute MaxN peaks
  - Confusing intervals (high count, low confidence)
  - Empty intervals (false-negative check)
  - Video-start intervals (baseline)
"""
import logging
import os
from pathlib import Path

import pandas as pd

from spyfish.config import config


# ── helpers ─────────────────────────────────────────────────────────────────

def _time_to_seconds(time_str: str) -> int:
    if pd.isna(time_str):
        return 0
    h, m, s = map(int, time_str.split(":"))
    return h * 3600 + m * 60 + s


def _seconds_to_time(seconds: float) -> str:
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


# ── main function ────────────────────────────────────────────────────────────

def select_zooniverse_clips(
    maxn_csv_path: str,
    output_selections_path: str,
    drop_id: str,
    sampling_start: int = 0,
) -> pd.DataFrame | None:
    """
    Selects 10-second intervals from the MaxN CSV to send to Zooniverse.

    Times in the MaxN CSV are relative to SamplingStart (T=0 = start of usable footage).
    The selections CSV records these relative times. When ffmpeg later cuts clips from the
    full source video, sampling_start must be added back to seek to the correct position.

    Args:
        maxn_csv_path: Path to the MaxN CSV for this drop.
        output_selections_path: Where to write the selections CSV.
        drop_id: Deployment identifier.
        sampling_start: SamplingStart offset in seconds (stored in DB per deployment).

    Returns:
        DataFrame of selected clips, or None if nothing to select.
    """
    logging.info(f"Selecting Zooniverse clips for {drop_id} (sampling_start={sampling_start}s)")

    if not os.path.exists(maxn_csv_path):
        raise FileNotFoundError(f"MaxN CSV not found: {maxn_csv_path}")

    df = pd.read_csv(maxn_csv_path)
    if df.empty:
        logging.warning(f"Empty MaxN CSV for {drop_id}, no clips to select.")
        return None

    z_config = config.get("zooniverse_extraction", {}) if hasattr(config, "get") else {}

    # Clip interval width — always match what the pipeline used
    interval_sec = config.interval_seconds

    # Use sub-second precise timing from the raw ML output if available.
    df["TimeOfMaxnMs"] = df["time_of_maxn_ms"].astype(float)
    df["ConfidenceAgreement"] = df["ConfidenceAgreement"].replace(0, 0.001)
    df["ConfusionScore"] = df["MaxInterval"] / df["ConfidenceAgreement"]

    selected_clip_starts: set = set()
    selection_rows: list = []

    def _add_interval(row, reason: str, species: str = "All") -> bool:
        clip_start = int((row["TimeOfMaxnMs"] // interval_sec) * interval_sec)
        if clip_start in selected_clip_starts:
            return False
        selected_clip_starts.add(clip_start)
        selection_rows.append({
            "DropID": drop_id,
            "SamplingStart": sampling_start,
            "ClipStartRelative": clip_start,                       # seconds since SamplingStart (snapped to interval)
            "ClipEndRelative": clip_start + interval_sec,
            "TimeOfMaxnMs": float(row.get("TimeOfMaxnMs", clip_start)),  # exact ML peak seconds (sub-second precision)
            "StartTime": _seconds_to_time(clip_start),            # HH:MM:SS relative
            "EndTime": _seconds_to_time(clip_start + interval_sec),
            "TargetSpecies": species,
            "SelectionReason": reason,
            "MaxCount": row["MaxInterval"],
            "Confidence": row["ConfidenceAgreement"],
        })
        return True

    def _temporal_ok(candidate_sec: float, spacing: int) -> bool:
        candidate_start = int((candidate_sec // interval_sec) * interval_sec)
        return all(abs(candidate_start - s) >= spacing for s in selected_clip_starts)

    unique_species = df["ScientificName"].unique()
    is_binary = len(unique_species) <= 1
    logging.info(f"{'Binary' if is_binary else 'Multi-class'} strategy ({len(unique_species)} species)")

    if is_binary:
        strat = z_config.get("binary_strategy", {})
        n_maxn       = strat.get("maxn_clips", 10)
        n_confusing  = strat.get("confusing_clips", 20)
        n_empty      = strat.get("empty_clips", 5)
        n_start      = strat.get("start_clips", 2)
        spacing      = strat.get("temporal_spacing_seconds", 30)

        for _, row in df.nlargest(n_maxn, "MaxInterval").iterrows():
            _add_interval(row, "Absolute MaxN", row["ScientificName"])

        added = 0
        for _, row in df.sort_values("ConfusionScore", ascending=False).iterrows():
            if added >= n_confusing:
                break
            if _temporal_ok(row["TimeSeconds"], spacing):
                if _add_interval(row, "Confusing (high count, low conf)", row["ScientificName"]):
                    added += 1

        empty_df = df[df["MaxInterval"] == 0]
        if not empty_df.empty:
            for _, row in empty_df.sample(min(n_empty, len(empty_df))).iterrows():
                _add_interval(row, "Empty (false-negative check)", row["ScientificName"])

        start_df = df[df["TimeSeconds"] < 60]
        if not start_df.empty:
            for _, row in start_df.sample(min(n_start, len(start_df))).iterrows():
                _add_interval(row, "Video Start", row["ScientificName"])

    else:
        strat = z_config.get("multiclass_strategy", {})
        n_maxn_per_sp  = strat.get("per_species_maxn_clips", 5)
        n_conf_per_sp  = strat.get("per_species_confusing_clips", 10)
        n_empty        = strat.get("per_video_empty_clips", 3)
        n_start        = strat.get("per_video_start_clips", 2)
        spacing        = strat.get("temporal_spacing_seconds", 30)

        for species in unique_species:
            sp_df = df[df["ScientificName"] == species]
            for _, row in sp_df.nlargest(n_maxn_per_sp, "MaxInterval").iterrows():
                _add_interval(row, f"MaxN ({species})", species)
            added = 0
            for _, row in sp_df.sort_values("ConfusionScore", ascending=False).iterrows():
                if added >= n_conf_per_sp:
                    break
                if _temporal_ok(row["TimeSeconds"], spacing):
                    if _add_interval(row, f"Confusing ({species})", species):
                        added += 1

        global_counts = df.groupby("TimeSeconds")["MaxInterval"].sum().reset_index()
        empty_times = global_counts[global_counts["MaxInterval"] == 0]["TimeSeconds"]
        for t in empty_times.sample(min(n_empty, len(empty_times))):
            _add_interval({"TimeSeconds": t, "MaxInterval": 0, "ConfidenceAgreement": 1.0},
                          "Global Empty", "All")

        start_times = pd.Series([t for t in df["TimeSeconds"].unique() if t < 60])
        for t in start_times.sample(min(n_start, len(start_times))):
            _add_interval({"TimeSeconds": t, "MaxInterval": -1, "ConfidenceAgreement": -1.0},
                          "Global Video Start", "All")

    if not selection_rows:
        logging.warning(f"No clips selected for {drop_id}.")
        return None

    selections_df = pd.DataFrame(selection_rows)
    selections_df = selections_df.sort_values("ClipStartRelative").reset_index(drop=True)

    Path(output_selections_path).parent.mkdir(parents=True, exist_ok=True)
    selections_df.to_csv(output_selections_path, index=False)
    logging.info(f"Selected {len(selections_df)} clips for {drop_id} → {output_selections_path}")
    return selections_df


def main():
    if "snakemake" in globals():
        pass
    else:
        logging.info("Running clip selection in standalone test mode.")
        repo_root = Path(__file__).parent.parent.parent
        drop_id = config.test_drops[0][0]
        model_name = Path(config.model_path or config.mock_model_path).stem

        annotations_dir = repo_root / config.local_manifest_dir_path
        input_maxn = str(annotations_dir / f"{drop_id}_{model_name}_maxn.csv")
        output_selections = str(annotations_dir / f"{drop_id}_zooniverse_selections.csv")

        select_zooniverse_clips(input_maxn, output_selections, drop_id, sampling_start=0)


if __name__ == "__main__":
    main()
