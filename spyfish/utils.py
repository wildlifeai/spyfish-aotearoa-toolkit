import logging
import os
from pathlib import Path
from typing import Any, Iterable, List, Optional

import numpy as np
import pandas as pd

from spyfish.config.base import PipelineStatus
from spyfish.config.wrapper import config


def delete_file(filename: str):
    """
    Deletes a file from the filesystem if it exists.
    """
    try:
        if os.path.exists(filename):
            os.remove(filename)
        else:
            logging.debug(f"File '{filename}' did not exist, nothing to delete.")
    except (PermissionError, OSError) as e:
        logging.error(f"Failed to remove file '{filename}': {e}")


def filter_file_paths_by_extension(
    file_paths: Iterable[str], valid_extensions: Iterable[str]
) -> List[str]:
    """
    Filter a collection of file paths, returning only those that match the given file extensions.
    """
    filtered_file_paths = []

    for file_path in file_paths:
        ext = os.path.splitext(file_path)[-1].lower().lstrip(".")
        if ext in valid_extensions:
            filtered_file_paths.append(file_path)

    return filtered_file_paths


def get_unique_entries_df_column(
    buv_deployment_df: pd.DataFrame,
    column_name_to_extract: str,
    drop_na: bool = True,
    column_filter: Optional[str] = None,
    column_value: Optional[Any] = None,
) -> set:
    """
    Return a set of unique values from a specified DataFrame column,
    with optional filtering and NaN removal.
    """
    if column_filter:
        buv_deployment_df = buv_deployment_df[
            buv_deployment_df[column_filter] == column_value
        ]
    if drop_na:
        csv_filepaths = set(
            buv_deployment_df[column_name_to_extract].dropna().astype(str)
        )
    else:
        csv_filepaths = set(buv_deployment_df[column_name_to_extract])

    return csv_filepaths


def extract_survey_id(drop_id_series: pd.Series) -> pd.Series:
    """
    Extracts the SurveyID from a pandas Series of DropIDs using the config format regex.
    """
    # The SurveyID pattern is a full-match pattern (e.g. ^KSF_20240124_BUV$).
    # We strip the anchors and wrap in a capture group for str.extract.
    raw = config.get_validation_pattern("survey_id")
    pattern = raw.lstrip("^").rstrip("$")
    if not pattern.startswith("("):
        pattern = f"({pattern})"

    extracted = drop_id_series.str.extract(pattern, expand=True)
    return extracted[0].fillna("UNKNOWN")


def normalize_file_name(file_name: Any) -> str:
    """Normalize file name by extracting just the filename from a path."""
    if isinstance(file_name, (str, Path)):
        return Path(file_name).name
    return str(file_name)


def convert_int_num_columns_to_int(df: pd.DataFrame) -> pd.DataFrame:
    """Convert numeric columns with whole numbers to nullable integers in-place."""
    for col in df.select_dtypes(include=[np.number]).columns:
        series_no_na = df[col].dropna()
        if not series_no_na.empty and np.all(series_no_na == series_no_na.astype(int)):
            df[col] = df[col].astype("Int64")
    return df


def time_to_seconds(time_str: str) -> float:
    """Converts a time string (HH:MM:SS or HH:MM:SS.mmm) to seconds."""
    if pd.isna(time_str):
        return 0.0
    parts = str(time_str).split(":")
    if len(parts) == 3:
        h = int(parts[0])
        m = int(parts[1])
        s = float(parts[2])
        return h * 3600.0 + m * 60.0 + s
    elif len(parts) == 2:
        m = int(parts[0])
        s = float(parts[1])
        return m * 60.0 + s
    else:
        return float(parts[0])


def seconds_to_time(seconds: float) -> str:
    """Converts seconds to HH:MM:SS.mmm format."""
    if pd.isna(seconds):
        return "00:00:00.000"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds % 1) * 1000))
    if ms == 1000:
        s += 1
        ms = 0
        if s == 60:
            s = 0
            m += 1
            if m == 60:
                m = 0
                h += 1
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def get_survey_summary(deployment_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregates deployment data to summarize status and progress by Survey."""
    if deployment_df.empty:
        return pd.DataFrame()

    survey_summary = (
        deployment_df.groupby("SurveyID")
        .agg(
            TotalDeployments=("DropID", "nunique"),
            CompleteDeployments=("Complete", "sum"),
            BadDeployments=("IsBadDeployment", "sum"),
            NeedsAction=("NeedsAction", "sum"),
            VideosPresent=(
                "Status",
                lambda x: x.isin(PipelineStatus.VIDEO_PRESENT_STATUSES).sum(),
            ),
            MLAnnotated=("MlAnnotations", lambda x: (x > 0).sum()),
            CitSciAnnotated=("CitSciAnnotations", lambda x: (x > 0).sum()),
            ExpertAnnotated=("ExpertAnnotations", lambda x: (x > 0).sum()),
        )
        .reset_index()
    )

    # Calculate percentages cleanly: (Bad + Expert) / Total
    survey_summary["CompletionPct"] = (
        (
            (survey_summary["BadDeployments"] + survey_summary["ExpertAnnotated"])
            / survey_summary["TotalDeployments"]
        )
        * 100
    ).round(1).astype(str) + "%"

    return survey_summary


def generate_clip_filename(drop_id: str, duration: float, start_seconds: float) -> str:
    """Standardizes Zooniverse/Biigle MP4 clip naming."""
    return f"{drop_id}__clip_{int(duration):02d}s_{int(start_seconds):05d}s.mp4"


def generate_frame_filename(drop_id: str, time_seconds: float) -> str:
    """Standardizes Zooniverse/Biigle JPEG frame naming."""
    return f"{drop_id}__frame_{time_seconds:.3f}s.jpg"


def validate_model_path(model_path: str | Path) -> Path:
    """
    Validates that a model path is a .pt file located within trusted local directories.
    Prevents path traversal and loading of arbitrary files.
    """
    # TODO: review whether this guard is still needed once we standardise on a single models/ root.
    # Currently prevents path traversal and loading of arbitrary files from untrusted sources.
    path = Path(model_path).resolve()

    # 1. Check extension
    if path.suffix != ".pt":
        raise ValueError(
            f"Security Alert: Invalid model extension: '{path.suffix}'. Only .pt files are allowed."
        )

    # 2. Check for path traversal characters in the input string (redundant but safe)
    if ".." in str(model_path):
        raise ValueError(
            f"Security Alert: Potential path traversal in model path: '{model_path}'"
        )

    # 3. Define trusted roots
    local_training_root = config.local_training_dir.resolve()
    models_root = config.models_root_dir.resolve()

    # 4. Check if path is within trusted roots
    is_safe = False
    for root in [local_training_root, models_root]:
        try:
            path.relative_to(root)
            is_safe = True
            break
        except ValueError:
            continue

    if not is_safe:
        # Check for allowed default weights (names only)
        allowed_defaults = [
            "yolov8n.pt",
            "yolov8s.pt",
            "yolov8m.pt",
            "yolov8l.pt",
            "yolov8x.pt",
            "yolov11n.pt",
            "yolov11s.pt",
            "yolov11m.pt",
            "yolov11l.pt",
            "yolov11x.pt",
            "yolov12n.pt",
            "yolov12s.pt",
            "yolov12m.pt",
            "yolov12x.pt",
        ]
        if path.name in allowed_defaults:
            return path

        raise ValueError(
            f"Security Alert: Model path '{path}' is not within a trusted directory.\n"
            f"Trusted roots: {local_training_root}, {models_root}"
        )

    return path


