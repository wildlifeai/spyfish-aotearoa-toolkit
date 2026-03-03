import os
import logging
from typing import Set, Iterable, List, Optional, Any
import pandas as pd
import numpy as np
from pathlib import Path
from spyfish.config import config


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
    pattern = getattr(config, 'validation_rules', {}).get('formats', {}).get('SurveyID', r'^([A-Z]{3}_\d{8}_BUV)')

    if pattern.endswith('$'):
        pattern = pattern[:-1]
    if not pattern.startswith('('):
        pattern = f"({pattern})"

    # Extract returns a dataframe of capture groups. We just want the first capture group.
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

def write_data_to_file(data_str: str, output_path: str) -> None:
    """Write data to a text file."""
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(data_str)
        logging.info(f"Wrote data to {output_path}")
    except (IOError, OSError) as e:
        logging.error(f"Failed to write file paths to {output_path}: {e}")
        raise

def time_to_seconds(time_str: str) -> float:
    """Converts a time string (HH:MM:SS or HH:MM:SS.mmm) to seconds."""
    if pd.isna(time_str):
        return 0.0
    parts = str(time_str).split(':')
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
