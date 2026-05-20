import logging
import os
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, List, Optional, cast

import numpy as np
import pandas as pd


def normalize_file_name(file_name: Any) -> str:
    """
    Normalize file name by extracting just the filename from a path.

    Args:
        file_name: File name or path (str, Path, or other)

    Returns:
        str: Just the filename portion, or original value if not a path
    """
    if isinstance(file_name, (str, Path)):
        return Path(file_name).name
    return file_name


def flatten_list(lst: list[list]) -> list:
    """
    Flatten a list of lists.

    Args:
        lst (list[list]): A list of lists.

    Returns:
        list: A flattened list.
    """
    flattened = []
    for item in lst:
        item = flatten_list(item) if isinstance(item, list) else [item]
        flattened.extend(item)
    return flattened


def read_file_to_df(
    file_path: str, sheet_name: str | int | list | None = 0
) -> pd.DataFrame | dict:
    """Reads a CSV or Excel file into a Pandas DataFrame.

    If you don't know the name of your sheet, you can set sheet_name=None which
    returns a dictionary with all the sheet names as keys and content of each
    sheet in dfs as values. Using <output>.keys() outputs all the sheet names.
    """
    if file_path.endswith(".csv"):
        return pd.read_csv(file_path)
    return pd.read_excel(file_path, sheet_name=sheet_name)


def is_format_match(pattern, string):
    """Checks if a string matches a regex pattern.

    Args:
        pattern (str): The regex pattern to match against.
        string (str): The string to check.

    Returns:
        bool: True if the string matches the pattern, False otherwise.
    """
    if pd.isna(string):
        return False
    return bool(re.fullmatch(pattern, string))


class EnvironmentVariableError(Exception):
    """Custom exception for missing environment variables."""


def get_env_var(name: str) -> str:
    """
    Gets an environment variable and raises an error if not found.

    Args:
        name (str): The name of the environment variable.

    Returns:
        The value of the environment variable.

    Raises:
        EnvironmentVariableError: If the environment variable is not set.
    """
    value = os.getenv(name)
    if value is None:
        raise EnvironmentVariableError(f"Environment variable '{name}' not set.")
    return cast(str, value)


def str_to_bool(value: Optional[str]) -> bool:
    if value is None:
        return False
    return value.strip().lower() == "true"


def delete_file(filename: str):
    """
    Deletes a file from the filesystem if it exists.

    Args:
        filename (str): The path to the file to delete.

    Logs:
        - A debug message if the file does not exist.
        - An error message if the file could not be deleted due to permission or OS-related issues.
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

    Parameters:
        file_paths (Iterable): A set or list of file paths (strings) to filter.
        valid_extensions (Iterable): An iterable containing valid file
            extensions (e.g., ['mp4', 'mov']).

    Returns:
        list: A list of file paths that have an extension matching one of the valid extensions.
    """
    filtered_file_paths = []

    for file_path in file_paths:
        # Extract the file extension (e.g., 'mp4', 'jpg'), remove the leading dot
        ext = os.path.splitext(file_path)[-1].lower().lstrip(".")

        # Include file path if its extension is in the valid list
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

    Parameters:
        buv_deployment_df (pd.DataFrame): The DataFrame to process.
        column_name_to_extract (str): The column from which to extract unique values.
        drop_na (bool): Whether to drop NaN values (default is True).
        column_filter (Optional, str): Optional column to filter the DataFrame by.
        column_value (Optional:Any): Value that the filter column must equal.
            Only keeps the rows that are equal to this value.

    Returns:
        A set of unique values from the specified column.
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


# Function to check if a float column contains only whole numbers
def convert_int_num_columns_to_int(df: pd.DataFrame) -> pd.DataFrame:
    """Convert numeric columns with whole numbers to nullable integers in-place.

    This function iterates through all numeric columns of a DataFrame. If a column's
    non-null values are all whole numbers, it converts the column to pandas'
    nullable 'Int64' dtype. This modification is done in-place.

    Args:
        df: The DataFrame to modify.

    Returns:
        The modified DataFrame with integer columns converted.
    """
    for col in df.select_dtypes(include=[np.number]).columns:
        series_no_na = df[col].dropna()
        if not series_no_na.empty and np.all(series_no_na == series_no_na.astype(int)):
            df[col] = df[col].astype("Int64")  # Use pandas nullable Int type
    return df


def write_data_to_file(data_str: str, output_path: str) -> None:
    """
    Write data to a text file.

    Args:
        data_str: Data to write to the file
        output_path: Path to the output text file

    Side Effects:
        - Creates or overwrites the file at output_path
        - Logs the operation
    """
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(data_str)
        logging.info(f"Wrote data to {output_path}")
    except (IOError, OSError) as e:
        logging.error(f"Failed to write file paths to {output_path}: {e}")
        raise


def ping():
    logging.info("The Spyfish Aotearoa Toolkit is installed")


@contextmanager
def temp_file_manager(filenames: list[str]):
    """Context manager to handle temporary file cleanup."""
    try:
        yield
    finally:
        for filename in filenames:
            delete_file(filename)
