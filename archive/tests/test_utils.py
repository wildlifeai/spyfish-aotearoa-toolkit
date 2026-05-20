import os
import tempfile
from unittest.mock import patch

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from sftk.utils import (
    EnvironmentVariableError,
    delete_file,
    flatten_list,
    get_env_var,
    get_unique_entries_df_column,
    is_format_match,
    normalize_file_name,
    read_file_to_df,
    temp_file_manager,
    write_data_to_file,
)

survey_pattern = r"^[A-Z]{3}_\d{8}_BUV$"


@pytest.fixture
def sample_dataframe():
    return pd.DataFrame({"Name": ["Alice", "Bob"], "Age": [30, 25]})


def test_flatten_list():
    """flatten_list should handle simple and nested lists."""
    simple_list = [[1, 2], [3, 4], [5, 6]]
    assert flatten_list(simple_list) == [1, 2, 3, 4, 5, 6]

    nested_list = [[1, 2], [3, 4], [5, [6, 7]]]
    assert flatten_list(nested_list) == [1, 2, 3, 4, 5, 6, 7]


def test_is_format_match():
    """is_format_match should validate strings against regex patterns."""
    assert is_format_match(survey_pattern, "ABC_20250101_BUV")
    assert not is_format_match(survey_pattern, "AB_20250101_BUV")


def test_read_file_to_df(sample_dataframe):
    """read_file_to_df should read CSV and Excel files."""
    # Test CSV
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
        sample_dataframe.to_csv(tmp.name, index=False)
        df = read_file_to_df(tmp.name)
        assert_frame_equal(df, sample_dataframe)

    # Test Excel with multiple sheets
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
        with pd.ExcelWriter(tmp.name) as writer:
            sample_dataframe.to_excel(writer, sheet_name="Sheet1", index=False)
            sample_dataframe.to_excel(writer, sheet_name="Sheet2", index=False)
        sheets = read_file_to_df(tmp.name, sheet_name=None)
        assert isinstance(sheets, dict)
        assert "Sheet1" in sheets and "Sheet2" in sheets
        assert_frame_equal(sheets["Sheet1"], sample_dataframe)


def test_get_env_var():
    """get_env_var should retrieve environment variables or raise error."""
    with patch.dict(os.environ, {"TEST_VAR": "test_value"}):
        assert get_env_var("TEST_VAR") == "test_value"

    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(
            EnvironmentVariableError, match="Environment variable 'MISSING_VAR' not set"
        ):
            get_env_var("MISSING_VAR")


def test_delete_file():
    """delete_file should delete existing files and handle non-existent files."""
    # Test deleting existing file
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp_path = tmp.name
    assert os.path.exists(tmp_path)
    delete_file(tmp_path)
    assert not os.path.exists(tmp_path)

    # Test deleting non-existent file (should not raise)
    delete_file("non_existent_file.txt")


def test_get_unique_entries_df_column():
    """get_unique_entries_df_column should extract unique values with filtering."""
    mock_df = pd.DataFrame(
        {
            "col1": ["value1", "value2", "value1", None],
            "col2": [1, 2, 3, 4],
            "IsBadDeployment": [True, False, False, False],
        }
    )

    # Test with drop_na=True (default)
    result = get_unique_entries_df_column(mock_df, "col1")
    assert result == {"value1", "value2"}

    # Test with drop_na=False
    result = get_unique_entries_df_column(mock_df, "col1", drop_na=False)
    assert None in result and len(result) == 3

    # Test filtering by column value
    result = get_unique_entries_df_column(
        mock_df, "col1", column_filter="IsBadDeployment", column_value=False
    )
    assert result == {"value1", "value2"}


def test_temp_file_manager():
    """temp_file_manager should clean up files even with exceptions."""
    # Create temporary files
    temp_files = []
    for _ in range(3):
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            temp_files.append(tmp.name)

    # Test normal cleanup
    with temp_file_manager(temp_files):
        for file in temp_files:
            assert os.path.exists(file)
    for file in temp_files:
        assert not os.path.exists(file)

    # Test cleanup with exception
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        temp_file = tmp.name
    with pytest.raises(ValueError):
        with temp_file_manager([temp_file]):
            raise ValueError("Test exception")
    assert not os.path.exists(temp_file)


def test_write_data_to_file():
    """write_data_to_file should write data to file."""
    file_set = {"path/to/file1.mp4", "path/to/file2.mp4", "another/file3.mp4"}
    file_str = "\n".join(sorted(list(file_set)))

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as tmp:
        output_path = tmp.name

    try:
        write_data_to_file(file_str, output_path)
        with open(output_path, "r", encoding="utf-8") as f:
            lines = f.read().strip().split("\n")
        assert lines == sorted(file_set)
    finally:
        if os.path.exists(output_path):
            os.remove(output_path)


def test_normalize_file_name():
    """normalize_file_name should extract filename from paths."""
    from pathlib import Path

    assert normalize_file_name("/path/to/file.csv") == "file.csv"
    assert normalize_file_name(Path("/path/to/file.csv")) == "file.csv"
    assert normalize_file_name(123) == 123  # Non-path values returned as-is


def test_str_to_bool():
    """str_to_bool should convert 'true' to True, everything else to False."""
    from sftk.utils import str_to_bool

    assert str_to_bool("true") is True
    assert str_to_bool("True") is True
    assert str_to_bool(" true ") is True
    assert str_to_bool("false") is False
    assert str_to_bool(None) is False


def test_filter_file_paths_by_extension():
    """filter_file_paths_by_extension should filter files by extension."""
    from sftk.utils import filter_file_paths_by_extension

    file_paths = ["video1.mp4", "video2.mov", "image1.jpg", "video3.mp4"]

    video_files = filter_file_paths_by_extension(file_paths, ["mp4", "mov"])
    assert len(video_files) == 3
    assert "video1.mp4" in video_files

    # Test case insensitivity
    mixed_case = ["FILE.MP4", "file.mp4"]
    result = filter_file_paths_by_extension(mixed_case, ["mp4"])
    assert len(result) == 2


def test_convert_int_num_columns_to_int():
    """convert_int_num_columns_to_int should convert whole number floats to Int64."""
    from sftk.utils import convert_int_num_columns_to_int

    df = pd.DataFrame(
        {
            "int_col": [1.0, 2.0, 3.0],
            "float_col": [1.5, 2.5, 3.5],
            "string_col": ["a", "b", "c"],
        }
    )

    result = convert_int_num_columns_to_int(df)
    assert result["int_col"].dtype == "Int64"
    assert list(result["int_col"]) == [1, 2, 3]
    assert result["float_col"].dtype == "float64"  # Should remain float
