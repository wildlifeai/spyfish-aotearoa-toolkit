"""
Consolidated validation tests.

Tests all validation types with comprehensive fake data showing all different issues:
- Required: missing values
- Unique: duplicate values
- Foreign keys: values not found in reference dataset
- Formats: regex pattern validation
- Values: value range validation
- Column relationships: computed column validation
- File presence: S3 file presence checks
- Clean row tracking: identifying error-free rows
"""

from unittest.mock import Mock, patch

import pandas as pd
import pytest

from sftk.common import VALIDATION_PATTERNS
from sftk.validation_strategies import (
    CleanRowTracker,
    DatasetValidator,
    ErrorSource,
    FilePresenceValidator,
    ValidationConfig,
)


class TestCleanRowTracker:
    """Tests for CleanRowTracker."""

    def test_initialization_and_tracking(self):
        """CleanRowTracker should initialize, track, and retrieve clean row indices."""
        tracker = CleanRowTracker()
        assert tracker.clean_row_indices == {}

        df = pd.DataFrame({"name": ["Alice", "Bob", "Charlie"]})
        tracker.initialize_dataset("test", df)
        assert tracker.clean_row_indices["test"] == {0, 1, 2}

        tracker.mark_row_as_error(1, "test")
        assert tracker.get_clean_indices("test") == {0, 2}
        assert tracker.get_clean_indices("nonexistent") == set()


class TestComprehensiveValidation:
    """
    Integration tests for all validation types using comprehensive fake data.

    Test dataset contains the following issues:
    - Row 0: Valid (clean row)
    - Row 1: Missing required DropID
    - Row 2: Invalid DropID format + relationship mismatch
    - Row 3: Valid (clean row)
    - Row 4: Duplicate DropID (same as Row 5)
    - Row 5: Duplicate DropID (same as Row 4) + Latitude/Longitude out of range
    - Row 6: Foreign key violation (invalid SurveyID reference) + out of range coords
    - Row 7: Replicate mismatch only (DropID ends with wrong replicate number)
    """

    @pytest.fixture
    def test_data(self):
        """Create comprehensive test dataset with various validation issues."""
        deployments_df = pd.DataFrame(
            {
                "DropID": [
                    "AKA_20231215_BUV_AKA_001_01",  # Row 0: Valid (unique)
                    None,  # Row 1: Missing required
                    "INVALID_FORMAT",  # Row 2: Invalid format
                    "AKA_20231215_BUV_AKA_001_02",  # Row 3: Valid (unique)
                    "AKA_20231216_BUV_AKA_002_01",  # Row 4: Duplicate of Row 5
                    "AKA_20231216_BUV_AKA_002_01",  # Row 5: Duplicate of Row 4 + bad coords
                    "AKA_20231217_BUV_AKA_003_01",  # Row 6: Valid format, FK violation
                    "AKA_20231218_BUV_AKA_004_99",  # Row 7: Replicate mismatch (99 vs 01)
                ],
                "SurveyID": [
                    "AKA_20231215_BUV",  # Row 0: Valid
                    "AKA_20231215_BUV",  # Row 1: Valid
                    "AKA_20231215_BUV",  # Row 2: Valid
                    "AKA_20231215_BUV",  # Row 3: Valid
                    "AKA_20231216_BUV",  # Row 4: Valid
                    "AKA_20231216_BUV",  # Row 5: Valid
                    "INVALID_SURVEY",  # Row 6: FK violation - not in surveys table
                    "AKA_20231218_BUV",  # Row 7: Valid
                ],
                "SiteID": [
                    "AKA_001",
                    "AKA_001",
                    "AKA_001",
                    "AKA_001",
                    "AKA_002",
                    "AKA_002",
                    "AKA_003",
                    "AKA_004",
                ],
                "ReplicateWithinSite": [1, 1, 1, 2, 1, 1, 1, 1],
                "Latitude": [
                    -40.5,
                    -40.0,
                    -41.0,
                    -42.0,
                    -40.5,
                    -50.0,  # Row 5: OUT OF RANGE (< -46)
                    -35.0,  # Row 6: OUT OF RANGE (> -36)
                    -40.0,  # Row 7: Valid
                ],
                "Longitude": [
                    174.0,
                    174.5,
                    175.0,
                    176.0,
                    174.0,
                    165.0,  # Row 5: OUT OF RANGE (< 170)
                    179.0,  # Row 6: OUT OF RANGE (> 178.5)
                    175.0,  # Row 7: Valid
                ],
            }
        )

        # Reference dataset for foreign key validation
        surveys_df = pd.DataFrame(
            {
                "SurveyID": [
                    "AKA_20231215_BUV",
                    "AKA_20231216_BUV",
                    "AKA_20231217_BUV",
                    "AKA_20231218_BUV",
                ],
            }
        )

        rules = {
            "file_name": "deployments.csv",
            "required": ["DropID", "SurveyID"],
            "unique": ["DropID"],
            "info_columns": ["SurveyID", "SiteID"],
            "formats": ["DropID"],
            "foreign_keys": {"surveys": "SurveyID"},
            "values": [
                {
                    "column": "Latitude",
                    "rule": "value_range",
                    "range": [-46, -36],
                    "allowed_values": [0],
                },
                {
                    "column": "Longitude",
                    "rule": "value_range",
                    "range": [170, 178.5],
                    "allowed_values": [0],
                },
            ],
            "relationships": [
                {
                    "column": "DropID",
                    "rule": "equals",
                    "template": "{SurveyID}_{SiteID}_{ReplicateWithinSite:02}",
                },
            ],
        }

        validation_rules = {
            "deployments": rules,
            "surveys": {"file_name": "surveys.csv", "dataset": surveys_df},
        }

        return deployments_df, surveys_df, rules, validation_rules

    def test_all_validations(self, test_data):
        """DatasetValidator should run all validations and track clean rows."""
        df, _, rules, validation_rules = test_data
        tracker = CleanRowTracker()
        tracker.initialize_dataset("deployments", df)

        validator = DatasetValidator(
            rules, VALIDATION_PATTERNS, validation_rules, tracker
        )
        errors = validator.validate(df, "deployments")

        # Check required validation (Row 1 has missing DropID)
        required_errors = [
            e for e in errors if e.ErrorType == ErrorSource.MISSING_REQUIRED_VALUE.value
        ]
        assert len(required_errors) == 1
        assert "DropID" in required_errors[0].ErrorMessage

        # Check unique validation (Rows 4 and 5 have duplicate DropID)
        unique_errors = [
            e for e in errors if e.ErrorType == ErrorSource.DUPLICATE_VALUE.value
        ]
        assert len(unique_errors) == 2

        # Check format validation (Row 2 has invalid DropID format)
        format_errors = [
            e for e in errors if e.ErrorType == ErrorSource.INVALID_FORMAT.value
        ]
        assert len(format_errors) == 1
        assert "INVALID_FORMAT" in format_errors[0].InvalidValue

        # Check foreign key validation (Row 6 has invalid SurveyID)
        fk_errors = [
            e for e in errors if e.ErrorType == ErrorSource.FOREIGN_KEY_NOT_FOUND.value
        ]
        assert len(fk_errors) == 1
        assert "INVALID_SURVEY" in fk_errors[0].ErrorMessage

        # Check value range validation (Rows 5 and 6 have out-of-range lat/long)
        range_errors = [
            e for e in errors if e.ErrorType == ErrorSource.VALUE_OUT_OF_RANGE.value
        ]
        assert len(range_errors) == 4  # 2 lat + 2 lon errors

        # Check replicate mismatch (Row 7 has wrong replicate in DropID)
        replicate_errors = [
            e for e in errors if e.ErrorType == ErrorSource.REPLICATE_MISMATCH.value
        ]
        assert len(replicate_errors) == 1
        assert "99" in replicate_errors[0].ErrorMessage  # Wrong replicate suffix

        # Check relationship mismatch (Row 2 has completely wrong DropID format)
        relationship_errors = [
            e for e in errors if e.ErrorType == ErrorSource.RELATIONSHIP_MISMATCH.value
        ]
        assert len(relationship_errors) >= 1

        # Only rows 0 and 3 should be clean
        clean_indices = tracker.get_clean_indices("deployments")
        assert 0 in clean_indices, "Row 0 should be clean"
        assert 3 in clean_indices, "Row 3 should be clean"
        assert (
            len(clean_indices) == 2
        ), f"Expected 2 clean rows, got {len(clean_indices)}"

    def test_missing_columns(self):
        """DatasetValidator should report missing columns once (deduplicated)."""
        df = pd.DataFrame({"SurveyID": ["AKA_20231215_BUV"]})  # Missing DropID column

        rules = {
            "file_name": "deployments.csv",
            "required": ["DropID", "SurveyID"],  # DropID column doesn't exist
            "formats": ["DropID"],  # DropID column doesn't exist
        }

        validator = DatasetValidator(rules, VALIDATION_PATTERNS, {}, None)
        errors = validator.validate(df, "deployments")

        missing_col_errors = [
            e for e in errors if e.ErrorType == ErrorSource.MISSING_COLUMN.value
        ]
        # Missing column reported once (deduplicated across required/formats/etc)
        assert len(missing_col_errors) == 1
        assert missing_col_errors[0].ColumnName == "DropID"
        assert "not found in dataset" in missing_col_errors[0].ErrorMessage

    def test_file_presence_validation(self):
        """FilePresenceValidator should find missing and extra files."""
        mock_s3_handler = Mock()
        mock_s3_handler.get_paths_from_csv.return_value = {
            "all": {"file1.mp4", "file2.mp4", "file3.mp4"},
            "filtered": {"file1.mp4", "file2.mp4"},
        }
        mock_s3_handler.get_paths_from_s3.return_value = {"file1.mp4", "file4.mp4"}

        validator = FilePresenceValidator(mock_s3_handler)
        rules = {
            "file_presence": {
                "csv_filename": "test.csv",
                "csv_column_to_extract": "video_path",
                "valid_extensions": ["mp4"],
                "path_prefix": "videos/",
                "s3_sharepoint_path": "sharepoint",
                "bucket": "test-bucket",
            }
        }

        errors = validator.validate(rules)

        # file2.mp4 missing from S3, file4.mp4 extra in S3
        assert len(errors) == 2

        # Check error sources
        missing_errors = [
            e for e in errors if e.ErrorType == ErrorSource.FILE_MISSING.value
        ]
        extra_errors = [
            e for e in errors if e.ErrorType == ErrorSource.FILE_EXTRA.value
        ]
        assert len(missing_errors) == 1
        assert len(extra_errors) == 1
        assert "file2.mp4" in missing_errors[0].ErrorMessage
        assert "file4.mp4" in extra_errors[0].ErrorMessage


class TestDataValidator:
    """Tests for DataValidator orchestrator."""

    def test_validate_with_config(self):
        """DataValidator should accept ValidationConfig and return errors DataFrame."""
        from sftk.data_validator import DataValidator

        with patch("sftk.data_validator.S3Handler") as mock_s3:
            mock_s3.return_value.read_df_from_s3_csv.return_value = pd.DataFrame()

            validator = DataValidator()
            config = ValidationConfig()

            result = validator.validate_with_config(config)
            assert isinstance(result, pd.DataFrame)

    def test_clean_dataframe_extraction(self):
        """DataValidator should extract clean dataframes when enabled."""
        from sftk.data_validator import DataValidator

        test_df = pd.DataFrame(
            {"name": ["Alice", None, "Charlie", "David"], "age": [25, 30, 35, 40]}
        )

        with patch("sftk.data_validator.S3Handler") as mock_s3:
            mock_s3.return_value.read_df_from_s3_csv.return_value = test_df
            with patch(
                "sftk.data_validator.VALIDATION_RULES",
                {
                    "test_dataset": {
                        "file_name": "test.csv",
                        "required": ["name"],
                        "unique": [],
                        "info_columns": [],
                        "foreign_keys": {},
                        "relationships": [],
                    }
                },
            ):
                validator = DataValidator()
                config = ValidationConfig(
                    file_presence=False,
                    extract_clean_dataframes=True,
                )

                errors_df = validator.validate_with_config(config)
                assert len(errors_df) == 1  # Missing name in row 1

                clean_df = validator.get_clean_dataframe("test_dataset")
                assert len(clean_df) == 3
                assert list(clean_df["name"]) == ["Alice", "Charlie", "David"]

    def test_run_validation_and_export(self):
        """DataValidator should run validation and export to CSV."""
        import os
        import tempfile

        from sftk.common import ERRORS_FILENAME
        from sftk.data_validator import DataValidator

        test_df = pd.DataFrame({"name": ["Alice", None, "Bob"], "id": [1, 2, 3]})

        with patch("sftk.data_validator.S3Handler") as mock_s3:
            mock_s3.return_value.read_df_from_s3_csv.return_value = test_df
            with patch("sftk.data_validator.EXPORT_LOCAL", True):
                with tempfile.TemporaryDirectory() as temp_dir:
                    with patch("sftk.data_validator.LOCAL_DATA_FOLDER_PATH", temp_dir):
                        validator = DataValidator()

                        validator.run_validation(file_presence=False)

                        assert validator.errors_df is not None
                        csv_path = os.path.join(temp_dir, ERRORS_FILENAME)
                        assert os.path.exists(csv_path)
