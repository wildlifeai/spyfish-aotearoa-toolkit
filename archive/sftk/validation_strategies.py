"""
Data validation module.

Provides simple validation functions and a DatasetValidator class:
- validate_required: Check required columns have values
- validate_unique: Check uniqueness constraints
- validate_formats: Check format patterns
- validate_values: Check value ranges
- validate_relationships: Check column relationships
- validate_foreign_keys: Check foreign key references
- DatasetValidator: Orchestrates all validations for a single dataset
- FilePresenceValidator: Validates file presence in S3 storage
- CleanRowTracker: Tracks rows without validation errors
"""

import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Set

import pandas as pd

from sftk.common import DROP_ID_COLUMN, REPLICATE_COLUMN, SURVEY_ID_COLUMN
from sftk.utils import normalize_file_name

# =============================================================================
# Core Types
# =============================================================================


@dataclass
class ErrorChecking:
    """Data class for validation errors for dashboard display."""

    # Record identifiers (for tracing back to source data)
    SurveyID: Optional[str]
    DropID: Optional[str]

    # Error classification
    ErrorType: str  # Category of error (e.g., "Missing Required Value")
    FileName: str  # Source file where error occurred
    ColumnName: Optional[str]  # Column that has the error

    # Error details
    ErrorMessage: str  # Human-readable error description
    InvalidValue: Optional[str]  # The problematic value


class ErrorSource(Enum):
    """Enumeration of specific error sources for validation."""

    # Data validation errors - specific types
    MISSING_REQUIRED_VALUE = "Missing Required Value"
    DUPLICATE_VALUE = "Duplicate Value"
    INVALID_FORMAT = "Invalid Format"
    VALUE_OUT_OF_RANGE = "Value Out of Range"
    FOREIGN_KEY_NOT_FOUND = "Foreign Key Not Found"
    RELATIONSHIP_MISMATCH = "Relationship Mismatch"
    REPLICATE_MISMATCH = "Replicate Mismatch"
    DATASET_LOAD_ERROR = "Dataset Load Error"
    MISSING_COLUMN = "Missing Column"

    # File presence errors
    FILE_MISSING = "File Missing from S3"
    FILE_EXTRA = "Extra File in S3"


@dataclass
class ValidationConfig:
    """Configuration for validation operations."""

    file_presence: bool = False
    remove_duplicates: bool = True
    extract_clean_dataframes: bool = False
    max_errors_per_validator: int = 10000


class CleanRowTracker:
    """Tracks which rows remain clean (error-free) during validation."""

    def __init__(self):
        self.clean_row_indices: Dict[str, Set[int]] = {}

    def initialize_dataset(self, dataset_name: str, df: pd.DataFrame) -> None:
        """Initialize clean indices with all row indices for a dataset."""
        self.clean_row_indices[dataset_name] = set(df.index)

    def mark_row_as_error(self, row_index: int, dataset_name: str) -> None:
        """Remove a row index from the clean set."""
        if dataset_name in self.clean_row_indices:
            self.clean_row_indices[dataset_name].discard(row_index)

    def get_clean_indices(self, dataset_name: str) -> Set[int]:
        """Get clean row indices for a dataset."""
        return self.clean_row_indices.get(dataset_name, set())


# =============================================================================
# Error Creation Helper
# =============================================================================


def create_error(
    survey_id: Optional[str] = None,
    drop_id: Optional[str] = None,
    message: str = "",
    error_source: str = "",
    file_name: str = "",
    column_name: Optional[str] = None,
    relevant_column_value: Any = None,
) -> ErrorChecking:
    """Create an ErrorChecking object - single consolidated error factory."""
    return ErrorChecking(
        SurveyID=survey_id,
        DropID=drop_id,
        ErrorType=error_source,
        FileName=normalize_file_name(file_name),
        ColumnName=column_name,
        ErrorMessage=message,
        InvalidValue=(
            str(relevant_column_value) if relevant_column_value is not None else None
        ),
    )


# =============================================================================
# Validation Functions
# =============================================================================


def validate_unique(
    df: pd.DataFrame,
    rules: Dict[str, Any],
    tracker: Optional[CleanRowTracker] = None,
    dataset_name: Optional[str] = None,
    max_errors: int = 10000,
) -> List[ErrorChecking]:
    """Validate unique column constraints."""
    errors = []
    file_name = normalize_file_name(rules.get("file_name", ""))
    unique_cols = rules.get("unique", [])

    for col in unique_cols:
        duplicated = df[df[col].duplicated(keep=False) & df[col].notna()]
        for idx, row in duplicated.iterrows():
            if len(errors) >= max_errors:
                break
            dup_value = row[col]
            errors.append(
                create_error(
                    survey_id=row.get(SURVEY_ID_COLUMN, ""),
                    drop_id=row.get(DROP_ID_COLUMN, ""),
                    message=f"Duplicate value '{dup_value}' in unique column '{col}'",
                    error_source=ErrorSource.DUPLICATE_VALUE.value,
                    column_name=col,
                    relevant_column_value=dup_value,
                    file_name=file_name,
                )
            )
            if tracker and dataset_name:
                tracker.mark_row_as_error(idx, dataset_name)

    return errors


def validate_foreign_keys(
    df: pd.DataFrame,
    rules: Dict[str, Any],
    all_validation_rules: Dict[str, Any],
    tracker: Optional[CleanRowTracker] = None,
    dataset_name: Optional[str] = None,
    max_errors: int = 10000,
) -> List[ErrorChecking]:
    """Validate foreign key relationships."""
    errors = []
    source_file = normalize_file_name(rules.get("file_name", ""))
    foreign_keys = rules.get("foreign_keys", {})

    for target_name, fk_col in foreign_keys.items():
        target_rules = all_validation_rules.get(target_name)
        if not target_rules:
            errors.append(
                create_error(
                    message=f"Foreign key check skipped: target dataset '{target_name}' not found",
                    error_source=ErrorSource.DATASET_LOAD_ERROR.value,
                    column_name=fk_col,
                    file_name=source_file,
                )
            )
            continue

        target_df = target_rules.get("dataset", pd.DataFrame())
        if target_df.empty:
            errors.append(
                create_error(
                    message=f"Foreign key check skipped: target dataset '{target_name}' is empty",
                    error_source=ErrorSource.DATASET_LOAD_ERROR.value,
                    column_name=fk_col,
                    file_name=source_file,
                )
            )
            continue

        if fk_col not in df.columns:
            errors.append(
                create_error(
                    message=f"Foreign key column '{fk_col}' not found in source file '{source_file}'",
                    error_source=ErrorSource.MISSING_COLUMN.value,
                    column_name=fk_col,
                    file_name=source_file,
                )
            )
            continue

        if fk_col not in target_df.columns:
            errors.append(
                create_error(
                    message=f"Foreign key column '{fk_col}' not found in target dataset '{target_name}'",
                    error_source=ErrorSource.MISSING_COLUMN.value,
                    column_name=fk_col,
                    file_name=source_file,
                )
            )
            continue

        missing = df[~df[fk_col].isin(target_df[fk_col])]
        fk_file_name = normalize_file_name(target_rules.get("file_name", ""))

        for idx, row in missing.iterrows():
            if len(errors) >= max_errors:
                break
            errors.append(
                create_error(
                    survey_id=row.get(SURVEY_ID_COLUMN, ""),
                    drop_id=row.get(DROP_ID_COLUMN, ""),
                    message=f"Foreign key '{fk_col}' = '{row[fk_col]}' not found in '{fk_file_name}'",
                    error_source=ErrorSource.FOREIGN_KEY_NOT_FOUND.value,
                    column_name=fk_col,
                    relevant_column_value=row[fk_col],
                    file_name=source_file,
                )
            )
            if tracker and dataset_name:
                tracker.mark_row_as_error(idx, dataset_name)

    return errors


def validate_required(
    df: pd.DataFrame,
    rules: Dict[str, Any],
    file_name: str,
    tracker: Optional[CleanRowTracker] = None,
    dataset_name: Optional[str] = None,
) -> List[ErrorChecking]:
    """Check required columns have values (vectorized)."""
    errors = []
    required_cols = rules.get("required", [])
    info_columns = rules.get("info_columns", [])

    for col in required_cols:
        # Vectorized: find rows with missing values (NA, empty string, or None)
        is_missing = df[col].isna() | (df[col] == "")
        missing_rows = df[is_missing]

        for idx, row in missing_rows.iterrows():
            help_info = ", ".join(
                [f"{ic}: {row.get(ic, '')}" for ic in info_columns if ic in df.columns]
            )
            message = f"Missing value in required column '{col}'"
            if help_info:
                message += f" ({help_info})"
            errors.append(
                create_error(
                    survey_id=row.get(SURVEY_ID_COLUMN, ""),
                    drop_id=row.get(DROP_ID_COLUMN, ""),
                    message=message,
                    error_source=ErrorSource.MISSING_REQUIRED_VALUE.value,
                    column_name=col,
                    relevant_column_value=None,
                    file_name=file_name,
                )
            )
            if tracker and dataset_name:
                tracker.mark_row_as_error(idx, dataset_name)

    return errors


def validate_formats(
    df: pd.DataFrame,
    rules: Dict[str, Any],
    patterns: Dict[str, str],
    file_name: str,
    tracker: Optional[CleanRowTracker] = None,
    dataset_name: Optional[str] = None,
) -> List[ErrorChecking]:
    """Check format patterns (vectorized)."""
    errors = []
    format_columns = rules.get("formats", [])

    for col in format_columns:
        pattern = patterns.get(col)
        if not pattern:
            continue

        # Vectorized: filter non-empty values, then check pattern match
        non_empty_mask = df[col].notna() & (df[col] != "")
        non_empty_df = df[non_empty_mask]

        if non_empty_df.empty:
            continue

        # Vectorized regex match using str.match()
        matches = non_empty_df[col].astype(str).str.match(pattern)
        invalid_rows = non_empty_df[~matches]

        for idx, row in invalid_rows.iterrows():
            value = row[col]
            errors.append(
                create_error(
                    survey_id=row.get(SURVEY_ID_COLUMN, ""),
                    drop_id=row.get(DROP_ID_COLUMN, ""),
                    message=f"Value '{value}' does not match required format for {col}",
                    error_source=ErrorSource.INVALID_FORMAT.value,
                    column_name=col,
                    relevant_column_value=value,
                    file_name=file_name,
                )
            )
            if tracker and dataset_name:
                tracker.mark_row_as_error(idx, dataset_name)

    return errors


def validate_values(
    df: pd.DataFrame,
    rules: Dict[str, Any],
    file_name: str,
    tracker: Optional[CleanRowTracker] = None,
    dataset_name: Optional[str] = None,
) -> List[ErrorChecking]:
    """Check value ranges (vectorized)."""
    errors = []
    value_rules = rules.get("values", [])

    for value_rule in value_rules:
        col = value_rule["column"]
        rule = value_rule["rule"]
        if rule != "value_range":
            continue

        value_range = value_rule.get("range")
        if not value_range or len(value_range) != 2:
            continue

        allowed_values = value_rule.get("allowed_values", [])

        # Vectorized: convert to numeric, filter out allowed values and NA
        numeric_col = pd.to_numeric(df[col], errors="coerce")
        is_valid_numeric = numeric_col.notna()

        # Exclude allowed values
        if allowed_values:
            is_allowed = df[col].isin(allowed_values)
            is_valid_numeric = is_valid_numeric & ~is_allowed

        # Vectorized range check
        out_of_range = is_valid_numeric & (
            (numeric_col < value_range[0]) | (numeric_col > value_range[1])
        )
        invalid_rows = df[out_of_range]

        for idx, row in invalid_rows.iterrows():
            actual = row[col]
            actual_numeric = float(actual)
            errors.append(
                create_error(
                    survey_id=row.get(SURVEY_ID_COLUMN, ""),
                    drop_id=row.get(DROP_ID_COLUMN, ""),
                    message=f"{col} value {actual_numeric} is outside valid range [{value_range[0]}, {value_range[1]}]",
                    error_source=ErrorSource.VALUE_OUT_OF_RANGE.value,
                    column_name=col,
                    relevant_column_value=actual,
                    file_name=file_name,
                )
            )
            if tracker and dataset_name:
                tracker.mark_row_as_error(idx, dataset_name)

    return errors


def _is_replicate_mismatch_only(actual: str, expected: str) -> bool:
    """Check if mismatch is only in the replicate part (last 2 digits)."""
    if len(actual) != len(expected) or len(actual) < 2:
        return False
    return actual[:-2] == expected[:-2] and actual[-2:] != expected[-2:]


def validate_relationships(
    df: pd.DataFrame,
    rules: Dict[str, Any],
    file_name: str,
    tracker: Optional[CleanRowTracker] = None,
    dataset_name: Optional[str] = None,
) -> List[ErrorChecking]:
    """Check column relationships (vectorized where possible)."""
    errors = []
    relationships = rules.get("relationships", [])

    for relationship in relationships:
        col = relationship["column"]
        rule = relationship["rule"]
        if rule != "equals":
            continue

        template = relationship.get("template", "")
        allowed_values = relationship.get("allowed_values", [])
        is_null_allowed = relationship.get("allow_null")

        # Pre-filter: exclude allowed values and nulls (if allowed)
        needs_check = pd.Series(True, index=df.index)
        if allowed_values:
            needs_check = needs_check & ~df[col].isin(allowed_values)
        if is_null_allowed:
            needs_check = needs_check & df[col].notna()

        rows_to_check = df[needs_check]

        for idx, row in rows_to_check.iterrows():
            actual = row[col]
            survey_id = row.get(SURVEY_ID_COLUMN, "")
            drop_id = row.get(DROP_ID_COLUMN, "")

            try:
                expected = template.format(**row)
            except KeyError as e:
                errors.append(
                    create_error(
                        survey_id=survey_id,
                        drop_id=drop_id,
                        message=f"Missing column {col} for relationship template: {str(e)}",
                        error_source=ErrorSource.MISSING_COLUMN.value,
                        column_name=col,
                        file_name=file_name,
                    )
                )
                if tracker and dataset_name:
                    tracker.mark_row_as_error(idx, dataset_name)
                continue

            if str(actual) != str(expected):
                if col == DROP_ID_COLUMN and _is_replicate_mismatch_only(
                    str(actual), str(expected)
                ):
                    message = f"{REPLICATE_COLUMN} mismatch: {col} should end with '{str(expected)[-2:]}' but ends with '{str(actual)[-2:]}'. Full {col} should be '{expected}', but is '{actual}'"
                    error_source = ErrorSource.REPLICATE_MISMATCH.value
                else:
                    message = f"{col} should be '{expected}', but is '{actual}'"
                    error_source = ErrorSource.RELATIONSHIP_MISMATCH.value
                errors.append(
                    create_error(
                        survey_id=survey_id,
                        drop_id=drop_id,
                        message=message,
                        error_source=error_source,
                        column_name=col,
                        relevant_column_value=actual,
                        file_name=file_name,
                    )
                )
                if tracker and dataset_name:
                    tracker.mark_row_as_error(idx, dataset_name)

    return errors


# =============================================================================
# File Presence Validator (needs S3 handler state)
# =============================================================================


class FilePresenceValidator:
    """Validates file presence in S3 storage."""

    def __init__(self, s3_handler, max_errors: int = 10000):
        self.s3_handler = s3_handler
        self.max_errors = max_errors

    def _extract_survey_drop_from_path(
        self, file_path: str
    ) -> tuple[Optional[str], Optional[str]]:
        """Extract SurveyID and DropID from a file path (media/{SurveyID}/{DropID}/...)."""
        try:
            parts = file_path.split("/")
            if len(parts) >= 3 and parts[0] == "media":
                return parts[1] if parts[1] else None, parts[2] if parts[2] else None
        except (IndexError, TypeError) as e:
            logging.warning(
                f"Could not extract survey/drop ID from path '{file_path}': {e}"
            )
        return None, None

    def validate(self, rules: Dict[str, Any]) -> List[ErrorChecking]:
        """Validate file presence between CSV references and S3 storage."""
        errors: List[ErrorChecking] = []
        file_presence_config = rules.get("file_presence", {})
        if not file_presence_config:
            return errors

        csv_filename = file_presence_config.get("csv_filename")
        try:
            _, missing_files, extra_files = self.get_file_differences(rules)

            for file_path in missing_files:
                if len(errors) >= self.max_errors:
                    break
                survey_id, drop_id = self._extract_survey_drop_from_path(file_path)
                errors.append(
                    create_error(
                        survey_id=survey_id,
                        drop_id=drop_id,
                        message=f"File {file_path} not found in AWS, but found in {csv_filename}",
                        relevant_column_value=file_path,
                        error_source=ErrorSource.FILE_MISSING.value,
                        file_name=csv_filename,
                    )
                )

            for file_path in extra_files:
                if len(errors) >= self.max_errors:
                    break
                survey_id, drop_id = self._extract_survey_drop_from_path(file_path)
                errors.append(
                    create_error(
                        survey_id=survey_id,
                        drop_id=drop_id,
                        message=f"File {file_path} found in AWS but not in {csv_filename}",
                        relevant_column_value=file_path,
                        error_source=ErrorSource.FILE_EXTRA.value,
                        file_name=csv_filename,
                    )
                )
        except Exception as e:
            errors.append(
                create_error(
                    message=f"File presence validation failed: {str(e)}",
                    file_name=csv_filename,
                    error_source=ErrorSource.DATASET_LOAD_ERROR.value,
                )
            )
        return errors

    def get_file_differences(self, rules: Dict[str, Any]) -> tuple[set, set, set]:
        """Get file differences between CSV references and S3 storage."""
        file_presence_config = rules.get("file_presence", {})
        if not file_presence_config:
            return set(), set(), set()

        csv_filename = file_presence_config.get("csv_filename")
        csv_column_to_extract = file_presence_config.get("csv_column_to_extract")
        valid_extensions = file_presence_config.get("valid_extensions", [])
        path_prefix = file_presence_config.get("path_prefix", "")
        column_filter = file_presence_config.get("column_filter")
        column_value = file_presence_config.get("column_value")
        s3_sharepoint_path = file_presence_config.get("s3_sharepoint_path", "")
        bucket = file_presence_config.get("bucket")

        if not all([csv_filename, csv_column_to_extract, bucket]):
            raise ValueError(
                "File presence validation requires csv_filename, csv_column_to_extract, and bucket"
            )

        csv_s3_path = os.path.join(s3_sharepoint_path, csv_filename)
        csv_paths_result = self.s3_handler.get_paths_from_csv(
            csv_s3_path=csv_s3_path,
            csv_column=csv_column_to_extract,
            column_filter=column_filter,
            column_value=column_value,
        )
        csv_filepaths_all = csv_paths_result["all"]
        csv_filepaths_filtered = csv_paths_result["filtered"]

        s3_video_filepaths = self.s3_handler.get_paths_from_s3(
            path_prefix=path_prefix, valid_extensions=valid_extensions
        )

        missing_files = csv_filepaths_filtered - s3_video_filepaths
        extra_files = s3_video_filepaths - csv_filepaths_all

        return s3_video_filepaths, missing_files, extra_files


# =============================================================================
# Dataset Validator (orchestrates all validations for one dataset)
# =============================================================================


class DatasetValidator:
    """Validates a single dataset with all rules."""

    def __init__(
        self,
        rules: Dict[str, Any],
        patterns: Dict[str, str],
        all_validation_rules: Dict[str, Any],
        tracker: Optional[CleanRowTracker] = None,
    ):
        self.rules = rules
        self.patterns = patterns
        self.all_validation_rules = all_validation_rules
        self.tracker = tracker
        self.file_name = normalize_file_name(rules.get("file_name", ""))

    def validate(self, df: pd.DataFrame, dataset_name: str) -> List[ErrorChecking]:
        """Validate the dataset with all rules (vectorized)."""
        # Check for missing columns first - stop if any are missing
        missing_col_errors = self._check_missing_columns(df)
        if missing_col_errors:
            return missing_col_errors

        errors = []

        # Dataset-level validations
        errors.extend(validate_unique(df, self.rules, self.tracker, dataset_name))
        errors.extend(
            validate_foreign_keys(
                df, self.rules, self.all_validation_rules, self.tracker, dataset_name
            )
        )

        # Vectorized validations
        errors.extend(
            validate_required(
                df, self.rules, self.file_name, self.tracker, dataset_name
            )
        )
        errors.extend(
            validate_formats(
                df,
                self.rules,
                self.patterns,
                self.file_name,
                self.tracker,
                dataset_name,
            )
        )
        errors.extend(
            validate_values(df, self.rules, self.file_name, self.tracker, dataset_name)
        )
        errors.extend(
            validate_relationships(
                df, self.rules, self.file_name, self.tracker, dataset_name
            )
        )

        return errors

    def _check_missing_columns(self, df: pd.DataFrame) -> List[ErrorChecking]:
        """Check if all columns needed for validation exist in the dataframe.

        Consolidates column existence checks for: required, formats, unique,
        values, and relationships validations. Foreign key column checks are
        handled separately in validate_foreign_keys due to cross-dataset logic.
        """
        errors = []
        df_columns = set(df.columns)

        # Collect all columns that need to exist
        columns_to_check = []
        columns_to_check.extend(self.rules.get("required", []))
        columns_to_check.extend(self.rules.get("formats", []))
        columns_to_check.extend(self.rules.get("unique", []))
        for value_rule in self.rules.get("values", []):
            columns_to_check.append(value_rule["column"])
        for relationship in self.rules.get("relationships", []):
            columns_to_check.append(relationship["column"])

        # Check each unique column
        for col in set(columns_to_check):
            if col not in df_columns:
                errors.append(
                    create_error(
                        message=f"Column '{col}' not found in dataset",
                        error_source=ErrorSource.MISSING_COLUMN.value,
                        column_name=col,
                        file_name=self.file_name,
                    )
                )

        return errors
