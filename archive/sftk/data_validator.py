"""
Data validation orchestrator module.

This module contains the DataValidator class that orchestrates comprehensive
data validation using the simplified validation functions and DatasetValidator.
"""

import copy
import logging
import os
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from sftk.common import (
    ERRORS_FILENAME,
    EXPORT_LOCAL,
    EXTRA_FILES_FILENAME,
    FILE_PRESENCE_RULES,
    LOCAL_DATA_FOLDER_PATH,
    MISSING_FILES_FILENAME,
    S3_ERRORS_CSV,
    S3_EXTRA_FILES,
    S3_MISSING_FILES,
    VALIDATION_PATTERNS,
    VALIDATION_RULES,
)
from sftk.s3_handler import S3FileNotFoundError, S3Handler
from sftk.utils import (
    convert_int_num_columns_to_int,
    normalize_file_name,
    write_data_to_file,
)
from sftk.validation_strategies import (
    CleanRowTracker,
    DatasetValidator,
    ErrorChecking,
    ErrorSource,
    FilePresenceValidator,
    ValidationConfig,
    create_error,
)


class DataValidator:
    """
    Main data validation orchestrator.

    Coordinates comprehensive data validation by managing validation rules,
    datasets, and running validations. Uses DatasetValidator for each dataset.
    """

    def __init__(self):
        """Initialize the DataValidator with default settings and validation rules."""
        self.errors = []
        self.errors_df = None
        self.patterns = VALIDATION_PATTERNS
        self.s3_handler = S3Handler()
        self.validation_rules = self._get_validation_rules()
        self.clean_row_tracker = None
        self.file_presence_validator = FilePresenceValidator(self.s3_handler)
        # Local folder path for exports (only used when EXPORT_LOCAL=True)
        self.local_folder_path = LOCAL_DATA_FOLDER_PATH
        # Cache for file differences (to avoid duplicate S3 calls)
        self._file_differences_cache = None

    def run_validation(
        self,
        file_presence: bool = False,
        remove_duplicates: bool = True,
        extract_clean_dataframes: bool = False,
    ):
        """Main function to run data validation."""
        logging.info("Error validation started")
        config = ValidationConfig(
            file_presence=file_presence,
            remove_duplicates=remove_duplicates,
            extract_clean_dataframes=extract_clean_dataframes,
        )

        result_df = self.validate_with_config(config)
        logging.info(f"Error validation completed, {result_df.shape[0]} errors found")
        self.export_to_csv()

        if config.extract_clean_dataframes:
            self.export_clean_dataframes_to_csv()
            summary = self.get_clean_summary()
            logging.info(f"Data info: {summary}")

        if config.file_presence:
            self.export_file_differences()

        logging.info("Error validation process completed, files created/uploaded.")

    def _get_validation_rules(self) -> Dict[str, Any]:
        """Load validation rules with their associated reference datasets from S3."""
        validation_rules = copy.deepcopy(VALIDATION_RULES)
        for dataset_name, rule_set in validation_rules.items():
            try:
                df = self.s3_handler.read_df_from_s3_csv(rule_set["file_name"])
            except S3FileNotFoundError as e:
                error = create_error(
                    message=f"Dataset '{dataset_name}' could not be downloaded from '{rule_set['file_name']}', error: {e}",
                    error_source=ErrorSource.DATASET_LOAD_ERROR.value,
                    file_name=dataset_name,
                )
                self.errors.append(error)
                df = pd.DataFrame()
            validation_rules[dataset_name]["dataset"] = df
        return validation_rules

    def _process_datasets(self) -> None:
        """Process each dataset with all validations."""
        for dataset_name, rules in self.validation_rules.items():
            file_name = normalize_file_name(rules.get("file_name", ""))
            df = rules.get("dataset", pd.DataFrame())

            if df.empty:
                error = create_error(
                    message="Dataset could not be loaded.",
                    error_source=ErrorSource.DATASET_LOAD_ERROR.value,
                    file_name=file_name,
                )
                self.errors.append(error)
                continue

            # Convert numerical values to integers where possible
            df = convert_int_num_columns_to_int(df)

            # Initialize clean indices for this dataset if tracker is enabled
            if self.clean_row_tracker:
                self.clean_row_tracker.initialize_dataset(dataset_name, df)

            # Run validation using DatasetValidator
            validator = DatasetValidator(
                rules=rules,
                patterns=self.patterns,
                all_validation_rules=self.validation_rules,
                tracker=self.clean_row_tracker,
            )
            dataset_errors = validator.validate(df, dataset_name)
            self.errors.extend(dataset_errors)

    def validate_with_config(self, config: ValidationConfig) -> pd.DataFrame:
        """Orchestrate comprehensive data validation using ValidationConfig."""
        # Initialize clean row tracker if requested
        if config.extract_clean_dataframes:
            self.clean_row_tracker = CleanRowTracker()
        else:
            self.clean_row_tracker = None

        # Execute validation on all datasets
        self._process_datasets()

        if config.file_presence:
            file_presence_errors = self.file_presence_validator.validate(
                FILE_PRESENCE_RULES
            )
            self.errors.extend(file_presence_errors)

        # Export and return results
        self._export_errors_from_list_to_df(config.remove_duplicates)
        return self.errors_df

    def _export_errors_from_list_to_df(self, remove_duplicates):
        """
        Convert the errors list to a pandas DataFrame and optionally remove duplicates.

        Transforms the list of ErrorChecking objects into a DataFrame for easier
        analysis and export. Clears the errors list after conversion.

        Args:
            remove_duplicates (bool): Whether to remove duplicate error entries
                from the resulting DataFrame

        Side Effects:
            - Sets self.errors_df to a DataFrame containing all error data
            - Calls _deduplicate_errors() if remove_duplicates is True
            - Clears self.errors list after conversion

        Note:
            Uses the __dict__ attribute of ErrorChecking objects to create
            DataFrame columns matching the dataclass fields.
        """
        self.errors_df = pd.DataFrame([e.__dict__ for e in self.errors])

        if remove_duplicates:
            self._deduplicate_errors()
        self.errors = []

    def _deduplicate_errors(self):
        """
        Remove duplicate error entries from the errors DataFrame.

        Uses all fields from the ErrorChecking dataclass to identify and remove
        duplicate error entries. This helps reduce noise in validation reports
        when the same error occurs multiple times.

        Side Effects:
            - Modifies self.errors_df by removing duplicate rows
            - Resets DataFrame index after deduplication
            - Does nothing if errors_df is empty

        Note:
            Uses all ErrorChecking dataclass fields as the subset for
            duplicate detection, ensuring only truly identical errors are removed.
        """
        if self.errors_df.empty:
            return

        key_cols = list(ErrorChecking.__dataclass_fields__.keys())
        self.errors_df = self.errors_df.drop_duplicates(
            subset=key_cols, ignore_index=True
        )

    def export_to_csv(self):
        """
        Export validation errors to a CSV file or S3 bucket.

        Saves the errors DataFrame to a CSV file for analysis and reporting.
        The CSV will contain all validation errors with their associated metadata.

        Side Effects:
            - Creates a CSV file with the specified name
            - Logs the export operation

        Note:
            The CSV is exported without the DataFrame index to keep the
            output clean and focused on the error data.
        """
        if EXPORT_LOCAL:
            path = Path(self.local_folder_path) / ERRORS_FILENAME
            path.parent.mkdir(parents=True, exist_ok=True)
            self.errors_df.to_csv(path, index=False)
            logging.info(f"Errors exported locally to csv file {path}.")
        else:
            self.s3_handler.upload_updated_df_to_s3(
                df=self.errors_df,
                key=S3_ERRORS_CSV,
                filename=ERRORS_FILENAME,
                keep_df_index=False,
            )
            logging.info(f"Errors exported to S3: {S3_ERRORS_CSV}")

    def export_clean_dataframes_to_csv(self) -> None:
        """
        Export all clean dataframes to CSV files (local only for now).

        Exports each clean dataframe (rows with no validation errors) to separate
        CSV files. Files are named with the pattern "clean_{dataset_name}.csv".

        Raises:
            ValueError: If no clean dataframes are available (clean row tracking not enabled)

        Note:
            - Only available when extract_clean_dataframes was enabled in ValidationConfig
            - CSV files are exported without DataFrame index
            - Empty dataframes are skipped
        """
        # TODO: Add S3 export for clean dataframes (will go to kso folder eventually)
        if not EXPORT_LOCAL:
            logging.warning(
                "Clean dataframe export to S3 not yet implemented. Skipping."
            )
            return

        if not self.clean_row_tracker:
            raise ValueError(
                "Clean dataframes not available. Enable extract_clean_dataframes "
                "in ValidationConfig before running validation."
            )

        clean_dataframes = self.get_all_clean_dataframes()
        if not clean_dataframes:
            logging.info("No clean dataframes to export.")
            return

        for dataset_name, clean_df in clean_dataframes.items():
            if clean_df.empty:
                logging.info(f"Skipping empty clean dataframe for {dataset_name}")
                continue

            current_filename = f"clean_{dataset_name}.csv"
            output_path = os.path.join(self.local_folder_path, current_filename)
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            clean_df.to_csv(output_path, index=False)
            logging.info(f"Clean {dataset_name} dataframe exported to {output_path}")

        logging.info(
            f"Exported {len(clean_dataframes)} clean dataframes to {self.local_folder_path}"
        )

    def get_file_differences(
        self,
        file_presence_rules: Dict[str, Any] = FILE_PRESENCE_RULES,
    ) -> tuple:
        """Get file differences, using cache if available.

        Args:
            file_presence_rules: Rules defining which files to check.

        Returns:
            tuple: (all_files_set, missing_files_set, extra_files_set)
        """
        if self._file_differences_cache is None:
            self._file_differences_cache = (
                self.file_presence_validator.get_file_differences(file_presence_rules)
            )
        return self._file_differences_cache

    def export_file_differences(
        self,
        file_presence_rules: Dict[str, Any] = FILE_PRESENCE_RULES,
    ) -> tuple:
        """Export file differences to separate text files.

        Uses cached file differences if available to avoid duplicate S3 calls.

        Args:
            file_presence_rules: Rules defining which files to check.

        Returns:
            tuple: (all_files_set, missing_files_set, extra_files_set)
        """
        try:
            all_files_set, missing_files_set, extra_files_set = (
                self.get_file_differences(file_presence_rules)
            )
            missing_files_data = "\n".join(sorted(missing_files_set))
            extra_files_data = "\n".join(sorted(extra_files_set))

            if EXPORT_LOCAL:
                missing_path = os.path.join(
                    self.local_folder_path, MISSING_FILES_FILENAME
                )
                extra_path = os.path.join(self.local_folder_path, EXTRA_FILES_FILENAME)
                write_data_to_file(missing_files_data, missing_path)
                write_data_to_file(extra_files_data, extra_path)
            else:
                self.s3_handler.upload_data_to_s3(missing_files_data, S3_MISSING_FILES)
                self.s3_handler.upload_data_to_s3(extra_files_data, S3_EXTRA_FILES)

            logging.info(
                f"File differences exported: {len(missing_files_set)} missing, {len(extra_files_set)} extra"
            )
        except Exception as e:
            logging.error(f"Failed to export file differences: {e}")
            raise

        return all_files_set, missing_files_set, extra_files_set

    def get_clean_dataframe(self, dataset_name: str) -> pd.DataFrame:
        """
        Get clean dataframe for a specific dataset.

        Args:
            dataset_name: Name of the dataset to get clean rows for

        Returns:
            DataFrame containing only rows with no validation errors,
            empty DataFrame if no tracker or dataset not found
        """
        if not self.clean_row_tracker:
            return pd.DataFrame()

        clean_indices = self.clean_row_tracker.get_clean_indices(dataset_name)
        if not clean_indices:
            return pd.DataFrame()

        # Get the original dataset from validation rules
        dataset_rules = self.validation_rules.get(dataset_name)
        if not dataset_rules or "dataset" not in dataset_rules:
            return pd.DataFrame()

        original_df = dataset_rules["dataset"]
        if original_df.empty:
            return pd.DataFrame()

        # Filter to clean rows only
        return original_df.loc[list(clean_indices)].copy()

    def get_all_clean_dataframes(self) -> Dict[str, pd.DataFrame]:
        """
        Get all clean dataframes.

        Returns:
            Dictionary mapping dataset names to clean DataFrames,
            empty dict if no tracker is initialized
        """
        if not self.clean_row_tracker:
            return {}

        clean_dataframes = {}
        for dataset_name in self.clean_row_tracker.clean_row_indices.keys():
            clean_df = self.get_clean_dataframe(dataset_name)
            if not clean_df.empty:
                clean_dataframes[dataset_name] = clean_df

        return clean_dataframes

    def get_clean_summary(self) -> Dict[str, Any]:
        """
        Get summary of clean vs error rows.

        Returns:
            Dictionary with summary statistics for all datasets
        """
        if not self.clean_row_tracker:
            return {
                "message": "Clean dataframe extraction was not enabled during validation",
                "datasets": {},
            }

        summary: Dict[str, Any] = {"datasets": {}}

        for dataset_name in self.clean_row_tracker.clean_row_indices.keys():
            dataset_rules = self.validation_rules.get(dataset_name)
            if dataset_rules and "dataset" in dataset_rules:
                original_df = dataset_rules["dataset"]
                clean_indices = self.clean_row_tracker.get_clean_indices(dataset_name)

                total_rows = len(original_df)
                clean_rows = len(clean_indices)
                error_rows = total_rows - clean_rows

                summary["datasets"][dataset_name] = {
                    "total_rows": total_rows,
                    "clean_rows": clean_rows,
                    "error_rows": error_rows,
                    "clean_percentage": (
                        (clean_rows / total_rows * 100) if total_rows > 0 else 0
                    ),
                    "error_percentage": (
                        (error_rows / total_rows * 100) if total_rows > 0 else 0
                    ),
                }

        return summary
