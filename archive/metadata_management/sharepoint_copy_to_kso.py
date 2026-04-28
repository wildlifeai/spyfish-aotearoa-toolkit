"""
This module processes and transforms SharePoint CSV data for compatibility
with the KSO software, storing the results in S3.

Steps:
    - Connect to S3.
    - Download csv files with SharePoint- and KSO-formatted data
    - Check new or different data from latest sharepoint copy
    - Update KSO-formatted data with latest sharepoint copy data
"""

import logging
import sys
from typing import Optional

import pandas as pd

# These get used to retrieve the paths sharepoint and kso csv files from S3
# for the various keywords (survey, site, movie, species or test) using getattr
import sftk.common

# Import centralized logging configuration
from sftk import log_config  # noqa: F401
from sftk.common import DEV_MODE
from sftk.s3_handler import S3FileConfig, S3FileNotFoundError, S3Handler
from sftk.utils import EnvironmentVariableError, temp_file_manager


def process_s3_files(
    s3_handler, keywords: list[str]
) -> dict[str, tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]]:
    """
    Downloads and reads KSO and SharePoint CSV files from S3 for given keywords.

    Args:
        s3_handler: The S3 handler object. TODO if object, no need to pass this around
        keywords: A list of keywords to process.

    Returns:
        A dictionary mapping keywords to tuples of (KSO DataFrame, SharePoint DataFrame).
    """
    results = {}

    for keyword in keywords:
        kso_df = None
        sharepoint_df = None

        try:
            config = S3FileConfig.from_keyword(keyword)
            # TODO error management if no var, similar to get_env_var to raise EnvironmentVariableError
            with temp_file_manager([config.kso_filename, config.sharepoint_filename]):
                try:
                    # Download and read KSO file
                    kso_df = s3_handler.download_and_read_s3_file(
                        key=getattr(sftk.common, config.kso_env_var),
                        filename=config.kso_filename,
                    )

                    # Download and read Sharepoint file
                    sharepoint_df = s3_handler.download_and_read_s3_file(
                        key=getattr(sftk.common, config.sharepoint_env_var),
                        filename=config.sharepoint_filename,
                    )

                except S3FileNotFoundError as e:
                    msg = f"CSV file not found in S3 for keyword '{keyword}': {e}"
                    logging.warning(msg)
                    raise S3FileNotFoundError(msg) from e

        # TODO fix these errors and exceptions
        except EnvironmentVariableError as e:
            msg = f"Missing environment variable for keyword '{keyword}': {e}"
            logging.error(msg)
            raise EnvironmentVariableError(msg) from e

        except Exception as e:
            msg = f"Unexpected error while processing keyword '{keyword}': {e}"
            logging.error(msg)
            raise RuntimeError(msg) from e

        results[keyword] = (kso_df, sharepoint_df)

    return results


def validate_and_filter_dfs(
    movie_sharepoint_df: pd.DataFrame, site_sharepoint_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Check for missing values in specified columns of movie and site DataFrames, log the missing values, and filter them out.

    Args:
        movie_sharepoint_df: DataFrame containing movie SharePoint data.
        site_sharepoint_df: DataFrame containing site SharePoint data.

    Returns:
        Tuple of filtered movie_sharepoint_df and site_sharepoint_df.
    """
    try:
        # Columns to validate for each DataFrame
        movie_required_columns = [
            "DropID",
            "SurveyID",
            "SiteID",
            "Latitude",
            "Longitude",
        ]
        site_required_columns = ["SiteID", "LinkToMarineReserve"]

        # Validate movie DataFrame
        if movie_sharepoint_df is not None:
            missing_movie = movie_sharepoint_df[
                movie_sharepoint_df[movie_required_columns].isna().any(axis=1)
            ]

            if not missing_movie.empty:
                logging.info(
                    f"The following {len(missing_movie)} rows with "
                    f"missing values in the {movie_required_columns} "
                    "required columns of the movie SharePoint list copy"
                    " will be dropped."
                )

                # Filter out rows with missing values
                movie_sharepoint_df = movie_sharepoint_df.dropna(
                    subset=movie_required_columns
                )

            else:
                logging.info("No missing values found for movie SharePoint list")
        # Validate site DataFrame
        if site_sharepoint_df is not None:
            missing_site = site_sharepoint_df[
                site_sharepoint_df[site_required_columns].isna().any(axis=1)
            ]

            if not missing_site.empty:
                logging.info(
                    f"Found the following {len(missing_site)} rows with "
                    f"missing values in the {site_required_columns} "
                    "required columns of the site SharePoint list copy."
                )
                # Filter out rows with missing values
                site_sharepoint_df = site_sharepoint_df.dropna(
                    subset=site_required_columns
                )
            else:
                logging.info("No missing values found for site SharePoint list")

        return movie_sharepoint_df, site_sharepoint_df

    except Exception as e:
        logging.error(f"Error occurred during DataFrame validation: {e}", exc_info=True)
        raise


def standarise_sharepoint_to_kso(
    results: dict[str, tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]],
) -> dict[str, tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]]:
    """
    Standardise and integrate data from SharePoint copies to align with KSO data structure.

    This function combines latitude and longitude columns from the `movie` SharePoint DataFrame
    into the `site` SharePoint DataFrame based on the `SiteID` column.

    Args:
        results: Dictionary containing tuples of (kso_df, sharepoint_df) for each keyword.

    Returns:
        Updated dictionary with standardised SharePoint DataFrames.
    """
    try:
        # Extract relevant DataFrames
        movie_sharepoint_df = results.get("movie", (None, None))[1]
        site_sharepoint_df = results.get("site", (None, None))[1]
        survey_sharepoint_df = results.get("survey", (None, None))[1]

        if movie_sharepoint_df is None or site_sharepoint_df is None:
            logging.warning(
                "Movie or Site SharePoint DataFrame is missing. Skipping standardization."
            )
            return results

        logging.info("Checking for potential missing values from movies and sites.")
        # Filter out empty rows for critical values
        movie_sharepoint_df, site_sharepoint_df = validate_and_filter_dfs(
            movie_sharepoint_df, site_sharepoint_df
        )

        logging.info(
            "Overwriting site coordinates from site sharepoint copy with actual drop"
            " coordinates from deployment sharepoint copy."
        )

        # Ensure the relevant columns exist
        if not {"SiteID", "Latitude", "Longitude"}.issubset(
            movie_sharepoint_df.columns
        ):
            logging.warning(
                "Required columns ('SiteID', 'Latitude', 'Longitude')\
                are missing in movie DataFrame."
            )
            return results
        if "SiteID" not in site_sharepoint_df.columns:
            logging.warning("Required column 'SiteID' is missing in site DataFrame.")
            return results
        # Drop latitude and longitude from site DataFrame to avoid duplicates
        site_sharepoint_df = site_sharepoint_df.drop(
            columns=["TargetedLatitude", "TargetedLongitude"]
        )

        # Rename SiteID from movie and site DataFrame to avoid duplicates
        movie_sharepoint_df, site_sharepoint_df = (
            df.rename(columns={"SiteID": "orig_SiteID"})
            for df in (movie_sharepoint_df, site_sharepoint_df)
        )

        # TODO why is this getting the year/Site ID out? is this necessary now that we have dropIDs
        # Extract the year from Survey
        movie_sharepoint_df["year"] = movie_sharepoint_df["SurveyID"].str[6:8]

        # Create new SiteID on movie DataFrame based on orig SiteID and year of the survey
        movie_sharepoint_df["SiteID"] = (
            movie_sharepoint_df["orig_SiteID"].astype(str)
            + "_"
            + movie_sharepoint_df["year"]
        )

        # Merge latitude and longitude into site DataFrame
        lat_long_df = movie_sharepoint_df[
            ["orig_SiteID", "SiteID", "Latitude", "Longitude"]
        ].drop_duplicates(subset="SiteID")
        site_sharepoint_df = site_sharepoint_df.merge(
            lat_long_df, on="orig_SiteID", how="right"
        )

        # Drop latitude and longitude from site DataFrame to avoid duplicates
        site_sharepoint_df = site_sharepoint_df.drop(columns=["orig_SiteID"])
        movie_sharepoint_df = movie_sharepoint_df.drop(
            columns=["orig_SiteID", "year", "Longitude", "Latitude"]
        )

        try:
            survey_sharepoint_df = survey_sharepoint_df.drop(
                columns=["LinkToMarineReserve"]
            )
        except KeyError:
            logging.warning(
                "Column 'LinkToMarineReserve' not found in survey_sharepoint_df"
            )

        # Update the results dictionary with the modified site DataFrame
        results["site"] = (results["site"][0], site_sharepoint_df)
        results["movie"] = (results["movie"][0], movie_sharepoint_df)
        results["survey"] = (results["survey"][0], survey_sharepoint_df)

        logging.info("Movie/BUV Drop coordinates have been successfully updated.")
        return results

    except Exception as e:
        logging.error(
            "Error occurred in standarise_sharepoint_to_kso: %s", str(e), exc_info=True
        )
        raise e


def validate_dataframes(
    kso_df: pd.DataFrame, sharepoint_df: pd.DataFrame, unique_id_column: str
) -> list:
    """
    Validate input DataFrames and the presence and uniqueness of the specified identifier column.

    Args:
        kso_df: KSO DataFrame
        sharepoint_df: Sharepoint DataFrame
        unique_id_column: Name of the column used as a unique identifier for data records.

    Returns:
        A list of validation errors encountered, empty if no issues found.
    """
    validation_log = []

    # Check for None DataFrames
    if kso_df is None or sharepoint_df is None:
        raise ValueError("Cannot compare DataFrames: One or both DataFrames are None")

    # Validate unique identifier column presence
    if unique_id_column not in kso_df.columns:
        validation_log.append(f"Column '{unique_id_column}' is missing in kso_df.")

    if unique_id_column not in sharepoint_df.columns:
        validation_log.append(
            f"Column '{unique_id_column}' is missing in sharepoint_df."
        )

    # Check for duplicates in unique identifier column
    if kso_df[unique_id_column].duplicated().any():
        validation_log.append(
            f"kso_df has duplicate values in the unique identifier column {unique_id_column}."
        )

    if sharepoint_df[unique_id_column].duplicated().any():
        validation_log.append(
            f"sharepoint_df has duplicate values in the unique identifier column {unique_id_column}."
        )

    return validation_log


def align_dataframe_columns(kso_df: pd.DataFrame, sharepoint_df: pd.DataFrame) -> None:
    """
    Align and preprocess DataFrame columns by identifying and handling missing or extra columns.

    Args:
        kso_df: KSO DataFrame to modify.
        sharepoint_df: Sharepoint DataFrame to modify.
    """

    # Identify and handle column mismatches
    missing_columns = set(kso_df.columns) - set(sharepoint_df.columns)
    extra_columns = set(sharepoint_df.columns) - set(kso_df.columns)

    # Handle extra columns
    if extra_columns:
        logging.info(
            f"Dropping extra columns from SharePoint DataFrame: {extra_columns}"
        )
        sharepoint_df.drop(columns=list(extra_columns), inplace=True)

    # Handle missing columns
    if missing_columns:
        logging.info(
            f"Adding missing columns to SharePoint DataFrame: {missing_columns}"
        )
        for col in missing_columns:
            sharepoint_df[col] = None


def compare_and_update_dataframes(
    kso_df: pd.DataFrame,
    sharepoint_df: pd.DataFrame,
    s3_handler: S3Handler,
    keyword: str,
) -> pd.DataFrame:
    """
    Compare KSO and Sharepoint DataFrames, log differences, and update the KSO DataFrame
    based on the Sharepoint data. Uploads the updated KSO DataFrame to S3 if changes were made.

    Args:
        kso_df: DataFrame containing KSO data.
        sharepoint_df: DataFrame containing Sharepoint data.
        s3_handler: S3Handler instance.
        keyword: String identifier for the type of data (e.g., "survey", "site").

    """

    # Configuration for different data types
    keyword_lookup = {
        "survey": "SurveyID",
        "site": "SiteID",
        "movie": "DropID",
        "species": "species_id",
        "test": "SurveyID",
    }

    column_mappings = {
        "survey": {},
        "site": {},
        "movie": {"ID": "id"},
        # TODO use aphiaID rather than DOC_TaxonID as unique identifier?
        "species": {"DOC_TaxonID": "species_id"},
        "test": {},
    }

    # Validate input
    if keyword not in keyword_lookup or keyword not in column_mappings:
        raise ValueError(f"Unknown keyword: {keyword}")

    unique_id_column = keyword_lookup[keyword]

    # Rename the column names in sharepoint_df based on the dictionary
    sharepoint_df.rename(columns=column_mappings[keyword], inplace=True)

    # If keyword is "site", add a new column with incremental IDs
    if keyword == "site":
        sharepoint_df["schema_site_id"] = range(1, len(sharepoint_df) + 1)

    # Log initial DataFrame information
    logging.info("Initial DataFrame sizes:")
    logging.info(f"KSO DataFrame: {kso_df.shape[0]} rows")
    logging.info(f"Sharepoint DataFrame: {sharepoint_df.shape[0]} rows")

    # Validate DataFrames
    validation_issues = validate_dataframes(kso_df, sharepoint_df, unique_id_column)
    if validation_issues:
        kso_df = kso_df[~kso_df[unique_id_column].duplicated(keep=False)].copy()
        sharepoint_df = sharepoint_df[
            ~sharepoint_df[unique_id_column].duplicated(keep=False)
        ].copy()
        logging.error("Validation issues found:")
        for issue in validation_issues:
            logging.error("- %s", issue)

    # Align columns
    align_dataframe_columns(kso_df, sharepoint_df)

    # Set unique identifier as index
    kso_df.set_index(unique_id_column, inplace=True)
    sharepoint_df.set_index(unique_id_column, inplace=True)

    # Find rows in sharepoint_df that are not in kso_df
    new_rows = sharepoint_df.loc[~sharepoint_df.index.isin(kso_df.index)]

    if not new_rows.empty:
        logging.info(f"Adding {len(new_rows)} new rows from Sharepoint")
        kso_df = pd.concat([kso_df, new_rows])

    # Find rows in kso_df that are not in sharepoint_df
    missing_rows = kso_df.loc[~kso_df.index.isin(sharepoint_df.index)]

    if not missing_rows.empty:
        logging.info(
            f"Temporarily adding {len(new_rows)} new rows from kso_df. Consider updating the Sharepoint list"
        )
        sharepoint_df = pd.concat([sharepoint_df, missing_rows])

    # Ensure consistent column set
    common_columns = list(set(kso_df.columns) & set(sharepoint_df.columns))

    # Create a copy of KSO DataFrame to modify
    updated_kso_df = kso_df.copy()

    # Prepare DataFrames for comparison
    # 1. Use only common columns
    kso_compare = kso_df[common_columns]
    sharepoint_compare = sharepoint_df[common_columns]

    # 2. Combine indexes to create a unified index
    all_indexes = sorted(set(kso_df.index) | set(sharepoint_df.index))

    # 3. Reindex both DataFrames with the combined index
    kso_reindexed = kso_compare.reindex(all_indexes)
    sharepoint_reindexed = sharepoint_compare.reindex(all_indexes)

    dataframe_changed = False

    # Compare dataframes
    try:
        # Compare DataFrames with identical labels
        differences = kso_reindexed.compare(sharepoint_reindexed, keep_equal=False)

        # Process differences
        if not differences.empty:
            detailed_differences = []

            # Iterate through differences
            for idx in differences.index:
                for col in common_columns:
                    # Check if there's a difference in this column
                    kso_val = kso_reindexed.loc[idx, col]
                    sharepoint_val = sharepoint_reindexed.loc[idx, col]

                    # Log and track differences
                    if pd.notna(sharepoint_val) and (
                        pd.isna(kso_val) or kso_val != sharepoint_val
                    ):
                        dataframe_changed = True
                        detailed_differences.append(
                            {
                                "index": idx,
                                "column": col,
                                "old_value": kso_val,
                                "new_value": sharepoint_val,
                            }
                        )

                        # Update the value in the original DataFrame
                        updated_kso_df.loc[idx, col] = sharepoint_val

                        if DEV_MODE == "DEBUG":
                            # Detailed logging
                            logging.info(
                                f"Difference in {keyword} DataFrame: "
                                f"Row {idx}, Column {col}: "
                                f"Changed from '{kso_val}' "
                                f"to '{sharepoint_val}'"
                            )

            # Log summary of differences
            logging.info(f"Total differences found: {len(detailed_differences)}")

    except Exception as e:
        logging.error(f"Error comparing DataFrames for {keyword}: {e}")
        return

    # Check for new rows
    if not new_rows.empty:
        dataframe_changed = True

    # Upload to S3 only if changes were made
    if dataframe_changed:
        # Upload to S3
        try:
            config = S3FileConfig.from_keyword(keyword)
            s3_handler.upload_updated_df_to_s3(
                df=updated_kso_df,
                key=getattr(sftk.common, config.kso_env_var),
                keyword=keyword,
            )
            logging.info(f"Updated {keyword} DataFrame uploaded to S3")
        except Exception as e:
            logging.error(f"Error uploading updated DataFrame to S3: {e}")
    else:
        logging.info(f"No differences found in {keyword} DataFrame. No updates needed.")


def main():
    """Main function to process each sharepoint list and update KSO data if needed."""
    logging.info("Starting main function")
    keywords = ["survey", "site", "movie", "species"]
    if DEV_MODE == "TEST":
        keywords = ["test"]

    logging.info(f"Processing csv files in S3 with the following keywords: {keywords}")

    try:
        logging.info("Initializing S3 handler...")
        s3_handler = S3Handler()
        logging.info("S3 handler initialized successfully")

        logging.info("Download and check S3 files...")
        results = process_s3_files(s3_handler, keywords)
        logging.info("S3 files were successfully downloaded and checked.")

        logging.info(
            "Formatting survey, movie and site info from sharepoint copy to match KSO requirements..."
        )
        results = standarise_sharepoint_to_kso(results)
        logging.info("survey, movie and site info has been successfully formatted")

        for keyword, (kso_df, sharepoint_df) in results.items():
            logging.info(f"Processing {keyword} data...")
            if kso_df is None:
                logging.info("KSO DataFrame is not available.")
            if sharepoint_df is None:
                logging.info("Sharepoint DataFrame is not available.")

            if kso_df is not None and sharepoint_df is not None:
                # Compare DataFrames and update and upload if different
                compare_and_update_dataframes(
                    kso_df=kso_df,
                    sharepoint_df=sharepoint_df,
                    s3_handler=s3_handler,
                    keyword=keyword,
                )

            else:
                logging.warning(
                    f"Skipping comparison for {keyword}: "
                    "One or both DataFrames are None"
                )

    except Exception as e:
        logging.error(
            "An error occurred during processing of sharepoint list info: %s",
            str(e),
            exc_info=True,
        )
        sys.exit(1)


if __name__ == "__main__":
    logging.info("Script started")
    main()
    logging.info("Script completed")
