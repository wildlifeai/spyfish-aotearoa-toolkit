"""
S3 Handler Module
=================

Provides a singleton interface for AWS S3 operations. This module is designed to
handle all interactions with S3, including file uploads, downloads, and listings.

Required Environment Variables:
    AWS_ACCESS_KEY_ID: AWS access key.
    AWS_SECRET_ACCESS_KEY: AWS secret key.
    S3_BUCKET: Default S3 bucket name.

Optional Environment Variables:
    S3_KSO_{TYPE}_CSV: KSO CSV file paths (e.g., S3_KSO_SURVEY_CSV).
    S3_SHAREPOINT_{TYPE}_CSV: SharePoint CSV paths (e.g., S3_SHAREPOINT_SURVEY_CSV).

Dependencies:
    - boto3: AWS SDK for Python.
    - pandas: For data manipulation, especially reading CSVs from S3.
    - tqdm: For displaying progress bars during file transfers.
"""

import datetime
import io
import logging
import mimetypes
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Union

import boto3
import pandas as pd
from boto3.s3.transfer import TransferConfig
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

from sftk.common import AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, S3_BUCKET
from sftk.utils import (
    delete_file,
    filter_file_paths_by_extension,
    get_unique_entries_df_column,
)


class S3FileNotFoundError(Exception):
    """Custom exception for S3 file not found scenarios."""

    pass


class ProgressTracker:
    """Thread-safe progress tracker for S3 transfers."""

    def __init__(self, filename: str, total_size: int, log_interval: float = 10.0):
        """
        Initialize progress tracker.

        Args:
            filename: Name of file being transferred
            total_size: Total size in bytes
            log_interval: Seconds between progress logs (default 10s)
        """
        self.filename = filename
        self.total_size = total_size
        self.transferred = 0
        self.lock = threading.Lock()
        self.start_time = time.time()
        self.last_log_time = time.time()
        self.log_interval = log_interval

    def __call__(self, bytes_amount: int):
        """Callback called by boto3 during transfer."""
        with self.lock:
            self.transferred += bytes_amount
            current_time = time.time()

            # Only log periodically to avoid spam
            if current_time - self.last_log_time >= self.log_interval:
                self._log_progress()
                self.last_log_time = current_time

    def _log_progress(self):
        """Log current progress."""
        if self.total_size > 0:
            percent = (self.transferred / self.total_size) * 100
            elapsed = time.time() - self.start_time
            speed_mbps = (
                (self.transferred / (1024 * 1024)) / elapsed if elapsed > 0 else 0
            )
            eta_seconds = (
                ((self.total_size - self.transferred) / (self.transferred / elapsed))
                if self.transferred > 0 and elapsed > 0
                else 0
            )
            # 1. Calculate the timedelta object
            eta_delta = datetime.timedelta(seconds=max(0, eta_seconds))
            # 2. Format the timedelta object into HH:MM:SS string
            # This converts the delta to H:MM:SS format (e.g., 1:05:30)
            eta_formatted = str(eta_delta)

            logging.info(
                f"   📥 {self.filename}: {percent:.1f}% "
                f"({self.transferred / (1024 * 1024):.1f}/{self.total_size / (1024 * 1024):.1f} MB) "
                f"@ {speed_mbps:.2f} MB/s - ETA: {eta_formatted}"
            )

    def complete(self):
        """Log final completion."""
        elapsed = time.time() - self.start_time
        speed_mbps = (self.total_size / (1024 * 1024)) / elapsed if elapsed > 0 else 0
        logging.info(
            f"   ✅ {self.filename}: {self.total_size / (1024 * 1024):.1f} MB "
            f"in {elapsed:.1f}s (avg {speed_mbps:.2f} MB/s)"
        )


@dataclass
class S3FileConfig:
    """
    Configuration class for S3 file operations containing file paths and environment variables.

    Attributes:
        keyword (str): Identifier for the type of data (e.g., 'survey', 'site', 'movie')
        kso_env_var (str): Environment variable name for KSO file path
        sharepoint_env_var (str): Environment variable name for Sharepoint file path
        kso_filename (str): Temporary filename for KSO data
        sharepoint_filename (str): Temporary filename for Sharepoint data
    """

    keyword: str
    kso_env_var: str
    sharepoint_env_var: str
    kso_filename: str
    sharepoint_filename: str

    @classmethod
    def from_keyword(cls, keyword: str) -> "S3FileConfig":
        """
        Creates a configuration object for S3 file operations based on
        a keyword (e.g., "survey", "site").

        Args:
            keyword(str): String identifier for the type of data

        Returns:
            S3FileConfig: The configuration information for S3 handler.
        """
        upper = keyword.upper()
        return cls(
            keyword=keyword,
            kso_env_var=f"S3_KSO_{upper}_CSV",
            sharepoint_env_var=f"S3_SHAREPOINT_{upper}_CSV",
            kso_filename=f"{keyword}_kso_temp.csv",
            sharepoint_filename=f"{keyword}_sharepoint_temp.csv",
        )


class S3Handler:
    """
    Singleton class for interacting with an S3 bucket.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs) -> "S3Handler":
        """
        Create a new instance of the class if one does not already exist.

        Args:
            bucket (str): The S3 bucket name.
        Returns:
            S3Handler: The instance of the class.
        """
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False  # Add initialization flag
            return cls._instance

    def __init__(self, s3_client: Optional[Any] = None, bucket: Optional[str] = None):
        if self._initialized:
            return

        self.bucket = bucket or S3_BUCKET
        if not self.bucket:
            raise ValueError(
                "S3_BUCKET environment variable not set or bucket not provided."
            )

        # Configure boto3 with larger connection pool
        boto_config = Config(
            max_pool_connections=50,  # Increase from default 10
            retries={"max_attempts": 3, "mode": "adaptive"},
        )

        self.s3 = s3_client or boto3.client(
            "s3",
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            config=boto_config,  # Add config here
        )
        self._initialized = True
        logging.info("S3Handler initialized for bucket: %s", self.bucket)

    def __repr__(self) -> str:
        """
        Return a string representation of the class.

        Returns:
            str: The string representation of the class.
        """
        return f"S3Handler({self.s3})"

    def download_object_from_s3(
        self, key: str, filename: str, callback=None, version_id: Optional[str] = None
    ) -> bool:
        """
        Downloads an object from S3 with progress bar and error handling.

        Args:
            key: The S3 object key (path within the bucket).
            filename: Local filesystem path where the file will be saved.
            callback: Optional callback function for progress tracking.
                    Will be called with bytes_transferred as argument.
            version_id: Optional version ID for versioned buckets.

        Returns:
            bool: True if download was successful, False otherwise.

        Raises:
            ValueError: If key or filename is empty.

        Side Effects:
            - Creates a local file at `filename`.
            - Displays a progress bar to stdout (if no callback provided).

        Example:
            >>> handler = S3Handler()
            >>> handler.download_object_from_s3(
            ...     key="media/survey1/video.mp4",
            ...     filename="local_video.mp4"
            ... )
            local_video.mp4: 100%|████████| 50.0M/50.0M [00:05<00:00, 9.8MB/s]

        Thread Safety:
            This method is thread-safe for different files, but not for the
            same filename due to filesystem constraints.
        """
        if not key or not filename:
            raise ValueError("S3 key and local filename must be provided.")

        try:

            kwargs: Dict[str, Any] = {"Bucket": self.bucket, "Key": key}
            if version_id:
                kwargs["VersionId"] = version_id

            try:
                object_size = self.s3.head_object(**kwargs)["ContentLength"]
            except ClientError as e:
                logging.error(f"Could not get size for {key}: {e}")
                return False

            # Configure transfer for better performance with large files
            config = TransferConfig(
                max_concurrency=5,
                multipart_threshold=8 * 1024 * 1024,
                multipart_chunksize=8 * 1024 * 1024,
                num_download_attempts=3,
                use_threads=True,
            )

            if callback:
                # Use custom callback (for parallel processing with minimal logging)
                self.s3.download_file(
                    Bucket=self.bucket,
                    Key=key,
                    Filename=filename,
                    Callback=callback,
                    Config=config,
                )
            else:
                # Use tqdm progress bar (for interactive/sequential downloads)
                progress = ProgressTracker(Path(filename).name, object_size)
                self.s3.download_file(
                    Bucket=self.bucket,
                    Key=key,
                    Filename=filename,
                    Callback=progress,
                    Config=config,
                )
                progress.complete()

            return True

        except BotoCoreError as e:
            logging.error("Failed to download %s from S3: %s", key, e)
            return False
        except Exception as e:
            logging.error("Unexpected error downloading %s from S3: %s", key, e)
            return False

    def download_and_read_s3_file(self, key: str, filename: str) -> pd.DataFrame:
        """
        Downloads an S3 object and reads it into a Pandas DataFrame.

        Args:
            key (str): The S3 object key.
            filename (str): The local filename to save the downloaded object.

        Returns:
            pd.DataFrame: The DataFrame read from the downloaded file.

        Raises:
            S3FileNotFoundError: If errors occur during download or reading.
        """
        try:
            self.download_object_from_s3(key=key, filename=filename)
            return pd.read_csv(filename)
        except Exception as e:
            logging.warning("Failed to process S3 file %s: %s", key, str(e))
            raise S3FileNotFoundError(
                f"Failed to download or read S3 file {key}: {e}"
            ) from e

    def upload_data_to_s3(
        self,
        text_data: str,
        s3_path: str,
    ) -> bool:
        """
        Uploads a string to S3.

        Args:
            text_data (str): The data to upload.
            s3_path (str): The S3 object key.

        Returns:
            bool: True if upload succeeded, False otherwise.
        """
        logging.info("Uploading data to S3 %s", s3_path)
        try:
            self.s3.put_object(Bucket=self.bucket, Key=s3_path, Body=text_data)
            logging.info("Successfully uploaded data to S3 %s", s3_path)
            return True
        except BotoCoreError as e:
            logging.error("Failed to upload data to S3 %s, with error: %s", s3_path, e)
            return False

    def upload_file_to_s3(
        self,
        filename: str,
        key: str,
        callback=None,
        delete_file_after_upload=False,
        content_type: Optional[str] = None,
    ) -> bool:
        """
        Uploads a file to S3 with progress bar and error handling.

        Args:
            filename (str): The local filename to upload.
            key (str): The S3 object key.
            callback: Optional callback function for progress tracking.
            delete_file_after_upload (bool): If True, deletes the local file after a successful upload.
            content_type (Optional[str]): The content type of the file. If not provided, it's guessed.

        Returns:
            bool: True if upload succeeded, False otherwise.

        Raises:
            ValueError: If filename or key is empty.
            FileNotFoundError: If the local file does not exist.
        """
        if not filename or not key:
            raise ValueError("Local filename and S3 key must be provided.")
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Local file not found: {filename}")
        if content_type:
            content_args = {"ContentType": content_type}
        else:
            ct, _ = mimetypes.guess_type(filename)
            content_args = {"ContentType": ct or "application/octet-stream"}
        logging.info("Uploading file %s to S3", filename)
        try:
            file_size = os.path.getsize(filename)
            if callback:
                self.s3.upload_file(
                    Filename=filename,
                    Bucket=self.bucket,
                    Key=key,
                    ExtraArgs=content_args,
                    Callback=callback,
                )
            else:
                progress = ProgressTracker(Path(filename).name, file_size)
                self.s3.upload_file(
                    Filename=filename,
                    Bucket=self.bucket,
                    Key=key,
                    ExtraArgs=content_args,
                    Callback=progress,
                )
                progress.complete()

            logging.info("Successfully uploaded file %s to S3", filename)
            return True
        except BotoCoreError as e:
            logging.error("Failed to upload file %s to S3: %s", filename, e)
            return False
        finally:
            if delete_file_after_upload:
                delete_file(filename)

    def upload_updated_df_to_s3(
        self,
        df: pd.DataFrame,
        key: str,
        filename: Optional[str] = None,
        keyword: Optional[str] = None,
        keep_df_index=True,
    ) -> None:
        """
        Upload an updated DataFrame to S3 with progress bar and error handling.

        Args:
            df (pd.DataFrame): DataFrame to upload.
            key (str): S3 key for the file.
            keyword (str): String identifier for the type of data (e.g., "survey", "site").
            keep_df_index (bool): Whether to write the DataFrame index to the CSV.
        """
        if keyword:
            temp_filename = f"updated_{keyword}_kso_temp.csv"
        elif filename:
            temp_filename = filename
        else:
            raise ValueError("Either keyword or filename must be provided.")

        try:
            df.to_csv(temp_filename, index=keep_df_index)
            self.upload_file_to_s3(temp_filename, key)
        except (BotoCoreError, IOError) as e:
            logging.error("Failed to upload updated %s data to S3: %s", keyword, e)
        finally:
            delete_file(temp_filename)

    def get_file_paths_set_from_s3(
        self, prefix: str = "", suffixes: tuple = ()
    ) -> Set[str]:
        """Retrieve a set of object keys from S3."""
        keys = self.get_objects_from_s3(
            prefix, suffixes, keys_only=True, file_names_only=False
        )
        return keys if isinstance(keys, set) else set()

    def get_objects_from_s3(
        self,
        prefix: str = "",
        suffixes: tuple = (),
        keys_only: bool = False,
        file_names_only: bool = False,
    ) -> Union[Set[str], List[Dict[str, Any]]]:
        """
        Unified method for retrieving S3 objects or their keys.

        Args:
            prefix (str, optional): Folder path to filter objects. Defaults to "".
            suffixes (tuple, optional): File suffixes to filter by.
            keys_only (bool): If True, returns a set of object keys. Otherwise, a list of object dicts.
            file_names_only (bool): If True and keys_only is True, returns only file names.

        Returns:
            Union[Set[str], List[Dict[str, Any]]]: A set of S3 object keys or a list of S3 object dictionaries.
        """
        results: Union[Set[str], List[Dict[str, Any]]] = set() if keys_only else []
        paginator = self.s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                if not suffixes or obj["Key"].lower().endswith(tuple(suffixes)):
                    if keys_only:
                        key = Path(obj["Key"]).name if file_names_only else obj["Key"]
                        # We know results is a set here
                        results.add(key)  # type: ignore
                    else:
                        # We know results is a list here
                        results.append(obj)  # type: ignore
        return results

    def get_paths_from_csv(
        self,
        csv_s3_path: str,
        csv_column: str,
        column_filter: Optional[str] = None,
        column_value: Optional[Any] = None,  # type: ignore
    ) -> Dict[str, set]:
        """
        Extract unique file paths from a CSV file stored in S3.

        This method reads a CSV file from S3 and extracts unique file paths from a specified
        column. It returns two sets of paths: all unique paths from the column, and paths
        excluding those from rows that match optional filter criteria.

        Args:
            csv_s3_path: S3 path to the CSV file (without bucket name)
            csv_column: Name of the column containing file paths to extract
            column_filter: Optional column name to filter rows by
            column_value: Value that the filter column must equal to exclude rows

        Returns:
            Dictionary with two keys:
            - 'all': Set of all unique file paths from the specified column
            - 'filtered': Set of unique file paths excluding rows where
              column_filter equals column_value

        Example:
            >>> handler = S3Handler()
            >>> result = handler.get_paths_from_csv(
            ...     csv_s3_path="data/BUV Deployments.csv",
            ...     csv_column="LinkToVideoFile",
            ...     column_filter="IsBadDeployment",
            ...     column_value=False
            ... )
            >>> print(f"All paths: {len(result['all'])}")
            >>> print(f"Valid paths: {len(result['filtered'])}")

        Note:
            The 'all' set is used to check for extra files in S3 (a file in S3 is not
            extra if it appears anywhere in the CSV). The 'filtered' set is used to
            check for missing files (a file is only missing if it's from a valid deployment).
        """
        logging.info(f"Processing CSV: {csv_s3_path}.")

        # Load dataframe from AWS
        csv_df = self.read_df_from_s3_csv(csv_s3_path)

        # Load unique file paths from the CSV column excluding the filtered values
        csv_filepaths_without_filtered_values = get_unique_entries_df_column(
            csv_df,
            csv_column,
            column_filter=column_filter,
            column_value=column_value,
        )
        # Load all unique file paths from the CSV column
        csv_filepaths_all = get_unique_entries_df_column(csv_df, csv_column)

        logging.info(f"Unique file paths from CSV: {len(csv_filepaths_all)}.")
        logging.info(
            f"Unique file paths from CSV, without filtered value {column_filter} as {column_value}: {len(csv_filepaths_without_filtered_values)}."
        )

        return {
            "all": csv_filepaths_all,
            "filtered": csv_filepaths_without_filtered_values,
        }

    def get_paths_from_s3(
        self,
        valid_extensions: Iterable[str] = [],
        path_prefix: str = "",
    ) -> Set[str]:
        logging.info("Processing the files in the bucket: %s.", self.bucket)

        # Get all file paths currently in S3
        s3_filepaths = self.get_file_paths_set_from_s3(prefix=path_prefix)

        # Filter only video files based on their extension
        if valid_extensions:
            s3_filepaths = filter_file_paths_by_extension(
                s3_filepaths, valid_extensions
            )

        logging.info("Found %d files in S3 matching criteria.", len(s3_filepaths))
        return set(s3_filepaths)

    def rename_s3_objects_from_dict(
        self,
        rename_pairs: Dict[str, str],
        prefix: str = "",
        suffixes: Iterable = (),
        try_run: bool = False,
    ) -> None:
        files_from_aws = self.get_file_paths_set_from_s3(prefix, suffixes)
        failed_renames = []

        for old_name, new_name in rename_pairs.items():

            if old_name in files_from_aws:
                logging.info(f"⛭ Renaming: {old_name} ➜ {new_name}")
                try:
                    if not try_run:
                        try:
                            self.s3.head_object(Bucket=self.bucket, Key=new_name)
                            failed_renames.append(("FileExists", old_name, new_name))
                            logging.info(
                                f"❌ There is a file under the existing name: {new_name}, skipping rename."
                            )
                        except ClientError as e:
                            if e.response["Error"]["Code"] == "404":
                                copy_source = {"Bucket": self.bucket, "Key": old_name}
                                self.s3.copy(copy_source, self.bucket, new_name)
                                # Delete the old object
                                self.s3.delete_object(Bucket=self.bucket, Key=old_name)
                                logging.info(f"✅ Renamed: {old_name} ➜ {new_name}")
                            else:
                                raise

                except BotoCoreError as e:
                    logging.warning(
                        f"Failed to rename {old_name} to {new_name}, error: {str(e)}"
                    )
                    failed_renames.append(("BotoCoreError", old_name, new_name))
                except ClientError as e:
                    logging.warning(
                        f"Failed to rename {old_name} to {new_name}, error: {str(e)}"
                    )
                    failed_renames.append(("ClientError", old_name, new_name))
            else:
                logging.info(f"File not found in the {self.bucket} bucket: {old_name}.")

        logging.info(
            f"Rename complete, failed to rename {len(failed_renames)} out of {len(rename_pairs)} files."
        )
        logging.info(f"Failed renames: {failed_renames}")

    def read_df_from_s3_csv(self, csv_s3_path: str) -> pd.DataFrame:
        """
        Downloads a CSV file from S3 and loads it into a pandas DataFrame.

        Parameters:
            csv_s3_path (str): The S3 key for the CSV file.

        Returns:
            pd.DataFrame: The loaded DataFrame.
        """
        try:
            # Download the object to memory
            response = self.s3.get_object(Bucket=self.bucket, Key=csv_s3_path)
            return pd.read_csv(io.BytesIO(response["Body"].read()))
        except BotoCoreError as e:
            logging.error("Failed to read CSV %s from S3: %s", csv_s3_path, e)
            raise S3FileNotFoundError(
                f"Failed to read CSV {csv_s3_path} from S3: {e}"
            ) from e

    def generate_presigned_url(self, key: str, expiration: int = 3600) -> Optional[str]:
        """
        Generate a presigned URL to share an S3 object.

        Args:
            key (str): The key of the object for which to generate the URL.
            expiration (int): Time in seconds for the presigned URL to remain valid.

        Returns:
            Optional[str]: The presigned URL, or None if an error occurred.
        """
        try:
            response = self.s3.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket, "Key": key},
                ExpiresIn=expiration,
            )
            return response
        except BotoCoreError as e:
            logging.error(f"Failed to generate presigned URL for {key}: {e}")
            return None
