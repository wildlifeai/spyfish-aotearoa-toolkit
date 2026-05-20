"""
BIIGLE API Handler for Spyfish Aotearoa project.

This module provides functionality to interact with the BIIGLE API for:
- Creating and managing volumes
- Setting up label trees with scientific names
- Exporting and processing annotations
- Integrating with S3 for video file management

Example usage:
    # Initialize handler
    biigle_handler = BiigleHandler()

    # Get projects
    projects = biigle_handler.get_projects()

    # Create volume with files from S3
    volume = biigle_handler.create_volume_from_s3_files(
        project_id=3711,
        volume_name="My Volume",
        s3_url=biigle_handler.build_s3_url("biigle_clips/my_folder/"),
        files=["video1.mp4", "video2.mp4"]
    )


    # Export annotations
    annotations_df = biigle_handler.export_report_to_df(
        resource="volumes", resource_id=12345
    )
"""

import io
import logging
import time
import zipfile
from typing import Any, Dict, List, Literal, Optional

import pandas as pd
from requests.exceptions import HTTPError

from sftk.common import BIIGLE_ANNOTATION_REPORT_TYPE, BIIGLE_DISK_ID, BIIGLE_PROJECT_ID
from sftk.external.biigle_api import Api

ResourceType = Literal["volumes", "projects"]
MAX_DEPTH = 2


class BiigleHandler:
    """Handler for BIIGLE API operations."""

    def __init__(self, email: Optional[str] = None, token: Optional[str] = None):
        """
        Initialize BiigleHandler with API credentials.

        Raises:
            Exception: If credentials are not provided and not found in environment.
        """
        self.email = email
        self.token = token

        try:
            self.api = Api(self.email, self.token)
        except Exception as e:
            raise Exception(f"Failed to initialize BIIGLE API: {e}") from e

        logging.info("BiigleHandler initialized successfully")

    def get_projects(self) -> List[Dict[str, Any]]:
        """
        Get all projects that the user can access.

        Returns:
            List[Dict[str, Any]]: List of project dictionaries with id, name, description, etc.

        Raises:
            Exception: If the API call fails.
        """
        try:
            response = self.api.get("projects")
            projects = response.json()
            logging.info(f"Retrieved {len(projects)} projects")
            return projects
        except Exception as e:
            logging.error(f"Failed to get projects: {e}")
            raise

    def create_pending_volume(
        self, project_id: int = BIIGLE_PROJECT_ID, media_type: str = "video"
    ) -> Dict[str, Any]:
        """
        Create a pending volume in a project.

        Args:
            project_id: The ID of the project to create the volume in.
            media_type: The type of media for the volume (default: "video").

        Returns:
            Dict[str, Any]: The created pending volume information.

        Raises:
            Exception: If the API call fails.
        """
        try:
            response = self.api.post(
                f"projects/{project_id}/pending-volumes",
                json={"media_type": media_type},
            )
            pending_volume = response.json()
            logging.info(f"Created pending volume with ID: {pending_volume['id']}")
            return pending_volume
        except Exception as e:
            logging.error(f"Failed to create pending volume: {e}")
            raise

    def setup_volume_with_files(
        self, pending_volume_id: int, volume_name: str, s3_url: str, files: List[str]
    ) -> Dict[str, Any]:
        """
        Configure a pending volume with name, S3 URL, and file list.

        Args:
            pending_volume_id: The ID of the pending volume to configure.
            volume_name: The name for the volume.
            s3_url: The S3 URL where the files are located.
            files: List of file names to include in the volume.

        Returns:
            Dict[str, Any]: The configured volume information.

        Raises:
            Exception: If the API call fails.
        """
        try:
            payload = {"name": volume_name, "url": s3_url, "files": files}
            response = self.api.put(
                f"pending-volumes/{pending_volume_id}", json=payload
            )
            volume_info = response.json()
            logging.info(f"Configured volume '{volume_name}' with {len(files)} files")
            return volume_info
        except Exception as e:
            logging.error(f"Failed to setup volume with files: {e}")
            raise

    def create_label_tree(
        self,
        csv_path: str,
        tree_name: str,
        tree_description: str,
        project_id: int = BIIGLE_PROJECT_ID,
    ) -> Dict[str, Any]:
        """
        Create a label tree and populate it with scientific names from a CSV file.

        TODO: add aphiaID to label tree. It doesn't seem to pick up the info.

        Args:
            project_id: The ID of the project to create the label tree in.
            csv_path: Path to CSV file containing label data (name, color, source_id columns).
            tree_name: Name for the label tree.
            tree_description: Description for the label tree.

        Returns:
            Dict[str, Any]: Dictionary containing tree_id and labels information.

        Raises:
            Exception: If the API call fails or CSV processing fails.
        """
        try:
            # Create the label tree
            tree_config = {
                "name": tree_name,
                "description": tree_description,
                "visibility_id": 2,  # Private by default, can change later in the label tree settings.
                "project_id": project_id,
            }

            tree_response = self.api.post("label-trees", json=tree_config)
            tree_info = tree_response.json()
            tree_id = tree_info["id"]
            logging.info(f"Created label tree '{tree_name}' with ID: {tree_id}")

            # Read labels from CSV and add them to the tree
            labels_df = pd.read_csv(csv_path)
            created_labels = {}

            for idx, row in labels_df.iterrows():
                label_config = {
                    "name": row["name"],
                    "color": row["color"],
                    "source_id": row["source_id"],
                    "label_source_id": 999,
                }

                label_response = self.api.post(
                    f"label-trees/{tree_id}/labels", json=label_config
                )
                label_info = label_response.json()
                created_labels[row["name"]] = label_info

            logging.info(f"Added {len(created_labels)} labels to tree '{tree_name}'")

            return {
                "tree_id": tree_id,
                "tree_info": tree_info,
                "labels": created_labels,
            }

        except Exception as e:
            logging.error(f"Failed to create label tree: {e}")
            raise

    def create_report(
        self,
        resource: ResourceType,
        resource_id: int,
        type_id: int,
    ) -> int:
        """
        Create a BIIGLE report for a volume or project.
        Example endpoints:
            volumes/{volume_id}/reports
            projects/{project_id}/reports
        """
        resp = self.api.post(
            f"{resource}/{resource_id}/reports", json={"type_id": type_id}
        )
        resp.raise_for_status()
        report_id = resp.json()["id"]
        logging.info(
            f"Created report {report_id} for {resource.rstrip('s')} {resource_id} "
            f"(type_id={type_id})"
        )
        return report_id

    def download_report_zip_bytes(
        self,
        report_id: int,
        max_tries: int = 60,  # e.g. up to 2 minutes if poll_interval=2
        poll_interval: float = 2.0,
    ) -> bytes:
        """
        Download the report ZIP by its ID and return raw bytes.

        BIIGLE reports are generated asynchronously, so this polls the API until
        the report file is available.

        Endpoint: reports/{report_id}
        """
        for attempt in range(1, max_tries + 1):
            # IMPORTANT: disable auto raise_for_status so we can handle 404 ourselves
            resp = self.api.get(f"reports/{report_id}", raise_for_status=False)

            status = resp.status_code

            if status == 200:
                logging.info(
                    f"Downloaded ZIP for report {report_id} "
                    f"after {attempt} attempt(s)."
                )
                return resp.content

            # "Not ready yet" cases – BIIGLE may respond with 404 until generated
            if status in (202, 404):
                logging.info(
                    f"Report {report_id} not ready yet (status {status}), "
                    f"attempt {attempt}/{max_tries}. Waiting {poll_interval}s..."
                )
                time.sleep(poll_interval)
                continue

            # Any other status is treated as an actual error
            try:
                resp.raise_for_status()
            except HTTPError as e:
                logging.error(
                    f"Error fetching report {report_id}: {e} "
                    f"(status {status}, attempt {attempt})"
                )
                raise

        raise TimeoutError(
            f"Report {report_id} was not ready after "
            f"{max_tries * poll_interval:.0f} seconds."
        )

    def read_csvs_from_zip_bytes(
        self,
        zip_bytes: bytes,
        allow_nested: bool = True,
        _depth: int = 0,
    ) -> Dict[str, pd.DataFrame]:
        """
        Read all CSV files from a ZIP (given as bytes).

        - If allow_nested=True, will also read CSVs from nested ZIPs.
        - Returns { 'name.csv': DataFrame }.
        """
        csv_dfs: Dict[str, pd.DataFrame] = {}

        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            for info in zf.infolist():
                name = info.filename

                # CSV directly inside this ZIP
                if name.lower().endswith(".csv"):
                    logging.info(f"Found CSV in ZIP: {name}")
                    with zf.open(info) as f:
                        csv_dfs[name] = pd.read_csv(f)

                # Nested ZIP – recurse if allowed
                elif (
                    allow_nested
                    and name.lower().endswith(".zip")
                    and _depth < MAX_DEPTH
                ):
                    logging.info(f"Found nested ZIP in ZIP: {name}")
                    nested_bytes = zf.read(name)
                    nested_csvs = self.read_csvs_from_zip_bytes(
                        zip_bytes=nested_bytes,
                        allow_nested=allow_nested,
                        _depth=_depth + 1,
                    )

                    for inner_name, inner_df in nested_csvs.items():
                        csv_dfs[inner_name] = inner_df

        return csv_dfs

    def concat_csv_dict(
        self,
        csv_dfs: Dict[str, pd.DataFrame],
        source_col: str = "source_file",
    ) -> pd.DataFrame:
        """
        Concatenate multiple CSV DataFrames into one, annotating
        each row with its origin (key name) in a new column.
        """
        frames: List[pd.DataFrame] = []
        for name, df in csv_dfs.items():
            df = df.copy()
            df[source_col] = name
            frames.append(df)
        if not frames:
            raise FileNotFoundError("No CSV DataFrames to concatenate.")
        return pd.concat(frames, ignore_index=True)

    def export_report_to_df(
        self,
        resource: ResourceType,
        resource_id: int,
        type_id: int = BIIGLE_ANNOTATION_REPORT_TYPE,
        source_col: str = "source_file",
    ) -> pd.DataFrame:
        """
        High-level exporter:

        1. Create BIIGLE report for given resource (volume/project)
        2. Download the report ZIP as bytes
        3. Read all CSVs from the ZIP (optionally nested)
        4. Concatenate them into a single DataFrame with a `source_col` that
        indicates which CSV each row came from.

        Args:
            resource: "volumes" or "projects"
            resource_id: volume_id or project_id
            type_id: BIIGLE report type
            source_col: name of the column that stores CSV origin

        Returns:
            pd.DataFrame: concatenated DataFrame of all CSVs in the report.
        """
        # 1) Create report
        report_id = self.create_report(resource, resource_id, type_id)

        # 2) Download ZIP
        zip_bytes = self.download_report_zip_bytes(report_id)

        # 3) Read all CSVs (flat or nested)
        allow_nested = resource == "projects"
        csv_dict = self.read_csvs_from_zip_bytes(
            zip_bytes=zip_bytes, allow_nested=allow_nested
        )

        if not csv_dict:
            raise FileNotFoundError(
                f"No CSV files found in report for {resource.rstrip('s')} {resource_id} for report_id {report_id}."
            )

        # 4) Concatenate and annotate with source filename
        csv_df = self.concat_csv_dict(csv_dict, source_col=source_col)
        logging.info(
            f"Exported report for {resource.rstrip('s')} {resource_id}: "
            f"{len(csv_dict)} CSV(s), {len(csv_df)} total rows."
        )
        return csv_df

    def get_volumes(self, project_id: int = BIIGLE_PROJECT_ID) -> List[Dict[str, Any]]:
        """
        Get all volumes in a project.

        Args:
            project_id: The ID of the project to get volumes from.

        Returns:
            List[Dict[str, Any]]: List of volume dictionaries.

        Raises:
            Exception: If the API call fails.
        """
        try:
            response = self.api.get(f"projects/{project_id}/volumes")
            volumes = response.json()
            logging.info(f"Retrieved {len(volumes)} volumes from project {project_id}")
            return volumes
        except Exception as e:
            logging.error(f"Failed to get volumes for project {project_id}: {e}")
            raise

    def create_volume_from_s3_files(
        self,
        project_id: int,
        volume_name: str,
        s3_url: str,
        files: List[str],
        media_type: str = "video",
    ) -> Dict[str, Any]:
        """
        Complete workflow to create a volume with files from S3.

        This is a convenience method that combines create_pending_volume and
        setup_volume_with_files.

        Args:
            project_id: The ID of the project to create the volume in.
            volume_name: The name for the volume.
            s3_url: The S3 URL where the files are located.
            files: List of file names to include in the volume.
            media_type: The type of media for the volume (default: "video").

        Returns:
            Dict[str, Any]: The created volume information.

        Raises:
            Exception: If any step of the process fails.
        """
        try:
            # Step 1: Create pending volume
            pending_volume = self.create_pending_volume(project_id, media_type)
            pending_volume_id = pending_volume["id"]

            # Step 2: Setup volume with files
            volume_info = self.setup_volume_with_files(
                pending_volume_id, volume_name, s3_url, files
            )

            logging.info(
                f"Successfully created volume '{volume_name}' with {len(files)} files"
            )
            return volume_info

        except Exception as e:
            logging.error(f"Failed to create volume from S3 files: {e}")
            raise

    def build_s3_url(self, s3_path: str, disk_id: int = BIIGLE_DISK_ID) -> str:
        """
        Build a BIIGLE-compatible S3 URL from an S3 path.

        Args:
            s3_path: The S3 path (e.g., "biigle_clips/TON_20221205_BUV/TON_20221205_BUV_TON_044_01/")
            disk_id: The BIIGLE disk ID for the S3 bucket.

        Returns:
            str: The BIIGLE-compatible S3 URL.
        """
        # Ensure path ends with /
        if not s3_path.endswith("/"):
            s3_path += "/"

        return f"disk-{disk_id}://{s3_path}"
