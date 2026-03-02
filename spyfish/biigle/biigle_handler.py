"""
BIIGLE API Handler for Spyfish Aotearoa.

Provides functionality to interact with the Biigle API:
- Creating and managing volumes (S3-backed)
- Setting up label trees from a CSV of scientific names
- Exporting annotation reports and reading them as DataFrames
"""
import io
import logging
import time
import zipfile
from typing import Any, Dict, List, Literal, Optional

import pandas as pd
from requests.exceptions import HTTPError

from spyfish.biigle.external.biigle_api import Api
from spyfish.config import config

ResourceType = Literal["volumes", "projects"]
MAX_DEPTH = 2


class BiigleHandler:
    """Handler for BIIGLE API operations."""

    def __init__(self, email: Optional[str] = None, token: Optional[str] = None):
        """
        Initialize BiigleHandler with API credentials.
        Credentials fall back to config (BIIGLE_API_EMAIL / BIIGLE_API_TOKEN env vars).
        """
        self.email = email or config.biigle_email
        self.token = token or config.biigle_token
        try:
            self.api = Api(self.email, self.token)
        except Exception as e:
            raise Exception(f"Failed to initialize BIIGLE API: {e}") from e
        logging.info("BiigleHandler initialized successfully")

    # ── Projects & Volumes ────────────────────────────────────────────────────

    def get_projects(self) -> List[Dict[str, Any]]:
        """Get all projects accessible to the authenticated user."""
        try:
            response = self.api.get("projects")
            projects = response.json()
            logging.info(f"Retrieved {len(projects)} projects")
            return projects
        except Exception as e:
            logging.error(f"Failed to get projects: {e}")
            raise

    def get_volumes(self, project_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get all volumes in a project (defaults to config.biigle_project_id)."""
        project_id = project_id or config.biigle_project_id
        try:
            response = self.api.get(f"projects/{project_id}/volumes")
            volumes = response.json()
            logging.info(f"Retrieved {len(volumes)} volumes from project {project_id}")
            return volumes
        except Exception as e:
            logging.error(f"Failed to get volumes for project {project_id}: {e}")
            raise

    def create_pending_volume(
        self, project_id: Optional[int] = None, media_type: str = "video"
    ) -> Dict[str, Any]:
        """Create a pending volume in a project."""
        project_id = project_id or config.biigle_project_id
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
        """Configure a pending volume with name, S3 URL, and file list."""
        try:
            payload = {"name": volume_name, "url": s3_url, "files": files}
            response = self.api.put(f"pending-volumes/{pending_volume_id}", json=payload)
            volume_info = response.json()
            logging.info(f"Configured volume '{volume_name}' with {len(files)} files")
            return volume_info
        except Exception as e:
            logging.error(f"Failed to setup volume with files: {e}")
            raise

    def create_volume_from_s3_files(
        self,
        volume_name: str,
        s3_url: str,
        files: List[str],
        project_id: Optional[int] = None,
        media_type: str = "video",
    ) -> Dict[str, Any]:
        """
        Convenience method: create_pending_volume + setup_volume_with_files in one call.

        Args:
            volume_name: Display name for the new volume.
            s3_url: Biigle-compatible S3 URL (see build_s3_url()).
            files: List of filenames within the S3 folder.
            project_id: Defaults to config.biigle_project_id.
            media_type: "video" or "image".
        """
        project_id = project_id or config.biigle_project_id
        try:
            pending = self.create_pending_volume(project_id, media_type)
            volume_info = self.setup_volume_with_files(
                pending["id"], volume_name, s3_url, files
            )
            logging.info(f"Created volume '{volume_name}' with {len(files)} files")
            return volume_info
        except Exception as e:
            logging.error(f"Failed to create volume from S3 files: {e}")
            raise

    def build_s3_url(self, s3_path: str, disk_id: Optional[int] = None) -> str:
        """
        Build a Biigle-compatible S3 URL from an S3 path.

        Example:
            handler.build_s3_url("biigle_frames/KSF_20240601/")
            → "disk-42://biigle_frames/KSF_20240601/"
        """
        disk_id = disk_id or config.biigle_disk_id
        s3_path = s3_path.rstrip("/") + "/"
        return f"disk-{disk_id}://{s3_path}"

    # ── Label Trees ───────────────────────────────────────────────────────────

    def create_label_tree(
        self,
        csv_path: str,
        tree_name: str,
        tree_description: str,
        project_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Create a label tree and populate it with labels from a CSV.

        CSV must have columns: name, color, source_id
        (source_id = WoRMS AphiaID or similar external reference).
        """
        project_id = project_id or config.biigle_project_id
        try:
            tree_response = self.api.post(
                "label-trees",
                json={
                    "name": tree_name,
                    "description": tree_description,
                    "visibility_id": 2,  # Private
                    "project_id": project_id,
                },
            )
            tree_info = tree_response.json()
            tree_id = tree_info["id"]
            logging.info(f"Created label tree '{tree_name}' (id={tree_id})")

            labels_df = pd.read_csv(csv_path)
            created_labels = {}
            for _, row in labels_df.iterrows():
                label_response = self.api.post(
                    f"label-trees/{tree_id}/labels",
                    json={
                        "name": row["name"],
                        "color": row["color"],
                        "source_id": row["source_id"],
                        "label_source_id": 999,
                    },
                )
                created_labels[row["name"]] = label_response.json()

            logging.info(f"Added {len(created_labels)} labels to tree '{tree_name}'")
            return {"tree_id": tree_id, "tree_info": tree_info, "labels": created_labels}
        except Exception as e:
            logging.error(f"Failed to create label tree: {e}")
            raise

    # ── Reports ───────────────────────────────────────────────────────────────

    def create_report(
        self,
        resource: ResourceType,
        resource_id: int,
        type_id: int = config.biigle_annotation_report_type_video,
    ) -> int:
        """
        Request a BIIGLE annotation report. Returns the report ID.
        Reports are generated asynchronously — use download_report_zip_bytes() to fetch.
        Defaults to the video annotation CSV report type (config.biigle_annotation_report_type_video).
        """
        resp = self.api.post(
            f"{resource}/{resource_id}/reports", json={"type_id": type_id}
        )
        resp.raise_for_status()
        report_id = resp.json()["id"]
        logging.info(
            f"Requested report {report_id} for {resource.rstrip('s')} {resource_id} "
            f"(type_id={type_id})"
        )
        return report_id

    def download_report_zip_bytes(
        self,
        report_id: int,
        max_tries: int = 60,
        poll_interval: float = 2.0,
    ) -> bytes:
        """
        Poll until a report ZIP is ready and return its raw bytes.
        BIIGLE reports are generated asynchronously (may take up to ~2 min).
        """
        for attempt in range(1, max_tries + 1):
            resp = self.api.get(f"reports/{report_id}", raise_for_status=False)
            status = resp.status_code

            if status == 200:
                logging.info(f"Report {report_id} ready after {attempt} attempt(s).")
                return resp.content

            if status in (202, 404):
                logging.info(
                    f"Report {report_id} not ready (status {status}), "
                    f"attempt {attempt}/{max_tries}. Waiting {poll_interval}s..."
                )
                time.sleep(poll_interval)
                continue

            try:
                resp.raise_for_status()
            except HTTPError as e:
                logging.error(f"Error fetching report {report_id}: {e}")
                raise

        raise TimeoutError(
            f"Report {report_id} not ready after {max_tries * poll_interval:.0f}s."
        )

    def read_csvs_from_zip_bytes(
        self,
        zip_bytes: bytes,
        allow_nested: bool = True,
        _depth: int = 0,
    ) -> Dict[str, pd.DataFrame]:
        """
        Read all CSV files from a ZIP blob (including nested ZIPs if allow_nested=True).
        Returns {filename: DataFrame}.
        """
        csv_dfs: Dict[str, pd.DataFrame] = {}
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            for info in zf.infolist():
                name = info.filename
                if name.lower().endswith(".csv"):
                    with zf.open(info) as f:
                        csv_dfs[name] = pd.read_csv(f)
                elif allow_nested and name.lower().endswith(".zip") and _depth < MAX_DEPTH:
                    nested_csvs = self.read_csvs_from_zip_bytes(
                        zf.read(name), allow_nested=allow_nested, _depth=_depth + 1
                    )
                    csv_dfs.update(nested_csvs)
        return csv_dfs

    def concat_csv_dict(
        self, csv_dfs: Dict[str, pd.DataFrame], source_col: str = "source_file"
    ) -> pd.DataFrame:
        """Concatenate multiple CSV DataFrames, tagging each row with its source filename."""
        frames = [df.assign(**{source_col: name}) for name, df in csv_dfs.items()]
        if not frames:
            raise FileNotFoundError("No CSV DataFrames to concatenate.")
        return pd.concat(frames, ignore_index=True)

    def export_report_to_df(
        self,
        resource: ResourceType,
        resource_id: int,
        type_id: int = config.biigle_annotation_report_type_video,
        source_col: str = "source_file",
    ) -> pd.DataFrame:
        """
        High-level helper: request report → wait → download ZIP → read all CSVs → return DataFrame.
        Defaults to the video annotation CSV report type.
        """
        report_id = self.create_report(resource, resource_id, type_id)
        zip_bytes = self.download_report_zip_bytes(report_id)
        allow_nested = resource == "projects"
        csv_dict = self.read_csvs_from_zip_bytes(zip_bytes, allow_nested=allow_nested)
        if not csv_dict:
            raise FileNotFoundError(
                f"No CSVs in report for {resource.rstrip('s')} {resource_id} (report {report_id})."
            )
        df = self.concat_csv_dict(csv_dict, source_col=source_col)
        logging.info(
            f"Exported report for {resource.rstrip('s')} {resource_id}: "
            f"{len(csv_dict)} CSV(s), {len(df)} rows."
        )
        return df
