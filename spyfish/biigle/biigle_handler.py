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
from requests.exceptions import HTTPError  # type: ignore

from spyfish.biigle.external.biigle_api import Api
from spyfish.config.wrapper import config

ResourceType = Literal["volumes", "projects"]
MAX_DEPTH = 2


class BiigleHandler:
    """Handler for BIIGLE API operations."""

    def __init__(self, email: Optional[str] = None, token: Optional[str] = None):
        """
        Initialize BiigleHandler with API credentials.
        Credentials fall back to config (BIIGLE_API_EMAIL / BIIGLE_API_TOKEN env vars).
        """
        self.email = email or config.email
        self.token = token or config.token
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

    def get_pending_volumes(
        self, project_id: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get all pending volumes for a project."""
        project_id = project_id or config.biigle_project_id
        try:
            # Try project-specific endpoint first
            response = self.api.get(f"projects/{project_id}/pending-volumes")
            return response.json()
        except HTTPError as e:
            if e.response.status_code == 405:
                logging.info(
                    f"GET projects/{project_id}/pending-volumes returned 405, trying GET pending-volumes..."
                )
                try:
                    # Fallback to global endpoint
                    response = self.api.get("pending-volumes")
                    pending = response.json()
                    logging.info(
                        f"Global fallback returned {len(pending)} pending volumes."
                    )
                    return pending
                except Exception as e2:
                    logging.error(f"Failed to get pending volumes from fallback: {e2}")
            else:
                logging.error(
                    f"Failed to get pending volumes (project {project_id}): {e}"
                )
            raise
        except Exception as e:
            logging.error(f"Failed to get pending volumes: {e}")
            raise

    def create_pending_volume(self, project_id: int, media_type: str) -> Dict[str, Any]:
        """
        Create a pending volume in a project.
        If a pending volume already exists, returns the existing one.
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
            # Handle "Only a single pending volume can be created at a time"
            if "Only a single pending volume" in str(e):
                logging.info(
                    "Pending volume already exists. Attempting to fetch existing one..."
                )
                try:
                    existing_pending = self.get_pending_volumes(project_id)
                    logging.info(
                        f"Found {len(existing_pending)} existing pending volumes."
                    )
                    for pv in existing_pending:
                        logging.info(
                            f"Checking pending volume: id={pv.get('id')}, media_type={pv.get('media_type')}"
                        )
                        if pv.get("media_type") == media_type:
                            logging.info(
                                f"Re-using existing pending volume ID: {pv['id']} (media_type={media_type})"
                            )
                            return pv
                    if existing_pending:
                        logging.warning(
                            f"Found pending volume {existing_pending[0]['id']} but media_type differs ({existing_pending[0].get('media_type')} != {media_type})."
                        )
                        return existing_pending[0]
                except Exception as list_err:
                    logging.warning(f"Could not list pending volumes: {list_err}")

                logging.error("Could not find existing pending volume to reuse.")
                raise RuntimeError(
                    "A pending volume already exists in Biigle for this user/project, and I could not retrieve its ID. Please delete it in the Biigle UI (under Projects > Pending Volumes) and try again."
                ) from e

            logging.error(f"Failed to create pending volume: {e}")
            raise

    def setup_volume_with_files(
        self, pending_volume_id: int, volume_name: str, s3_url: str, files: List[str]
    ) -> Dict[str, Any]:
        """Configure a pending volume with name, S3 URL, and file list."""
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

    def create_volume_from_s3_files(
        self,
        volume_name: str,
        s3_url: str,
        files: List[str],
        project_id: int,
        media_type: str,
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
            pending_id = pending["id"]
            volume_info = self.setup_volume_with_files(
                pending_id, volume_name, s3_url, files
            )
            logging.info(f"Created volume '{volume_name}' with {len(files)} files")

            # The pending volume ID is temporary — resolve the real (finalized) volume ID
            # by looking it up in the project's volume list.
            real_id = self.resolve_real_volume_id(volume_name, project_id)
            if real_id and real_id != pending_id:
                logging.info(
                    f"Resolved real volume ID: {pending_id} (pending) → {real_id} (finalized)"
                )
                volume_info["id"] = real_id
            elif not real_id:
                logging.warning(
                    f"Could not resolve real volume ID for '{volume_name}'. "
                    f"Storing pending ID {pending_id} — sync checks may fail until Biigle processes the volume."
                )

            return volume_info
        except Exception as e:
            logging.error(f"Failed to create volume from S3 files: {e}")
            raise

    def resolve_real_volume_id(
        self, volume_name: str, project_id: Optional[int] = None
    ) -> Optional[int]:
        """
        After creating a pending volume, Biigle assigns the finalized volume a different ID.
        This method polls the project's real volumes list until it finds one matching
        `volume_name`, returning the real volume ID.

        Args:
            volume_name: The name used when creating the volume.
            project_id: The Biigle project to search in.

        Returns:
            The real volume ID, or None if not found within the retry window.
        """
        max_tries = config.volume_finalize_max_retries
        poll_interval = config.volume_finalize_retry_interval_secs
        project_id = project_id or config.biigle_project_id
        for attempt in range(1, max_tries + 1):
            try:
                volumes = self.get_volumes(project_id)
                # Collect all volumes with a matching name
                matches = [v for v in volumes if v.get("name") == volume_name]

                if matches:
                    # Sort by created_at descending (ISO format strings sort correctly)
                    # Example: "2024-03-01T10:00:00Z" > "2024-02-01T10:00:00Z"
                    matches.sort(key=lambda x: x.get("created_at", ""), reverse=True)
                    best_match = matches[0]

                    if len(matches) > 1:
                        logging.info(
                            f"Multiple volumes found with name '{volume_name}'. Selecting most recent: ID {best_match['id']} (created {best_match.get('created_at')})"
                        )

                    return best_match["id"]
            except Exception as e:
                logging.warning(
                    f"Attempt {attempt}/{max_tries}: failed to list volumes — {e}"
                )

            if attempt < max_tries:
                logging.info(
                    f"Volume '{volume_name}' not yet visible in project list. "
                    f"Attempt {attempt}/{max_tries}. Retrying in {poll_interval}s..."
                )
                time.sleep(poll_interval)

        logging.warning(
            f"Could not find finalized volume '{volume_name}' after {max_tries} attempts."
        )
        return None

    def build_s3_url(self, s3_path: str, disk_id: Optional[int] = None) -> str:
        """
        Build a Biigle-compatible S3 URL from an S3 path.

        """
        disk_id = disk_id or config.disk_id
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
            return {
                "tree_id": tree_id,
                "tree_info": tree_info,
                "labels": created_labels,
            }
        except Exception as e:
            logging.error(f"Failed to create label tree: {e}")
            raise

    def get_label_tree_labels(self, tree_id: int) -> List[Dict[str, Any]]:
        """
        Get all labels in a label tree.
        Uses GET /api/v1/label-trees/{id}/labels
        """
        try:
            response = self.api.get(f"label-trees/{tree_id}/labels")
            return response.json()
        except Exception as e:
            logging.error(f"Failed to list labels for label tree {tree_id}: {e}")
            raise

    # ── Reports ───────────────────────────────────────────────────────────────

    def create_report(
        self,
        resource: ResourceType,
        resource_id: int,
        type_id: int = config.annotation_report_type_video,
    ) -> int:
        """
        Request a BIIGLE annotation report. Returns the report ID.
        Reports are generated asynchronously — use download_report_zip_bytes() to fetch.
        Defaults to the video annotation CSV report type (config.annotation_report_type_video).
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
                elif (
                    allow_nested
                    and name.lower().endswith(".zip")
                    and _depth < MAX_DEPTH
                ):
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
        type_id: int = config.annotation_report_type_video,
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

    def get_volume_info(self, volume_id: int) -> Dict[str, Any]:
        """
        Get metadata for a single volume, including its media_type ('image' or 'video').
        Uses GET /api/v1/volumes/{id}
        """
        try:
            response = self.api.get(f"volumes/{volume_id}")
            return response.json()
        except Exception as e:
            logging.error(f"Failed to get info for volume {volume_id}: {e}")
            raise

    def get_volume_file_ids(self, volume_id: int) -> List[int]:
        """
        Get the list of file IDs (video or image) for a volume.
        Uses GET /api/v1/volumes/{id}/files
        """
        try:
            response = self.api.get(f"volumes/{volume_id}/files")
            file_ids = response.json()
            logging.info(f"Volume {volume_id} has {len(file_ids)} file(s).")
            return file_ids
        except Exception as e:
            logging.error(f"Failed to get files for volume {volume_id}: {e}")
            raise

    def get_volume_images(self, volume_id: int) -> List[Dict[str, Any]]:
        """
        Get the full information for all images in a volume by first getting the list of file IDs,
        then fetching each image's details concurrently.
        Returns list of {"id": int, "filename": str, ...}
        """
        import concurrent.futures

        try:
            file_ids = self.get_volume_file_ids(volume_id)
            if not file_ids:
                return []

            def _fetch_img(fid: int):
                try:
                    return self.api.get(f"images/{fid}").json()
                except Exception as e:
                    logging.warning(f"Could not fetch image {fid}: {e}")
                    return None

            images = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
                for img in executor.map(_fetch_img, file_ids):
                    if img:
                        images.append(img)

            logging.info(
                f"Retrieved {len(images)} images concurrently for volume {volume_id}."
            )
            return images
        except Exception as e:
            logging.error(f"Failed to get images for volume {volume_id}: {e}")
            raise

    def get_video_labels(self, video_id: int) -> List[Dict[str, Any]]:
        """
        Get whole-video labels for a specific video file.
        These are the "Done" / status labels applied to the entire video (not individual annotations).
        Uses GET /api/v1/videos/{id}/labels
        """
        try:
            response = self.api.get(f"videos/{video_id}/labels")
            labels = response.json()
            logging.info(f"Video {video_id} has {len(labels)} video label(s).")
            return labels
        except Exception as e:
            logging.error(f"Failed to get labels for video {video_id}: {e}")
            raise

    def get_image_labels(self, image_id: int) -> List[Dict[str, Any]]:
        """
        Get whole-image labels for a specific image file.
        Uses GET /api/v1/images/{id}/labels
        """
        try:
            response = self.api.get(f"images/{image_id}/labels")
            labels = response.json()
            logging.info(f"Image {image_id} has {len(labels)} image label(s).")
            return labels
        except Exception as e:
            logging.error(f"Failed to get labels for image {image_id}: {e}")
            raise

    def volume_is_done(self, volume_id: int):
        """
        Check whether the first file in a volume has a whole-file label
        indicating it is done. The done labels are configured via
        config.done_labels (defaults: 'Done Volume' and 'Done QA Review').
        Uses the volume's media_type (from its metadata) to call the correct label endpoint.

        Returns:
            Tuple of (is_done: bool, media_type: str) where media_type is 'image' or 'video'.
        """
        done_labels = config.done_labels

        try:
            # Get media_type directly from volume metadata — no guessing needed
            volume_info = self.get_volume_info(volume_id)
            media_type = volume_info.get("media_type", "image")

            file_ids = self.get_volume_file_ids(volume_id)
            if not file_ids:
                logging.warning(f"Volume {volume_id} has no files.")
                return False, media_type

            # Only the first file has the status labels
            first_id = file_ids[0]
            get_labels = (
                self.get_image_labels
                if media_type == "image"
                else self.get_video_labels
            )

            try:
                labels = get_labels(first_id)
            except Exception as e:
                logging.warning(
                    f"Could not get {media_type} labels for file {first_id}: {e}"
                )
                return False, media_type

            # Both labels must be present (exact match, as they appear in Biigle)
            found_labels = {
                label_entry.get("label", {}).get("name", "") for label_entry in labels
            }
            if all(lbl in found_labels for lbl in done_labels):
                logging.info(
                    f"All done labels {done_labels} found on {media_type} {first_id} in volume {volume_id}."
                )
                return True, media_type

            return False, media_type

        except Exception as e:
            logging.error(f"Error checking done status for volume {volume_id}: {e}")
            raise

    # ── Annotations Upload ────────────────────────────────────────────────────

    def upload_image_annotations(
        self, volume_id: int, annotations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Upload multiple image annotations via bulk endpoint.
        Uses POST /api/v1/image-annotations

        Input list format:
        [
            {
                "image_id": 12345,
                "shape_id": 5,          # 5 = Rectangle
                "points": [x1, y1, x2, y1, x2, y2, x1, y2], # 8 points flat mapped
                "label_id": 6789,
                "confidence": 0.85
            },
            ...
        ]
        """
        if not annotations:
            logging.warning("No annotations to bulk upload.")
            return {}

        logging.info(
            f"Bulk uploading {len(annotations)} image annotations to volume {volume_id} in batches of 100..."
        )

        final_result = []
        try:
            for i in range(0, len(annotations), 100):
                batch = annotations[i : i + 100]
                logging.info(
                    f"Uploading batch {i // 100 + 1}/{(len(annotations) - 1) // 100 + 1} ({len(batch)} items)"
                )
                response = self.api.post("image-annotations", json=batch)

                # Biigle API returns an array, but if we encounter errors, we capture the exception below
                batch_result = response.json()
                if isinstance(batch_result, list):
                    final_result.extend(batch_result)
                elif isinstance(batch_result, dict):
                    final_result.append(batch_result)

            logging.info(
                f"Successfully uploaded {len(annotations)} annotations in {(len(annotations) - 1) // 100 + 1} batches."
            )
            return {"uploaded_count": len(final_result), "details": final_result}

        except Exception as e:
            logging.error(
                f"Failed to bulk upload image annotations to volume {volume_id}: {e}"
            )
            if hasattr(e, "response") and hasattr(e.response, "text"):
                logging.error(f"Response: {e.response.text}")
            raise
