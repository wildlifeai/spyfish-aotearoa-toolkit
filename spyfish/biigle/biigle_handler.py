"""
BIIGLE API Handler for Spyfish Aotearoa.

Provides functionality to interact with the Biigle API:
- Creating and managing volumes (S3-backed)
- Setting up label trees from a CSV of scientific names
- Exporting annotation reports and reading them as DataFrames
"""

import io
import logging
import threading
import time
import zipfile
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Dict, List, Literal, Optional, Tuple

import pandas as pd
from requests.exceptions import HTTPError  # type: ignore

from spyfish.biigle.external.biigle_api import Api
from spyfish.config.wrapper import config

ResourceType = Literal["volumes", "projects"]
MAX_DEPTH = 2


class BiigleRateLimitError(HTTPError):
    """BIIGLE kept answering 429 after the bounded waits; stop, do not retry.

    An HTTPError carrying the 429 response, so handlers that look at
    `e.response.status_code` keep working. Once raised, every later BIIGLE
    request in this process raises it again without being sent (see
    `_RateLimit`): the account was locked on 2026-10-16 by code that carried
    on after a 429.
    """


class _RateLimit:
    """Process-wide 429 state, shared by every handler and every thread.

    `resume_at` is the moment (time.monotonic) before which no request may
    start: one 429 pauses all threads for the server's Retry-After, not only
    the thread that received it. `gave_up` holds the 429 response that used up
    the retries; from then on nothing is sent for the rest of the process.
    """

    lock = threading.Lock()
    resume_at = 0.0
    gave_up: Any = None

    @classmethod
    def reset(cls) -> None:
        """Clear the state (tests; a long-lived session after the limit lifts)."""
        with cls.lock:
            cls.resume_at = 0.0
            cls.gave_up = None


def _retry_after_secs(response: Any) -> Optional[float]:
    """Seconds the 429's Retry-After header asks for, None if absent/unreadable.

    The header is either a number of seconds or an HTTP date.
    """
    raw = (getattr(response, "headers", None) or {}).get("Retry-After")
    if raw is None:
        return None
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        pass
    try:
        when = parsedate_to_datetime(str(raw))
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return max(0.0, (when - datetime.now(timezone.utc)).total_seconds())


class _ApiWithTimeout(Api):
    """Thin subclass that injects a timeout and 429 handling into every request.

    The vendored biigle_api.py must not be modified (upstream file). This
    subclass overrides call() to set a default timeout so no request can hang
    indefinitely. Callers can still override per-request by passing timeout=N.

    A 429 is waited out for as long as its Retry-After says (or
    `fallback_wait` when the header is missing) and the request is sent again,
    at most `max_retries` times. A 429 means the request was not processed, so
    resending is safe for POSTs too. Still 429 after that, or a Retry-After
    longer than `max_wait`, raises `BiigleRateLimitError` and ends BIIGLE
    traffic for the process. No other status is retried.
    """

    def __init__(
        self,
        *args: Any,
        timeout: int = 30,
        max_retries: int = 0,
        fallback_wait: float = 60.0,
        max_wait: float = 300.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._timeout = timeout
        self._max_retries = max_retries
        self._fallback_wait = fallback_wait
        self._max_wait = max_wait

    @staticmethod
    def _wait_until_allowed(url: str) -> None:
        """Sleep through a pause another request's 429 set; raise once given up."""
        while True:
            with _RateLimit.lock:
                if _RateLimit.gave_up is not None:
                    raise BiigleRateLimitError(
                        f"BIIGLE rate limit (429) was hit earlier in this run and "
                        f"the retries are used up; not sending {url}. Wait before "
                        "running again.",
                        response=_RateLimit.gave_up,
                    )
                delay = _RateLimit.resume_at - time.monotonic()
            if delay <= 0:
                return
            time.sleep(delay)

    def call(
        self,
        method: Any,
        url: str,
        raise_for_status: bool = True,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        kwargs.setdefault("timeout", self._timeout)
        attempt = 0
        while True:
            self._wait_until_allowed(url)
            try:
                response = super().call(method, url, raise_for_status, *args, **kwargs)
            except HTTPError as e:
                if e.response is None or e.response.status_code != 429:
                    raise
                response = e.response
            if response.status_code != 429:
                return response

            asked = _retry_after_secs(response)
            wait = self._fallback_wait if asked is None else asked
            if attempt >= self._max_retries or wait > self._max_wait:
                with _RateLimit.lock:
                    _RateLimit.gave_up = response
                raise BiigleRateLimitError(
                    f"BIIGLE rate limit (429) on {url}: "
                    + (
                        f"Retry-After {wait:.0f}s is over the "
                        f"{self._max_wait:.0f}s this run will wait"
                        if wait > self._max_wait
                        else f"still limited after {attempt} wait(s)"
                    )
                    + ". No further BIIGLE requests will be sent by this run.",
                    response=response,
                )
            attempt += 1
            with _RateLimit.lock:
                _RateLimit.resume_at = max(
                    _RateLimit.resume_at, time.monotonic() + wait
                )
            logging.warning(
                f"BIIGLE rate limit (429) on {url}: pausing all BIIGLE requests "
                f"{wait:.0f}s "
                + ("(Retry-After)" if asked is not None else "(no Retry-After header)")
                + f", retry {attempt}/{self._max_retries}."
            )


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
            self.api = _ApiWithTimeout(
                self.email,
                self.token,
                timeout=config.request_timeout_secs,
                max_retries=config.rate_limit_max_retries,
                fallback_wait=config.rate_limit_fallback_wait_secs,
                max_wait=config.rate_limit_max_wait_secs,
            )
        except Exception as e:
            raise Exception(f"Failed to initialize BIIGLE API: {e}") from e
        logging.info("BiigleHandler initialized successfully")

    # ── Projects & Volumes ────────────────────────────────────────────────────

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
        """Configure a *pending* volume with name, S3 URL, and file list.

        Pending volumes are configured atomically in a single PUT (name + url +
        files). After Biigle finalizes the volume, this endpoint no longer
        applies, to append more files later, use `add_files_to_volume`.
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

            # The pending volume ID is temporary, resolve the real (finalized) volume ID
            # by looking it up in the project's volume list.
            real_id = self.resolve_real_volume_id(volume_name, project_id)
            if not real_id:
                raise RuntimeError(
                    f"Biigle volume '{volume_name}' was submitted but did not appear in the "
                    f"project volume list after {config.volume_finalize_max_retries} attempts. "
                    "The pending volume ID is transient and cannot be used for future sync. "
                    "Check the Biigle project manually and retry."
                )
            if real_id != pending_id:
                logging.info(
                    f"Resolved real volume ID: {pending_id} (pending) → {real_id} (finalized)"
                )
            volume_info["id"] = real_id

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
            except BiigleRateLimitError:
                raise  # polling on would only add requests to a limited account
            except Exception as e:
                logging.warning(
                    f"Attempt {attempt}/{max_tries}: failed to list volumes, {e}"
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

    def add_files_to_volume(
        self, volume_id: int, files: List[str]
    ) -> List[Dict[str, Any]]:
        """Add filenames to an existing (finalized) image volume.

        Counterpart to `setup_volume_with_files`: that one configures a
        *pending* volume in one shot via PUT, this one appends to a
        *finalized* one. Use the right method for the lifecycle stage.

        API: ``POST volumes/{id}/files`` with body ``{"files": [...]}``,
        verified against the Biigle apidoc (group "Volumes", title
        "Add images/videos"). Caller must be a project admin. Filenames must
        already be present in the volume's S3 folder; this call only registers
        them in the Biigle volume.

        Args:
            volume_id: Finalized Biigle volume ID (NOT a pending-volume ID).
            files: Filenames (basenames) to add.

        Returns:
            List of ``{"id": <image_id>, "filename": <name>}`` for the
            newly-added images, per the documented response shape.
        """
        if not files:
            logging.info(f"add_files_to_volume({volume_id}): no files to add")
            return []
        try:
            response = self.api.post(
                f"volumes/{volume_id}/files", json={"files": files}
            )
            payload = response.json() if response.content else []
            logging.info(
                f"Added {len(files)} file(s) to volume {volume_id} "
                f"(API returned {len(payload) if isinstance(payload, list) else '?'} entries)"
            )
            return payload if isinstance(payload, list) else []
        except Exception as e:
            logging.error(
                f"Failed to add {len(files)} file(s) to volume {volume_id}: {e}"
            )
            raise

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
                    "visibility": 2,  # Private; BIIGLE renamed visibility_id -> visibility
                    "visibility_id": 2,  # older servers
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

        Biigle's `label-trees/{id}/labels` route is POST-only (create a label).
        To list, fetch the tree itself and pluck its nested `labels` array.
        """
        try:
            response = self.api.get(f"label-trees/{tree_id}")
            return response.json().get("labels", [])
        except Exception as e:
            logging.error(f"Failed to list labels for label tree {tree_id}: {e}")
            raise

    # ── Reports ───────────────────────────────────────────────────────────────

    def create_report(
        self,
        resource: ResourceType,
        resource_id: int,
        type_id: Optional[int] = None,
    ) -> int:
        """
        Request a BIIGLE annotation report. Returns the report ID.
        Reports are generated asynchronously, use download_report_zip_bytes() to fetch.
        Defaults to the video annotation CSV report type (config.annotation_report_type_video).

        `type_id` defaults to None and is resolved to config at call time rather
        than at class-definition time, evaluating `config.x` in a default arg
        runs at module import, which is brittle if import order changes.
        """
        if type_id is None:
            type_id = config.annotation_report_type_video
        resp = self.api.post(
            f"{resource}/{resource_id}/reports",
            # BIIGLE renamed type_id -> type when reports moved into core
            # (2026-10); send both so old and new servers accept it
            json={"type": type_id, "type_id": type_id},
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
        max_tries: Optional[int] = None,
        poll_interval: Optional[float] = None,
    ) -> bytes:
        """
        Poll until a report ZIP is ready and return its raw bytes.
        BIIGLE reports are generated asynchronously (may take up to ~2 min).

        Retry parameters default to None and are resolved from config at call
        time (`biigle.report_download_max_retries` and
        `biigle.report_download_retry_interval_secs`).
        """
        if max_tries is None:
            max_tries = config.report_download_max_retries
        if poll_interval is None:
            poll_interval = config.report_download_retry_interval_secs
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
        type_id: Optional[int] = None,
        source_col: str = "source_file",
    ) -> pd.DataFrame:
        """
        High-level helper: request report → wait → download ZIP → read all CSVs → return DataFrame.
        Defaults to the video annotation CSV report type (resolved at call time).
        """
        # `create_report` resolves type_id=None → config.annotation_report_type_video
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
        Every image of a volume as ``{"id": int, "filename": str}``.

        One request: GET /api/v1/volumes/{id}/filenames returns the whole
        id -> filename map. The previous approach (``/files`` for the ids, then
        one ``/images/{id}`` per image) cost a request per image, hit BIIGLE's
        rate limit (429) on a 79-volume backfill, and silently dropped every
        image whose request failed. Falls back to that per-image path only if
        the endpoint is missing, and then raises rather than return a list with
        holes: callers write "frames the expert was shown" from this, and a
        short list is wrong data, not partial data.
        """
        import requests

        try:
            response = self.api.get(f"volumes/{volume_id}/filenames")
            mapping = response.json()
            images = [{"id": int(i), "filename": f} for i, f in mapping.items()]
            logging.info(f"Volume {volume_id}: {len(images)} filename(s).")
            return images
        except requests.exceptions.HTTPError as e:
            if e.response is None or e.response.status_code != 404:
                logging.error(f"Failed to get filenames for volume {volume_id}: {e}")
                raise
        logging.warning(
            f"Volume {volume_id}: /filenames unavailable, falling back to one "
            f"request per image."
        )
        return self._get_volume_images_per_image(volume_id)

    def _get_volume_images_per_image(self, volume_id: int) -> List[Dict[str, Any]]:
        """Per-image fallback for get_volume_images; raises if any image fails."""
        import concurrent.futures

        file_ids = self.get_volume_file_ids(volume_id)
        if not file_ids:
            return []
        failed: List[int] = []

        def _fetch_img(fid: int):
            try:
                return self.api.get(f"images/{fid}").json()
            except Exception as e:
                logging.warning(f"Could not fetch image {fid}: {e}")
                failed.append(fid)
                return None

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            images = [img for img in executor.map(_fetch_img, file_ids) if img]
        if failed:
            raise RuntimeError(
                f"Volume {volume_id}: {len(failed)} of {len(file_ids)} image "
                f"lookup(s) failed; refusing to return an incomplete list."
            )
        logging.info(f"Retrieved {len(images)} images for volume {volume_id}.")
        return images

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

    def create_label(
        self,
        tree_id: int,
        name: str,
        color: str,
        parent_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Create one label in a label tree. POST label-trees/{id}/labels

        Returns the created label dict. Biigle answers this endpoint with a
        LIST containing the new label (not the bare object), so the list is
        unwrapped here and callers get the shape they expect.

        `label_source_id` is deliberately not sent: it is only meaningful
        alongside a `source_id` from an external authority such as WoRMS, and
        these are internal workflow labels with no such reference.

        Note two Biigle constraints before designing around this: a label's
        `parent_id` cannot later be set back to null, and a label with children
        cannot be deleted. A hierarchy is easy to create and awkward to undo.
        """
        try:
            payload: Dict[str, Any] = {"name": name, "color": color}
            if parent_id is not None:
                payload["parent_id"] = int(parent_id)

            response = self.api.post(f"label-trees/{tree_id}/labels", json=payload)
            created = response.json()
            if isinstance(created, list):
                if not created:
                    raise ValueError(
                        f"Biigle returned an empty list creating label {name!r} "
                        f"in tree {tree_id}."
                    )
                created = created[0]

            logging.info(
                f"Created label {name!r} (id={created.get('id')}) in tree {tree_id}"
                + (f" under parent {parent_id}" if parent_id else "")
            )
            return created
        except Exception as e:
            logging.error(f"Failed to create label {name!r} in tree {tree_id}: {e}")
            raise

    def attach_image_label(self, image_id: int, label_id: int) -> Optional[dict]:
        """Attach a whole-image label. POST images/{id}/labels

        Returns the created image-label dict, or None when the attach failed.
        Failure is returned rather than raised because these labels are
        review metadata: losing one costs the expert a hint, while raising
        would abort an upload whose frames and annotations already landed.

        Re-attaching a label the image already carries is the expected re-run
        case and is reported by Biigle as an error; the caller counts those
        separately rather than pre-fetching every image's labels, which would
        double the request count to prevent a harmless no-op.

        The one failure that IS raised is `BiigleRateLimitError`: a hint is not
        worth another request to an account that is being rate limited.
        """
        try:
            response = self.api.post(
                f"images/{image_id}/labels", json={"label_id": int(label_id)}
            )
            return response.json()
        except BiigleRateLimitError:
            raise
        except Exception as e:
            logging.warning(
                f"Could not attach label {label_id} to image {image_id}: {e}"
            )
            return None

    def attach_image_labels(self, pairs: List[Tuple[int, int]]) -> int:
        """Attach many whole-image labels. Returns the success count.

        Biigle has no bulk image-label endpoint (unlike annotations, which take
        an array), so this is genuinely one request per pair. They go out on
        `biigle.image_label_workers` threads: a few, because a drop is only
        some tens of pairs and a wide burst is what a rate limit punishes.

        Raises `BiigleRateLimitError` if BIIGLE stays at 429 after the bounded
        waits. Pairs not yet sent are then never sent.
        """
        if not pairs:
            return 0

        import concurrent.futures

        workers = config.image_label_workers
        logging.info(
            f"Attaching {len(pairs)} whole-image label(s) on {workers} thread(s)..."
        )
        stop = threading.Event()

        def _attach(pair: Tuple[int, int]) -> Optional[dict]:
            if stop.is_set():
                return None
            try:
                return self.attach_image_label(pair[0], pair[1])
            except BiigleRateLimitError:
                stop.set()
                raise

        attached = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            results = executor.map(_attach, pairs)
            for result in results:
                if result is not None:
                    attached += 1

        if attached < len(pairs):
            logging.warning(
                f"Attached {attached}/{len(pairs)} image label(s); the rest were "
                "already present or rejected (see warnings above)."
            )
        else:
            logging.info(f"Attached {attached} image label(s).")
        return attached

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
            # Get media_type directly from volume metadata, no guessing needed
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

            # Both labels must be present (exact match, as they appear in Biigle).
            # Biigle whole-file label entries are nested: {"label": {"name": "..."}, ...}.
            # Flat fallback ("name" at top level) guards against API shape changes.
            found_labels = set()
            for label_entry in labels:
                name = label_entry.get("label", {}).get("name", "")
                if not name:
                    name = label_entry.get("name", "")
                if name:
                    found_labels.add(name)

            if labels and not found_labels:
                logging.warning(
                    f"Could not extract label names from {len(labels)} label entries "
                    f"on {media_type} {first_id} (volume {volume_id}). "
                    f"Sample entry: {labels[0]}"
                )

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
                "shape": 5,             # 5 = Rectangle (BIIGLE renamed shape_id -> shape)
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
