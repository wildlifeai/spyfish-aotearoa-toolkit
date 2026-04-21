import logging
from typing import Dict, Optional, Set

import pandas as pd

from spyfish.config.base import (
    IngestStatus,
    MlStatus,
    VideoPresence,
)
from spyfish.config.wrapper import config
from spyfish.database.manager import DatabaseManager
from spyfish.storage.s3_handler import S3Handler
from spyfish.validation.data_validator import DataValidator


def check_pending_arrivals(
    known_files: Optional[Set[str]] = None,
    media_file_info: Optional[Dict[str, str]] = None,
):
    """
    Advances ml_status for drops waiting on video arrival or DEEP_ARCHIVE restore.

    Picks up any drop with ml_status='pending' and video_presence in (absent, archived).
    For each, re-checks S3:
      - not present → leave as-is
      - present + DEEP_ARCHIVE → video_presence=archived (no ml advance)
      - present + downloadable → video_presence=present, advance ml_status to ready.
    """
    db = DatabaseManager()
    pending_all = db.get_deployments_eligible("ml_status", [MlStatus.PENDING])
    pending = [
        d
        for d in pending_all
        if d.get("video_presence") in (VideoPresence.ABSENT, VideoPresence.ARCHIVED)
    ]

    if not pending:
        logging.info("No pending-arrival drops found.")
        return

    logging.info(f"Checking S3 for {len(pending)} pending drops...")

    if known_files is None or media_file_info is None:
        storage = S3Handler(bucket=config.s3_bucket)
        logging.info(
            f"Downloading master file list from S3 bucket (prefix: {config.media_s3_prefix})..."
        )
        media_objects = storage.get_objects_from_s3(
            prefix=config.media_s3_prefix, keys_only=False
        )
        known_files = {obj["Key"] for obj in media_objects}
        media_file_info = {
            obj["Key"]: obj.get("StorageClass", "STANDARD") for obj in media_objects
        }

    updated_count = 0
    for drop in pending:
        drop_id = drop["drop_id"]
        video_path = drop["video_path"]
        current_presence = drop.get("video_presence")

        if not (video_path and video_path in known_files):
            continue

        storage_class = media_file_info.get(video_path)
        if storage_class == "DEEP_ARCHIVE":
            if current_presence != VideoPresence.ARCHIVED:
                db.update_deployment_fields(
                    drop_id, video_presence=VideoPresence.ARCHIVED
                )
                logging.info(f"{drop_id}: video in DEEP_ARCHIVE — marked archived.")
            continue

        logging.info(f"✅ Video confirmed for {drop_id}. Advancing ml_status → ready.")
        db.update_deployment_fields(drop_id, video_presence=VideoPresence.PRESENT)
        db.advance_status(drop_id, MlStatus.COLUMN, MlStatus.READY)
        updated_count += 1

    logging.info(
        f"Arrival check complete. Advanced {updated_count} drops to ml_status=ready."
    )


def run_ingestion():
    logging.info("Starting Spyfish Pipeline Ingestion...")

    mapping = config.csv_mapping
    if not mapping:
        raise ValueError(
            "Missing 'csv_mapping' in config.yaml. Cannot ingest CSV without column mappings."
        )

    drop_col = mapping.get("drop_id_column")
    bad_col = mapping.get("is_bad_deployment_column")
    video_col = mapping.get("video_file_link_column")

    if not all([drop_col, bad_col, video_col]):
        raise KeyError(
            f"Missing required CSV column mappings in config.yaml. Found: DropID={drop_col}, Bad={bad_col}, Video={video_col}"
        )

    db = DatabaseManager()
    storage = S3Handler(bucket=config.s3_bucket)

    logging.info("Fetching known media files and master deployments list from S3...")
    media_objects = storage.get_objects_from_s3(
        prefix=config.media_s3_prefix, keys_only=False
    )
    known_files = {obj["Key"] for obj in media_objects}
    media_file_info = {
        obj["Key"]: obj.get("StorageClass", "STANDARD") for obj in media_objects
    }

    csv_path = config.s3_sharepoint_deployment_csv
    deployments_df = storage.read_df_from_s3_csv(csv_path)

    logging.debug(
        f"Running full cross-dataset DataValidation on {len(deployments_df)} loaded deployment records...."
    )
    validator = DataValidator()
    validator.run_validation(
        file_presence=False,
        remove_duplicates=True,
        extract_clean_dataframes=False,
        known_files=known_files,
    )

    structural_error_drops = set()
    structured_errors = []

    if validator.errors_df is not None and not validator.errors_df.empty:
        # Dict-style access (e["SurveyID"]) rather than attribute-style
        # (e.SurveyID) — the latter silently breaks when a column name
        # collides with a pandas Series method (e.g. a column named `name`,
        # `count`, `size` would return the method instead of the value).
        for _, e in validator.errors_df.iterrows():
            structured_errors.append(
                {
                    "SurveyID": str(e["SurveyID"]) if pd.notna(e["SurveyID"]) else "",
                    "DropID": str(e["DropID"]) if pd.notna(e["DropID"]) else "",
                    "ErrorType": str(e["ErrorType"]),
                    "FileName": str(e["FileName"]),
                    "ColumnName": (
                        str(e["ColumnName"]) if pd.notna(e["ColumnName"]) else ""
                    ),
                    "ErrorMessage": str(e["ErrorMessage"]),
                    "InvalidValue": (
                        str(e["InvalidValue"]) if pd.notna(e["InvalidValue"]) else ""
                    ),
                }
            )
            if pd.notna(e["DropID"]):
                structural_error_drops.add(str(e["DropID"]).strip())

    logging.debug(
        f"Found {len(structural_error_drops)} DropIDs with structural CSV errors, logging them into DB."
    )
    db.clear_validation_errors()
    db.add_validation_errors(structured_errors)

    try:
        sites_df = storage.read_df_from_s3_csv(config.s3_sharepoint_site_csv)
        db.upsert_sites(sites_df)
    except Exception as e:
        logging.warning(
            f"Failed to load sites CSV from S3: {e}. Site metadata will not be updated in DB."
        )

    if not config.class_map_path.exists():
        try:
            from spyfish.biigle.class_map import (
                build_class_map_from_species,
                save_class_map,
            )

            species_df = storage.read_df_from_s3_csv(config.s3_sharepoint_species_csv)
            save_class_map(
                build_class_map_from_species(species_df), config.class_map_path
            )
        except Exception as e:
            logging.warning(
                f"Failed to seed class_map.json from species CSV: {e}. "
                f"Run `python -m spyfish.biigle.class_map` to retry."
            )

    _sync_deployments_to_db(
        deployments_df,
        db,
        structural_error_drops,
        known_files,
        media_file_info,
        mapping,
    )
    logging.info(
        f"Ingestion complete. Synchronized {len(deployments_df)} records into the pipeline database."
    )

    # After ingestion, check for video arrivals and DEEP_ARCHIVE restores
    check_pending_arrivals(known_files=known_files, media_file_info=media_file_info)


def _sync_deployments_to_db(
    deployments_df,
    db,
    structural_error_drops: set,
    known_files,
    media_file_info,
    mapping,
):
    """Upsert deployment rows into the pipeline DB.

    `structural_error_drops` is read-only here — drops flagged by the
    validator before we started. Per-row sampling-parse failures are
    tracked locally rather than mutated back into the caller's set, so
    this function stays pure with respect to its inputs.
    """
    drop_col = mapping.get("drop_id_column")
    bad_col = mapping.get("is_bad_deployment_column")
    video_col = mapping.get("video_file_link_column")

    new_count = 0

    for _, row in deployments_df.iterrows():
        try:
            raw_drop_id = str(row.get(drop_col, "")).strip()
            drop_id = config.validate_drop_id(raw_drop_id)
        except ValueError:
            logging.debug(
                f"Skipping row with invalid DropID format: {raw_drop_id!r} "
                "(error surfaced by DataValidator)"
            )
            continue

        video_path = str(row.get(video_col, "")).strip()
        if not video_path:
            survey_id = config.get_survey_id_from_drop(drop_id)
            video_path = f"media/{survey_id}/{drop_id}/{drop_id}.mp4"
        is_bad_deployment = str(row.get(bad_col, "")).strip() == "True"

        try:
            sampling_start = float(row[config.csv_sampling_start_column])
            sampling_end = float(row[config.csv_sampling_end_column])
            sampling_parse_failed = False
        except (ValueError, TypeError, KeyError):
            sampling_start = None
            sampling_end = None
            sampling_parse_failed = True

        sampling_window_errors: list[str] = []
        if not sampling_parse_failed and not is_bad_deployment:
            sampling_window_errors = config.validate_sampling_window(
                drop_id, sampling_start, sampling_end
            )

        # Determine ingest_status from data quality checks.
        # Sources of VALIDATION_ERROR:
        #   - drop was flagged by the validator upstream (structural_error_drops)
        #   - sampling_start / sampling_end couldn't be parsed from the row
        #   - sampling window is out of range (validate_sampling_window)
        if is_bad_deployment:
            ingest_status = IngestStatus.EXCLUDED
        elif (
            drop_id in structural_error_drops
            or sampling_parse_failed
            or sampling_window_errors
        ):
            ingest_status = IngestStatus.VALIDATION_ERROR
        else:
            ingest_status = IngestStatus.OK

        for msg in sampling_window_errors:
            db.add_validation_error(
                survey_id=config.get_survey_id_from_drop(drop_id),
                drop_id=drop_id,
                error_type=IngestStatus.VALIDATION_ERROR,
                column_name="sampling_window",
                error_message=msg,
            )

        # Determine video_presence from current S3 state.
        # For bad deployments: use NO_VIDEO_BAD_DEP only if the video is truly absent;
        # if a video was uploaded for a bad deployment, record it as PRESENT.
        if video_path and video_path in known_files:
            storage_class = media_file_info.get(video_path)
            if storage_class == "DEEP_ARCHIVE":
                video_presence = VideoPresence.ARCHIVED
            else:
                video_presence = VideoPresence.PRESENT
        elif is_bad_deployment:
            video_presence = VideoPresence.NO_VIDEO_BAD_DEP
        else:
            video_presence = VideoPresence.ABSENT

        # Determine initial ml_status. Only applied on INSERT — existing records
        # preserve their section statuses via the ON CONFLICT clause in
        # add_or_update_deployment. get_deployments_eligible filters ingest_status='ok'
        # so bad/excluded deployments won't be picked up by processing stages.
        # ml_status starts at 'ready' only when the video is confirmed present
        # and the deployment is in good standing; other sections default to
        # their respective 'pending' values via SQL defaults.
        if ingest_status == IngestStatus.OK and video_presence == VideoPresence.PRESENT:
            ml_status = MlStatus.READY
        else:
            ml_status = MlStatus.PENDING

        db.add_or_update_deployment(
            drop_id=drop_id,
            ingest_status=ingest_status,
            ml_status=ml_status,
            video_path=video_path,
            video_presence=video_presence,
            is_bad_deployment=is_bad_deployment,
            sampling_start=sampling_start,
            sampling_end=sampling_end,
        )
        new_count += 1

    logging.info(
        f"Ingestion complete. Synchronized {new_count} records into the pipeline database. "
        "Expert annotation counts will be populated by sync_annotation_counts() "
        "via ingest_legacy_expert_annotations()."
    )


if __name__ == "__main__":
    run_ingestion()
