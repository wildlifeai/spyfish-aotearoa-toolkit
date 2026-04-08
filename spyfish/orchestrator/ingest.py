import logging
from typing import Optional, Set

import pandas as pd

from spyfish.config.base import PipelineStatus, SourceStatus, VideoPresence
from spyfish.config.wrapper import config
from spyfish.database.manager import DatabaseManager
from spyfish.storage.s3_handler import S3Handler
from spyfish.validation.data_validator import DataValidator


def check_pending_arrivals(known_files: Optional[Set[str]] = None):
    """
    Checks S3 for videos of deployments in PENDING_ARRIVAL status.
    Advances them to READY_FOR_ML if found.

    Args:
        known_files: Optional pre-fetched set of S3 'media/' file paths.
                    If None, it will be fetched from S3.
    """
    db = DatabaseManager()
    pending = db.get_deployments_by_status(PipelineStatus.PENDING_ARRIVAL)

    if not pending:
        logging.info("No PENDING_ARRIVAL drops found.")
        return

    logging.info(f"Checking S3 for {len(pending)} PENDING_ARRIVAL drops...")

    if known_files is None:
        storage = S3Handler(bucket=config.s3_bucket)
        logging.info("Downloading master file list from S3 bucket (prefix: media/)...")
        known_files = set(storage.get_file_paths_set_from_s3(prefix="media/"))

    updated_count = 0
    for drop in pending:
        drop_id = drop["drop_id"]
        video_path = drop["video_path"]

        # Skip deployments with source data issues — they shouldn't enter the pipeline
        if drop.get("source_status", SourceStatus.OK) != SourceStatus.OK:
            continue

        if video_path and video_path in known_files:
            logging.info(
                f"✅ Video confirmed for {drop_id}. Updating status to {PipelineStatus.READY_FOR_ML}."
            )
            db.update_deployment_fields(drop_id, video_presence=VideoPresence.PRESENT)
            db.advance_status(drop_id, PipelineStatus.READY_FOR_ML)
            updated_count += 1

    logging.info(
        f"Arrival check complete. Advanced {updated_count} drops to {PipelineStatus.READY_FOR_ML}."
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

    # Fetch File List & Deployment CSV (One Journey, One S3 Scan)
    logging.info("Fetching known media files and master deployments list from S3...")
    known_files = set(storage.get_file_paths_set_from_s3(prefix="media/"))

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
        for idx, e in validator.errors_df.iterrows():
            structured_errors.append(
                {
                    "SurveyID": str(e.SurveyID) if pd.notna(e.SurveyID) else "",
                    "DropID": str(e.DropID) if pd.notna(e.DropID) else "",
                    "ErrorType": str(e.ErrorType),
                    "FileName": str(e.FileName),
                    "ColumnName": str(e.ColumnName) if pd.notna(e.ColumnName) else "",
                    "ErrorMessage": str(e.ErrorMessage),
                    "InvalidValue": (
                        str(e.InvalidValue) if pd.notna(e.InvalidValue) else ""
                    ),
                }
            )
            if pd.notna(e.DropID):
                structural_error_drops.add(str(e.DropID).strip())

    logging.debug(
        f"Found {len(structural_error_drops)} DropIDs with structural CSV errors, logging them into DB."
    )
    db.clear_validation_errors()
    db.add_validation_errors(structured_errors)

    # Load expert annotations and count per DropID (always from S3)
    logging.debug("Fetching expert annotations from S3...")
    expert_counts = {}
    try:
        annotations_df = storage.read_df_from_s3_csv(
            config.s3_sharepoint_annotations_legacy_experts_csv
        )
        if not annotations_df.empty:
            expert_counts = (
                annotations_df[config.drop_id_column].value_counts().to_dict()
            )
            logging.info(
                f"Loaded {len(annotations_df)} annotation rows covering {len(expert_counts)} deployments."
            )
    except Exception as e:
        logging.warning(
            f"Failed to load KSO annotations from S3: {e}. Expert counts will default to 0."
        )

    # Load sites CSV and cache in DB so upload step doesn't need S3
    try:
        sites_df = storage.read_df_from_s3_csv(config.s3_sharepoint_site_csv)
        db.upsert_sites(sites_df)
    except Exception as e:
        logging.warning(
            f"Failed to load sites CSV from S3: {e}. Site metadata will not be updated in DB."
        )

    _sync_deployments_to_db(
        deployments_df, db, structural_error_drops, known_files, expert_counts, mapping
    )
    logging.info(
        f"Ingestion complete. Synchronized {len(deployments_df)} records into the pipeline database."
    )

    # After full ingestion, check for arrivals of drops that WERE already PENDING in the DB
    check_pending_arrivals(known_files=known_files)


def _sync_deployments_to_db(
    deployments_df, db, structural_error_drops, known_files, expert_counts, mapping
):
    drop_col = mapping.get("drop_id_column")
    bad_col = mapping.get("is_bad_deployment_column")
    video_col = mapping.get("video_file_link_column")

    new_count = 0
    expt_count = 0
    existing_deployments = db.get_all_deployments_map()

    # Iterate over all drops and sync them into our SQLite brain
    for _, row in deployments_df.iterrows():
        # Strict validation of drop_id to prevent path traversal and ensure format
        try:
            raw_drop_id = str(row.get(drop_col, "")).strip()
            drop_id = config.validate_drop_id(raw_drop_id)
        except ValueError:
            continue  # invalid DropID format — error surfaced by DataValidator

        video_path = str(row.get(video_col, "")).strip()

        # Parse IsBadDeployment column strictly
        is_bad_deployment = str(row.get(bad_col, "")).strip() == "True"

        # Parse Sampling offsets — these MUST exist in the BUV Deployment CSV
        try:
            # We strictly require these to be numeric
            sampling_start = float(row[config.csv_sampling_start_column])
            sampling_end = float(row[config.csv_sampling_end_column])
        except (ValueError, TypeError, KeyError):
            if drop_id not in structural_error_drops:
                structural_error_drops.add(drop_id)
            sampling_start = None
            sampling_end = None

        expert_anns = expert_counts.get(drop_id, 0)
        ml_anns = 0
        citsci_anns = 0

        # Determine source_status from data quality checks (orthogonal to pipeline stage)
        if is_bad_deployment:
            source_status = SourceStatus.EXCLUDED
        elif drop_id in structural_error_drops:
            source_status = SourceStatus.VALIDATION_ERROR
        else:
            source_status = SourceStatus.OK

        # Determine video_presence from current S3 state (updated every ingest run)
        if is_bad_deployment:
            video_presence = VideoPresence.NO_VIDEO_BAD_DEP
        elif video_path and video_path in known_files:
            video_presence = VideoPresence.PRESENT
        else:
            video_presence = VideoPresence.ABSENT

        # Determine pipeline status
        if expert_anns > 0:
            # Legacy data: deployments that already have expert annotations (pre-pipeline)
            # are set to PIPELINE_COMPLETE immediately, skipping ML and citsci stages.
            # TODO: verify this doesn't interfere with drops that have expert annotations
            # but still need ML inference (e.g. re-annotated surveys).
            status = PipelineStatus.PIPELINE_COMPLETE
        elif source_status != SourceStatus.OK:
            # Don't advance source-problematic deployments through the pipeline.
            # Preserve existing stage if already in DB, otherwise default to PENDING_ARRIVAL.
            existing_record = existing_deployments.get(drop_id)
            status = (
                existing_record["status"]
                if existing_record
                else PipelineStatus.PENDING_ARRIVAL
            )
        else:
            existing_record = existing_deployments.get(drop_id)
            if existing_record and existing_record["status"] not in [
                PipelineStatus.PENDING_ARRIVAL,
                PipelineStatus.ERROR,
            ]:
                # It's already moving through the pipeline, leave it alone.
                status = existing_record["status"]
            else:
                # Immediate File Check Optimization!
                if video_path and video_path in known_files:
                    status = PipelineStatus.READY_FOR_ML
                else:
                    status = PipelineStatus.PENDING_ARRIVAL

        db.add_or_update_deployment(
            drop_id=drop_id,
            status=status,
            source_status=source_status,
            video_path=video_path,
            video_presence=video_presence,
            is_bad_deployment=is_bad_deployment,
            error_message=(
                "Found in structural errors"
                if source_status == SourceStatus.VALIDATION_ERROR
                else ""
            ),
            sampling_start=sampling_start,
            sampling_end=sampling_end,
            expert_annotations=expert_anns,
            ml_annotations=ml_anns,
            citsci_annotations=citsci_anns,
        )
        new_count += 1
        expt_count += expert_anns

    logging.info(
        f"Ingestion complete. Synchronized {new_count} records into the pipeline database, with {expt_count} of {sum(expert_counts.values())} expert annotations."
    )


if __name__ == "__main__":
    run_ingestion()
