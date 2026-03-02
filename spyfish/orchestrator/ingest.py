import os
import argparse
import logging
import yaml
from typing import Set
from pathlib import Path
import pandas as pd

from spyfish.config import config, PipelineStatus
from spyfish.database.manager import DatabaseManager
from spyfish.validation.data_validator import DataValidator
from spyfish.storage.s3_handler import S3Handler
from spyfish.storage.local import LocalStorageHandler

def run_ingestion():
    logging.info("Starting Spyfish Pipeline Ingestion...")

    mapping = config.csv_mapping
    if not mapping:
        raise ValueError("Missing 'csv_mapping' in config.yaml. Cannot ingest CSV without column mappings.")

    drop_col = mapping.get("drop_id_column")
    bad_col = mapping.get("is_bad_deployment_column")
    video_col = mapping.get("video_file_link_column")

    if not all([drop_col, bad_col, video_col]):
        raise KeyError(f"Missing required CSV column mappings in config.yaml. Found: DropID={drop_col}, Bad={bad_col}, Video={video_col}")

    db = DatabaseManager()
    # Setup Storage precisely like loop.py
    mode = config.storage.get("mode", "local")
    if mode == "aws":

        storage = S3Handler(bucket=config.storage.get("bucket_name"))
        s3 = storage
    else:
        storage = LocalStorageHandler(video_folder=config.storage.get("local_video_dir", "."))
        s3 = None  # We only need s3 to download the csv, handled differently below

    logging.info("2. Fetching the master BUV Deployments list...")
    if mode == "aws":
        csv_path = config.storage.get("sharepoint_deployment_csv_key")
        deployments_df = storage.read_df_from_s3_csv(csv_path)
    else:

        csv_path = config.storage.get("local_deployment_csv_key")
        deployments_df = pd.read_csv(csv_path)

    logging.info(f"Loaded {len(deployments_df)} deployment records.")

    logging.info("3. Running full cross-dataset DataValidation...")
    validator = DataValidator()
    validator.run_validation(file_presence=False, remove_duplicates=True, extract_clean_dataframes=False)

    structural_error_drops = set()
    structured_errors = []

    if validator.errors_df is not None and not validator.errors_df.empty:
        for idx, e in validator.errors_df.iterrows():
            structured_errors.append({
                "SurveyID": str(e.SurveyID) if pd.notna(e.SurveyID) else "",
                "DropID": str(e.DropID) if pd.notna(e.DropID) else "",
                "ErrorType": str(e.ErrorType),
                "FileName": str(e.FileName),
                "ColumnName": str(e.ColumnName) if pd.notna(e.ColumnName) else "",
                "ErrorMessage": str(e.ErrorMessage),
                "InvalidValue": str(e.InvalidValue) if pd.notna(e.InvalidValue) else ""
            })
            if pd.notna(e.DropID):
                structural_error_drops.add(str(e.DropID).strip())

    logging.info(f"Found {len(structural_error_drops)} DropIDs with structural CSV errors.")

    logging.info(f"Logging {len(structured_errors)} validation errors to SQLite...")
    db.clear_validation_errors()
    db.add_validation_errors(structured_errors)

    logging.info(f"4. Batch mapping known media files in {mode} storage...")
    if mode == "aws":
        known_files = set(storage.get_file_paths_set_from_s3(prefix="media/"))
    else:
        known_files = storage.get_all_videos()

    # User test injection
    test_drops = config.test_drops
    for test_id, test_key, _, _ in test_drops:
        if test_key:
            known_files.add(test_key)

    # 5. Load expert annotations and count per DropID (always from S3)
    logging.info("5. Fetching expert annotations from S3...")
    expert_counts = {}
    try:
        s3_handler = S3Handler(bucket=config.storage.get("bucket_name"))
        annotations_df = s3_handler.read_df_from_s3_csv(config.s3_sharepoint_annotations_legacy_experts_csv)
        if not annotations_df.empty:
            expert_counts = annotations_df["DropID"].value_counts().to_dict()
            logging.info(f"Loaded {len(annotations_df)} annotation rows covering {len(expert_counts)} deployments.")
    except Exception as e:
        logging.warning(f"Failed to load KSO annotations from S3: {e}. Expert counts will default to 0.")

    # Inject User Test DataFrame rows matching the mock files
    for test_id, test_key, sampling_start, sampling_end in test_drops:
        fake_row = pd.DataFrame([{
            "DropID": test_id,
            "SurveyID": test_id[:16],
            "SiteID": test_id[17:24],
            "IsBadDeployment": "False",
            "LinkToVideoFile": test_key,
            "SamplingStart": sampling_start,
            "SamplingEnd": sampling_end
        }])
        deployments_df = pd.concat([deployments_df, fake_row], ignore_index=True)

    # 6. Synchronize into SQLite
    logging.info("6. Synchronizing all deployment records into SQLite...")
    _sync_deployments_to_db(deployments_df, db, structural_error_drops, known_files, expert_counts, mapping)

def _sync_deployments_to_db(deployments_df, db, structural_error_drops, known_files, expert_counts, mapping):
    drop_col = mapping.get("drop_id_column")
    bad_col = mapping.get("is_bad_deployment_column")
    video_col = mapping.get("video_file_link_column")

    new_count = 0
    expt_count = 0
    # Iterate over all drops and sync them into our SQLite brain
    for _, row in deployments_df.iterrows():
        drop_id = str(row.get(drop_col, "")).strip()
        if not drop_id or drop_id == "nan":
            continue

        video_path = str(row.get(video_col, "")).strip()

        # Parse IsBadDeployment column strictly
        is_bad_deployment = str(row.get(bad_col, "")).strip() == "True"

        # Parse Sampling offsets — these MUST exist in the BUV Deployment CSV
        try:
            sampling_start = int(pd.to_numeric(row.get("SamplingStart")))
            sampling_end = int(pd.to_numeric(row.get("SamplingEnd")))
        except (ValueError, TypeError) as e:
            if drop_id in structural_error_drops:
                continue
            else:
                logging.warning(
                    f"Missing or invalid SamplingStart/SamplingEnd for {drop_id}. "
                    f"Got SamplingStart={row.get('SamplingStart')}, SamplingEnd={row.get('SamplingEnd')}. "
                    f"This deployment will be flagged as ERROR."
                )
                structural_error_drops.add(drop_id)
            sampling_start = None
            sampling_end = None

        expert_anns = expert_counts.get(drop_id, 0)
        ml_anns = 0
        citsci_anns = 0

        # Determine initial status
        if expert_anns > 0:
            # TODO Legacy annotations, check if doing this first might ever be an issue
            status = PipelineStatus.PIPELINE_COMPLETE
        elif is_bad_deployment:
            status = PipelineStatus.EXCLUDED
        elif drop_id in structural_error_drops:
            status = PipelineStatus.ERROR
        else:
            existing_record = db.get_deployment(drop_id)
            if existing_record and existing_record["status"] not in [
                PipelineStatus.PENDING_ARRIVAL,
                PipelineStatus.ERROR,
                PipelineStatus.MISSING_METADATA
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
            video_path=video_path,
            is_bad_deployment=is_bad_deployment,
            error_message="Found in structural errors" if status == PipelineStatus.ERROR else "",
            sampling_start=sampling_start,
            sampling_end=sampling_end,
            expert_annotations=expert_anns,
            ml_annotations=ml_anns,
            citsci_annotations=citsci_anns
        )
        new_count += 1
        expt_count += expert_anns

    logging.info(f"Ingestion complete. Synchronized {new_count} records into the pipeline database, with {expt_count} of {sum(expert_counts.values())} expert annotations.")
if __name__ == "__main__":
    run_ingestion()
