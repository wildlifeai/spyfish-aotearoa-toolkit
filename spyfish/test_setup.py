import argparse
import logging
import os
import sys

import pandas as pd

from spyfish.config.base import PipelineStatus
from spyfish.config.wrapper import config
from spyfish.database.manager import DatabaseManager
from spyfish.storage.db_sync import upload_db

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def process_csv_targets(
    csv_path: str, default_stage: str = None, push_s3: bool = False
):
    """
    Reads a CSV file containing Drop IDs and their target pipeline stage,
    and updates the local database accordingly.

    Expected CSV format:
    DropID,PipelineStatus
    KSF_20240124_BUV_KSF_085_01,READY_FOR_ML
    KSF_20240124_BUV_KSF_085_02,AWAITING_EXPERT_REVIEW

    Row order implies priority — first row is processed first. Priorities are
    assigned as descending integers (N, N-1, ..., 1) so all CSV entries rank
    above unset deployments (default priority 0).
    """
    if not os.path.exists(csv_path):
        logging.error(f"CSV file not found: {csv_path}")
        sys.exit(1)

    db = DatabaseManager()
    valid_stages = [s[0] for s in PipelineStatus.STAGE_ORDER]

    if default_stage and default_stage not in valid_stages:
        logging.error(
            f"Invalid default stage: {default_stage}. Must be one of: {', '.join(valid_stages)}"
        )
        sys.exit(1)

    drop_col = config.drop_id_column
    status_col = config.pipeline_status_column

    df = pd.read_csv(csv_path)

    if drop_col not in df.columns:
        logging.error(
            f"Expected column '{drop_col}' not found in CSV. "
            f"Available columns: {list(df.columns)}. "
            f"Check csv_mapping.drop_id_column in config.yaml."
        )
        sys.exit(1)

    if status_col not in df.columns and not default_stage:
        logging.error(
            f"Expected column '{status_col}' not found in CSV and no --stage default provided. "
            f"Available columns: {list(df.columns)}. "
            f"Check csv_mapping.pipeline_status_column in config.yaml."
        )
        sys.exit(1)

    updates = []
    for _, row in df.iterrows():
        drop_id = str(row[drop_col]).strip()
        if not drop_id or drop_id == "nan":
            continue

        stage = default_stage
        if (
            status_col in df.columns
            and pd.notna(row.get(status_col))
            and str(row[status_col]).strip()
        ):
            stage = str(row[status_col]).strip()

        if not stage:
            logging.warning(
                f"DropID '{drop_id}' has no target stage and no default was provided. Skipping."
            )
            continue

        if stage not in valid_stages:
            logging.warning(
                f"DropID '{drop_id}' has invalid stage '{stage}'. "
                f"Must be one of: {', '.join(valid_stages)}. Skipping."
            )
            continue

        updates.append((drop_id, stage))

    if not updates:
        logging.warning("No valid Drop IDs and stages found in the CSV.")
        sys.exit(0)

    drop_ids = [u[0] for u in updates]
    existing_records = db.get_deployments_by_ids(drop_ids)

    success_count = 0
    missing_count = 0
    n = len(updates)
    # TODO: priority values grow with each set-targets run (base + n per call).
    # In practice this is fine since only relative order matters, but if it ever
    # becomes unwieldy consider a periodic normalisation pass.
    base_priority = db.get_max_priority()  # new entries always land above existing ones

    for i, (drop_id, stage) in enumerate(updates):
        if drop_id not in existing_records:
            logging.warning(
                f"DropID '{drop_id}' not found in the local database. Skipping."
            )
            missing_count += 1
            continue

        try:
            priority = base_priority + n - i  # first row gets highest priority
            db.update_status(drop_id, stage)
            db.update_deployment_fields(drop_id, priority=priority)
            success_count += 1
            logging.info(f"Updated '{drop_id}' → status='{stage}', priority={priority}")
        except Exception as e:
            logging.error(f"Failed to update '{drop_id}': {e}")

    logging.info(
        f"Summary: Successfully updated {success_count} deployments. ({missing_count} skipped/not found)."
    )

    if push_s3 and success_count > 0:
        logging.info("Pushing updated database to S3...")
        if upload_db():
            logging.info("✅ Database uploaded to S3 successfully.")
        else:
            logging.error("❌ Failed to upload database to S3.")


def main():
    parser = argparse.ArgumentParser(
        description="Bulk update pipeline stages from a CSV file."
    )
    parser.add_argument(
        "csv_file",
        help=f"Path to the CSV file (must have '{config.drop_id_column}' and '{config.pipeline_status_column}' columns).",
    )
    parser.add_argument(
        "--stage",
        "-s",
        help="Default target Pipeline Stage to apply when the CSV has no status column.",
    )
    parser.add_argument(
        "--push-s3",
        action="store_true",
        help="Push the modified database back to S3 after updating.",
    )

    args = parser.parse_args()
    process_csv_targets(args.csv_file, args.stage, args.push_s3)


if __name__ == "__main__":
    main()
