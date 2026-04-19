import argparse
import logging
import os
import sys

import pandas as pd

from spyfish.config.wrapper import config
from spyfish.database.manager import DatabaseManager
from spyfish.storage.db_sync import upload_db

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

_SECTION_COLUMNS = {
    "ml_status",
    "citsci_status",
    "biigle_status",
    "reporting_status",
    "ingest_status",
}


def process_csv_targets(csv_path: str, push_s3: bool = False):
    """
    Reads a CSV file containing DropIDs with optional section statuses and priority,
    and updates the local database accordingly.

    Expected CSV format (all columns except DropID are optional):
        DropID,ml_status,priority
        KSF_20240124_BUV_KSF_085_01,ml_ready,10
        KSF_20240124_BUV_KSF_085_02,,5

    Row order implies priority — first row has highest priority. If a 'priority'
    column is present it takes precedence; otherwise row order is used.
    """
    if not os.path.exists(csv_path):
        logging.error(f"CSV file not found: {csv_path}")
        sys.exit(1)

    db = DatabaseManager()
    drop_col = config.drop_id_column

    df = pd.read_csv(csv_path)

    if drop_col not in df.columns:
        logging.error(
            f"Expected column '{drop_col}' not found in CSV. "
            f"Available columns: {list(df.columns)}."
        )
        sys.exit(1)

    section_cols = [c for c in df.columns if c in _SECTION_COLUMNS]
    has_priority_col = "priority" in df.columns

    updates = []
    for _, row in df.iterrows():
        drop_id = str(row[drop_col]).strip()
        if not drop_id or drop_id == "nan":
            continue
        entry = {"drop_id": drop_id, "sections": {}, "priority": None}
        for col in section_cols:
            val = row.get(col)
            if pd.notna(val) and str(val).strip():
                entry["sections"][col] = str(val).strip()
        if has_priority_col and pd.notna(row.get("priority")):
            entry["priority"] = int(row["priority"])
        updates.append(entry)

    if not updates:
        logging.warning("No valid Drop IDs found in the CSV.")
        sys.exit(0)

    drop_ids = [u["drop_id"] for u in updates]
    existing_records = db.get_deployments_by_ids(drop_ids)

    success_count = 0
    missing_count = 0
    n = len(updates)
    base_priority = db.get_max_priority()

    for i, entry in enumerate(updates):
        drop_id = entry["drop_id"]
        if drop_id not in existing_records:
            logging.warning(
                f"DropID '{drop_id}' not found in the local database. Skipping."
            )
            missing_count += 1
            continue

        try:
            # Priority: explicit column value > row order
            priority = (
                entry["priority"]
                if entry["priority"] is not None
                else base_priority + n - i
            )
            db.update_deployment_fields(drop_id, priority=priority)

            for section, value in entry["sections"].items():
                db.update_section_status(drop_id, section, value)

            success_count += 1
            logging.info(
                f"Updated '{drop_id}' — priority={priority}"
                + (f", sections={entry['sections']}" if entry["sections"] else "")
            )
        except Exception as e:
            logging.error(f"Failed to update '{drop_id}': {e}")

    logging.info(
        f"Summary: Updated {success_count} deployments. ({missing_count} skipped/not found)."
    )

    if push_s3 and success_count > 0:
        logging.info("Pushing updated database to S3...")
        if upload_db():
            logging.info("✅ Database uploaded to S3 successfully.")
        else:
            logging.error("❌ Failed to upload database to S3.")


def main():
    parser = argparse.ArgumentParser(
        description="Bulk update pipeline section statuses and priorities from a CSV file."
    )
    parser.add_argument(
        "csv_file",
        help=f"Path to the CSV file (must have '{config.drop_id_column}' column; "
        f"optional: {sorted(_SECTION_COLUMNS)}, priority).",
    )
    parser.add_argument(
        "--push-s3",
        action="store_true",
        help="Push the modified database back to S3 after updating.",
    )

    args = parser.parse_args()
    process_csv_targets(args.csv_file, args.push_s3)


if __name__ == "__main__":
    main()
