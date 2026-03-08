import logging
import pandas as pd
from typing import Optional, List
from pathlib import Path
from spyfish.config import config
from spyfish.storage.s3_handler import S3Handler
from spyfish.database.annotation_manager import AnnotationDatabaseManager
from spyfish.database.manager import DatabaseManager

def ingest_legacy_expert_annotations():
    """
    Downloads legacy expert annotations from S3 and populates the annotation database.
    Also syncs the aggregated expert_annotations count back to the main pipeline DB.
    """
    logging.info("Starting legacy expert annotation ingestion...")

    s3 = S3Handler()
    bucket = config.s3_bucket
    s3_key = config.s3_sharepoint_annotations_legacy_experts_csv

    local_csv = config.project_root / "temp_legacy_annotations.csv"

    try:
        # 1. Download from S3
        logging.info(f"Downloading legacy annotations from s3://{bucket}/{s3_key}")
        s3.download_object_from_s3(s3_key, str(local_csv))

        # 2. Parse CSV
        df = pd.read_csv(local_csv)
        logging.info(f"Loaded {len(df)} legacy annotation records.")

        # Expected columns: DropID, ScientificName, TimeOfMax, MaxInterval, AnnotatedBy, IntervalAnnotation, ConfidenceAgreement
        # We map these to our annotation schema
        annotations = []
        for _, row in df.iterrows():
            scientific_name = row[config.csv_scientific_name_column] if not pd.isna(row[config.csv_scientific_name_column]) else None
            conf = row[config.csv_confidence_agreement_column]
            confidence = None if (pd.isna(conf) or conf == "NA") else float(conf)

            annotations.append({
                "drop_id": row[config.drop_id_column],
                "scientific_name": scientific_name,
                "time_of_max": row[config.csv_maxn_time_column] if not pd.isna(row[config.csv_maxn_time_column]) else None,
                "max_interval": row[config.csv_max_interval_column] if not pd.isna(row[config.csv_max_interval_column]) else 0,
                "annotated_by": "expert",
                "interval_annotation": row.get(config.csv_interval_annotation_column, None) if not pd.isna(row.get(config.csv_interval_annotation_column)) else None,
                "confidence_agreement": confidence,
                "external_id": "legacy"  # distinguishes these from Biigle-synced expert annotations
            })

        # 3. Insert into Annotation DB
        ann_db = AnnotationDatabaseManager()
        # Clear only legacy expert annotations to avoid wiping Biigle-synced expert data
        with ann_db.get_connection() as conn:
            conn.execute("DELETE FROM annotations WHERE annotated_by = 'expert' AND external_id = 'legacy'")

        ann_db.add_annotations(annotations)
        logging.info(f"Successfully ingested {len(annotations)} expert annotations into spyfish_annotations.db")

        # 4. Sync counts back to main pipeline DB
        main_db = DatabaseManager()
        main_db.sync_annotation_counts()

    except Exception as e:
        logging.error(f"Failed legacy ingestion: {e}")
        raise
    finally:
        if local_csv.exists():
            local_csv.unlink()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ingest_legacy_expert_annotations()
