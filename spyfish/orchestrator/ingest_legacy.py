import logging

import pandas as pd

from spyfish.config.wrapper import config
from spyfish.database.annotation_manager import AnnotationDatabaseManager
from spyfish.database.manager import DatabaseManager
from spyfish.storage.s3_handler import S3Handler


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
        conf_col = config.csv_confidence_agreement_column
        intv_col = config.csv_interval_annotation_column

        annotations = []
        for row in df.to_dict("records"):
            conf = row.get(conf_col)
            confidence = None if (pd.isna(conf) or conf == "NA") else float(conf)

            sci = row.get(config.csv_scientific_name_column)
            t_max = row.get(config.csv_maxn_time_column)
            m_intv = row.get(config.csv_max_interval_column)
            intv_ann = row.get(intv_col)

            annotations.append(
                {
                    "drop_id": row[config.drop_id_column],
                    "scientific_name": None if pd.isna(sci) else sci,
                    "time_of_max": None if pd.isna(t_max) else t_max,
                    "max_interval": 0 if pd.isna(m_intv) else m_intv,
                    "annotated_by": "expert",
                    "interval_annotation": None if pd.isna(intv_ann) else intv_ann,
                    "confidence_agreement": confidence,
                    "external_id": "legacy",  # distinguishes these from Biigle-synced expert annotations
                }
            )

        # 3. Insert into Annotation DB
        ann_db = AnnotationDatabaseManager()
        # Clear only legacy expert annotations to avoid wiping Biigle-synced expert data
        with ann_db.get_connection() as conn:
            conn.execute(
                "DELETE FROM annotations WHERE annotated_by = 'expert' AND external_id = 'legacy'"
            )

        ann_db.add_annotations(annotations)
        logging.info(
            f"Successfully ingested {len(annotations)} expert annotations into spyfish_annotations.db"
        )

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
