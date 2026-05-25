import logging
import tempfile
from pathlib import Path

import pandas as pd

from spyfish.config.wrapper import config
from spyfish.database.annotation_manager import AnnotationDatabaseManager
from spyfish.database.manager import DatabaseManager
from spyfish.storage.s3_handler import S3Handler


def ingest_legacy_expert_annotations():
    """Download legacy expert annotations from S3 and ingest them.

    Four-step flow:
      1. Download legacy CSV from S3.
      2. Parse and normalise rows.
      3. Replace existing legacy rows in spyfish_annotations.db (DELETE+INSERT,
         keyed on annotated_by='expert' AND external_id='legacy' so this never
         touches BIIGLE-synced expert rows).
      4. `sync_annotation_counts()` — writes the aggregated per-drop counts
         back to the deployments table AND advances expert_status to
         expert_complete for any drop that gained annotations (data presence
         is the source of truth).

    **Recovery semantics.** Every step is idempotent: re-running the whole
    function recovers from any partial failure. Step 3 is replace-on-conflict,
    step 4 is a full recount with idempotent status advancement. If you see
    a partial state, just re-run `--legacy-experts`.
    """
    logging.info("Starting legacy expert annotation ingestion...")

    s3 = S3Handler()
    bucket = config.s3_bucket
    s3_key = config.s3_sharepoint_annotations_legacy_experts_csv

    # Use a per-run temp file so concurrent pipeline invocations don't collide.
    with tempfile.NamedTemporaryFile(
        prefix="legacy_annotations_", suffix=".csv", delete=False
    ) as tmp:
        local_csv = Path(tmp.name)

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
        # Clear only legacy expert annotations to avoid wiping Biigle-synced expert data.
        with ann_db.get_writable_connection() as conn:
            conn.execute(
                "DELETE FROM annotations WHERE annotated_by = 'expert' AND external_id = 'legacy'"
            )

        ann_db.add_annotations(annotations)
        logging.info(
            f"Successfully ingested {len(annotations)} expert annotations into spyfish_annotations.db"
        )

        # 4. Sync counts back to main pipeline DB. This also advances
        # expert_status → expert_complete for any drop with annotations.
        drop_ids = list({a["drop_id"] for a in annotations})
        main_db = DatabaseManager()
        main_db.sync_annotation_counts(drop_ids)

    except Exception as e:
        logging.error(f"Failed legacy ingestion: {e}")
        raise
    finally:
        if local_csv.exists():
            local_csv.unlink()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ingest_legacy_expert_annotations()
