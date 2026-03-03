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
    s3_key = "spyfish_metadata/sharepoint_lists/BUV Annotations Legacy Experts.csv"

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
            scientific_name = row["ScientificName"] if not pd.isna(row["ScientificName"]) else None
            conf = row["ConfidenceAgreement"]
            confidence = None if (pd.isna(conf) or conf == "NA") else float(conf)

            annotations.append({
                "drop_id": row["DropID"],
                "scientific_name": scientific_name,
                "timestamp": row["TimeOfMax"] if not pd.isna(row["TimeOfMax"]) else None,
                "count": row["MaxInterval"] if not pd.isna(row["MaxInterval"]) else 0,
                "source": "expert",
                "confidence": confidence,
                "external_id": "legacy"  # distinguishes these from Biigle-synced expert annotations
            })

        # 3. Insert into Annotation DB
        ann_db = AnnotationDatabaseManager()
        # Clear only legacy expert annotations to avoid wiping Biigle-synced expert data
        with ann_db.get_connection() as conn:
            conn.execute("DELETE FROM annotations WHERE source = 'expert' AND external_id = 'legacy'")

        ann_db.add_annotations(annotations)
        logging.info(f"Successfully ingested {len(annotations)} expert annotations into spyfish_annotations.db")

        # 4. Sync counts back to main pipeline DB
        sync_annotations_to_main_db()

    except Exception as e:
        logging.error(f"Failed legacy ingestion: {e}")
        raise
    finally:
        if local_csv.exists():
            local_csv.unlink()

def sync_annotations_to_main_db(drop_ids: Optional[List[str]] = None):
    """
    Aggregates counts from spyfish_annotations.db and updates the deployments table in spyfish_pipeline.db.
    If drop_ids is provided, only updates those specific deployments.
    """
    logging.info(f"Syncing annotation counts to main pipeline database{' (incremental)' if drop_ids else ''}...")
    ann_db = AnnotationDatabaseManager()
    main_db = DatabaseManager()

    query = '''
        SELECT drop_id, source, SUM(count) as total
        FROM annotations
    '''
    params = []
    if drop_ids:
        placeholders = ', '.join(['?'] * len(drop_ids))
        query += f" WHERE drop_id IN ({placeholders})"
        params.extend(drop_ids)

    query += " GROUP BY drop_id, source"

    with ann_db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query, params)
        results = cursor.fetchall()

    # Group results by drop_id
    counts_by_drop = {d: {"ml": 0, "expert": 0, "citsci": 0} for d in (drop_ids or [])}
    for row in results:
        drop_id = row['drop_id']
        source = row['source']
        count = row['total']

        if drop_id not in counts_by_drop:
            counts_by_drop[drop_id] = {"ml": 0, "expert": 0, "citsci": 0}

        if source == "ml":
            counts_by_drop[drop_id]["ml"] = count
        elif source == "expert":
            counts_by_drop[drop_id]["expert"] = count
        elif source == "citsci":
            counts_by_drop[drop_id]["citsci"] = count

    # Update main DB
    with main_db.get_connection() as conn:
        cursor = conn.cursor()
        for drop_id, counts in counts_by_drop.items():
            cursor.execute('''
                UPDATE deployments
                SET ml_annotations = ?,
                    expert_annotations = ?,
                    citsci_annotations = ?
                WHERE drop_id = ?
            ''', (counts["ml"], counts["expert"], counts["citsci"], drop_id))
        conn.commit()

    logging.info(f"Updated annotation counts for {len(counts_by_drop)} deployments.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ingest_legacy_expert_annotations()
