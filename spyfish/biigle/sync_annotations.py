import logging
import pandas as pd
from typing import List, Dict, Any

from spyfish.biigle.biigle_handler import BiigleHandler
from spyfish.database.manager import DatabaseManager
from spyfish.database.annotation_manager import AnnotationDatabaseManager
from spyfish.orchestrator.ingest_legacy import sync_annotations_to_main_db
from spyfish.config import PipelineStatus, config

def sync_biigle_annotations():
    """
    Checks all deployments with a biigle_volume_id that are not yet complete.
    Downloads reports, checks for 'Done' label, and ingests annotations.
    """
    logging.info("Starting Biigle annotation sync...")

    db = DatabaseManager()
    ann_db = AnnotationDatabaseManager()
    handler = BiigleHandler()

    # 1. Get deployments with biigle_volume_id that are NOT yet complete
    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(f"""
            SELECT drop_id, biigle_volume_id
            FROM deployments
            WHERE biigle_volume_id IS NOT NULL
              AND status != '{PipelineStatus.PIPELINE_COMPLETE}'
        """)
        deployments = cursor.fetchall()

    if not deployments:
        logging.info("No active deployments with Biigle volumes found to check.")
        return

    processed_drops = []
    for dep in deployments:
        drop_id = dep["drop_id"]
        volume_id = int(dep["biigle_volume_id"])

        logging.info(f"Checking Biigle volume {volume_id} for {drop_id}")

        try:
            # 2. Check 'Done' label via file-level labels (NOT from annotation report)
            # In Biigle, "Done" is a whole-file label applied via the label panel,
            # which lives at GET /api/v1/images/{id}/labels or /videos/{id}/labels.
            # volume_is_done also detects whether this is an image or video volume.
            is_done, media_type = handler.volume_is_done(volume_id)

            if not is_done:
                logging.info(f"  Volume {volume_id} for {drop_id} not marked 'Done' yet. Skipping.")
                continue

            logging.info(f"  ✅ Volume {volume_id} for {drop_id} is DONE ({media_type} volume). Downloading annotation report...")

            # 3. Download annotation report using the correct type for this volume's media
            if media_type == "video":
                report_type = config.biigle_annotation_report_type_video
            else:
                report_type = config.biigle_annotation_report_type_images
            report_df = handler.export_report_to_df("volumes", volume_id, type_id=report_type)

            if report_df.empty:
                logging.info(f"  No annotations found for {drop_id} (volume may be done but have no annotations).")
                db.update_status(drop_id, PipelineStatus.PIPELINE_COMPLETE)
                continue

            # 4. Find the label column
            label_col = None
            for col in ['label_name', 'label', 'Label', 'Scientific Name', 'scientific_name']:
                if col in report_df.columns:
                    label_col = col
                    break

            if not label_col:
                logging.warning(f"  Could not find label column in report for {drop_id}. Columns: {list(report_df.columns)}")
                continue

            fish_annotations = report_df

            annotations_to_add = []
            for _, row in fish_annotations.iterrows():
                timestamp = None
                if 'image_filename' in row:
                    fname = str(row['image_filename'])
                    if 'extracted_frame_' in fname:
                        try:
                            secs = float(fname.split('extracted_frame_')[-1].split('.jpg')[0])
                            h = int(secs // 3600)
                            m = int((secs % 3600) // 60)
                            s = int(secs % 60)
                            ms = int((secs % 1) * 1000)
                            timestamp = f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"
                        except (ValueError, IndexError) as e:
                            logging.warning(f"Could not parse timestamp from filename '{fname}': {e}")

                annotations_to_add.append({
                    "drop_id": drop_id,
                    "scientific_name": row[label_col],
                    "timestamp": timestamp,
                    "count": 1,
                    "source": "expert",
                    "confidence": 1.0,
                    "external_id": str(row.get('id', ''))
                })

            if annotations_to_add:
                # Clear previous syncs for this drop (using a distinct connection to avoid locks)
                with ann_db.get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("DELETE FROM annotations WHERE drop_id = ? AND source = 'expert' AND external_id IS NOT NULL", (drop_id,))
                    conn.commit()

                # Add new annotations
                ann_db.add_annotations(annotations_to_add)

                processed_drops.append(drop_id)
                logging.info(f"  Ingested {len(annotations_to_add)} annotations for {drop_id}")

                # Advance status to PIPELINE_COMPLETE
                db.update_status(drop_id, PipelineStatus.PIPELINE_COMPLETE)
                logging.info(f"  Advanced {drop_id} to {PipelineStatus.PIPELINE_COMPLETE}")

        except Exception as e:
            logging.error(f"  Failed to sync volume {volume_id} for {drop_id}: {e}")

    # 5. Final sync of counts to main DB
    if processed_drops:
        sync_annotations_to_main_db(processed_drops)
        logging.info(f"Biigle sync complete. Processed {len(processed_drops)} drops.")
    else:
        logging.info("No new 'Done' volumes found to process.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sync_biigle_annotations()
