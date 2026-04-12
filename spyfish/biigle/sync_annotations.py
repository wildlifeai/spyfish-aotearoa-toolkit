import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from spyfish.biigle.biigle_handler import BiigleHandler
from spyfish.biigle.biigle_parser import BiigleParser
from spyfish.config.base import BiigleStatus
from spyfish.config.wrapper import config
from spyfish.database.annotation_manager import AnnotationDatabaseManager
from spyfish.database.manager import DatabaseManager
from spyfish.utils import seconds_to_time


def _extract_timestamp_from_filename(row: pd.Series, fname_col: str) -> Optional[str]:
    """Helper to parse timestamps from Biigle's image or video snippet filenames."""
    if not fname_col or fname_col not in row:
        return None

    fname = str(row[fname_col])
    try:
        secs = None
        # Image frame formatting from biigle test
        if "__frame_" in fname:
            match = re.search(r"__frame_([\d\.]+)s\.jpg", fname)
            if match:
                secs = float(match.group(1))
        # Video clip format from biigle video annotations
        elif "_clip_" in fname:
            match = re.search(r"_clip_([\d\.]+)s\.", fname)
            if match:
                secs = float(match.group(1))

            if "frames" in row:
                frame_str = str(row["frames"]).strip("[]")
                if frame_str and frame_str != "nan":
                    secs += float(frame_str)  # type: ignore

        if secs is not None:
            return seconds_to_time(secs)
    except (ValueError, IndexError) as e:
        logging.warning(f"Could not parse timestamp from filename '{fname}': {e}")

    return None


def _map_biigle_to_spyfish_schema(
    row: pd.Series, label_col: str, drop_id: str, timestamp: Optional[str]
) -> Tuple[Tuple[str, str], Dict[str, Any]]:
    """Maps a single Biigle annotation row to the Spyfish schema and returns (aggregation_key, mapped_dict)."""
    species = str(row.get(label_col, "unknown_species")).strip()
    # Clean up "Kina - Evechinus chloroticus" to "Evechinus chloroticus"
    if " - " in species:
        parts = species.split(" - ", 1)
        if len(parts) == 2:
            species = parts[1]

    # Use empty string instead of None to allow sorting
    sortable_time = timestamp or ""

    key = (sortable_time, species)

    mapped_item = {
        "drop_id": drop_id,
        "scientific_name": species,
        "time_of_max": timestamp,
        "max_interval": 0,  # Will be incremented during aggregation
        "annotated_by": "expert",
        "interval_annotation": "",
        "confidence_agreement": 1.0,
        "external_id": str(
            row.get("annotation_id", row.get("id", ""))
        ),  # Prefer video annotation ID, fallback to image annotation ID
    }
    return key, mapped_item


def _aggregate_annotations(
    fish_annotations_df: pd.DataFrame, drop_id: str
) -> List[Dict[str, Any]]:
    """Aggregates raw DataFrame rows from Biigle into MaxN counts per timestamp and species."""
    label_col = "label_name"
    fname_col = "filename"

    aggregated_annotations = {}
    for _, row in fish_annotations_df.iterrows():
        timestamp = _extract_timestamp_from_filename(row, fname_col)
        key, mapped_item = _map_biigle_to_spyfish_schema(
            row, label_col, drop_id, timestamp
        )

        if key not in aggregated_annotations:
            aggregated_annotations[key] = mapped_item

        aggregated_annotations[key]["max_interval"] += 1

    # Convert to list and sort by drop_id (already same) and time_of_max
    annotations_to_add = list(aggregated_annotations.values())
    annotations_to_add.sort(key=lambda x: (x["drop_id"], x["time_of_max"] or ""))
    return annotations_to_add


def sync_biigle_annotations():
    """
    Checks all deployments with a biigle_volume_id that are not yet complete.
    Downloads reports, checks for 'Done' label, and ingests annotations.

    Biigle output columns: 'annotation_label_id', 'label_id', 'label_name',
    'label_hierarchy','user_id', 'firstname', 'lastname', 'image_id', 'filename',
    'image_longitude', 'image_latitude', 'shape_id', 'shape_name', 'points',
    'attributes', 'annotation_id', 'created_at', 'source_file'
    """
    logging.info("Starting Biigle annotation sync...")

    db = DatabaseManager()
    ann_db = AnnotationDatabaseManager()
    handler = BiigleHandler()

    # 1. Get deployments with biigle_volume_id that are in biigle_status=uploaded
    deployments = db.get_biigle_volumes_awaiting_sync(BiigleStatus.UPLOADED)

    if not deployments:
        logging.info("No active deployments with Biigle volumes found to check.")
        return

    processed_drops = []
    for dep in deployments:
        drop_id = dep["drop_id"]
        volume_id = int(dep["biigle_volume_id"])

        logging.debug(f"Checking Biigle volume {volume_id} for {drop_id}")

        try:
            # 2. Check presence of labels that define the volume as done via file-level labels
            is_done, media_type = handler.volume_is_done(volume_id)
            if not is_done:
                logging.debug(
                    f"  Volume {volume_id} for {drop_id} not marked Done yet. Skipping."
                )
                continue

            logging.info(
                f"  ✅ Volume {volume_id} for {drop_id} is DONE ({media_type} volume). Downloading annotation report (with caching)..."
            )

            # 3. Download annotation report (using Parser for per-drop caching)
            parser = BiigleParser(drop_id=drop_id)
            report_type = (
                config.annotation_report_type_video
                if media_type == "video"
                else config.annotation_report_type_images
            )

            # 3b. Download annotations via public API — uses per-drop cache
            fish_annotations_df = parser.download_volume_annotations(
                volume_id=volume_id, type_id=report_type
            )

            if fish_annotations_df.empty:
                logging.debug(
                    f"  No annotations found for {drop_id} (volume may be done but have no annotations)."
                )
                db.advance_status(drop_id, BiigleStatus.COLUMN, BiigleStatus.COMPLETE)
                continue

            # Save the raw Biigle report for YOLO retraining (bounding boxes)
            drop_ann_dir = config.get_drop_annotations_dir(drop_id)
            drop_ann_dir.mkdir(parents=True, exist_ok=True)
            raw_path = drop_ann_dir / f"{drop_id}_biigle_expert_raw.csv"
            fish_annotations_df.to_csv(raw_path, index=False)
            logging.info(
                f"  Exported expert raw annotations for ML retraining: {raw_path}"
            )

            # 4. Process and aggregate annotations
            annotations_to_add = _aggregate_annotations(fish_annotations_df, drop_id)

            if not annotations_to_add:
                # Report had rows but all were filtered (e.g. only Scale Bar labels).
                # Volume is done — advance without writing annotations.
                logging.info(
                    f"  No fish annotations after aggregation for {drop_id} "
                    "(only non-fish labels present). Advancing to PIPELINE_COMPLETE."
                )
                db.advance_status(drop_id, BiigleStatus.COLUMN, BiigleStatus.COMPLETE)
                continue

            # Replace only Biigle-sourced expert annotations (external_id IS NOT NULL).
            # Manually-entered expert annotations (external_id = NULL) are preserved.
            ann_db.clear_synced_annotations(drop_id, "expert")

            # Add new annotations
            ann_db.add_annotations(annotations_to_add)

            # Export the MaxN CSV for ML retraining
            maxn_df = pd.DataFrame(annotations_to_add).rename(
                columns={
                    "drop_id": config.drop_id_column,
                    "scientific_name": config.csv_scientific_name_column,
                    "time_of_max": config.csv_maxn_time_column,
                    "max_interval": config.csv_max_interval_column,
                    "annotated_by": config.csv_annotated_by_column,
                    "interval_annotation": config.csv_interval_annotation_column,
                    "confidence_agreement": config.csv_confidence_agreement_column,
                }
            )
            maxn_path = drop_ann_dir / f"{drop_id}_biigle_expert_maxn.csv"
            maxn_df.to_csv(maxn_path, index=False)
            logging.info(
                f"  Exported expert MaxN annotations for ML retraining: {maxn_path}"
            )

            processed_drops.append(drop_id)
            logging.info(
                f"  Ingested {len(annotations_to_add)} annotations for {drop_id}"
            )

            db.advance_status(drop_id, BiigleStatus.COLUMN, BiigleStatus.COMPLETE)
            logging.info(
                f"  Advanced {drop_id} to biigle_status={BiigleStatus.COMPLETE!r}"
            )

        except Exception as e:
            logging.error(
                f"  Failed to sync volume {volume_id} for {drop_id}: {e}",
                exc_info=True,
            )

    # 5. Final sync of counts to main DB
    if processed_drops:
        db.sync_annotation_counts(processed_drops)
        logging.info(f"Biigle sync complete. Processed {len(processed_drops)} drops.")
    else:
        logging.info("No new 'Done' volumes found to process.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sync_biigle_annotations()
