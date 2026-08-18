import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from spyfish.biigle.biigle_handler import BiigleHandler
from spyfish.biigle.biigle_parser import BiigleParser
from spyfish.biigle.biigle_to_yolo import biigle_to_yolo
from spyfish.config.base import ExpertStatus
from spyfish.config.species import species_registry
from spyfish.config.wrapper import config
from spyfish.database.annotation_manager import AnnotationDatabaseManager
from spyfish.database.manager import DatabaseManager
from spyfish.utils import seconds_to_time, time_to_seconds


def _drop_substrate_rows(df: pd.DataFrame, substrate_label_ids: set) -> pd.DataFrame:
    """Remove CMECS substrate annotations from a report DataFrame.

    Substrate is measured as percent-cover (process_substrate), not counted as
    a species, so its rows must be excluded before MaxN aggregation, or each
    substrate polygon would inflate a bogus "species" count. Identified by the
    same label-tree membership used in process_substrate.
    """
    if not substrate_label_ids or "label_id" not in df.columns:
        return df
    is_substrate = pd.to_numeric(df["label_id"], errors="coerce").isin(
        substrate_label_ids
    )
    return df[~is_substrate].copy()


def _extract_timestamp_from_filename(row: pd.Series, fname_col: str) -> Optional[str]:
    """Parse timestamps from Biigle's image or video snippet filenames."""
    if not fname_col or fname_col not in row:
        return None

    fname = str(row[fname_col])
    try:
        secs = None
        if "__frame_" in fname:
            match = re.search(r"__frame_([\d\.]+)s\.jpg", fname)
            if match:
                secs = float(match.group(1))
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
    row: pd.Series,
    label_col: str,
    drop_id: str,
    timestamp: Optional[str],
    frame_key: str,
) -> Optional[Tuple[Tuple[str, str], Dict[str, Any]]]:
    """Maps a Biigle annotation row to the Spyfish schema. Returns (aggregation_key, mapped_dict).

    Returns None for rows whose label is not a species: label-tree species are
    named "Common - Scientific", so a bare label the species registry doesn't
    know is a workflow label ("Done Volume", "Review"), letting it through
    would write it into the annotations DB as a scientific name.

    Names the registry DOES know are resolved to their canonical scientific
    name, not kept as written. The registry treats "Fish: final" as an alias of
    the `fish` bucket; recognising it and then storing the alias put a second
    spelling of one class into the database, where every reader has to know
    both.

    `frame_key` is used as the per-frame identifier when `timestamp` is None,
    e.g. for image volumes (like UUID-named uploads) where no clip/frame-seconds
    pattern is in the filename. Preserves per-frame uniqueness for MaxN aggregation.
    """
    label = str(row.get(label_col, "unknown_species")).strip()

    # The registry first, on the WHOLE label. It knows the label-tree strings
    # as aliases, so "Fish - Final" resolves to `fish` and "Snapper - Pagrus
    # auratus" to `Pagrus auratus`. Splitting on " - " first would take the
    # second half blindly and turn "Fish - Final" into a species called
    # "Final".
    registry = species_registry()
    known = registry.get(label) or registry.get(label.lower())
    if known is not None:
        species = known.scientific_name
    elif " - " in label:
        # Not in the registry, but it carries the label tree's
        # "Common - Scientific" shape, so the half after the separator is the
        # scientific name. This is how a species new to the tree arrives before
        # anyone adds it to the registry.
        species = label.split(" - ", 1)[1]
    else:
        # A bare label the registry does not know is a workflow label
        # ("Done Volume", "Interesting Sighting"); letting it through would
        # write it into the annotations database as a scientific name.
        logging.warning(
            f"{drop_id}: skipping BIIGLE annotation with non-species label "
            f"{label!r} (no 'Common - Scientific' form, unknown to the "
            "species registry)."
        )
        return None

    sortable_time = timestamp or frame_key
    key = (sortable_time, species)

    time_str = timestamp or frame_key
    try:
        toms = time_to_seconds(time_str) if time_str else None
    except Exception:
        toms = None
    mapped_item = {
        "drop_id": drop_id,
        "scientific_name": species,
        "time_of_max": time_str,
        "time_of_max_seconds": toms,
        "max_interval": 0,
        "annotated_by": "expert",
        "interval_annotation": "",
        "confidence_agreement": 1.0,
        "external_id": str(row.get("annotation_id", row.get("id", ""))),
    }
    return key, mapped_item


def aggregate_raw_to_maxn_rows(
    fish_annotations_df: pd.DataFrame, drop_id: str
) -> List[Dict[str, Any]]:
    """Aggregate raw Biigle annotation rows into MaxN rows (one per frame × species).

    Public entry point reused by the Biigle sync pipeline AND by the
    download-volume bundle synthesizer. Returns a list of dicts using the
    Spyfish lowercase schema (drop_id, scientific_name, time_of_max, ...);
    use `maxn_rows_to_df` to rename for CSV output.
    """
    label_col = "label_name"
    fname_col = "filename"

    aggregated_annotations = {}
    for _, row in fish_annotations_df.iterrows():
        fname = str(row.get(fname_col, ""))
        timestamp = _extract_timestamp_from_filename(row, fname_col)
        mapped = _map_biigle_to_spyfish_schema(
            row, label_col, drop_id, timestamp, fname
        )
        if mapped is None:
            continue
        key, mapped_item = mapped

        if key not in aggregated_annotations:
            aggregated_annotations[key] = mapped_item

        aggregated_annotations[key]["max_interval"] += 1

    annotations_to_add = list(aggregated_annotations.values())
    annotations_to_add.sort(key=lambda x: (x["drop_id"], x["time_of_max"] or ""))
    return annotations_to_add


def maxn_rows_to_df(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """Build a MaxN CSV-ready DataFrame from `aggregate_raw_to_maxn_rows` output.

    Renames lowercase schema keys to the canonical CSV column names used
    downstream by `prepare_from_annotations` (config-driven).
    """
    return pd.DataFrame(rows).rename(
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


def sync_biigle_annotations():
    """
    Sync annotations from Biigle volumes that the annotator has marked as done.

    For each deployment with expert_status=uploaded and a biigle_volume_id:
    - Confirm the volume lives in the `done` project (3711). In-progress
      volumes (4942) are skipped so the annotator's WIP isn't pulled prematurely.
    - When `biigle.require_done_label` is true, ALSO require the legacy
      `Done Volume` whole-file label gate (belt-and-braces during transition).
    - Download the annotation report, save raw CSV, aggregate to MaxN, ingest
      into the annotations DB, export MaxN CSV per drop, rebuild YOLO labels.
    """
    logging.info("Starting Biigle annotation sync...")

    db = DatabaseManager()
    ann_db = AnnotationDatabaseManager()
    handler = BiigleHandler()

    deployments = db.get_biigle_volumes_awaiting_sync(ExpertStatus.UPLOADED)

    if not deployments:
        logging.info("No active deployments with Biigle volumes found to check.")
        return

    # Project-membership gate: only volumes currently in `done` (3711) are ready
    # to sync. One API call instead of per-volume get_volume_info.
    done_project_id = config.biigle_done_project_id
    # Cache full volume dicts (not just IDs) so we can reuse media_type below
    # without a per-volume get_volume_info() round-trip.
    done_volumes = {v["id"]: v for v in handler.get_volumes(done_project_id)}
    logging.info(f"Project {done_project_id} (done) has {len(done_volumes)} volume(s)")

    # Substrate (CMECS) label-id set, fetched once for the whole run, these are
    # the labels whose annotations are measured as percent-cover instead of being
    # counted as a species. Empty set (e.g. API hiccup) cleanly disables substrate
    # processing without aborting the species sync.
    substrate_tree_id = config.biigle_substrate_label_tree_id
    try:
        substrate_label_ids = {
            int(lbl["id"]) for lbl in handler.get_label_tree_labels(substrate_tree_id)
        }
        logging.info(
            f"Substrate tree {substrate_tree_id} has "
            f"{len(substrate_label_ids)} label(s)"
        )
    except Exception as e:
        logging.warning(
            f"Could not fetch substrate label tree {substrate_tree_id}: {e}. "
            "Substrate percent-cover will be skipped this run."
        )
        substrate_label_ids = set()

    processed_drops = []
    for dep in deployments:
        drop_id = dep["drop_id"]
        volume_id = int(dep["biigle_volume_id"])

        logging.debug(f"Checking Biigle volume {volume_id} for {drop_id}")

        try:
            if volume_id not in done_volumes:
                logging.debug(
                    f"  Volume {volume_id} for {drop_id} not in project "
                    f"{done_project_id} (done) yet. Skipping."
                )
                continue

            if config.biigle_require_done_label:
                is_done, media_type = handler.volume_is_done(volume_id)
                if not is_done:
                    logging.debug(
                        f"  Volume {volume_id} for {drop_id} in done project but "
                        "Done-label gate enabled and label missing. Skipping."
                    )
                    continue
                logging.info(
                    f"  Volume {volume_id} for {drop_id} is DONE ({media_type}). Downloading report..."
                )
            else:
                # Reuse cached metadata, saves a get_volume_info() round-trip per volume
                media_type = done_volumes[volume_id].get("media_type", "image")
                logging.info(
                    f"  Volume {volume_id} for {drop_id} in done project "
                    f"({media_type}). Downloading report..."
                )

            parser = BiigleParser()
            report_type = (
                config.annotation_report_type_video
                if media_type == "video"
                else config.annotation_report_type_images
            )

            fish_annotations_df = parser.download_volume_annotations(
                volume_id=volume_id, type_id=report_type
            )

            if fish_annotations_df.empty:
                logging.debug(f"  No annotations found for {drop_id}.")
                db.advance_status(drop_id, ExpertStatus.COLUMN, ExpertStatus.COMPLETE)
                continue

            # Save raw Biigle report (used by YOLO label generation)
            config.get_drop_annotations_dir(drop_id).mkdir(parents=True, exist_ok=True)
            raw_path = config.get_biigle_expert_raw_csv_path(drop_id)
            fish_annotations_df.to_csv(raw_path, index=False)
            logging.info(f"  Raw expert annotations → {raw_path}")

            # Substrate (CMECS) is an area-cover statistic, computed separately
            # from species MaxN. Export per-drop percentages, then drop substrate
            # rows so they never count as a "species" in the fish aggregation.
            substrate_df = parser.process_substrate(
                fish_annotations_df, substrate_label_ids
            )
            if not substrate_df.empty:
                substrate_path = config.get_biigle_expert_substrate_csv_path(drop_id)
                substrate_df.to_csv(substrate_path, index=False)
                logging.info(
                    f"  Expert substrate ({len(substrate_df)} rows) → {substrate_path}"
                )
            fish_annotations_df = _drop_substrate_rows(
                fish_annotations_df, substrate_label_ids
            )

            # Aggregate into MaxN counts
            annotations_to_add = aggregate_raw_to_maxn_rows(
                fish_annotations_df, drop_id
            )

            if not annotations_to_add:
                logging.info(
                    f"  No fish annotations after aggregation for {drop_id} "
                    "(only non-fish labels). Advancing to complete."
                )
                db.advance_status(drop_id, ExpertStatus.COLUMN, ExpertStatus.COMPLETE)
                continue

            # Replace only Biigle-sourced expert annotations (external_id IS NOT NULL).
            # Manually-entered expert annotations (external_id = NULL) are preserved.
            ann_db.clear_synced_annotations(drop_id, "expert")
            ann_db.add_annotations(annotations_to_add)

            # Export MaxN CSV per drop
            maxn_df = maxn_rows_to_df(annotations_to_add)
            maxn_path = config.get_biigle_expert_maxn_csv_path(drop_id)
            maxn_df.to_csv(maxn_path, index=False)
            logging.info(f"  Expert MaxN → {maxn_path}")

            processed_drops.append(drop_id)
            logging.info(
                f"  Ingested {len(annotations_to_add)} annotations for {drop_id}"
            )

            db.advance_status(drop_id, ExpertStatus.COLUMN, ExpertStatus.COMPLETE)

        except Exception as e:
            logging.error(
                f"  Failed to sync volume {volume_id} for {drop_id}: {e}",
                exc_info=True,
            )

    if processed_drops:
        db.sync_annotation_counts(processed_drops)

        # Rebuild YOLO labels from all expert CSVs (class map needs all drops)
        class_map_path = config.local_training_dir / "class_map.json"
        biigle_to_yolo(
            deployment_data_dir=config.deployment_data_dir,
            class_map_path=class_map_path,
        )

        logging.info(f"Biigle sync complete. Processed {len(processed_drops)} drops.")
    else:
        logging.info("No new 'Done' volumes found to process.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sync_biigle_annotations()
