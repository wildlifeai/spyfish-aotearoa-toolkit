"""Re-run MaxN pooling over every full-video raw CSV on disk.

One-off after the persistence-filter change (2026-08-21): pooling is pure
post-processing on raw CSVs, so historical drops can be brought onto the new
MaxN semantics (persistence window, gap fill, bait exclusion, fish union)
without any video download or inference. Rewrites each drop's MaxN CSV and
its annotations-DB rows (scoped by external_id = the model that produced the
raw CSV, so other sources are untouched), then syncs annotation counts.

The model name is derived from each raw CSV's filename rather than from
config.pipeline_model_path, because historical CSVs were written by older
models and re-pooling must not relabel their provenance.

Usage (from project root):
    python scripts/repool_maxn.py            # re-pool everything found
    python scripts/repool_maxn.py --dry-run  # just list what would run
"""

import argparse
import logging

from spyfish.config.wrapper import config
from spyfish.database.annotation_manager import AnnotationDatabaseManager
from spyfish.database.manager import DatabaseManager
from spyfish.log_config import log_header
from spyfish.ml.process_ml_annotations import process_one_drop


def find_raw_csvs() -> list:
    """(drop_id, model_name, path) for every full-video raw CSV on disk.

    Skips the Zooniverse-frame rerun CSVs and BIIGLE exports — they are
    per-frame files, not full-video timelines, and never feed MaxN.
    """
    found = []
    for raw in sorted(config.deployment_data_dir.glob("*/*/annotations/*_raw.csv")):
        if "zooniverse_frames" in raw.name or "biigle" in raw.name:
            continue
        drop_id = raw.parent.parent.name
        model_name = raw.stem.removeprefix(f"{drop_id}_").removesuffix("_raw")
        found.append((drop_id, model_name, raw))
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true", help="List targets without re-pooling."
    )
    args = parser.parse_args()

    targets = find_raw_csvs()
    logging.info(f"{len(targets)} raw CSV(s) found under {config.deployment_data_dir}")
    for drop_id, model_name, _ in targets:
        logging.info(f"  {drop_id}  ({model_name})")
    if args.dry_run or not targets:
        return

    ann_db = AnnotationDatabaseManager()
    done, failed = [], []
    for drop_id, model_name, _ in targets:
        try:
            process_one_drop(
                drop_id=drop_id,
                video_dir=config.media_dir,
                ann_db=ann_db,
                model_name=model_name,
                interval=config.interval_seconds,
                base_conf=config.confidence_threshold,
                maxn_conf=config.maxn_confidence_threshold,
                draw_images=False,  # no video needed: MaxN + DB ingest only
            )
            done.append(drop_id)
        except Exception as e:
            failed.append(drop_id)
            logging.error(f"{drop_id}: re-pool failed: {e}", exc_info=True)

    if done:
        DatabaseManager().sync_annotation_counts(done)
    logging.info(f"Re-pooled {len(done)} drop(s); {len(failed)} failed.")
    if failed:
        logging.warning(f"Failed drops: {failed}")


if __name__ == "__main__":
    log_header("Re-pool MaxN from raw CSVs")
    main()
