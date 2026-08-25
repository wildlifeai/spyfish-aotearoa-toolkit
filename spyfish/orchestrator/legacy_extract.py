import logging
import tempfile
from pathlib import Path

import pandas as pd

from spyfish.config.base import ExpertStatus
from spyfish.config.wrapper import config
from spyfish.database.annotation_manager import (
    AnnotationDatabaseManager,
    null_deployment_row,
)
from spyfish.database.manager import DatabaseManager
from spyfish.storage.s3_handler import S3Handler


def parse_legacy_rows(df: pd.DataFrame) -> tuple[list, list]:
    """Normalise legacy CSV rows into annotation records.

    Returns ``(annotations, placeholder_drop_ids)``, where the placeholders
    are ALSO present in ``annotations`` as absence records.

    Rows where ScientificName, TimeOfMax AND MaxInterval are all missing are
    the legacy export's "no expert data" marker: that database wrote the
    literal string "NULL" into every data field (pandas reads "NULL" as
    missing).

    These become `null_deployment_row` records — "expert reviewed this and
    saw nothing" (2026-08-22 decision, Kalindi). The evidence supports reading
    them as absence rather than as unknown: there are 34 of them, **exactly
    one row per deployment**, with **zero overlap** with the 294 deployments
    that carry real observations. A deployment that was never looked at would
    not have earned a row in an annotations export at all; a deliberate
    one-row-per-drop placeholder is what "processed, nothing to report" looks
    like.

    History, so this does not flip a third time: the original code ingested
    these as ordinary annotations, which fabricated an observation of an
    unnamed species and pushed a bogus count of 1 into the deployment. That
    was correctly reverted to a skip — but the skip lost the fact of review,
    leaving 34 deployments indistinguishable from never-reviewed. The absence
    row is the third option and the right one: it asserts the review, counts
    a real zero, and dashboards already map NULL_DEPLOYMENT to NaN so it
    never reads as a species.
    """
    conf_col = config.csv_confidence_agreement_column
    intv_col = config.csv_interval_annotation_column

    annotations, placeholders = [], []
    for row in df.to_dict("records"):
        sci = row.get(config.csv_scientific_name_column)
        t_max = row.get(config.csv_maxn_time_column)
        m_intv = row.get(config.csv_max_interval_column)

        if pd.isna(sci) and pd.isna(t_max) and pd.isna(m_intv):
            drop_id = row[config.drop_id_column]
            placeholders.append(drop_id)
            # external_id='legacy' matches the DELETE scope in the caller, so
            # a re-ingest replaces these rather than accumulating duplicates.
            annotations.append(
                null_deployment_row(drop_id, "expert", external_id="legacy")
            )
            continue

        conf = row.get(conf_col)
        confidence = None if (pd.isna(conf) or conf == "NA") else float(conf)
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
    return annotations, placeholders


def ingest_legacy_expert_annotations():
    """Download legacy expert annotations from S3 and ingest them.

    Five-step flow:
      1. Download legacy CSV from S3.
      2. Parse and normalise rows; all-NULL export placeholders become
         absence records (see `parse_legacy_rows`).
      3. Replace existing legacy rows in spyfish_annotations.db (DELETE+INSERT,
         keyed on annotated_by='expert' AND external_id='legacy' so this never
         touches BIIGLE-synced expert rows).
      4. `sync_annotation_counts()`, writes the aggregated per-drop counts
         back to the deployments table AND advances expert_status to
         expert_complete for any drop that gained annotations (data presence
         is the source of truth).
      5. Legacy repair for the pre-absence-row era: demote a placeholder drop
         to expert_pending only when it has NO expert rows at all. Since step 2
         now writes an absence row for every placeholder, this is inert on a
         normal run — it survives to repair DBs written by the older code.

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
        annotations, placeholders = parse_legacy_rows(df)
        if placeholders:
            logging.info(
                f"{len(placeholders)} placeholder row(s) had every data field "
                f"NULL — the export's 'no expert data' marker. Ingested as "
                f"absence records (expert reviewed, nothing seen): "
                f"{sorted(placeholders)}"
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
        # Placeholder drops are included so a prior ingest's bogus count of 1
        # is recounted down to its true value.
        drop_ids = list({a["drop_id"] for a in annotations} | set(placeholders))
        main_db = DatabaseManager()
        main_db.sync_annotation_counts(drop_ids)

        # 5. Repair status for placeholder drops. An earlier ingest of these
        # same rows counted each placeholder as an annotation and advanced the
        # drop to expert_complete, dropping it out of the review queue. The
        # count sync above only ever advances, so demote here — but only when
        # the drop has no expert rows from ANY path (a BIIGLE-synced review
        # still legitimately completes it).
        demoted = 0
        for drop_id in sorted(set(placeholders)):
            if ann_db.get_annotations_for_drop(drop_id, "expert"):
                continue
            dep = main_db.get_deployment(drop_id)
            if dep and dep.get(ExpertStatus.COLUMN) == ExpertStatus.COMPLETE:
                main_db.update_section_status(
                    drop_id, ExpertStatus.COLUMN, ExpertStatus.PENDING
                )
                demoted += 1
        if demoted:
            logging.info(
                f"Reverted expert_status → {ExpertStatus.PENDING} for {demoted} "
                "placeholder drop(s) with no expert data."
            )

    except Exception as e:
        logging.error(f"Failed legacy ingestion: {e}")
        raise
    finally:
        if local_csv.exists():
            local_csv.unlink()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ingest_legacy_expert_annotations()
