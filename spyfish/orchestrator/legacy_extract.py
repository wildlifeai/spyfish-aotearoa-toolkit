import logging
import tempfile
from pathlib import Path

import pandas as pd

from spyfish.config.base import ExpertStatus
from spyfish.config.wrapper import config
from spyfish.database.annotation_manager import (
    LEGACY_EXTERNAL_ID,
    AnnotationDatabaseManager,
    null_deployment_row,
)
from spyfish.database.manager import DatabaseManager
from spyfish.storage.s3_handler import S3Handler


def legacy_time_of_max_seconds(drop_ids: pd.Series, times: pd.Series) -> pd.Series:
    """Video-relative seconds for legacy TimeOfMax text.

    The single source of truth for reading legacy time text — the ingest
    below, the DB backfill script and the report dashboard all call this, so
    one row cannot parse to two different moments in two places.
    `tests/unit/test_one_time_parser.py` fails if a second reader appears.

    Deliberately NOT `spyfish.utils.time_to_seconds`: that one is for
    timestamps the pipeline wrote (already correct, no repair wanted) and
    returns 0.0 for missing input, which here would read as "seen in the
    first minute" rather than "time unknown".

    Most legacy TimeOfMax text is honest `HH:MM:SS` with a zero hours field.
    But a slice of the hand-entered workbooks reached the extractor as Excel
    time cells that had swallowed `MM:SS` entries as `HH:MM` — "23:45"
    meaning 23 minutes 45 seconds became the cell 23:45:00 — so their first
    field holds MINUTES (rule from Kalindi, 2026-10-02: hand legacy has no
    millisecond precision, and a nonzero first field there can only be
    minutes, since no BUV video is an hour long).

    Three cases, validated against the full import (72,694 timed rows: the
    minutes-first reading leaves zero repaired rows past 37 minutes, where
    reading them as written would leave 3,653):

    * A nonzero first field on a hand row (no fractional seconds) is minutes:
      seconds = first*60 + second. This also catches an analyst's "17:00"
      for seventeen minutes, which Excel stored as 17:00:00.
    * Within a deployment that holds any such row, the whole sheet went
      through Excel's coercion, so its `00:SS:00`-shaped rows are seconds,
      not minutes ("0:17" for seventeen seconds became 00:17:00). Rows with
      a nonzero THIRD field are immune: the coercion always produces :00
      seconds, so those are genuine `HH:MM:SS` even inside a flagged
      deployment — which also protects mis-flagged sheets from being
      corrupted by one typo.
    * Everything else (fractional-second player exports included) parses as
      written. Rows landing at 35-37 minutes are real observations from
      videos running past the nominal cutoff and are kept.

    Returns a float series aligned to the inputs; NaN where the text is
    missing or unparseable. `legacy_time_reading` returns the same seconds
    plus the as-written reading and the repair mask, for a caller that needs
    to report how much was repaired — so no caller ever parses the text a
    second time to find out.
    """
    return legacy_time_reading(drop_ids, times)["seconds"]


def legacy_time_reading(drop_ids: pd.Series, times: pd.Series) -> pd.DataFrame:
    """The one parse of legacy time text, with its workings exposed.

    `legacy_time_of_max_seconds` is the answer most callers want; this is the
    same computation returning three aligned columns:

    * `seconds` — the value to store and use.
    * `as_written` — what a naive `HH:MM:SS` read would have given.
    * `repaired` — True where the two differ because the Excel-coercion rule
      fired (see `legacy_time_of_max_seconds` for the rule itself).

    It exists so a report of "how many rows did the repair touch" costs no
    second parse. Two parses of one format is how the formats drift apart:
    the second copy is written from the first's docstring, not its code, and
    the next edge case is only ever fixed in one of them.
    """
    # Stripped first: the repair regex is anchored while `to_timedelta`
    # tolerates surrounding spaces, so " 23:45:00" used to parse as written
    # (23 hours) and dodge the repair that fixed its neighbours.
    text = times.astype(str).str.strip()
    as_written = pd.to_timedelta(text, errors="coerce").dt.total_seconds()
    fields = text.str.extract(r"^(\d+):(\d+):(\d+)(\.\d+)?$")
    first = pd.to_numeric(fields[0], errors="coerce")
    second = pd.to_numeric(fields[1], errors="coerce")
    third = pd.to_numeric(fields[2], errors="coerce")
    hand = fields[3].isna() & first.notna()

    minutes_first = hand & (first > 0)
    # A missing DropID must not form a pseudo-deployment: `isin` treats NaN
    # as equal to NaN, so without the guard every unkeyed row would share
    # one coercion flag.
    coerced_deps = set(drop_ids[minutes_first].dropna())
    same_dep = drop_ids.notna() & drop_ids.isin(coerced_deps)
    coerced = minutes_first | (hand & (third == 0) & same_dep)
    seconds = as_written.where(~coerced, first * 60 + second)
    return pd.DataFrame(
        {
            "seconds": seconds,
            "as_written": as_written,
            "repaired": coerced & seconds.notna() & (seconds != as_written),
        }
    )


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

    # Seconds are derived at ingest so every consumer of the DB gets the
    # repaired, video-relative value without re-deriving it from the text
    # (the text column is kept verbatim for audit).
    time_col = config.csv_maxn_time_column
    if time_col in df.columns:
        seconds = legacy_time_of_max_seconds(df[config.drop_id_column], df[time_col])
    else:
        seconds = pd.Series(float("nan"), index=df.index)

    annotations, placeholders = [], []
    for i, row in enumerate(df.to_dict("records")):
        sci = row.get(config.csv_scientific_name_column)
        t_max = row.get(config.csv_maxn_time_column)
        m_intv = row.get(config.csv_max_interval_column)

        if pd.isna(sci) and pd.isna(t_max) and pd.isna(m_intv):
            drop_id = row[config.drop_id_column]
            placeholders.append(drop_id)
            # external_id='legacy' matches the DELETE scope in the caller, so
            # a re-ingest replaces these rather than accumulating duplicates.
            annotations.append(
                null_deployment_row(drop_id, "expert", external_id=LEGACY_EXTERNAL_ID)
            )
            continue

        conf = row.get(conf_col)
        confidence = None if (pd.isna(conf) or conf == "NA") else float(conf)
        intv_ann = row.get(intv_col)
        secs = seconds.iloc[i]

        annotations.append(
            {
                "drop_id": row[config.drop_id_column],
                "scientific_name": None if pd.isna(sci) else sci,
                "time_of_max": None if pd.isna(t_max) else t_max,
                "time_of_max_seconds": None if pd.isna(secs) else float(secs),
                "max_interval": 0 if pd.isna(m_intv) else m_intv,
                "annotated_by": "expert",
                "interval_annotation": None if pd.isna(intv_ann) else intv_ann,
                "confidence_agreement": confidence,
                "external_id": LEGACY_EXTERNAL_ID,  # distinguishes these from Biigle-synced expert annotations
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
                "DELETE FROM annotations WHERE annotated_by = 'expert' AND external_id = ?",
                (LEGACY_EXTERNAL_ID,),
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
