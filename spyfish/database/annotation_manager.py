import logging
import re
import sqlite3
from contextlib import closing, contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from spyfish.config.base import NULL_DEPLOYMENT
from spyfish.config.wrapper import config
from spyfish.database.connection import connect_readonly

# ── Provenance: what `external_id` means per source ─────────────────────────
#
# `annotated_by` says WHO (ml / citsci / expert); `external_id` says WHERE FROM,
# and each writer owns a disjoint key space so no re-ingest can delete another
# writer's rows:
#
#   ml      → the model name            (`species_20260822_192536`)
#   expert  → `legacy` for the historical CSV backfill, or
#             `biigle_volume_{id}` for every row synced from that BIIGLE volume
#   citsci  → the Zooniverse subject id, a per-ROW key, not a source key
#
# Expert rows used to carry the BIIGLE annotation id, one per box. It was the
# first id of each aggregated (frame, species) group, nothing read it back, and
# worst, it made `clear_synced_annotations`'s "NOT NULL" test match the legacy
# rows too: the first BIIGLE sync on a drop with legacy expert data would have
# deleted the legacy account (found 2026-10-13, before it ever fired).
LEGACY_EXTERNAL_ID = "legacy"
# Ownership prefix: every `external_id` the BIIGLE sync writes starts with it,
# and `clear_synced_annotations` deletes by it. Matched with GLOB, not LIKE,
# because `_` is a single-character wildcard in LIKE.
BIIGLE_EXTERNAL_ID_PREFIX = "biigle_"


def biigle_external_id(volume_id) -> str:
    """The `external_id` every expert row synced from one BIIGLE volume carries.

    Volume-scoped, like the model name on ML rows: it names the review the
    row came from, so a re-sync of that volume replaces exactly its own rows
    (`clear_synced_annotations(..., volume_id=...)`) and two volumes that both
    hold the same drop coexist in the table, each with its own peak.
    """
    return f"{BIIGLE_EXTERNAL_ID_PREFIX}volume_{volume_id}"


def annotation_source(annotated_by: pd.Series, external_id: pd.Series) -> pd.Series:
    """One label per row saying which account it belongs to, for exports.

    `legacy` or `biigle_volume_N` for expert rows, the model name for ML rows,
    the source name itself where `external_id` is a per-row key (citsci) or
    missing. Readers of the exported CSV see this where `AnnotatedBy` alone
    cannot tell two expert accounts of one deployment apart.
    """
    ext = external_id.astype("string")
    is_expert = annotated_by.isin(("expert", "expert_training"))
    is_ml = annotated_by == "ml"
    expert_src = ext.where(
        (ext == LEGACY_EXTERNAL_ID) | ext.str.startswith(BIIGLE_EXTERNAL_ID_PREFIX),
        annotated_by,
    )
    out = annotated_by.astype("string")
    out = out.mask(is_expert, expert_src)
    out = out.mask(is_ml & ext.notna() & (ext != ""), ext)
    return out.astype(object)


def null_deployment_row(
    drop_id: str, annotated_by: str, external_id: Optional[str] = None
) -> Dict[str, Any]:
    """The absence record: this source reviewed the deployment and saw nothing.

    One shape for every writer (ML zero-detection, citsci all-NOTHINGHERE,
    empty BIIGLE review), so the convention cannot drift between sources.
    `scientific_name` and `time_of_max` carry the NULL_DEPLOYMENT marker —
    display fields, where a blank would read as missing data. The numeric
    fields stay honest numbers: the count for "nothing seen" IS zero, and
    there is no timestamp or confidence to report.
    """
    return {
        "drop_id": drop_id,
        "scientific_name": NULL_DEPLOYMENT,
        "time_of_max": NULL_DEPLOYMENT,
        "time_of_max_seconds": None,
        "max_interval": 0,
        "annotated_by": annotated_by,
        "interval_annotation": "",
        "confidence_agreement": None,
        "external_id": external_id,
    }


# The training stamp in a model name: `species_20260822_192536`,
# `cfd_binary_water_20260301`, `20260429_081503_species`. Date always, time
# only on some, and in no fixed position, so it is searched for, not split out.
def model_recency_key(model_name) -> tuple:
    """Sort key putting the most recently trained model last.

    Newest is read from the training stamp in the name, not from when the rows
    were inserted: re-running an old model on a drop today must not make it
    win over a newer one. A name with no stamp sorts as oldest. The name itself
    breaks ties, so the choice is deterministic.
    """
    name = "" if pd.isna(model_name) else str(model_name)
    return (config.model_name_stamp(name) or "", name)


def newest_model_only(
    df: pd.DataFrame, models: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """Keep, per deployment, only the newest model's ML rows.

    A drop can carry rows from several models (re-runs, binary then species).
    Reports must show one model's account per drop, not the larger count of
    any model per species, so every older model's ML rows are dropped. Expert
    and citsci rows are untouched. The model name is `external_id`.

    `models` (`drop_id`, `external_id` of ML rows) says which models ran on
    each drop when `df` is only a slice of the table. Pass it whenever `df`
    is filtered by species: if the newest model did not see that species,
    choosing from `df` alone would fall back to an older model that did.
    """
    is_ml = df["annotated_by"] == "ml"
    if not is_ml.any():
        return df
    source = df.loc[is_ml] if models is None else models
    ml = source[["drop_id", "external_id"]].drop_duplicates()
    ml["_key"] = ml["external_id"].map(model_recency_key)
    newest = ml.sort_values("_key").groupby("drop_id").tail(1)
    keep = pd.MultiIndex.from_frame(newest[["drop_id", "external_id"]])
    rows = pd.MultiIndex.from_frame(df[["drop_id", "external_id"]])
    return df[~is_ml | rows.isin(keep)]


class AnnotationDatabaseManager:
    """
    Manages the spyfish_annotations.db which stores detailed individual annotation records
    from ML, Expert (Biigle), and Citizen Science sources.
    """

    def __init__(self, db_path: Optional[str] = None, read_only: bool = False):
        """Open the annotations database, creating its schema unless ``read_only``.

        Readers pass ``read_only=True`` so a missing file raises rather than
        being created empty (see ``spyfish/database/connection.py``).
        """
        if db_path:
            self.db_path = str(Path(db_path).absolute())
        else:
            self.db_path = str(config.annotations_db_path)

        self.read_only = read_only
        if not read_only:
            self.init_db()

    def get_connection(self):
        """Returns a configured SQLite connection wrapped in contextlib.closing.

        ⚠️  DOES NOT auto-commit. Use this for SELECT queries. For DML
        (INSERT/UPDATE/DELETE), prefer get_writable_connection() which
        commits on exit and rolls back on exception.
        """
        if self.read_only:
            conn = connect_readonly(self.db_path)
        else:
            conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        return closing(conn)

    @contextmanager
    def get_writable_connection(self):
        """Yields a SQLite connection that auto-commits on clean exit.

        Use this for any DML (INSERT/UPDATE/DELETE). On exception, the
        transaction is rolled back and the exception re-raised.

        Added 2026-05-14 after legacy expert ingestion silently triplicated
        rows: callers using ``get_connection()`` for DML must remember to
        call ``conn.commit()`` explicitly, and one did not.
        """
        if self.read_only:
            raise RuntimeError(f"{self.db_path} was opened read-only")
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def init_db(self):
        """Initializes the annotations schema."""
        logging.info(f"Initializing Annotation DB at {self.db_path}")
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS annotations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    drop_id TEXT NOT NULL,
                    scientific_name TEXT,
                    time_of_max TEXT, -- HH:MM:SS
                    time_of_max_seconds REAL,
                    max_interval INTEGER DEFAULT 1,
                    annotated_by TEXT NOT NULL, -- e.g., 'expert'
                    interval_annotation TEXT,
                    confidence_agreement REAL,
                    external_id TEXT, -- e.g. Biigle annotation ID
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # Migrate existing DBs that pre-date the time_of_max_seconds column
            try:
                cursor.execute(
                    "ALTER TABLE annotations ADD COLUMN time_of_max_seconds REAL"
                )
            except Exception:
                pass  # column already exists

            # Index for fast aggregation by drop_id and source (annotated_by)
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_drop_source ON annotations(drop_id, annotated_by)"
            )
            # Covering index for `get_maxn_summary`'s correlated subquery, which
            # finds the peak row per (drop, source, species). Column order and
            # sort direction mirror that subquery's WHERE + ORDER BY exactly, so
            # SQLite reads the first matching entry instead of scanning every
            # interval row for each output row. Measured on the real 94k-row DB
            # (2026-08-22): full-table summary 4,371 ms → 43 ms, byte-identical
            # results, +5 MB on disk. Per-drop lookups were already fast, so this
            # is specifically the dashboard's whole-table read.
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_maxn_peak ON annotations"
                "(drop_id, annotated_by, scientific_name, max_interval DESC, id ASC)"
            )
            # Rows ingested before `_ingest_ml_annotations` wrote 'ml' carry the
            # model name in annotated_by. Move it to external_id so every ML row
            # reads `annotated_by = 'ml'`. Anything that is not one of the three
            # known sources IS a model name (a date in it was once required, but
            # an undated model slipped through as a fourth source); idempotent.
            cursor.execute(
                "UPDATE annotations "
                "SET external_id = COALESCE(external_id, annotated_by), "
                "annotated_by = 'ml' "
                "WHERE annotated_by IS NOT NULL "
                "AND annotated_by NOT IN ('ml', 'citsci', 'expert')"
            )
            if cursor.rowcount:
                logging.info(
                    f"Moved the model name out of annotated_by on {cursor.rowcount} "
                    "legacy ML row(s)"
                )
            self._migrate_biigle_external_ids(cursor)
            self._normalise_model_names(cursor)
            conn.commit()

    def _normalise_model_names(self, cursor) -> None:
        """Rewrite stamp-first model names to the `{type}_{stamp}` standard.

        Auto-promotion used to name the weights after the training run dir,
        `20260429_081503_species`, while hand-promoted files and the path
        resolver used `species_20260429_081503`. One model, two names, and a
        name starting with a digit also tripped the numeric-id escape hatch
        in `get_maxn_summary`. Idempotent: a canonical name does not match.
        """
        stamp_first = re.compile(r"^(20\d{6}(?:_\d{6})?)_([a-z][a-z0-9_]*)$")
        cursor.execute(
            "SELECT DISTINCT external_id FROM annotations "
            "WHERE annotated_by = 'ml' AND external_id IS NOT NULL"
        )
        for (name,) in cursor.fetchall():
            m = stamp_first.match(str(name))
            if not m:
                continue
            canonical = f"{m.group(2)}_{m.group(1)}"
            cursor.execute(
                "UPDATE annotations SET external_id = ? "
                "WHERE annotated_by = 'ml' AND external_id = ?",
                (canonical, name),
            )
            logging.info(
                f"Renamed model {name!r} -> {canonical!r} on {cursor.rowcount} ML row(s)"
            )

    def _migrate_biigle_external_ids(self, cursor) -> None:
        """Rewrite per-box BIIGLE annotation ids to the volume-scoped source key.

        Expert rows synced before 2026-10-13 hold a bare numeric annotation id
        in `external_id`. The volume each drop was reviewed in is on the
        pipeline DB's `deployments.biigle_volume_id`, so attach it and copy
        the id across; a drop the pipeline DB does not know keeps the BIIGLE
        prefix with no volume, which still marks it as BIIGLE-owned for
        `clear_synced_annotations`. Idempotent: only bare numeric ids match.
        """
        pipeline_db = Path(config.db_path)
        if not pipeline_db.exists() or str(pipeline_db) == self.db_path:
            return
        numeric = "annotated_by = 'expert' AND external_id GLOB '[0-9]*'"
        cursor.execute(f"SELECT COUNT(*) FROM annotations WHERE {numeric}")
        if cursor.fetchone()[0] == 0:
            return
        cursor.execute("ATTACH DATABASE ? AS pipeline", (str(pipeline_db),))
        try:
            cursor.execute(
                f"""
                UPDATE annotations
                SET external_id = ? || (
                    SELECT biigle_volume_id FROM pipeline.deployments d
                    WHERE d.drop_id = annotations.drop_id
                )
                WHERE {numeric} AND EXISTS (
                    SELECT 1 FROM pipeline.deployments d
                    WHERE d.drop_id = annotations.drop_id
                      AND d.biigle_volume_id IS NOT NULL
                )
                """,
                # `biigle_external_id("")` is the key minus the volume number,
                # so SQL and Python build the same string.
                (biigle_external_id(""),),
            )
            moved = cursor.rowcount
            cursor.execute(
                f"UPDATE annotations SET external_id = ? WHERE {numeric}",
                (biigle_external_id("unknown"),),
            )
            orphaned = cursor.rowcount
        finally:
            # ATTACH is per connection and must not outlive the migration.
            cursor.connection.commit()
            cursor.execute("DETACH DATABASE pipeline")
        logging.info(
            f"Rewrote BIIGLE annotation ids to volume keys on {moved} expert row(s)"
            + (f"; {orphaned} row(s) had no volume on record" if orphaned else "")
        )

    def add_annotations(self, annotations: List[Dict[str, Any]]):
        """Bulk insert annotation records."""
        if not annotations:
            return
        annotations = [
            {**a, "time_of_max_seconds": a.get("time_of_max_seconds")}
            for a in annotations
        ]
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany(
                """
                INSERT INTO annotations (drop_id, scientific_name, time_of_max, time_of_max_seconds, max_interval, annotated_by, interval_annotation, confidence_agreement, external_id)
                VALUES (:drop_id, :scientific_name, :time_of_max, :time_of_max_seconds, :max_interval, :annotated_by, :interval_annotation, :confidence_agreement, :external_id)
            """,
                annotations,
            )
            conn.commit()

    def clear_annotations(
        self,
        drop_id: str,
        annotated_by: str,
        external_id: Optional[str] = None,
    ):
        """Clear existing annotations for a given drop + source.

        When ``external_id`` is provided, only rows matching that exact
        external_id are deleted, useful for ML ingestion where each model
        writes rows tagged with its own name, and we want a re-run of one
        model to leave other models' rows intact. Without it, every row
        for the (drop_id, annotated_by) pair is cleared.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if external_id is None:
                cursor.execute(
                    "DELETE FROM annotations WHERE drop_id = ? AND annotated_by = ?",
                    (drop_id, annotated_by),
                )
            else:
                cursor.execute(
                    "DELETE FROM annotations "
                    "WHERE drop_id = ? AND annotated_by = ? AND external_id = ?",
                    (drop_id, annotated_by, external_id),
                )
            conn.commit()

    def clear_synced_annotations(self, drop_id: str, annotated_by: str, volume_id=None):
        """Clear the BIIGLE-synced rows a re-sync is about to replace.

        With `volume_id`, only that volume's rows (`external_id` =
        `biigle_volume_{id}`) go: a drop reviewed in two volumes keeps the
        other volume's account, and the two sit side by side with their own
        peaks. Without it, every `biigle_*` row of the drop goes (the old
        behaviour, kept for callers that have no volume in hand). Either way
        the legacy expert account (`legacy`) and hand-entered rows
        (`external_id` NULL) are never touched, the same ownership rule the
        legacy backfill and the ML refresh follow for their own keys.
        """
        owned = (
            biigle_external_id(volume_id)
            if volume_id is not None
            else f"{BIIGLE_EXTERNAL_ID_PREFIX}*"
        )
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM annotations "
                "WHERE drop_id = ? AND annotated_by = ? AND external_id GLOB ?",
                (drop_id, annotated_by, owned),
            )
            conn.commit()

    def get_annotations_for_drop(
        self, drop_id: str, annotated_by: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve annotations for a drop, optionally filtered by source."""
        query = "SELECT * FROM annotations WHERE drop_id = ?"
        params = [drop_id]
        if annotated_by:
            query += " AND annotated_by = ?"
            params.append(annotated_by)

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    # NOTE: `get_counts_per_source` (SUM(max_interval) per source) was deleted
    # 2026-08-22. Summing MaxN across time intervals is not a meaningful
    # quantity — it scales with how many intervals a deployment happens to
    # have, so a longer deployment reads as "more fish". It had no callers.
    # For the real per-deployment figure use `get_maxn_summary()` (the peak per
    # drop × species × source); for pipeline-monitoring row counts use the
    # `ml_annotations` / `citsci_annotations` / `expert_annotations` columns
    # that `DatabaseManager.sync_annotation_counts` maintains.

    def get_all_annotations_export_df(self) -> Optional[pd.DataFrame]:
        """
        Returns a DataFrame of all annotations formatted exactly for export/app display,
        with columns: DropID, ScientificName, CommonName, TimeOfMax, MaxInterval,
        AnnotatedBy, IntervalAnnotation, ConfidenceAgreement.

        CommonName is looked up in the species registry: this file is read by
        rangers and contractors, who know "snapper" rather than "Pagrus
        auratus". Blank for the absence sentinel and for any name the registry
        does not know.
        """
        import pandas as pd

        from spyfish.config.species import species_registry

        with self.get_connection() as conn:
            df = pd.read_sql_query("SELECT * FROM annotations", conn)

        if df.empty:
            return None

        # Map internal schema strictly to requested export columns using the exact casing
        df = df.rename(
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
        registry = species_registry()

        def common_name(scientific):
            species = registry.get(scientific)
            if species is None or species.is_bucket:
                return None
            return species.common_name

        df[config.csv_common_name_column] = df[config.csv_scientific_name_column].map(
            common_name
        )
        # Which account a row belongs to: `legacy` vs `biigle_volume_N` for
        # expert rows, the model name for ML. `AnnotatedBy` alone reads
        # "expert" for both expert accounts of one deployment.
        df[config.csv_source_column] = annotation_source(
            df[config.csv_annotated_by_column], df["external_id"]
        )

        # We drop any internal columns (like id, created_at) by strictly selecting
        export_cols = [
            config.drop_id_column,
            config.csv_scientific_name_column,
            config.csv_common_name_column,
            config.csv_maxn_time_column,
            config.csv_max_interval_column,
            config.csv_annotated_by_column,
            config.csv_source_column,
            config.csv_interval_annotation_column,
            config.csv_confidence_agreement_column,
        ]

        return df[export_cols]

    def get_maxn_summary(
        self,
        drop_id: Optional[str] = None,
        annotated_by: Optional[str] = None,
        all_models: bool = False,
    ) -> pd.DataFrame:
        """
        Returns the canonical MaxN per drop × species × source, i.e. the peak
        max_interval across all time intervals for each combination.

        This is the scientific result; contrast with deployments.ml_annotations
        which is a COUNT of MaxN records used only for pipeline monitoring.

        Args:
            drop_id: Filter to a single deployment. If None, returns all.
            annotated_by: Filter by source ('ml', 'expert', 'citsci'). If None, returns all.
            all_models: Keep every model's ML rows (one peak per model) instead
                of only the newest model's. For provenance views that compare
                models on one drop; reports leave it False.
        """
        # Correlated subquery picks the single row with the peak max_interval
        # per (drop_id, scientific_name, annotated_by), and for ML per model
        # too, so two models are never merged into one peak. Ties broken by
        # smallest id (earliest insert). This ensures time_of_max_seconds and
        # confidence_agreement come from the actual peak row, not an arbitrary
        # one (which is what a plain GROUP BY returns in SQLite).
        extra, params = [], []
        if drop_id:
            extra.append("a.drop_id = ?")
            params.append(drop_id)
        if annotated_by:
            extra.append("a.annotated_by = ?")
            params.append(annotated_by)
        extra_clause = (" AND ".join(extra) + " AND ") if extra else ""

        query = f"""
            SELECT a.drop_id, a.scientific_name, a.annotated_by,
                   a.max_interval AS maxn,
                   a.time_of_max,
                   a.time_of_max_seconds,
                   a.confidence_agreement,
                   a.external_id
            FROM annotations a
            WHERE {extra_clause}a.id = (
                SELECT id FROM annotations
                WHERE drop_id = a.drop_id
                  -- IS, not =: null-species rows record "reviewed, nothing
                  -- seen" and NULL = NULL is never true, so = silently drops
                  -- every absence record from the summary.
                  AND scientific_name IS a.scientific_name
                  AND annotated_by = a.annotated_by
                  -- external_id names the SOURCE on ML rows (model) and expert
                  -- rows (legacy / biigle_volume_N), so each account keeps its
                  -- own peak. On citsci rows it is a per-row subject id and
                  -- must not split the peak; nor may a bare numeric expert id
                  -- (pre-migration BIIGLE annotation id, seen by read-only
                  -- dashboards until a writable open has migrated the DB).
                  AND (a.annotated_by = 'citsci'
                       OR (a.annotated_by = 'expert'
                           AND a.external_id GLOB '[0-9]*')
                       OR external_id IS a.external_id)
                ORDER BY max_interval DESC, id ASC
                LIMIT 1
            )
            ORDER BY a.drop_id, maxn DESC
        """

        with self.get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=params)
        return df if all_models else newest_model_only(df)

    def export_to_csv(self, output_dir: Optional[str] = None) -> List[str]:
        """Export all annotation DB tables to CSV files. Returns list of written file paths."""
        out = Path(output_dir) if output_dir else Path(self.db_path).parent
        out.mkdir(parents=True, exist_ok=True)

        paths = []
        with self.get_connection() as conn:
            raw_path = out / "annotations.csv"
            pd.read_sql("SELECT * FROM annotations", conn).to_csv(raw_path, index=False)
            logging.info(f"Exported annotations → {raw_path}")
            paths.append(str(raw_path))

        maxn_path = out / "maxn_summary.csv"
        self.get_maxn_summary().to_csv(maxn_path, index=False)
        logging.info(f"Exported MaxN summary → {maxn_path}")
        paths.append(str(maxn_path))
        return paths
