import logging
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any, Dict, List, Optional

from spyfish.config.base import InvalidTransitionError, PipelineStatus, SourceStatus
from spyfish.config.wrapper import config


class DatabaseManager:
    """
    Core SQLite database manager for the Spyfish pipeline.
    Uses pure sqlite3 with dict-like row factories for simplicity.
    """

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = (
            str(config.db_path) if db_path is None else str(Path(db_path).absolute())
        )
        self.init_db()

    def get_connection(self):
        """Returns a configured SQLite connection wrapped in contextlib.closing."""
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row  # Access columns by name
        return closing(conn)

    def init_db(self):
        """Creates the core 'deployments' table if it does not exist."""
        logging.info(f"Initializing database at {self.db_path}")
        with self.get_connection() as conn:
            cursor = conn.cursor()
            # We track the bare minimum needed for orchestration. Metadata lives in BUV Deployments.csv
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS deployments (
                    drop_id TEXT PRIMARY KEY,
                    video_path TEXT,
                    status TEXT NOT NULL,
                    source_status TEXT NOT NULL DEFAULT 'OK',
                    is_bad_deployment BOOLEAN NOT NULL DEFAULT 0,
                    error_message TEXT,
                    sampling_start INTEGER,
                    sampling_end INTEGER,
                    ml_annotations INTEGER DEFAULT 0,
                    citsci_annotations INTEGER DEFAULT 0,
                    expert_annotations INTEGER DEFAULT 0,
                    biigle_volume_id TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS sites (
                    site_id TEXT PRIMARY KEY,
                    site_name TEXT,
                    link_to_marine_reserve TEXT,
                    protection_status TEXT
                )
            """
            )

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS validation_errors (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    SurveyID TEXT,
                    DropID TEXT,
                    ErrorType TEXT,
                    FileName TEXT,
                    ColumnName TEXT,
                    ErrorMessage TEXT,
                    InvalidValue TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # Create a trigger to automatically update updated_at
            cursor.execute(
                """
                CREATE TRIGGER IF NOT EXISTS update_deployments_timestamp
                AFTER UPDATE ON deployments
                BEGIN
                    UPDATE deployments SET updated_at = CURRENT_TIMESTAMP WHERE drop_id = NEW.drop_id;
                END;
            """
            )

            # TODO: delete this migration block once all envs have run it at least once
            # (source_status added 2026-04-05; safe to remove after next full deploy)
            cursor.execute("PRAGMA table_info(deployments)")
            existing_cols = {row[1] for row in cursor.fetchall()}
            if "source_status" not in existing_cols:
                cursor.execute(
                    "ALTER TABLE deployments ADD COLUMN source_status TEXT NOT NULL DEFAULT 'OK'"
                )
                logging.info("Migrated deployments table: added source_status column.")

            conn.commit()

    def upsert_sites(self, sites_df) -> None:
        """Replace all site metadata from BUV Survey Sites DataFrame.

        Full replace (delete + insert) rather than upsert — sites are config data with no
        pipeline state, so removed sites should not linger in the DB.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM sites")
            skipped = 0
            for _, row in sites_df.iterrows():
                site_id = str(row.get(config.site_id_column, "")).strip()
                if not site_id or site_id.lower() == "nan":
                    skipped += 1
                    continue
                cursor.execute(
                    """
                    INSERT INTO sites (site_id, site_name, link_to_marine_reserve, protection_status)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(site_id) DO UPDATE SET
                        site_name=excluded.site_name,
                        link_to_marine_reserve=excluded.link_to_marine_reserve,
                        protection_status=excluded.protection_status
                    """,
                    (
                        site_id,
                        str(row.get(config.site_name_column, "")),
                        str(row.get(config.link_to_marine_reserve_column, "")),
                        str(row.get(config.protection_status_column, "")),
                    ),
                )
            if skipped:
                logging.warning(f"Skipped {skipped} site rows with missing/empty site_id — check column mapping in config.yaml.")
            conn.commit()
        logging.info(f"Upserted {len(sites_df) - skipped} sites into DB.")

    def get_site(self, site_id: str) -> Optional[Dict[str, str]]:
        """Fetch site metadata by SiteID. Returns a config-keyed dict or None if not found."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM sites WHERE site_id = ?", (site_id,))
            row = cursor.fetchone()
        if row is None:
            return None
        return {
            config.site_id_column: row["site_id"],
            config.site_name_column: row["site_name"],
            config.link_to_marine_reserve_column: row["link_to_marine_reserve"],
            config.protection_status_column: row["protection_status"],
        }

    def add_or_update_deployment(
        self,
        drop_id: str,
        status: str,
        source_status: str = SourceStatus.OK,
        video_path: str = "",
        is_bad_deployment: bool = False,
        error_message: str = "",
        sampling_start: Optional[int] = None,
        sampling_end: Optional[int] = None,
        ml_annotations: int = 0,
        citsci_annotations: int = 0,
        expert_annotations: int = 0,
        biigle_volume_id: Optional[str] = None,
    ):
        """Upserts a deployment record."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO deployments (drop_id, video_path, status, source_status, is_bad_deployment, error_message, sampling_start, sampling_end, ml_annotations, citsci_annotations, expert_annotations, biigle_volume_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(drop_id) DO UPDATE SET
                    video_path=excluded.video_path,
                    status=excluded.status,
                    source_status=excluded.source_status,
                    is_bad_deployment=excluded.is_bad_deployment,
                    error_message=excluded.error_message,
                    sampling_start=excluded.sampling_start,
                    sampling_end=excluded.sampling_end,
                    ml_annotations=excluded.ml_annotations,
                    citsci_annotations=excluded.citsci_annotations,
                    expert_annotations=excluded.expert_annotations,
                    biigle_volume_id=COALESCE(excluded.biigle_volume_id, deployments.biigle_volume_id)
            """,
                (
                    drop_id,
                    video_path,
                    status,
                    source_status,
                    is_bad_deployment,
                    error_message,
                    sampling_start,
                    sampling_end,
                    ml_annotations,
                    citsci_annotations,
                    expert_annotations,
                    biigle_volume_id,
                ),
            )
            conn.commit()

    def update_status(self, drop_id: str, new_status: str):
        """Updates the status of a specific deployment.
        If transitioning away from ERROR, clears any PIPELINE_ERROR rows from validation_errors.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            # Check current status before updating
            cursor.execute(
                "SELECT status FROM deployments WHERE drop_id = ?", (drop_id,)
            )
            row = cursor.fetchone()
            current_status = row["status"] if row else None
            cursor.execute(
                "UPDATE deployments SET status = ? WHERE drop_id = ?",
                (new_status, drop_id),
            )
            # If we're moving away from ERROR, clear the PIPELINE_ERROR entries for this drop
            # TODO: confirm this is wanted — clearing errors on status transition means
            # re-running a drop that previously errored will always start with a clean slate.
            if current_status == "ERROR" and new_status != "ERROR":
                cursor.execute(
                    "DELETE FROM validation_errors WHERE DropID = ? AND ErrorType = 'PIPELINE_ERROR'",
                    (drop_id,),
                )
                logging.info(
                    f"Cleared PIPELINE_ERROR entries for {drop_id} (status reset to {new_status})"
                )
            conn.commit()

    def advance_status(self, drop_id: str, to_status: str) -> None:
        """Transition drop_id to to_status, validating against VALID_TRANSITIONS.

        Raises InvalidTransitionError if the transition is not permitted.
        Use update_status() directly only for admin/test tooling that needs to
        set arbitrary statuses (set_status.py, test_setup.py, conftest.py).
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT status FROM deployments WHERE drop_id = ?", (drop_id,)
            )
            row = cursor.fetchone()
            if row is None:
                raise KeyError(f"No deployment found with drop_id={drop_id!r}")
            current = row["status"]

        # ON_HOLD can transition to any status (it's a pause state, not a terminal)
        if current != PipelineStatus.ON_HOLD:
            allowed = PipelineStatus.VALID_TRANSITIONS.get(current, set())
            if to_status not in allowed:
                raise InvalidTransitionError(
                    f"{drop_id}: invalid transition {current!r} → {to_status!r}. "
                    f"Allowed from {current!r}: {sorted(allowed) if allowed else '(none)'}"
                )

        self.update_status(drop_id, to_status)

    def update_deployment_fields(self, drop_id: str, **fields) -> bool:
        """Update arbitrary columns on a deployment record. Returns False if drop_id not found."""
        allowed = {
            "status", "source_status", "sampling_start", "sampling_end", "video_path",
            "is_bad_deployment", "error_message", "biigle_volume_id",
        }
        invalid = set(fields) - allowed
        if invalid:
            raise ValueError(f"Unknown fields: {invalid}. Allowed: {allowed}")
        if not fields:
            return True
        set_clause = ", ".join(f"{col} = ?" for col in fields)
        values = list(fields.values()) + [drop_id]
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"UPDATE deployments SET {set_clause} WHERE drop_id = ?", values
            )
            conn.commit()
            return cursor.rowcount > 0

    def update_biigle_volume_id(self, drop_id: str, volume_id: str):
        """Sets the biigle_volume_id for a specific deployment."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE deployments SET biigle_volume_id = ? WHERE drop_id = ?",
                (str(volume_id), drop_id),
            )
            conn.commit()

    def get_deployments_by_status(self, status: str) -> List[Dict[str, Any]]:
        """Returns all deployments currently in the given status."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM deployments WHERE status = ?", (status,))
            return [dict(row) for row in cursor.fetchall()]

    def get_deployments_by_statuses(self, statuses: List[str]) -> List[Dict[str, Any]]:
        """Returns all deployments currently in any of the given statuses."""
        if not statuses:
            return []
        placeholders = ", ".join(["?"] * len(statuses))
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"SELECT * FROM deployments WHERE status IN ({placeholders})", statuses
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_deployment(self, drop_id: str) -> Optional[Dict[str, Any]]:
        """Fetch a specific deployment by drop_id."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM deployments WHERE drop_id = ?", (drop_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_deployments_by_ids(self, drop_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """Fetch multiple deployments in a single query. Returns {drop_id: record}."""
        if not drop_ids:
            return {}
        placeholders = ", ".join(["?"] * len(drop_ids))
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"SELECT * FROM deployments WHERE drop_id IN ({placeholders})", drop_ids
            )
            return {row["drop_id"]: dict(row) for row in cursor.fetchall()}

    def get_all_deployments_map(self) -> Dict[str, Dict[str, Any]]:
        """Fetch all deployment records and return them as a dictionary {drop_id: record}."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM deployments")
            return {row["drop_id"]: dict(row) for row in cursor.fetchall()}

    def clear_pipeline_errors(self, drop_id: str):
        """Remove all PIPELINE_ERROR rows from validation_errors for a specific drop.
        Call this when manually fixing a drop and retrying it.
        (This is also called automatically by update_status when moving away from ERROR.)
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM validation_errors WHERE DropID = ? AND ErrorType = 'PIPELINE_ERROR'",
                (drop_id,),
            )
            conn.commit()
            logging.info(f"Cleared PIPELINE_ERROR entries for {drop_id}")

    def clear_validation_errors(self):
        """Clears all validation errors from the database."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM validation_errors")
            conn.commit()

    def add_validation_errors(self, errors: List[Dict[str, Any]]):
        """Bulk inserts validation errors."""
        if not errors:
            return
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany(
                """
                INSERT INTO validation_errors (SurveyID, DropID, ErrorType, FileName, ColumnName, ErrorMessage, InvalidValue)
                VALUES (:SurveyID, :DropID, :ErrorType, :FileName, :ColumnName, :ErrorMessage, :InvalidValue)
            """,
                errors,
            )
            conn.commit()

    def get_all_validation_errors(self) -> List[Dict[str, Any]]:
        """Returns all stored validation errors, including deployment status where possible."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            query = """
                SELECT v.SurveyID, v.DropID, v.ErrorType, v.FileName, v.ColumnName, v.ErrorMessage, v.InvalidValue, d.status
                FROM validation_errors v
                LEFT JOIN deployments d ON v.DropID = d.drop_id
            """
            cursor.execute(query)
            return [dict(row) for row in cursor.fetchall()]

    def sync_annotation_counts(self, drop_ids: Optional[List[str]] = None):
        """
        Aggregates counts from spyfish_annotations.db and updates the deployments table in spyfish_pipeline.db.
        If drop_ids is provided, only updates those specific deployments.
        """
        logging.info(
            f"Syncing annotation counts to main pipeline database{' (incremental)' if drop_ids else ''}..."
        )

        # Import to avoid circular dependencies if any
        from spyfish.database.annotation_manager import AnnotationDatabaseManager

        ann_db = AnnotationDatabaseManager()

        query = """
            SELECT drop_id, annotated_by as source, COUNT(*) as total
            FROM annotations
        """
        params = []
        if drop_ids:
            placeholders = ", ".join(["?"] * len(drop_ids))
            query += f" WHERE drop_id IN ({placeholders})"
            params.extend(drop_ids)

        query += " GROUP BY drop_id, annotated_by"

        with ann_db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            results = cursor.fetchall()

        # Group results by drop_id
        counts_by_drop = {
            d: {"ml": 0, "expert": 0, "citsci": 0} for d in (drop_ids or [])
        }
        for row in results:
            drop_id = row["drop_id"]
            source = row["source"]
            count = row["total"]

            if drop_id not in counts_by_drop:
                counts_by_drop[drop_id] = {"ml": 0, "expert": 0, "citsci": 0}

            if source == "ml":
                counts_by_drop[drop_id]["ml"] = count
            elif source == "expert":
                counts_by_drop[drop_id]["expert"] = count
            elif source == "citsci":
                counts_by_drop[drop_id]["citsci"] = count

        # Update main DB
        with self.get_connection() as conn:
            cursor = conn.cursor()
            for drop_id, counts in counts_by_drop.items():
                cursor.execute(
                    """
                    UPDATE deployments
                    SET ml_annotations = ?,
                        expert_annotations = ?,
                        citsci_annotations = ?
                    WHERE drop_id = ?
                """,
                    (counts["ml"], counts["expert"], counts["citsci"], drop_id),
                )
            conn.commit()

        logging.info(
            f"Updated annotation counts for {len(counts_by_drop)} deployments."
        )

    def export_to_csv(self, output_dir: Optional[str] = None) -> List[str]:
        """Export all DB tables to CSV files. Returns list of written file paths."""
        import pandas as pd

        out = Path(output_dir) if output_dir else Path(self.db_path).parent
        out.mkdir(parents=True, exist_ok=True)

        tables = ["deployments", "validation_errors", "sites"]
        written = []
        with self.get_connection() as conn:
            for table in tables:
                path = out / f"{table}.csv"
                pd.read_sql(f"SELECT * FROM {table}", conn).to_csv(path, index=False)
                written.append(str(path))
                logging.info(f"Exported {table} → {path}")
        return written
