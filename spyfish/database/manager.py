import sqlite3
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from contextlib import closing

from spyfish.config import PipelineStatus, config

class DatabaseManager:
    """
    Core SQLite database manager for the Spyfish pipeline.
    Uses pure sqlite3 with dict-like row factories for simplicity.
    """
    def __init__(self, db_path: str = None):
        self.db_path = str(config.db_path) if db_path is None else str(Path(db_path).absolute())
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
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS deployments (
                    drop_id TEXT PRIMARY KEY,
                    video_path TEXT,
                    status TEXT NOT NULL,
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
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ml_jobs (
                    job_id TEXT PRIMARY KEY,
                    slurm_id TEXT,
                    status TEXT NOT NULL,
                    job_type TEXT NOT NULL,
                    drop_ids TEXT NOT NULL,
                    stdout_path TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    finished_at TIMESTAMP
                )
            ''')

            cursor.execute('''
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
            ''')

            # Create a trigger to automatically update updated_at
            cursor.execute('''
                CREATE TRIGGER IF NOT EXISTS update_deployments_timestamp
                AFTER UPDATE ON deployments
                BEGIN
                    UPDATE deployments SET updated_at = CURRENT_TIMESTAMP WHERE drop_id = NEW.drop_id;
                END;
            ''')
            conn.commit()

    def add_or_update_deployment(self, drop_id: str, status: str, video_path: str = "", is_bad_deployment: bool = False, error_message: str = "", sampling_start: int = None, sampling_end: int = None, ml_annotations: int = 0, citsci_annotations: int = 0, expert_annotations: int = 0, biigle_volume_id: str = None):
        """Upserts a deployment record."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO deployments (drop_id, video_path, status, is_bad_deployment, error_message, sampling_start, sampling_end, ml_annotations, citsci_annotations, expert_annotations, biigle_volume_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(drop_id) DO UPDATE SET
                    video_path=excluded.video_path,
                    status=excluded.status,
                    is_bad_deployment=excluded.is_bad_deployment,
                    error_message=excluded.error_message,
                    sampling_start=excluded.sampling_start,
                    sampling_end=excluded.sampling_end,
                    ml_annotations=excluded.ml_annotations,
                    citsci_annotations=excluded.citsci_annotations,
                    expert_annotations=excluded.expert_annotations,
                    biigle_volume_id=excluded.biigle_volume_id
            ''', (drop_id, video_path, status, is_bad_deployment, error_message, sampling_start, sampling_end, ml_annotations, citsci_annotations, expert_annotations, biigle_volume_id))
            conn.commit()

    def update_status(self, drop_id: str, new_status: str):
        """Updates the status of a specific deployment."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('UPDATE deployments SET status = ? WHERE drop_id = ?', (new_status, drop_id))
            conn.commit()

    def update_biigle_volume_id(self, drop_id: str, volume_id: str):
        """Sets the biigle_volume_id for a specific deployment."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('UPDATE deployments SET biigle_volume_id = ? WHERE drop_id = ?', (str(volume_id), drop_id))
            conn.commit()

    def get_deployments_by_status(self, status: str) -> List[Dict[str, Any]]:
        """Returns all deployments currently in the given status."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM deployments WHERE status = ?', (status,))
            return [dict(row) for row in cursor.fetchall()]

    def get_deployment(self, drop_id: str) -> Optional[Dict[str, Any]]:
        """Fetch a specific deployment by drop_id."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM deployments WHERE drop_id = ?', (drop_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def clear_validation_errors(self):
        """Clears all validation errors from the database."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM validation_errors')
            conn.commit()

    def add_validation_errors(self, errors: List[Dict[str, Any]]):
        """Bulk inserts validation errors."""
        if not errors: return
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany('''
                INSERT INTO validation_errors (SurveyID, DropID, ErrorType, FileName, ColumnName, ErrorMessage, InvalidValue)
                VALUES (:SurveyID, :DropID, :ErrorType, :FileName, :ColumnName, :ErrorMessage, :InvalidValue)
            ''', errors)
            conn.commit()

    def get_all_validation_errors(self) -> List[Dict[str, Any]]:
        """Returns all stored validation errors."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT SurveyID, DropID, ErrorType, FileName, ColumnName, ErrorMessage, InvalidValue FROM validation_errors')
            return [dict(row) for row in cursor.fetchall()]

    def sync_annotation_counts(self, drop_ids: Optional[List[str]] = None):
        """
        Aggregates counts from spyfish_annotations.db and updates the deployments table in spyfish_pipeline.db.
        If drop_ids is provided, only updates those specific deployments.
        """
        logging.info(f"Syncing annotation counts to main pipeline database{' (incremental)' if drop_ids else ''}...")

        # Import to avoid circular dependencies if any
        from spyfish.database.annotation_manager import AnnotationDatabaseManager
        ann_db = AnnotationDatabaseManager()

        query = '''
            SELECT drop_id, annotated_by as source, SUM(max_interval) as total
            FROM annotations
        '''
        params = []
        if drop_ids:
            placeholders = ', '.join(['?'] * len(drop_ids))
            query += f" WHERE drop_id IN ({placeholders})"
            params.extend(drop_ids)

        query += " GROUP BY drop_id, annotated_by"

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
        with self.get_connection() as conn:
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
