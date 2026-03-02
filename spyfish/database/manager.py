import sqlite3
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

from spyfish.config import PipelineStatus, config

class DatabaseManager:
    """
    Core SQLite database manager for the Spyfish pipeline.
    Uses pure sqlite3 with dict-like row factories for simplicity.
    """
    def __init__(self, db_path: str = None):
        self.db_path = str(config.db_path) if db_path is None else str(Path(db_path).absolute())
        self.mode = config.storage.get("mode", "local")
        self.init_db()

    def _download_if_aws(self):
        if self.mode == "aws":
            from spyfish.storage.db_sync import download_db
            download_db()

    def _upload_if_aws(self):
        if self.mode == "aws":
            from spyfish.storage.db_sync import upload_db
            upload_db()

    def get_connection(self):
        """Returns a configured SQLite connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Access columns by name
        return conn

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

            # Migration: Ensure biigle_volume_id exists (Legacy databases might be missing it)
            cursor.execute("PRAGMA table_info(deployments)")
            columns = [info[1] for info in cursor.fetchall()]
            if 'biigle_volume_id' not in columns:
                logging.info("Migrating database: Adding biigle_volume_id column to deployments table")
                cursor.execute('ALTER TABLE deployments ADD COLUMN biigle_volume_id TEXT')

            # Create a trigger to automatically update updated_at
            cursor.execute('''
                CREATE TRIGGER IF NOT EXISTS update_deployments_timestamp
                AFTER UPDATE ON deployments
                BEGIN
                    UPDATE deployments SET updated_at = CURRENT_TIMESTAMP WHERE drop_id = NEW.drop_id;
                END;
            ''')
            conn.commit()

    def add_or_update_deployment(self, drop_id: str, status: str, video_path: str = "", is_bad_deployment: bool = False, error_message: str = "", sampling_start: int = None, sampling_end: int = None, ml_annotations: int = 0, citsci_annotations: int = 0, expert_annotations: int = 0, biigle_volume_id: str = None, auto_sync: bool = True):
        """Upserts a deployment record."""
        if auto_sync: self._download_if_aws()
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
        if auto_sync: self._upload_if_aws()

    def update_status(self, drop_id: str, new_status: str, auto_sync: bool = True):
        """Updates the status of a specific deployment."""
        if auto_sync: self._download_if_aws()
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('UPDATE deployments SET status = ? WHERE drop_id = ?', (new_status, drop_id))
            conn.commit()
        if auto_sync: self._upload_if_aws()

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

    def clear_validation_errors(self, auto_sync: bool = True):
        """Clears all validation errors from the database."""
        if auto_sync: self._download_if_aws()
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM validation_errors')
            conn.commit()
        if auto_sync: self._upload_if_aws()

    def add_validation_errors(self, errors: List[Dict[str, Any]], auto_sync: bool = True):
        """Bulk inserts validation errors."""
        if not errors: return
        if auto_sync: self._download_if_aws()
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany('''
                INSERT INTO validation_errors (SurveyID, DropID, ErrorType, FileName, ColumnName, ErrorMessage, InvalidValue)
                VALUES (:SurveyID, :DropID, :ErrorType, :FileName, :ColumnName, :ErrorMessage, :InvalidValue)
            ''', errors)
            conn.commit()
        if auto_sync: self._upload_if_aws()

    def get_all_validation_errors(self) -> List[Dict[str, Any]]:
        """Returns all stored validation errors."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT SurveyID, DropID, ErrorType, FileName, ColumnName, ErrorMessage, InvalidValue FROM validation_errors')
            return [dict(row) for row in cursor.fetchall()]
