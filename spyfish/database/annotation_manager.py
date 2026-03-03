import sqlite3
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from contextlib import closing

from spyfish.config import config

class AnnotationDatabaseManager:
    """
    Manages the spyfish_annotations.db which stores detailed individual annotation records
    from ML, Expert (Biigle), and Citizen Science sources.
    """
    def __init__(self, db_path: str = None):
        if db_path:
            self.db_path = str(Path(db_path).absolute())
        else:
            # Use the same directory as the main pipeline DB (process_files/)
            self.db_path = str(config.project_root / "process_files" / "spyfish_annotations.db")

        self.init_db()

    def get_connection(self):
        """Returns a configured SQLite connection wrapped in contextlib.closing."""
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        return closing(conn)

    def init_db(self):
        """Initializes the annotations schema."""
        logging.info(f"Initializing Annotation DB at {self.db_path}")
        with self.get_connection() as conn:
            cursor = conn.cursor()
            # Drop and recreate if schema needs adjustment during implementation
            # cursor.execute('DROP TABLE IF EXISTS annotations')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS annotations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    drop_id TEXT NOT NULL,
                    scientific_name TEXT,
                    timestamp TEXT, -- HH:MM:SS
                    count INTEGER DEFAULT 1,
                    source TEXT NOT NULL, -- 'ml', 'expert', 'citsci'
                    confidence REAL,
                    external_id TEXT, -- e.g. Biigle annotation ID
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Index for fast aggregation by drop_id and source
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_drop_source ON annotations(drop_id, source)')
            conn.commit()

    def add_annotations(self, annotations: List[Dict[str, Any]]):
        """Bulk insert annotation records."""
        if not annotations:
            return
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany('''
                INSERT INTO annotations (drop_id, scientific_name, timestamp, count, source, confidence, external_id)
                VALUES (:drop_id, :scientific_name, :timestamp, :count, :source, :confidence, :external_id)
            ''', annotations)
            conn.commit()

    def clear_annotations(self, drop_id: str, source: str):
        """Clears existing annotations for a given drop and source."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM annotations WHERE drop_id = ? AND source = ?", (drop_id, source))
            conn.commit()

    def get_annotations_for_drop(self, drop_id: str, source: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve annotations for a drop, optionally filtered by source."""
        query = "SELECT * FROM annotations WHERE drop_id = ?"
        params = [drop_id]
        if source:
            query += " AND source = ?"
            params.append(source)

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def get_counts_per_source(self, drop_id: str) -> Dict[str, int]:
        """Get the total annotation count (sum of 'count' column) per source for a drop."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT source, SUM(count) as total
                FROM annotations
                WHERE drop_id = ?
                GROUP BY source
            ''', (drop_id,))
            return {row['source']: row['total'] for row in cursor.fetchall()}
