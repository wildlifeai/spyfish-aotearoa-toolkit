import logging
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from spyfish.config.wrapper import config


class AnnotationDatabaseManager:
    """
    Manages the spyfish_annotations.db which stores detailed individual annotation records
    from ML, Expert (Biigle), and Citizen Science sources.
    """

    def __init__(self, db_path: Optional[str] = None):
        if db_path:
            self.db_path = str(Path(db_path).absolute())
        else:
            self.db_path = str(config.annotations_db_path)

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
            # Schema migration is complete — do not drop table again
            # cursor.execute('DROP TABLE IF EXISTS annotations')
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS annotations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    drop_id TEXT NOT NULL,
                    scientific_name TEXT,
                    time_of_max TEXT, -- HH:MM:SS
                    max_interval INTEGER DEFAULT 1,
                    annotated_by TEXT NOT NULL, -- e.g., 'expert'
                    interval_annotation TEXT,
                    confidence_agreement REAL,
                    external_id TEXT, -- e.g. Biigle annotation ID
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # Index for fast aggregation by drop_id and source (annotated_by)
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_drop_source ON annotations(drop_id, annotated_by)"
            )
            conn.commit()

    def add_annotations(self, annotations: List[Dict[str, Any]]):
        """Bulk insert annotation records."""
        if not annotations:
            return
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany(
                """
                INSERT INTO annotations (drop_id, scientific_name, time_of_max, max_interval, annotated_by, interval_annotation, confidence_agreement, external_id)
                VALUES (:drop_id, :scientific_name, :time_of_max, :max_interval, :annotated_by, :interval_annotation, :confidence_agreement, :external_id)
            """,
                annotations,
            )
            conn.commit()

    def clear_annotations(self, drop_id: str, annotated_by: str):
        """Clears ALL existing annotations for a given drop and source (annotated_by)."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM annotations WHERE drop_id = ? AND annotated_by = ?",
                (drop_id, annotated_by),
            )
            conn.commit()

    def clear_synced_annotations(self, drop_id: str, annotated_by: str):
        """Clears only Biigle-sourced annotations (external_id IS NOT NULL) for a drop.

        Used by the Biigle sync path to replace previously downloaded annotations
        without touching any manually-entered rows (external_id = NULL).
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM annotations WHERE drop_id = ? AND annotated_by = ? AND external_id IS NOT NULL",
                (drop_id, annotated_by),
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

    def get_counts_per_source(self, drop_id: str) -> Dict[str, int]:
        """Get the total annotation count (sum of 'max_interval') per source ('annotated_by') for a drop."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT annotated_by as source, SUM(max_interval) as total
                FROM annotations
                WHERE drop_id = ?
                GROUP BY annotated_by
            """,
                (drop_id,),
            )
            return {row["source"]: row["total"] for row in cursor.fetchall()}

    def get_all_annotations_export_df(self) -> Optional[pd.DataFrame]:
        """
        Returns a DataFrame of all annotations formatted exactly for export/app display,
        with columns: DropID, ScientificName, TimeOfMax, MaxInterval, AnnotatedBy,
        IntervalAnnotation, ConfidenceAgreement.
        """
        import pandas as pd

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

        # We drop any internal columns (like id, external_id, created_at) by strictly selecting
        export_cols = [
            config.drop_id_column,
            config.csv_scientific_name_column,
            config.csv_maxn_time_column,
            config.csv_max_interval_column,
            config.csv_annotated_by_column,
            config.csv_interval_annotation_column,
            config.csv_confidence_agreement_column,
        ]

        return df[export_cols]

    def get_maxn_summary(
        self, drop_id: Optional[str] = None, annotated_by: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Returns the canonical MaxN per drop × species × source — i.e. the peak
        max_interval across all time intervals for each combination.

        This is the scientific result; contrast with deployments.ml_annotations
        which is a COUNT of MaxN records used only for pipeline monitoring.

        Args:
            drop_id: Filter to a single deployment. If None, returns all.
            annotated_by: Filter by source ('ml', 'expert', 'citsci'). If None, returns all.
        """
        query = """
            SELECT drop_id, scientific_name, annotated_by,
                   MAX(max_interval) as maxn,
                   confidence_agreement
            FROM annotations
        """
        conditions, params = [], []
        if drop_id:
            conditions.append("drop_id = ?")
            params.append(drop_id)
        if annotated_by:
            conditions.append("annotated_by = ?")
            params.append(annotated_by)
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " GROUP BY drop_id, scientific_name, annotated_by ORDER BY drop_id, maxn DESC"

        with self.get_connection() as conn:
            return pd.read_sql_query(query, conn, params=params)

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
