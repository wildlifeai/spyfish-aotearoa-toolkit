"""read_only=True must never create a database file.

The dashboard opens both databases this way. An empty file conjured before
the first S3 sync gets a fresh mtime, and db_sync then treats it as current
and never downloads the real one.
"""

import sqlite3

import pytest

from spyfish.database.annotation_manager import AnnotationDatabaseManager
from spyfish.database.manager import DatabaseManager


@pytest.mark.parametrize("manager", [DatabaseManager, AnnotationDatabaseManager])
def test_read_only_does_not_create_missing_file(tmp_path, manager):
    db_path = tmp_path / "absent.db"
    with pytest.raises(sqlite3.OperationalError):
        db = manager(db_path=str(db_path), read_only=True)
        # DatabaseManager reads the schema in __init__; the annotations
        # manager only connects on first query.
        with db.get_connection() as conn:
            conn.execute("SELECT 1")
    assert not db_path.exists()


def test_read_only_reads_existing_annotations(tmp_path):
    db_path = tmp_path / "annotations.db"
    AnnotationDatabaseManager(db_path=str(db_path))  # creates the schema
    df = AnnotationDatabaseManager(
        db_path=str(db_path), read_only=True
    ).get_maxn_summary()
    assert df.empty


@pytest.mark.parametrize("manager", [DatabaseManager, AnnotationDatabaseManager])
def test_read_only_refuses_writable_connection(tmp_path, manager):
    db_path = tmp_path / "x.db"
    manager(db_path=str(db_path))
    db = manager(db_path=str(db_path), read_only=True)
    with pytest.raises(RuntimeError):
        with db.get_writable_connection():
            pass
