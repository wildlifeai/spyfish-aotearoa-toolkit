"""Read-only SQLite connections.

The managers in this package open connections that can create and write:
``DatabaseManager.__init__`` calls ``init_db()``, so merely constructing one
creates the database file. That is correct for the pipeline, which owns these
files, and wrong for every read-only consumer — the dashboard, a notebook, a
reporting script — which must be able to fail when the database is absent
rather than quietly conjure an empty one.

So this is a plain function rather than a third method on the managers: a
reader must be able to open a database without instantiating something that
creates it.
"""

import sqlite3
from pathlib import Path


def connect_readonly(db_path: str | Path) -> sqlite3.Connection:
    """Open an existing SQLite database for reading. Never creates the file.

    ``sqlite3.connect(path)`` creates an empty database when the file is
    missing. For anything syncing from S3 that is worse than untidy: the new
    file's mtime makes ``db_sync`` treat the local copy as current and skip the
    download from then on, so one read before the first sync can block the real
    database from ever arriving.

    Raises ``sqlite3.OperationalError`` when the file does not exist. Callers
    should surface that as "sync from S3 first" rather than letting it escape
    as an unhandled error.
    """
    return sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
