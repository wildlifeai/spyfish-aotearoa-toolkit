import os
import sqlite3
from unittest.mock import MagicMock, patch

import pytest

from spyfish.database.manager import DatabaseManager


@pytest.fixture
def mock_db():
    """
    Provides a DatabaseManager connected to an in-memory SQLite database.
    This ensures tests do not touch the real filesystem database.
    """
    # Initialize DatabaseManager with :memory:
    db = DatabaseManager(db_path=":memory:")

    # Actually, DatabaseManager.get_connection() opens a new connection each time.
    # For :memory: databases, each new connection is a NEW empty database.
    # To share an in-memory database across methods, we need a shared URI
    # or to mock the connection generation itself.

    # The better way is to use a temporary file for the database per test.

    yield db


@pytest.fixture
def temp_db(tmp_path):
    """Provides a DatabaseManager connected to a temporary file database."""
    db_file = tmp_path / "test_pipeline.db"
    db = DatabaseManager(db_path=str(db_file))
    yield db


@pytest.fixture
def mock_s3_handler():
    """Mocks the S3Handler to prevent real AWS calls during testing."""
    with patch("spyfish.storage.s3_handler.S3Handler") as MockS3:
        mock_instance = MockS3.return_value
        mock_instance.download_object_from_s3.return_value = True
        yield mock_instance
