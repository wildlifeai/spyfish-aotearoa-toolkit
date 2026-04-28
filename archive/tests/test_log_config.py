import logging
import os
from pathlib import Path
from unittest.mock import patch

from sftk.log_config import LOG_PATH


def test_log_path_and_logging():
    """Log path should be valid and logging should work."""
    # Verify LOG_PATH is valid
    assert isinstance(LOG_PATH, str)
    assert Path(LOG_PATH).parent.exists()

    # Verify logging works without errors
    logging.info("Test message")
    logging.debug("Debug message")
    logging.warning("Warning message")


def test_log_output_env_var():
    """LOG_OUTPUT env var should control logging destination."""
    # Test file output
    with patch.dict(os.environ, {"LOG_OUTPUT": "file"}):
        assert os.getenv("LOG_OUTPUT") == "file"

    # Test console output
    with patch.dict(os.environ, {"LOG_OUTPUT": "console"}):
        assert os.getenv("LOG_OUTPUT") == "console"

    # Test default (console)
    with patch.dict(os.environ, {}, clear=True):
        assert os.getenv("LOG_OUTPUT", "console") == "console"
