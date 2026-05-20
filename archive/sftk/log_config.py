import logging
import os
from datetime import datetime as dt
from pathlib import Path


def _set_logging_path() -> str:
    """
    Retrieves the path to the log file.

    This function creates a directory in the user's home directory called ".sftk" and a subdirectory called "logs".
    The logfile is named with the current date and time in the format "YYYY-MM-DD_HH-MM-SS.log".

    This function is intended to be used internally by the module.

    Returns:
        str: The path to the log file.
    """
    log_dir = Path.home() / ".sftk" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_filename = dt.strftime(dt.now(), "%Y-%m-%d_%H-%M-%S") + ".log"
    log_file = log_dir / log_filename
    return str(log_file)


def clear_logs_directory() -> None:
    """
    Clear the logs directory by deleting all log files (except the current log file).
    """
    log_dir = Path(LOG_PATH).parent
    for log_file in log_dir.iterdir():
        if log_file != Path(LOG_PATH) and log_file.is_file():
            log_file.unlink()


# Configure the logging module in global scope to ensure that all modules use the same configuration.
LOG_PATH = _set_logging_path()

# Simple configuration: use file logging if LOG_OUTPUT=file, otherwise console (default)
format_string = "%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
if os.getenv("LOG_OUTPUT", "console").lower() == "file":
    logging.basicConfig(
        filename=LOG_PATH,
        level="INFO",
        format=format_string,
        force=True,
    )
else:
    # Default to console output
    logging.basicConfig(
        level="INFO",
        format=format_string,
        force=True,
    )
