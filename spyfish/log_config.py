import logging
from datetime import datetime as dt
from pathlib import Path

from spyfish.config.wrapper import config


def _set_logging_path() -> str:
    """
    Retrieves the path to the log file.

    This function creates a directory in the user's home directory called ".spyfish" and a subdirectory called "logs".
    The logfile is named with the current date and time in the format "YYYY-MM-DD_HH-MM-SS.log".

    Returns:
        str: The path to the log file.
    """

    log_dir = config.logs_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    log_filename = dt.strftime(dt.now(), "%Y-%m-%d_%H-%M-%S") + ".log"
    log_file = log_dir / log_filename
    return str(log_file)


def clear_logs_directory() -> None:
    """
    Clear the logs directory by deleting all log files (except the current log file).
    """
    if not LOG_PATH:
        return
    log_dir = Path(LOG_PATH).parent
    for log_file in log_dir.iterdir():
        if log_file != Path(LOG_PATH) and log_file.is_file():
            log_file.unlink()


LOG_PATH = None

# Simple configuration: use file logging if config.log_output is 'file', otherwise console
format_string = "%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
if config.log_output == "file":
    LOG_PATH = _set_logging_path()
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


def log_header(title: str, character: str = "═"):
    """Logs a visually distinct header for a pipeline step."""
    width = 60
    logging.info(character * width)
    logging.info(f" {title} ".center(width, character))
    logging.info(character * width)
