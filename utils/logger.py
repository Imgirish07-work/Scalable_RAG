"""
Centralised logging configuration for the application.

Design:
    Configures a single root logger with two handlers — a colourised console
    handler (via colorlog) and a rotating file handler. Child loggers in every
    module propagate to the root logger automatically, so no per-module handler
    setup is needed.

Chain of Responsibility:
    Called at import time by every module via `get_logger(__name__)`.
    Reads log_level and log_file from config.settings → writes to stdout and
    the rotating log file. No downstream module calls.

Dependencies:
    colorlog, config.settings
"""

import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler

import colorlog

from config.settings import settings


# Plain-text format for the rotating file — no ANSI codes in log files.
FILE_LOG_FORMAT = (
    "%(asctime)s | "
    "%(levelname)-8s | "
    "%(name)s:%(funcName)s:%(lineno)d | "
    "%(message)s"
)

# Colourised format for the console — ANSI codes added by colorlog.
CONSOLE_LOG_FORMAT = (
    "%(log_color)s%(asctime)s | "
    "%(levelname)-8s%(reset)s | "
    "%(cyan)s%(name)s:%(funcName)s:%(lineno)d%(reset)s | "
    "%(log_color)s%(message)s%(reset)s"
)

LOG_COLORS = {
    "DEBUG"   : "white",
    "INFO"    : "green",
    "WARNING" : "yellow",
    "ERROR"   : "red",
    "CRITICAL": "bold_red",
}

DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def _setup_root_logger() -> None:
    """Configure the root logger with console and file handlers.

    Idempotent — skips setup if handlers are already attached, so calling
    get_logger() from multiple modules does not duplicate output.
    """
    root = logging.getLogger()

    if root.handlers:
        return

    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)
    root.setLevel(log_level)

    # Console handler — colourised output to stdout.
    console_formatter = colorlog.ColoredFormatter(
        fmt=CONSOLE_LOG_FORMAT,
        datefmt=DATE_FORMAT,
        log_colors=LOG_COLORS,
        reset=True,
        style="%",
    )
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(console_formatter)
    root.addHandler(console_handler)

    # Rotating file handler — 10 MB per file, 5 backups, plain text.
    log_file_path = Path(settings.log_file)
    log_file_path.parent.mkdir(parents=True, exist_ok=True)

    file_formatter = logging.Formatter(
        fmt=FILE_LOG_FORMAT,
        datefmt=DATE_FORMAT
    )
    file_handler = RotatingFileHandler(
        filename=str(log_file_path),
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(file_formatter)
    root.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger that propagates to the root logger's handlers.

    Ensures the root logger is configured before returning the child logger.
    All modules should call this at import time with `get_logger(__name__)`.

    Args:
        name: Logger name, typically the module's __name__.

    Returns:
        A configured logging.Logger instance.
    """
    _setup_root_logger()
    return logging.getLogger(name)


logger = get_logger("scalable_rag_rlm")
