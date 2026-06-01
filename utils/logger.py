"""Centralised logging — root logger with colourised console + rotating file handlers."""

import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler

import colorlog

from config.settings import settings


FILE_LOG_FORMAT = (
    "%(asctime)s | "
    "%(levelname)-8s | "
    "%(name)s:%(funcName)s:%(lineno)d | "
    "%(message)s"
)

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
    """Configure the root logger with console and file handlers; idempotent."""
    root = logging.getLogger()

    if root.handlers:
        return

    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)
    root.setLevel(log_level)

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
    """Return a named logger; ensures root is configured first."""
    _setup_root_logger()
    return logging.getLogger(name)


logger = get_logger("scalable_rag_rlm")
