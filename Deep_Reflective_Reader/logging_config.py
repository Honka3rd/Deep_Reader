"""Application logging configuration with file persistence for diagnostics."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path


_DEFAULT_LOG_FORMAT = (
    "%(asctime)s %(levelname)s %(name)s "
    "pid=%(process)d thread=%(threadName)s %(message)s"
)


def configure_logging() -> Path:
    """Configure stdout and rotating file handlers, returning the log path."""
    log_dir = Path(os.environ.get("DEEP_READER_LOG_DIR", "data/logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "deep-reader.log"

    level_name = os.environ.get("DEEP_READER_LOG_LEVEL", "INFO").strip().upper()
    level = getattr(logging, level_name, logging.INFO)
    formatter = logging.Formatter(
        os.environ.get("DEEP_READER_LOG_FORMAT", _DEFAULT_LOG_FORMAT)
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
        handler.close()

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=int(os.environ.get("DEEP_READER_LOG_MAX_BYTES", "10485760")),
        backupCount=int(os.environ.get("DEEP_READER_LOG_BACKUP_COUNT", "5")),
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)
    root_logger.addHandler(file_handler)

    logging.getLogger(__name__).info("logging_configured log_path=%s", log_path)
    return log_path
