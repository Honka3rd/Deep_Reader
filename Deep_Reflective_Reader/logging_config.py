"""Application logging configuration with file persistence for diagnostics."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from contextvars import ContextVar
import json
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import time
from uuid import uuid4


request_id_context: ContextVar[str] = ContextVar("request_id", default="-")
stage_id_context: ContextVar[str] = ContextVar("stage_id", default="-")


class ExternalResponseFilter(logging.Filter):
    """Count completed SDK HTTP exchanges, not SDK calls or transport attempts."""

    def filter(self, record: logging.LogRecord) -> bool:
        if record.msg != 'HTTP Request: %s %s "%s %d %s"':
            return True
        args = record.args
        if not isinstance(args, tuple) or len(args) != 5:
            return True
        method, url, _, status, _ = args
        path = getattr(url, "path", "")
        endpoint = next((name for name in ("embeddings", "responses", "chat/completions")
                         if path.endswith("/" + name)), None)
        if method == "POST" and endpoint and isinstance(status, int):
            logging.getLogger("external_exchange").info(
                "external_response endpoint=%s status=%d stage_id=%s",
                endpoint, status, stage_id_context.get(),
            )
        return True


class RequestContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_context.get()
        return True


@contextmanager
def log_stage(logger: logging.Logger, stage: str, **fields):
    """Time a stage without changing its return/exception behavior or logging content."""
    stage_id = uuid4().hex[:12]
    started_at = time.perf_counter()
    metadata = dict(fields)
    metadata["parent_stage_id"] = stage_id_context.get()
    token = stage_id_context.set(stage_id)
    logger.info("stage_started stage=%s stage_id=%s metadata=%s", stage, stage_id,
                json.dumps(metadata, ensure_ascii=False))
    try:
        yield metadata
    except BaseException as error:
        logger.warning("stage_failed stage=%s stage_id=%s duration_ms=%.1f error_type=%s metadata=%s",
                       stage, stage_id, (time.perf_counter() - started_at) * 1000,
                       type(error).__name__, json.dumps(metadata, ensure_ascii=False))
        raise
    else:
        logger.info("stage_finished stage=%s stage_id=%s duration_ms=%.1f metadata=%s",
                    stage, stage_id, (time.perf_counter() - started_at) * 1000,
                    json.dumps(metadata, ensure_ascii=False))
    finally:
        stage_id_context.reset(token)


_DEFAULT_LOG_FORMAT = (
    "%(asctime)s %(levelname)s %(name)s "
    "pid=%(process)d thread=%(threadName)s request_id=%(request_id)s %(message)s"
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
    stream_handler.addFilter(RequestContextFilter())
    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=int(os.environ.get("DEEP_READER_LOG_MAX_BYTES", "10485760")),
        backupCount=int(os.environ.get("DEEP_READER_LOG_BACKUP_COUNT", "5")),
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    file_handler.addFilter(RequestContextFilter())
    root_logger.addHandler(stream_handler)
    root_logger.addHandler(file_handler)

    http_logger = logging.getLogger("httpx")
    if not any(isinstance(item, ExternalResponseFilter) for item in http_logger.filters):
        http_logger.addFilter(ExternalResponseFilter())

    logging.getLogger(__name__).info("logging_configured log_path=%s", log_path)
    return log_path
