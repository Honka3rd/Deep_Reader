"""Private-safe, best-effort timings for standalone cleanup preflight only."""

from contextlib import contextmanager

from logging_config import log_stage


_REASONS = frozenset({
    "cleanup_reconciliation_unavailable", "cleanup_identity_changed",
    "cleanup_operation_capacity", "cleanup_in_progress", "cleanup_document_missing",
    "cleanup_source_not_ready", "cleanup_preflight_unavailable",
    "capacity_exceeded", "capacity_unavailable", "native_source",
    "extraction_mode_unsupported", "ocr_evidence_missing", "ocr_evidence_mismatch",
})


def safe_reason(reason):
    if reason is None:
        return None
    return reason if isinstance(reason, str) and reason in _REASONS else "cleanup_preflight_unavailable"


class _BestEffortLogger:
    def __init__(self, logger):
        self.logger = logger

    def info(self, *args):
        try:
            self.logger.info(*args)
        except Exception:
            pass

    def warning(self, *args):
        try:
            self.logger.warning(*args)
        except Exception:
            pass


@contextmanager
def preflight_stage(logger, stage):
    with log_stage(_BestEffortLogger(logger), f"cleanup_preflight_{stage}",
                   origin="standalone", status="running", reason=None,
                   execution_available=False) as fields:
        try:
            yield fields
        except BaseException:
            fields.update(status="failed", reason="stage_exception", execution_available=False)
            raise
        else:
            if fields["status"] == "running":
                fields["status"] = "completed"
