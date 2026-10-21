"""Bounded process-local result reconciliation, not a durable job queue."""
from dataclasses import asdict
from threading import RLock
import hashlib
import json
import logging
import time

LOGGER = logging.getLogger(__name__)


class EnhancementOperations:
    def __init__(self, *, clock=time.time, limit=1024, retention=3600):
        self.clock, self.limit, self.retention = clock, limit, retention
        self._entries = {}
        self._lock = RLock()

    @staticmethod
    def _key(request):
        return hashlib.sha256(json.dumps([
            request["doc_name"], request.get("toc_hint", "auto"),
            request.get("enhancement_intent", "restore_source"),
            request.get("confirmation_token"),
        ]).encode()).hexdigest()

    def _prune(self):
        now = self.clock()
        self._entries = {key: value for key, value in self._entries.items()
                         if value["status"] == "running" or now - value["updated"] < self.retention}

    def register(self, request):
        from app.enhancement_preflight import EnhancementConflict
        with self._lock:
            self._prune()
            if len(self._entries) >= self.limit:
                raise EnhancementConflict("enhancement_operation_capacity", 503)
            self._entries[self._key(request)] = dict(
                status="not_started", result=None, reason=None, retryable=False,
                updated=self.clock())

    def lookup(self, request):
        with self._lock:
            self._prune()
            entry = self._entries.get(self._key(request))
            if entry is None:
                return dict(status="unknown", result=None,
                            reason="enhancement_outcome_unknown", retryable=False,
                            durability="process_local")
            return dict(status=entry["status"], result=asdict(entry["result"]) if entry["result"] else None,
                        reason=entry["reason"], retryable=entry["retryable"], durability="process_local")

    def execute(self, request, callback):
        from app.enhancement_preflight import EnhancementConflict
        if request.get("discard_interactions") is not True or not request.get("confirmation_token"):
            raise EnhancementConflict("confirmation_required")
        key = self._key(request)
        with self._lock:
            self._prune()
            entry = self._entries.get(key)
            if entry is None:
                raise EnhancementConflict("enhancement_outcome_unknown")
            if entry["status"] == "running":
                raise EnhancementConflict("enhancement_in_progress")
            if entry["status"] == "unknown":
                raise EnhancementConflict("enhancement_outcome_unknown")
            if entry["status"] == "completed" or (entry["status"] == "failed" and not entry["retryable"]):
                if entry["result"] is not None:
                    return entry["result"]
                raise EnhancementConflict(entry["reason"], entry.get("status_code", 409), entry.get("capacity"))
            entry.update(status="running", updated=self.clock())
        LOGGER.info("enhancement_operation status=running operation=%s", key)
        try:
            result = callback()
        except EnhancementConflict as error:
            with self._lock:
                entry.update(status="failed", reason=error.reason, status_code=error.status_code,
                             capacity=error.capacity,
                             retryable=False, updated=self.clock())
            LOGGER.info("enhancement_operation status=failed operation=%s reason=%s", key, error.reason)
            raise
        except BaseException:
            # A lost commit acknowledgement is not evidence of a rollback.
            with self._lock:
                entry.update(status="unknown", reason="enhancement_outcome_unknown",
                             retryable=False, updated=self.clock())
            LOGGER.warning("enhancement_operation status=unknown operation=%s", key)
            raise
        with self._lock:
            entry.update(status="completed" if result.success else "failed", result=result,
                         reason=result.error, retryable=bool(getattr(result, "retryable", False)),
                         updated=self.clock())
        LOGGER.info("enhancement_operation status=%s operation=%s change_kind=%s version=%s retryable=%s",
                    entry["status"], key, getattr(result, "change_kind", None),
                    getattr(result, "structure_version", None), entry["retryable"])
        return result
