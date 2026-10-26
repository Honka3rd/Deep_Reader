"""Process-local preparation activity, independent of browser request lifetime.

This is not a durable job queue. A service restart loses these observations;
persisted structure remains the authority for successful preparation.
"""
from contextlib import contextmanager
from threading import RLock


class PreparationBusyError(Exception):
    pass


class PreparationOperations:
    def __init__(self):
        self._states: dict[str, str] = {}
        self._lock = RLock()

    def snapshot(self) -> dict[str, str]:
        with self._lock:
            return dict(self._states)

    @contextmanager
    def track(self, doc_name: str):
        with self._lock:
            if self._states.get(doc_name) == "building":
                raise PreparationBusyError(doc_name)
            self._states[doc_name] = "building"
        outcome = {"success": False}
        try:
            yield outcome
        finally:
            with self._lock:
                if outcome["success"]:
                    # Never use an operation receipt to invent a ready hierarchy.
                    self._states.pop(doc_name, None)
                else:
                    self._states[doc_name] = "failed"

    def forget(self, *doc_names: str):
        with self._lock:
            for name in doc_names:
                if self._states.get(name) != "building":
                    self._states.pop(name, None)
