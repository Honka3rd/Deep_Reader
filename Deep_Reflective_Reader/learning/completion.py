"""Process-local completion notifications; durable operation state is authoritative."""
from concurrent.futures import Future
from threading import Lock


class LearningCompletion:
    def __init__(self):
        self._lock = Lock()
        self._listeners = {}

    def subscribe(self, document_id, operation_id):
        future = Future()
        with self._lock:
            self._listeners.setdefault((document_id, operation_id), set()).add(future)
        return future

    def unsubscribe(self, document_id, operation_id, future):
        with self._lock:
            key = (document_id, operation_id)
            listeners = self._listeners.get(key)
            if listeners is not None:
                listeners.discard(future)
                if not listeners:
                    del self._listeners[key]
            future.cancel()  # Observer only; never the worker's executor future.

    def notify(self, document_id, operation_id):
        with self._lock:
            for future in self._listeners.pop((document_id, operation_id), ()):
                if not future.done():
                    future.set_result(None)
