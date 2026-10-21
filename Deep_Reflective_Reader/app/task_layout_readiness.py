"""Readiness failure for a read-only task-layout request."""


class TaskLayoutNotReadyError(ValueError):
    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(
            f"Task layout is not ready ({reason}). Prepare the document in Library "
            "or call POST /documents/prepare-task-layout with the same split options."
        )
