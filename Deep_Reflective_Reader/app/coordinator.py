"""Backward-compatible alias module for QA coordinator naming migration."""

from app.qa_coordinator import AskExecutionResult, QACoordinator

# Compatibility alias: keep old import path working.
Coordinator = QACoordinator

__all__ = [
    "AskExecutionResult",
    "QACoordinator",
    "Coordinator",
]
