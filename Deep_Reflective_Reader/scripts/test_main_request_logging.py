#!/usr/bin/env python3
"""Regression checks for API request logging filters."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_health_success_request_logs_are_suppressed() -> None:
    from main import _should_suppress_success_request_log

    assert _should_suppress_success_request_log("/health")
    assert not _should_suppress_success_request_log("/documents")
    assert not _should_suppress_success_request_log("/documents/prepare")


if __name__ == "__main__":
    test_health_success_request_logs_are_suppressed()
    print("main request logging checks passed")
