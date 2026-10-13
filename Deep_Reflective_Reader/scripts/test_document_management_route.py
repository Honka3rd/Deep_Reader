#!/usr/bin/env python3
"""Regression tests for document management list/status API."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import main  # noqa: E402
from document_structure.document_artifact_repository import DocumentListItem  # noqa: E402


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


class _FakeDocumentRepository:
    def __init__(self) -> None:
        self.calls: list[tuple[str | None, int]] = []

    def list_documents(
        self,
        query: str | None = None,
        limit: int = 50,
    ) -> list[DocumentListItem]:
        self.calls.append((query, limit))
        candidates = [
            DocumentListItem(
                doc_name="Prepared Book",
                title="Prepared Book",
                source="structured_file",
            ),
            DocumentListItem(
                doc_name="Shared Book",
                title="Shared Book",
                source="structured_file",
            ),
        ]
        normalized_query = (query or "").casefold()
        if normalized_query:
            candidates = [
                item
                for item in candidates
                if normalized_query in item.doc_name.casefold()
                or normalized_query in (item.title or "").casefold()
            ]
        return candidates[:limit]


def _with_temp_raw_dir(test_func) -> None:
    original_raw_dir = main._RAW_DOCUMENT_BASE_DIR
    original_repository = main.document_artifact_repository
    with tempfile.TemporaryDirectory() as temp_dir:
        main._RAW_DOCUMENT_BASE_DIR = Path(temp_dir)
        main.document_artifact_repository = _FakeDocumentRepository()
        try:
            test_func(Path(temp_dir), main.document_artifact_repository)
        finally:
            main._RAW_DOCUMENT_BASE_DIR = original_raw_dir
            main.document_artifact_repository = original_repository


def test_management_route_merges_structured_readiness_and_raw_source_status() -> None:
    def run(raw_dir: Path, repository: _FakeDocumentRepository) -> None:
        (raw_dir / "Raw Only.txt").write_text("raw only", encoding="utf-8")
        (raw_dir / "Shared Book.pdf").write_bytes(b"%PDF-1.4\n")

        client = TestClient(main.app)
        response = client.get("/documents/management", params={"limit": 20})

        _assert(response.status_code == 200, f"unexpected status: {response.status_code}")
        payload = response.json()
        _assert(payload["query"] is None, f"unexpected query echo: {payload}")
        _assert(payload["total"] == 3, f"unexpected total: {payload}")
        items = {item["doc_name"]: item for item in payload["items"]}

        prepared = items["Prepared Book"]
        _assert(prepared["source"] == "structured_file", "prepared source mismatch")
        _assert(prepared["source_status"] == "unknown", "structured-only source status should be unknown")
        _assert(prepared["preparation_status"] == "ready", "structured-only row should be ready")
        _assert(prepared["source_filename"] is None, "structured-only row should not invent source filename")

        raw_only = items["Raw Only"]
        _assert(raw_only["source"] == "raw.txt", "raw-only source mismatch")
        _assert(raw_only["source_filename"] == "Raw Only.txt", "raw filename mismatch")
        _assert(raw_only["source_type"] == "txt", "raw type mismatch")
        _assert(raw_only["source_size_bytes"] == len("raw only"), "raw size mismatch")
        _assert(raw_only["source_status"] == "available", "raw source should be available")
        _assert(raw_only["preparation_status"] == "not_prepared", "raw-only row should not be prepared")

        shared = items["Shared Book"]
        _assert(shared["source"] == "raw.pdf+structured_file", f"merged source mismatch: {shared}")
        _assert(shared["source_filename"] == "Shared Book.pdf", "merged raw filename should survive")
        _assert(shared["source_type"] == "pdf", "merged raw source type should survive")
        _assert(shared["source_status"] == "available", "merged source should be available")
        _assert(shared["preparation_status"] == "ready", "structured readiness should win")

        for item in payload["items"]:
            for forbidden_key in [
                "raw_text",
                "chapters",
                "sections",
                "task_units",
                "content_blocks",
                "profile",
                "artifacts",
            ]:
                _assert(forbidden_key not in item, f"management row leaked {forbidden_key}: {item}")

        _assert(repository.calls == [(None, 200)], "route should over-fetch structured candidates before merge")

    _with_temp_raw_dir(run)


def test_management_route_filters_query_and_keeps_compact_shape() -> None:
    def run(raw_dir: Path, repository: _FakeDocumentRepository) -> None:
        (raw_dir / "Raw Only.txt").write_text("raw only", encoding="utf-8")
        (raw_dir / "Shared Book.pdf").write_bytes(b"%PDF-1.4\n")

        client = TestClient(main.app)
        response = client.get("/documents/management", params={"q": "shared", "limit": 10})

        _assert(response.status_code == 200, f"unexpected status: {response.status_code}")
        payload = response.json()
        _assert(payload["query"] == "shared", f"unexpected query echo: {payload}")
        _assert(payload["total"] == 1, f"unexpected total: {payload}")
        _assert(payload["items"][0]["doc_name"] == "Shared Book", "query should filter rows")
        _assert(repository.calls == [("shared", 200)], "route should pass query to repository")

    _with_temp_raw_dir(run)


if __name__ == "__main__":
    test_management_route_merges_structured_readiness_and_raw_source_status()
    test_management_route_filters_query_and_keeps_compact_shape()
    print("OK: document management route tests passed")
