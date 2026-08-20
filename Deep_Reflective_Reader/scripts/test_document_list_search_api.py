#!/usr/bin/env python3
"""Regression tests for lightweight document list/search API."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import main
from document_structure.document_artifact_repository import DocumentListItem
from document_structure.structured_document_artifact_repository import (
    StructuredDocumentArtifactRepository,
)


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
                doc_name="Madame Bovary",
                title="Madame Bovary",
                source="fake",
            ),
            DocumentListItem(
                doc_name="Moby Dick",
                title="Moby-Dick",
                source="fake",
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


def test_file_repository_lists_and_filters_structured_documents() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        base_dir = Path(temp_dir)
        (base_dir / "Madame Bovary.structured.json").write_text(
            json.dumps({"title": "Madame Bovary"}),
            encoding="utf-8",
        )
        (base_dir / "Moby Dick.structured.json").write_text(
            json.dumps({"title": "Moby-Dick"}),
            encoding="utf-8",
        )
        (base_dir / "notes.txt").write_text("ignored", encoding="utf-8")

        repository = StructuredDocumentArtifactRepository(base_dir=str(base_dir))

        all_items = repository.list_documents()
        _assert(
            [item.doc_name for item in all_items] == ["Madame Bovary", "Moby Dick"],
            f"unexpected document order: {all_items}",
        )
        _assert(all_items[0].title == "Madame Bovary", "title metadata should be read")
        _assert(all_items[0].source == "structured_file", "source should identify file backend")

        filtered_items = repository.list_documents(query="bov", limit=5)
        _assert(
            [item.doc_name for item in filtered_items] == ["Madame Bovary"],
            f"unexpected filtered documents: {filtered_items}",
        )

        limited_items = repository.list_documents(limit=1)
        _assert(len(limited_items) == 1, "limit should bound file repository results")


def test_raw_document_discovery_lists_supported_raw_files() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        base_dir = Path(temp_dir)
        (base_dir / "Raw Book.txt").write_text("raw text", encoding="utf-8")
        (base_dir / "Paper.pdf").write_bytes(b"%PDF-1.4\n")
        (base_dir / "notes.md").write_text("ignored", encoding="utf-8")

        items = main._list_raw_documents(base_dir=base_dir)

        _assert(
            [item.doc_name for item in items] == ["Paper", "Raw Book"],
            f"unexpected raw document candidates: {items}",
        )
        _assert(items[0].source == "raw.pdf", "source should identify raw pdf candidate")
        _assert(items[1].source == "raw.txt", "source should identify raw txt candidate")

        filtered = main._list_raw_documents(query="raw", base_dir=base_dir)
        _assert(
            [item.doc_name for item in filtered] == ["Raw Book"],
            f"unexpected raw query result: {filtered}",
        )


def test_rest_document_list_endpoint_maps_lightweight_response() -> None:
    original_repository = main.document_artifact_repository
    original_raw_lister = main._list_raw_documents
    fake_repository = _FakeDocumentRepository()
    main.document_artifact_repository = fake_repository
    main._list_raw_documents = lambda query=None, limit=200: [
        DocumentListItem(
            doc_name="Madame Bovary",
            title="Madame Bovary",
            source="raw.txt",
        )
    ]
    try:
        client = TestClient(main.app)
        response = client.get("/documents", params={"q": "mad", "limit": 10})
    finally:
        main.document_artifact_repository = original_repository
        main._list_raw_documents = original_raw_lister

    _assert(response.status_code == 200, f"unexpected status: {response.status_code}")
    payload = response.json()
    _assert(payload["query"] == "mad", f"unexpected query echo: {payload}")
    _assert(payload["total"] == 1, f"unexpected total: {payload}")
    _assert(
        payload["items"] == [
            {
                "doc_name": "Madame Bovary",
                "title": "Madame Bovary",
                "source": "fake+raw.txt",
            }
        ],
        f"unexpected document list payload: {payload}",
    )
    _assert(fake_repository.calls == [("mad", 200)], "route should over-fetch structured candidates before merge")


if __name__ == "__main__":
    test_file_repository_lists_and_filters_structured_documents()
    test_raw_document_discovery_lists_supported_raw_files()
    test_rest_document_list_endpoint_maps_lightweight_response()
    print("OK: document list/search API tests passed")
