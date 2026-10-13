#!/usr/bin/env python3
"""Regression tests for explicit document rename lifecycle route."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import main  # noqa: E402


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


class _FakeDocumentRepository:
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir


def _with_temp_document_dirs(test_func) -> None:
    original_raw_dir = main._RAW_DOCUMENT_BASE_DIR
    original_derived_dir = main._DERIVED_DOCUMENT_BASE_DIR
    original_repository = main.document_artifact_repository
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        raw_dir = root / "raw"
        structured_dir = root / "structured"
        derived_dir = root / "faiss"
        raw_dir.mkdir()
        structured_dir.mkdir()
        derived_dir.mkdir()
        main._RAW_DOCUMENT_BASE_DIR = raw_dir
        main._DERIVED_DOCUMENT_BASE_DIR = derived_dir
        main.document_artifact_repository = _FakeDocumentRepository(structured_dir)
        try:
            test_func(raw_dir, structured_dir, derived_dir)
        finally:
            main._RAW_DOCUMENT_BASE_DIR = original_raw_dir
            main._DERIVED_DOCUMENT_BASE_DIR = original_derived_dir
            main.document_artifact_repository = original_repository


def test_rename_moves_raw_structured_and_derived_artifacts() -> None:
    def run(raw_dir: Path, structured_dir: Path, derived_dir: Path) -> None:
        raw_path = raw_dir / "Book.txt"
        structured_path = structured_dir / "Book.structured.json"
        derived_path = derived_dir / "Book"
        raw_path.write_text("raw", encoding="utf-8")
        structured_path.write_text(
            json.dumps(
                {
                    "document_id": "Book",
                    "title": "Book",
                    "chapters": [],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        derived_path.mkdir()
        (derived_path / "profile.json").write_text("{}", encoding="utf-8")

        client = TestClient(main.app)
        response = client.post(
            "/documents/lifecycle/rename",
            json={
                "doc_name": "Book",
                "action": "rename",
                "new_doc_name": "New Book",
            },
        )

        _assert(response.status_code == 200, f"unexpected status: {response.status_code}")
        payload = response.json()
        _assert(payload["doc_name"] == "Book", f"unexpected payload: {payload}")
        _assert(payload["action"] == "rename", "action should round-trip")
        _assert(payload["status"] == "accepted", "rename should be accepted")
        _assert(payload["new_doc_name"] == "New Book", "new name should be exposed")
        _assert(payload["source_affected"] is True, "raw source should be affected")
        _assert(
            payload["structured_artifacts_affected"] is True,
            "structured artifact should be affected",
        )
        _assert(
            payload["derived_resources_affected"] is True,
            "derived resources should be affected",
        )
        _assert(not raw_path.exists(), "old raw file should be gone")
        _assert((raw_dir / "New Book.txt").exists(), "new raw file should exist")
        _assert(not structured_path.exists(), "old structured file should be gone")
        renamed_structured = structured_dir / "New Book.structured.json"
        _assert(renamed_structured.exists(), "new structured file should exist")
        renamed_payload = json.loads(renamed_structured.read_text(encoding="utf-8"))
        _assert(
            renamed_payload["document_id"] == "New Book",
            "structured document id should be updated",
        )
        _assert(renamed_payload["title"] == "New Book", "title should be updated")
        _assert(not derived_path.exists(), "old derived dir should be gone")
        _assert((derived_dir / "New Book" / "profile.json").exists(), "derived dir should move")
        for forbidden_key in [
            "raw_text",
            "chapters",
            "sections",
            "task_units",
            "content_blocks",
            "profile",
            "artifacts",
        ]:
            _assert(forbidden_key not in payload, f"rename response leaked {forbidden_key}")

    _with_temp_document_dirs(run)


def test_rename_conflict_preserves_existing_artifacts() -> None:
    def run(raw_dir: Path, structured_dir: Path, _: Path) -> None:
        old_raw = raw_dir / "Book.txt"
        new_raw = raw_dir / "Existing.txt"
        old_structured = structured_dir / "Book.structured.json"
        old_raw.write_text("old", encoding="utf-8")
        new_raw.write_text("existing", encoding="utf-8")
        old_structured.write_text('{"document_id": "Book"}', encoding="utf-8")

        client = TestClient(main.app)
        response = client.post(
            "/documents/lifecycle/rename",
            json={
                "doc_name": "Book",
                "action": "rename",
                "new_doc_name": "Existing",
            },
        )

        _assert(response.status_code == 409, f"unexpected status: {response.status_code}")
        payload = response.json()
        _assert(payload["status"] == "conflict", "conflicting rename should conflict")
        _assert(payload["reason"] == "conflict", "reason mismatch")
        _assert(old_raw.read_text(encoding="utf-8") == "old", "old raw should remain")
        _assert(new_raw.read_text(encoding="utf-8") == "existing", "target should remain")
        _assert(old_structured.exists(), "structured source should remain")

    _with_temp_document_dirs(run)


def test_rename_missing_document_returns_typed_404_response() -> None:
    def run(_: Path, __: Path, ___: Path) -> None:
        client = TestClient(main.app)
        response = client.post(
            "/documents/lifecycle/rename",
            json={
                "doc_name": "Missing Book",
                "action": "rename",
                "new_doc_name": "New Book",
            },
        )

        _assert(response.status_code == 404, f"unexpected status: {response.status_code}")
        payload = response.json()
        _assert(payload["status"] == "rejected", "missing rename should reject")
        _assert(payload["reason"] == "missing_document", "reason mismatch")

    _with_temp_document_dirs(run)


def test_rename_route_rejects_non_rename_actions() -> None:
    def run(raw_dir: Path, _: Path, __: Path) -> None:
        raw_path = raw_dir / "Book.txt"
        raw_path.write_text("raw", encoding="utf-8")

        client = TestClient(main.app)
        response = client.post(
            "/documents/lifecycle/rename",
            json={
                "doc_name": "Book",
                "action": "delete",
                "confirm": True,
            },
        )

        _assert(response.status_code == 400, f"unexpected status: {response.status_code}")
        payload = response.json()
        _assert(payload["status"] == "rejected", "non-rename action should reject")
        _assert(payload["reason"] == "unsupported_action", "reason mismatch")
        _assert(raw_path.exists(), "wrong action must not rename raw file")

    _with_temp_document_dirs(run)


if __name__ == "__main__":
    test_rename_moves_raw_structured_and_derived_artifacts()
    test_rename_conflict_preserves_existing_artifacts()
    test_rename_missing_document_returns_typed_404_response()
    test_rename_route_rejects_non_rename_actions()
    print("OK: document rename route tests passed")
