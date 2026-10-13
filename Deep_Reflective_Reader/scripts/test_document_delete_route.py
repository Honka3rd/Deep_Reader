#!/usr/bin/env python3
"""Regression tests for explicit document delete lifecycle route."""

from __future__ import annotations

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


def test_delete_removes_raw_structured_and_derived_artifacts() -> None:
    def run(raw_dir: Path, structured_dir: Path, derived_dir: Path) -> None:
        raw_path = raw_dir / "Book.txt"
        structured_path = structured_dir / "Book.structured.json"
        derived_path = derived_dir / "Book"
        raw_path.write_text("raw", encoding="utf-8")
        structured_path.write_text('{"title": "Book", "chapters": []}', encoding="utf-8")
        derived_path.mkdir()
        (derived_path / "profile.json").write_text("{}", encoding="utf-8")
        (derived_path / "records.json").write_text("{}", encoding="utf-8")

        client = TestClient(main.app)
        response = client.post(
            "/documents/lifecycle/delete",
            json={
                "doc_name": "Book",
                "action": "delete",
                "confirm": True,
            },
        )

        _assert(response.status_code == 200, f"unexpected status: {response.status_code}")
        payload = response.json()
        _assert(payload["doc_name"] == "Book", f"unexpected payload: {payload}")
        _assert(payload["action"] == "delete", "action should round-trip")
        _assert(payload["status"] == "accepted", "delete should be accepted")
        _assert(payload["source_affected"] is True, "raw source should be affected")
        _assert(
            payload["structured_artifacts_affected"] is True,
            "structured artifact should be affected",
        )
        _assert(
            payload["derived_resources_affected"] is True,
            "derived resources should be affected",
        )
        _assert(payload["deleted"] is True, "deleted flag should be true")
        _assert(not raw_path.exists(), "raw file should be deleted")
        _assert(not structured_path.exists(), "structured file should be deleted")
        _assert(not derived_path.exists(), "derived directory should be deleted")
        for forbidden_key in [
            "raw_text",
            "chapters",
            "sections",
            "task_units",
            "content_blocks",
            "profile",
            "artifacts",
        ]:
            _assert(forbidden_key not in payload, f"delete response leaked {forbidden_key}")

    _with_temp_document_dirs(run)


def test_delete_requires_explicit_confirmation() -> None:
    def run(raw_dir: Path, _: Path, __: Path) -> None:
        raw_path = raw_dir / "Book.txt"
        raw_path.write_text("raw", encoding="utf-8")

        client = TestClient(main.app)
        response = client.post(
            "/documents/lifecycle/delete",
            json={
                "doc_name": "Book",
                "action": "delete",
            },
        )

        _assert(response.status_code == 400, f"unexpected status: {response.status_code}")
        payload = response.json()
        _assert(payload["status"] == "rejected", "unconfirmed delete should reject")
        _assert(payload["reason"] == "validation_failure", "reason mismatch")
        _assert(raw_path.exists(), "unconfirmed delete must not remove raw file")

    _with_temp_document_dirs(run)


def test_delete_missing_document_returns_typed_404_response() -> None:
    def run(_: Path, __: Path, ___: Path) -> None:
        client = TestClient(main.app)
        response = client.post(
            "/documents/lifecycle/delete",
            json={
                "doc_name": "Missing Book",
                "action": "delete",
                "confirm": True,
            },
        )

        _assert(response.status_code == 404, f"unexpected status: {response.status_code}")
        payload = response.json()
        _assert(payload["status"] == "rejected", "missing delete should reject")
        _assert(payload["reason"] == "missing_document", "reason mismatch")
        _assert(payload["deleted"] is False, "missing delete should not report deleted")

    _with_temp_document_dirs(run)


def test_delete_route_rejects_non_delete_actions() -> None:
    def run(raw_dir: Path, _: Path, __: Path) -> None:
        raw_path = raw_dir / "Book.txt"
        raw_path.write_text("raw", encoding="utf-8")

        client = TestClient(main.app)
        response = client.post(
            "/documents/lifecycle/delete",
            json={
                "doc_name": "Book",
                "action": "rename",
                "new_doc_name": "New Book",
                "confirm": True,
            },
        )

        _assert(response.status_code == 400, f"unexpected status: {response.status_code}")
        payload = response.json()
        _assert(payload["status"] == "rejected", "non-delete action should reject")
        _assert(payload["reason"] == "unsupported_action", "reason mismatch")
        _assert(raw_path.exists(), "wrong action must not delete raw file")

    _with_temp_document_dirs(run)


if __name__ == "__main__":
    test_delete_removes_raw_structured_and_derived_artifacts()
    test_delete_requires_explicit_confirmation()
    test_delete_missing_document_returns_typed_404_response()
    test_delete_route_rejects_non_delete_actions()
    print("OK: document delete route tests passed")
