#!/usr/bin/env python3
"""Regression tests for explicit document upload route behavior."""

from __future__ import annotations

import hashlib
import sys
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import main  # noqa: E402


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _with_temp_raw_dir(test_func) -> None:
    original_raw_dir = main._RAW_DOCUMENT_BASE_DIR
    with tempfile.TemporaryDirectory() as temp_dir:
        main._RAW_DOCUMENT_BASE_DIR = Path(temp_dir)
        try:
            test_func(Path(temp_dir))
        finally:
            main._RAW_DOCUMENT_BASE_DIR = original_raw_dir


def test_upload_stores_raw_source_candidate_without_preparing() -> None:
    def run(raw_dir: Path) -> None:
        client = TestClient(main.app)
        response = client.post(
            "/documents/upload",
            files={"file": ("Reader Notes.txt", b"hello reader", "text/plain")},
            data={"title": " Reader Notes "},
        )

        _assert(response.status_code == 201, f"unexpected status: {response.status_code}")
        payload = response.json()
        _assert(payload["doc_name"] == "Reader Notes", f"unexpected payload: {payload}")
        _assert(payload["title"] == "Reader Notes", "title should be normalized")
        _assert(payload["source_filename"] == "Reader Notes.txt", "filename mismatch")
        _assert(payload["source_type"] == "txt", "source type mismatch")
        _assert(payload["source_status"] == "uploaded", "upload should succeed")
        _assert(
            payload["preparation_status"] == "not_prepared",
            "upload must not imply preparation readiness",
        )
        _assert(
            payload["source_hash"] == hashlib.sha256(b"hello reader").hexdigest(),
            "source hash should describe stored bytes",
        )
        _assert((raw_dir / "Reader Notes.txt").read_bytes() == b"hello reader", "file not stored")

        list_response = client.get("/documents", params={"q": "reader", "limit": 10})
        _assert(list_response.status_code == 200, "document list should remain readable")
        listed = list_response.json()["items"]
        _assert(
            any(item["doc_name"] == "Reader Notes" and item["source"] == "raw.txt" for item in listed),
            f"uploaded raw candidate should be discoverable: {listed}",
        )

    _with_temp_raw_dir(run)


def test_upload_rejects_duplicate_by_default() -> None:
    def run(raw_dir: Path) -> None:
        (raw_dir / "Book.txt").write_text("existing", encoding="utf-8")
        client = TestClient(main.app)

        response = client.post(
            "/documents/upload",
            files={"file": ("Book.txt", b"new", "text/plain")},
        )

        _assert(response.status_code == 409, f"unexpected status: {response.status_code}")
        payload = response.json()
        _assert(payload["source_status"] == "failed", "duplicate should fail")
        _assert(payload["reason"] == "duplicate_conflict", "reason mismatch")
        _assert((raw_dir / "Book.txt").read_text(encoding="utf-8") == "existing", "file should not be replaced")

    _with_temp_raw_dir(run)


def test_upload_replace_policy_overwrites_raw_source_only() -> None:
    def run(raw_dir: Path) -> None:
        (raw_dir / "Book.txt").write_text("existing", encoding="utf-8")
        client = TestClient(main.app)

        response = client.post(
            "/documents/upload",
            files={"file": ("Book.txt", b"replacement", "text/plain")},
            data={"conflict_policy": "replace"},
        )

        _assert(response.status_code == 201, f"unexpected status: {response.status_code}")
        payload = response.json()
        _assert(payload["conflict_policy"] == "replace", "policy should round-trip")
        _assert((raw_dir / "Book.txt").read_bytes() == b"replacement", "file should be replaced")

    _with_temp_raw_dir(run)


def test_upload_create_distinct_policy_avoids_overwrite() -> None:
    def run(raw_dir: Path) -> None:
        (raw_dir / "Book.txt").write_text("existing", encoding="utf-8")
        client = TestClient(main.app)

        response = client.post(
            "/documents/upload",
            files={"file": ("Book.txt", b"new", "text/plain")},
            data={"conflict_policy": "create-distinct"},
        )

        _assert(response.status_code == 201, f"unexpected status: {response.status_code}")
        payload = response.json()
        _assert(payload["source_filename"] == "Book (1).txt", f"unexpected payload: {payload}")
        _assert(payload["doc_name"] == "Book (1)", "distinct upload should expose unique doc name")
        _assert(
            payload["conflict_policy"] == "create_distinct",
            "distinct upload should round-trip normalized policy",
        )
        _assert((raw_dir / "Book.txt").read_text(encoding="utf-8") == "existing", "original should remain")
        _assert((raw_dir / "Book (1).txt").read_bytes() == b"new", "distinct upload missing")

    _with_temp_raw_dir(run)


def test_upload_rejects_unsupported_file_type_with_typed_reason() -> None:
    def run(_: Path) -> None:
        client = TestClient(main.app)

        response = client.post(
            "/documents/upload",
            files={"file": ("notes.md", b"# notes", "text/markdown")},
        )

        _assert(response.status_code == 415, f"unexpected status: {response.status_code}")
        payload = response.json()
        _assert(payload["source_status"] == "failed", "unsupported upload should fail")
        _assert(payload["reason"] == "unsupported_file_type", "reason mismatch")
        _assert(payload["source_type"] == "unknown", "unsupported type should not leak as supported")

    _with_temp_raw_dir(run)


def test_upload_rejects_unsafe_filename() -> None:
    def run(_: Path) -> None:
        client = TestClient(main.app)

        response = client.post(
            "/documents/upload",
            files={"file": ("../Book.txt", b"bad", "text/plain")},
        )

        _assert(response.status_code == 400, f"unexpected status: {response.status_code}")
        payload = response.json()
        _assert(payload["source_status"] == "failed", "unsafe filename should fail")
        _assert(payload["reason"] == "unsafe_filename", "reason mismatch")

    _with_temp_raw_dir(run)


if __name__ == "__main__":
    test_upload_stores_raw_source_candidate_without_preparing()
    test_upload_rejects_duplicate_by_default()
    test_upload_replace_policy_overwrites_raw_source_only()
    test_upload_create_distinct_policy_avoids_overwrite()
    test_upload_rejects_unsupported_file_type_with_typed_reason()
    test_upload_rejects_unsafe_filename()
    print("OK: document upload route tests passed")
