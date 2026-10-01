#!/usr/bin/env python3
"""REST regression tests for manual structure validation/preview route."""

from __future__ import annotations

from pathlib import Path
import sys

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import main  # noqa: E402
from app.section_task_coordinator import ManualStructureCommitResult  # noqa: E402
from document_structure.manual_structure_projection import ManualStructureIssue  # noqa: E402


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _page_range_reparse_payload(*, source_hash: str | None = None) -> dict:
    manual_structure: dict = {
        "entries": [
            {
                "title": "Chapter One",
                "level": 1,
                "anchor": {
                    "anchor_type": "page_range",
                    "page_start_index": 0,
                    "page_end_index": 0,
                },
            },
            {
                "title": "Section One",
                "level": 2,
                "anchor": {
                    "anchor_type": "page_range",
                    "page_start_index": 0,
                    "page_end_index": 1,
                },
            },
        ],
    }
    if source_hash is not None:
        manual_structure["source_hash"] = source_hash
    return {
        "doc_name": "Book.pdf",
        "parser_mode": "manual_structure",
        "manual_structure": manual_structure,
    }


def test_manual_structure_validate_route_returns_lightweight_preview() -> None:
    client = TestClient(main.app)

    response = client.post(
        "/documents/manual-structure/validate",
        json={
            "doc_name": "  Any Source.pdf  ",
            "manual_structure": {
                "source_hash": " source-hash ",
                "entries": [
                    {
                        "title": " Chapter One ",
                        "level": 1,
                        "anchor": {
                            "anchor_type": "char-range",
                            "char_start": 0,
                            "char_end": 100,
                        },
                        "external_id": " ch-1 ",
                    },
                    {
                        "title": "Section One",
                        "level": 2,
                        "anchor": {
                            "anchor_type": "page-range",
                            "page_start_index": 2,
                            "page_end_index": 3,
                        },
                    },
                ],
            },
        },
    )

    _assert(response.status_code == 200, f"unexpected status: {response.status_code}")
    payload = response.json()
    _assert(payload["doc_name"] == "Any Source.pdf", "doc_name should be normalized")
    _assert(payload["valid"] is True, "valid plan should return valid=true")
    _assert(payload["errors"] == [], "schema-only preview should not emit errors")
    _assert(payload["warnings"] == [], "schema-only preview should not emit warnings")
    _assert(
        payload["normalized_entries"][0]["anchor"]["anchor_type"] == "char_range",
        "char anchor type should normalize",
    )
    _assert(
        payload["normalized_entries"][1]["anchor"]["anchor_type"] == "page_range",
        "page anchor type should normalize",
    )
    _assert(
        payload["preview_chapters"] == [
            {
                "title": "Chapter One",
                "external_id": "ch-1",
                "entry_index": 0,
                "sections": [
                    {
                        "title": "Section One",
                        "external_id": None,
                        "entry_index": 1,
                    }
                ],
            }
        ],
        f"unexpected preview shape: {payload['preview_chapters']}",
    )
    _assert(
        payload["parse_provenance_preview"] == {
            "parser_mode": "manual_structure",
            "source_hash": "source-hash",
            "anchor_types": ["char_range", "page_range"],
        },
        "provenance preview should be lightweight and normalized",
    )
    _assert("raw_text" not in payload, "preview route must not expose raw_text")
    _assert("content" not in payload, "preview route must not expose heavy content")
    _assert("task_units" not in payload, "preview route must not expose task-layout payload")


def test_manual_structure_validate_route_rejects_invalid_plan() -> None:
    client = TestClient(main.app)

    response = client.post(
        "/documents/manual-structure/validate",
        json={
            "doc_name": "Book.pdf",
            "manual_structure": {
                "entries": [
                    {
                        "title": "Orphan Section",
                        "level": 2,
                        "anchor": {"anchor_type": "char_range", "char_start": 0},
                    }
                ],
            },
        },
    )

    _assert(response.status_code == 422, f"unexpected status: {response.status_code}")
    _assert(
        "manual structure section entry requires preceding chapter" in response.text,
        f"unexpected validation error: {response.text}",
    )


def test_manual_structure_validate_route_uses_projection_validation() -> None:
    client = TestClient(main.app)

    response = client.post(
        "/documents/manual-structure/validate",
        json={
            "doc_name": "Book.pdf",
            "manual_structure": {
                "entries": [
                    {
                        "title": "Chapter One",
                        "level": 1,
                        "anchor": {
                            "anchor_type": "char_range",
                            "char_start": 0,
                            "char_end": 100,
                        },
                    },
                    {
                        "title": "Chapter Two",
                        "level": 1,
                        "anchor": {
                            "anchor_type": "char_range",
                            "char_start": 90,
                            "char_end": 150,
                        },
                    },
                ],
            },
        },
    )

    _assert(response.status_code == 200, f"unexpected status: {response.status_code}")
    payload = response.json()
    _assert(payload["valid"] is False, "overlapping siblings should be invalid")
    _assert(payload["normalized_entries"] == [], "invalid projection should not preview entries")
    _assert(payload["preview_chapters"] == [], "invalid projection should not preview hierarchy")
    _assert(
        [error["code"] for error in payload["errors"]] == ["overlapping_range"],
        f"unexpected projection errors: {payload['errors']}",
    )


def test_manual_structure_reparse_route_reports_unimplemented_without_mutation() -> None:
    class _ManualCommitCoordinatorSpy:
        def __init__(self) -> None:
            self.manual_commit_calls = []

        def commit_manual_structure_reparse(self, *, doc_name, manual_structure):  # noqa: ANN001
            self.manual_commit_calls.append((doc_name, manual_structure))
            return ManualStructureCommitResult.not_implemented(doc_name=doc_name)

        def reparse_document_structure(self, doc_name, parser_mode):  # noqa: ANN001
            _ = (doc_name, parser_mode)
            raise RuntimeError("manual structure reparse should not reach coordinator")

    original_coordinator = main.section_task_coordinator
    coordinator_spy = _ManualCommitCoordinatorSpy()
    main.section_task_coordinator = coordinator_spy
    try:
        client = TestClient(main.app)
        response = client.post(
            "/documents/reparse-structure",
            json={
                "doc_name": "Book.pdf",
                "parser_mode": "manual_structure",
                "manual_structure": {
                    "entries": [
                        {
                            "title": "Chapter One",
                            "level": 1,
                            "anchor": {
                                "anchor_type": "char_range",
                                "char_start": 0,
                                "char_end": 10,
                            },
                        }
                    ],
                },
            },
        )
    finally:
        main.section_task_coordinator = original_coordinator

    _assert(response.status_code == 501, f"unexpected status: {response.status_code}")
    _assert(
        len(coordinator_spy.manual_commit_calls) == 1,
        "manual structure commit should be routed through coordinator boundary",
    )
    call_doc_name, call_plan = coordinator_spy.manual_commit_calls[0]
    _assert(call_doc_name == "Book.pdf", "manual commit doc_name should be passed through")
    _assert(len(call_plan.entries) == 1, "manual commit plan should preserve entries")
    _assert(
        call_plan.entries[0].anchor.anchor_type == "char_range",
        "manual commit anchor should be mapped into app DTO",
    )
    payload = response.json()
    _assert(payload["success"] is False, "manual commit should not report success")
    _assert(payload["doc_name"] == "Book.pdf", "doc_name should round-trip")
    _assert(
        payload["parser_mode"] == "manual_structure",
        "parser mode should identify manual structure commit path",
    )
    _assert(payload["structured_document_path"] is None, "no hierarchy should be persisted")
    _assert(payload["section_count"] is None, "no section count should be produced")
    _assert(
        "not implemented" in payload["error"],
        f"unexpected error message: {payload['error']}",
    )


def test_manual_structure_reparse_route_maps_commit_success() -> None:
    class _ManualCommitCoordinatorSpy:
        def __init__(self) -> None:
            self.manual_commit_calls = []

        def commit_manual_structure_reparse(self, *, doc_name, manual_structure):  # noqa: ANN001
            self.manual_commit_calls.append((doc_name, manual_structure))
            return ManualStructureCommitResult.committed(
                doc_name=doc_name,
                structured_document_path=None,
                section_count=1,
            )

        def reparse_document_structure(self, doc_name, parser_mode):  # noqa: ANN001
            _ = (doc_name, parser_mode)
            raise RuntimeError("manual structure reparse should not reach legacy coordinator")

    original_coordinator = main.section_task_coordinator
    coordinator_spy = _ManualCommitCoordinatorSpy()
    main.section_task_coordinator = coordinator_spy
    try:
        client = TestClient(main.app)
        response = client.post(
            "/documents/reparse-structure",
            json={
                "doc_name": "Book.pdf",
                "parser_mode": "manual_structure",
                "manual_structure": {
                    "entries": [
                        {
                            "title": "Chapter One",
                            "level": 1,
                            "anchor": {
                                "anchor_type": "char_range",
                                "char_start": 0,
                                "char_end": 10,
                            },
                        }
                    ],
                },
            },
        )
    finally:
        main.section_task_coordinator = original_coordinator

    _assert(response.status_code == 200, f"unexpected status: {response.status_code}")
    _assert(
        len(coordinator_spy.manual_commit_calls) == 1,
        "manual structure success should be routed through coordinator boundary",
    )
    payload = response.json()
    _assert(payload["success"] is True, "manual commit success should round-trip")
    _assert(payload["doc_name"] == "Book.pdf", "doc_name should round-trip")
    _assert(payload["parser_mode"] == "manual_structure", "parser mode should round-trip")
    _assert(payload["structured_document_path"] is None, "path may be backend-dependent")
    _assert(payload["section_count"] == 1, "section count should map from coordinator")
    _assert(payload["error"] is None, f"unexpected error: {payload['error']}")


def test_manual_structure_reparse_route_rejects_invalid_projection_before_commit() -> None:
    client = TestClient(main.app)

    response = client.post(
        "/documents/reparse-structure",
        json={
            "doc_name": "Book.pdf",
            "parser_mode": "manual_structure",
            "manual_structure": {
                "entries": [
                    {
                        "title": "Chapter One",
                        "level": 1,
                        "anchor": {
                            "anchor_type": "char_range",
                            "char_start": 0,
                            "char_end": 100,
                        },
                    },
                    {
                        "title": "Chapter Two",
                        "level": 1,
                        "anchor": {
                            "anchor_type": "char_range",
                            "char_start": 90,
                            "char_end": 150,
                        },
                    },
                ],
            },
        },
    )

    _assert(response.status_code == 422, f"unexpected status: {response.status_code}")
    payload = response.json()
    _assert(payload["success"] is False, "invalid manual commit should fail")
    _assert(payload["parser_mode"] == "manual_structure", "parser mode should identify manual path")
    _assert(payload["structured_document_path"] is None, "invalid plan must not persist hierarchy")
    _assert(payload["section_count"] is None, "invalid plan must not produce section count")
    _assert(
        "overlapping_range" in payload["error"],
        f"unexpected manual commit error: {payload['error']}",
    )


def test_manual_structure_reparse_route_maps_page_evidence_required_failure() -> None:
    class _ManualCommitCoordinatorSpy:
        def commit_manual_structure_reparse(self, *, doc_name, manual_structure):  # noqa: ANN001
            _assert(
                manual_structure.entries[0].anchor.anchor_type == "page_range",
                "route should preserve schema-valid page_range anchor",
            )
            return ManualStructureCommitResult.source_unavailable(
                doc_name=doc_name,
                error=(
                    "manual_structure plan is invalid: "
                    "page_boundaries_required[entry_index=0]: "
                    "manual page_range anchors require page boundary evidence"
                ),
                status_code=422,
            )

        def reparse_document_structure(self, doc_name, parser_mode):  # noqa: ANN001
            _ = (doc_name, parser_mode)
            raise RuntimeError("page-backed manual commit should not reach legacy reparse")

    original_coordinator = main.section_task_coordinator
    main.section_task_coordinator = _ManualCommitCoordinatorSpy()
    try:
        response = TestClient(main.app).post(
            "/documents/reparse-structure",
            json=_page_range_reparse_payload(),
        )
    finally:
        main.section_task_coordinator = original_coordinator

    _assert(response.status_code == 422, f"unexpected status: {response.status_code}")
    payload = response.json()
    _assert(payload["success"] is False, "unsupported page evidence should fail")
    _assert(payload["structured_document_path"] is None, "failure must not report a path")
    _assert(payload["section_count"] is None, "failure must not report section count")
    _assert(
        "page_boundaries_required" in payload["error"],
        f"unexpected error: {payload['error']}",
    )


def test_manual_structure_reparse_route_maps_stale_page_source_failure() -> None:
    class _ManualCommitCoordinatorSpy:
        def commit_manual_structure_reparse(self, *, doc_name, manual_structure):  # noqa: ANN001
            _assert(
                manual_structure.source_hash == "client-hash",
                "route should pass source_hash through to stale-source gate",
            )
            return ManualStructureCommitResult.stale_source_evidence(
                doc_name=doc_name,
                expected_hash="client-hash",
                actual_hash="server-hash",
            )

        def reparse_document_structure(self, doc_name, parser_mode):  # noqa: ANN001
            _ = (doc_name, parser_mode)
            raise RuntimeError("page-backed manual commit should not reach legacy reparse")

    original_coordinator = main.section_task_coordinator
    main.section_task_coordinator = _ManualCommitCoordinatorSpy()
    try:
        response = TestClient(main.app).post(
            "/documents/reparse-structure",
            json=_page_range_reparse_payload(source_hash="client-hash"),
        )
    finally:
        main.section_task_coordinator = original_coordinator

    _assert(response.status_code == 409, f"unexpected status: {response.status_code}")
    payload = response.json()
    _assert(payload["success"] is False, "stale source evidence should fail")
    _assert(
        "stale" in payload["error"] and "client-hash" in payload["error"],
        f"unexpected stale-source error: {payload['error']}",
    )


def test_manual_structure_reparse_route_maps_page_anchor_projection_failure() -> None:
    class _ManualCommitCoordinatorSpy:
        def commit_manual_structure_reparse(self, *, doc_name, manual_structure):  # noqa: ANN001
            _ = manual_structure
            return ManualStructureCommitResult.invalid_plan(
                doc_name=doc_name,
                issues=[
                    ManualStructureIssue(
                        code="out_of_range_anchor",
                        message="page_range anchor references page 99 outside available page evidence",
                        entry_index=1,
                    )
                ],
            )

        def reparse_document_structure(self, doc_name, parser_mode):  # noqa: ANN001
            _ = (doc_name, parser_mode)
            raise RuntimeError("page-backed manual commit should not reach legacy reparse")

    original_coordinator = main.section_task_coordinator
    main.section_task_coordinator = _ManualCommitCoordinatorSpy()
    try:
        response = TestClient(main.app).post(
            "/documents/reparse-structure",
            json=_page_range_reparse_payload(),
        )
    finally:
        main.section_task_coordinator = original_coordinator

    _assert(response.status_code == 422, f"unexpected status: {response.status_code}")
    payload = response.json()
    _assert(payload["success"] is False, "out-of-range page anchor should fail")
    _assert(
        "out_of_range_anchor" in payload["error"],
        f"unexpected page-anchor error: {payload['error']}",
    )


def test_manual_structure_reparse_route_maps_page_backed_commit_success() -> None:
    class _ManualCommitCoordinatorSpy:
        def commit_manual_structure_reparse(self, *, doc_name, manual_structure):  # noqa: ANN001
            _assert(doc_name == "Book.pdf", "doc_name should pass through")
            _assert(
                {entry.anchor.anchor_type for entry in manual_structure.entries}
                == {"page_range"},
                "route should preserve page_range anchors",
            )
            return ManualStructureCommitResult.committed(
                doc_name=doc_name,
                structured_document_path="/tmp/Book.structured.json",
                section_count=1,
            )

        def reparse_document_structure(self, doc_name, parser_mode):  # noqa: ANN001
            _ = (doc_name, parser_mode)
            raise RuntimeError("page-backed manual commit should not reach legacy reparse")

    original_coordinator = main.section_task_coordinator
    main.section_task_coordinator = _ManualCommitCoordinatorSpy()
    try:
        response = TestClient(main.app).post(
            "/documents/reparse-structure",
            json=_page_range_reparse_payload(),
        )
    finally:
        main.section_task_coordinator = original_coordinator

    _assert(response.status_code == 200, f"unexpected status: {response.status_code}")
    payload = response.json()
    _assert(payload["success"] is True, "page-backed commit success should round-trip")
    _assert(payload["parser_mode"] == "manual_structure", "parser mode should identify manual path")
    _assert(
        payload["structured_document_path"] == "/tmp/Book.structured.json",
        "structured path should map from coordinator result",
    )
    _assert(payload["section_count"] == 1, "section count should map from coordinator")
    _assert(payload["error"] is None, f"unexpected error: {payload['error']}")


if __name__ == "__main__":
    test_manual_structure_validate_route_returns_lightweight_preview()
    test_manual_structure_validate_route_rejects_invalid_plan()
    test_manual_structure_validate_route_uses_projection_validation()
    test_manual_structure_reparse_route_reports_unimplemented_without_mutation()
    test_manual_structure_reparse_route_maps_commit_success()
    test_manual_structure_reparse_route_rejects_invalid_projection_before_commit()
    test_manual_structure_reparse_route_maps_page_evidence_required_failure()
    test_manual_structure_reparse_route_maps_stale_page_source_failure()
    test_manual_structure_reparse_route_maps_page_anchor_projection_failure()
    test_manual_structure_reparse_route_maps_page_backed_commit_success()
    print("OK: manual structure validation route tests passed")
