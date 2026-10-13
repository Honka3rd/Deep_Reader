#!/usr/bin/env python3
"""Schema regression tests for document upload API contracts."""

from __future__ import annotations

from pathlib import Path
import sys

from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api_schemas import (  # noqa: E402
    DocumentUploadMetadataRequest,
    DocumentUploadResponse,
)


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _assert_metadata_validation_error(
    payload: dict[str, object],
    expected_message: str,
) -> None:
    try:
        DocumentUploadMetadataRequest.model_validate(payload)
    except ValidationError as error:
        _assert(
            expected_message in str(error),
            f"expected {expected_message!r} in metadata validation error: {error}",
        )
        return
    raise AssertionError(
        f"expected metadata validation error containing {expected_message!r}"
    )


def _assert_upload_response_validation_error(
    payload: dict[str, object],
    expected_message: str,
) -> None:
    try:
        DocumentUploadResponse.model_validate(payload)
    except ValidationError as error:
        _assert(
            expected_message in str(error),
            f"expected {expected_message!r} in upload response error: {error}",
        )
        return
    raise AssertionError(
        f"expected upload response validation error containing {expected_message!r}"
    )


def _upload_response_payload(**updates: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "doc_name": " Madame Bovary ",
        "title": " Madame Bovary ",
        "source_filename": " madame-bovary.pdf ",
        "source_type": " PDF ",
        "source_size_bytes": 1234,
        "source_hash": " source-hash ",
        "uploaded_at": "2026-10-13T12:00:00Z",
        "source_status": " UPLOADED ",
        "preparation_status": " not_prepared ",
        "conflict_policy": "create-distinct",
    }
    payload.update(updates)
    return payload


def test_upload_metadata_normalizes_optional_title_and_conflict_policy() -> None:
    metadata = DocumentUploadMetadataRequest.model_validate(
        {
            "title": "  Madame Bovary  ",
            "conflict_policy": "create-distinct",
        }
    )

    _assert(metadata.title == "Madame Bovary", "title should be stripped")
    _assert(
        metadata.conflict_policy == "create_distinct",
        "conflict policy should normalize hyphenated spelling",
    )


def test_upload_metadata_rejects_unknown_conflict_policy() -> None:
    _assert_metadata_validation_error(
        {
            "title": "Madame Bovary",
            "conflict_policy": "merge",
        },
        "conflict_policy must be one of",
    )


def test_upload_response_normalizes_lightweight_source_metadata() -> None:
    response = DocumentUploadResponse.model_validate(_upload_response_payload())

    _assert(response.doc_name == "Madame Bovary", "doc_name should be stripped")
    _assert(response.title == "Madame Bovary", "title should be stripped")
    _assert(
        response.source_filename == "madame-bovary.pdf",
        "source filename should be stripped",
    )
    _assert(response.source_type == "pdf", "source type should be normalized")
    _assert(response.source_status == "uploaded", "status should be normalized")
    _assert(
        response.preparation_status == "not_prepared",
        "upload success must not imply preparation success",
    )
    _assert(
        response.conflict_policy == "create_distinct",
        "conflict policy should be normalized",
    )


def test_upload_response_requires_failure_reason_for_failed_status() -> None:
    _assert_upload_response_validation_error(
        _upload_response_payload(
            source_status="failed",
            preparation_status="unknown",
            reason=None,
        ),
        "failed upload response requires reason",
    )

    response = DocumentUploadResponse.model_validate(
        _upload_response_payload(
            source_status="failed",
            preparation_status="unknown",
            reason="duplicate_conflict",
        )
    )
    _assert(response.reason == "duplicate_conflict", "failure reason should pass")


def test_upload_response_rejects_unknown_failure_reason() -> None:
    _assert_upload_response_validation_error(
        _upload_response_payload(
            source_status="failed",
            preparation_status="unknown",
            reason="mystery_failure",
        ),
        "reason must be one of",
    )


def test_successful_upload_response_rejects_failure_reason() -> None:
    _assert_upload_response_validation_error(
        _upload_response_payload(reason="storage_failure"),
        "successful upload response must not include reason",
    )


def test_upload_response_forbids_heavy_or_route_local_fields() -> None:
    _assert_upload_response_validation_error(
        _upload_response_payload(raw_text="Once upon a time"),
        "Extra inputs are not permitted",
    )
    _assert_upload_response_validation_error(
        _upload_response_payload(task_layout={"chapters": []}),
        "Extra inputs are not permitted",
    )
    _assert_upload_response_validation_error(
        _upload_response_payload(prepare_now=True),
        "Extra inputs are not permitted",
    )


if __name__ == "__main__":
    test_upload_metadata_normalizes_optional_title_and_conflict_policy()
    test_upload_metadata_rejects_unknown_conflict_policy()
    test_upload_response_normalizes_lightweight_source_metadata()
    test_upload_response_requires_failure_reason_for_failed_status()
    test_upload_response_rejects_unknown_failure_reason()
    test_successful_upload_response_rejects_failure_reason()
    test_upload_response_forbids_heavy_or_route_local_fields()
    print("OK: document upload API schema tests passed")
