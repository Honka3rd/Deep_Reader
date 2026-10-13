#!/usr/bin/env python3
"""Schema regression tests for document lifecycle API contracts."""

from __future__ import annotations

from pathlib import Path
import sys

from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api_schemas import (  # noqa: E402
    DocumentLifecycleActionRequest,
    DocumentLifecycleActionResponse,
)


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _assert_request_validation_error(
    payload: dict[str, object],
    expected_message: str,
) -> None:
    try:
        DocumentLifecycleActionRequest.model_validate(payload)
    except ValidationError as error:
        _assert(
            expected_message in str(error),
            f"expected {expected_message!r} in request validation error: {error}",
        )
        return
    raise AssertionError(
        f"expected request validation error containing {expected_message!r}"
    )


def _assert_response_validation_error(
    payload: dict[str, object],
    expected_message: str,
) -> None:
    try:
        DocumentLifecycleActionResponse.model_validate(payload)
    except ValidationError as error:
        _assert(
            expected_message in str(error),
            f"expected {expected_message!r} in response validation error: {error}",
        )
        return
    raise AssertionError(
        f"expected response validation error containing {expected_message!r}"
    )


def _request_payload(**updates: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "doc_name": " Madame Bovary ",
        "action": "archive",
        "expected_source_hash": " source-hash ",
        "expected_structure_version": " 7 ",
        "confirm": True,
    }
    payload.update(updates)
    return payload


def _response_payload(**updates: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "doc_name": " Madame Bovary ",
        "action": "archive",
        "status": "accepted",
        "source_affected": True,
        "structured_artifacts_affected": False,
        "derived_resources_affected": True,
        "archived": True,
        "deleted": False,
    }
    payload.update(updates)
    return payload


def test_lifecycle_request_normalizes_action_and_optional_evidence() -> None:
    request = DocumentLifecycleActionRequest.model_validate(
        _request_payload(action=" ARCHIVE ")
    )

    _assert(request.doc_name == "Madame Bovary", "doc_name should be stripped")
    _assert(request.action == "archive", "action should be normalized")
    _assert(
        request.expected_source_hash == "source-hash",
        "expected source hash should be stripped",
    )
    _assert(
        request.expected_structure_version == "7",
        "expected structure version should be stripped",
    )


def test_lifecycle_request_requires_rename_target() -> None:
    _assert_request_validation_error(
        _request_payload(action="rename"),
        "rename action requires new_doc_name",
    )

    request = DocumentLifecycleActionRequest.model_validate(
        _request_payload(action="rename", new_doc_name=" Madame Bovary 2 ")
    )
    _assert(request.action == "rename", "rename action should pass")
    _assert(
        request.new_doc_name == "Madame Bovary 2",
        "new_doc_name should be stripped",
    )


def test_lifecycle_request_rejects_route_local_or_heavy_fields() -> None:
    _assert_request_validation_error(
        _request_payload(action="delete", new_doc_name="Not Allowed"),
        "new_doc_name is only valid for rename action",
    )
    _assert_request_validation_error(
        _request_payload(raw_text="Once upon a time"),
        "Extra inputs are not permitted",
    )
    _assert_request_validation_error(
        _request_payload(task_layout={"chapters": []}),
        "Extra inputs are not permitted",
    )
    _assert_request_validation_error(
        _request_payload(artifacts=[]),
        "Extra inputs are not permitted",
    )


def test_lifecycle_response_normalizes_lightweight_status() -> None:
    response = DocumentLifecycleActionResponse.model_validate(
        _response_payload(action="archive", status=" ACCEPTED ")
    )

    _assert(response.doc_name == "Madame Bovary", "doc_name should be stripped")
    _assert(response.action == "archive", "action should be normalized")
    _assert(response.status == "accepted", "status should be normalized")
    _assert(response.source_affected is True, "source effect should be explicit")
    _assert(
        response.derived_resources_affected is True,
        "derived-resource effect should be explicit",
    )


def test_lifecycle_response_requires_reason_for_terminal_failure_status() -> None:
    _assert_response_validation_error(
        _response_payload(status="failed"),
        "failed lifecycle response requires reason",
    )

    response = DocumentLifecycleActionResponse.model_validate(
        _response_payload(status="conflict", reason="conflict")
    )
    _assert(response.reason == "conflict", "conflict reason should pass")


def test_lifecycle_response_rejects_unknown_reason_or_success_reason() -> None:
    _assert_response_validation_error(
        _response_payload(status="failed", reason="mystery_failure"),
        "reason must be one of",
    )
    _assert_response_validation_error(
        _response_payload(status="accepted", reason="storage_failure"),
        "accepted lifecycle response must not include reason",
    )


def test_lifecycle_response_rename_target_rules() -> None:
    _assert_response_validation_error(
        _response_payload(action="rename"),
        "accepted rename response requires new_doc_name",
    )

    response = DocumentLifecycleActionResponse.model_validate(
        _response_payload(action="rename", new_doc_name=" Madame Bovary 2 ")
    )
    _assert(
        response.new_doc_name == "Madame Bovary 2",
        "accepted rename should expose normalized new_doc_name",
    )

    _assert_response_validation_error(
        _response_payload(action="delete", new_doc_name="Not Allowed"),
        "new_doc_name is only valid for rename action",
    )


def test_lifecycle_response_forbids_heavy_payload_fields() -> None:
    _assert_response_validation_error(
        _response_payload(raw_text="Once upon a time"),
        "Extra inputs are not permitted",
    )
    _assert_response_validation_error(
        _response_payload(hierarchy={"chapters": []}),
        "Extra inputs are not permitted",
    )
    _assert_response_validation_error(
        _response_payload(content_blocks=[]),
        "Extra inputs are not permitted",
    )


if __name__ == "__main__":
    test_lifecycle_request_normalizes_action_and_optional_evidence()
    test_lifecycle_request_requires_rename_target()
    test_lifecycle_request_rejects_route_local_or_heavy_fields()
    test_lifecycle_response_normalizes_lightweight_status()
    test_lifecycle_response_requires_reason_for_terminal_failure_status()
    test_lifecycle_response_rejects_unknown_reason_or_success_reason()
    test_lifecycle_response_rename_target_rules()
    test_lifecycle_response_forbids_heavy_payload_fields()
    print("OK: document lifecycle API schema tests passed")
