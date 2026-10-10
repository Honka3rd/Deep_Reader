#!/usr/bin/env python3
"""Route regressions for the inline insight/analysis vertical slice."""

from __future__ import annotations

from pathlib import Path
import sys

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import main  # noqa: E402
from app.section_task_coordinator import (  # noqa: E402
    ReadingInteractionResponseDTO,
    ReadingInteractionTargetDTO,
)


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


class _FakeSectionTaskCoordinator:
    def __init__(self) -> None:
        self.read_calls: list[dict[str, object]] = []
        self.generate_calls: list[dict[str, object]] = []
        self.refresh_calls: list[dict[str, object]] = []
        self.read_error: Exception | None = None
        self.generate_error: Exception | None = None
        self.refresh_error: Exception | None = None
        self.read_response: ReadingInteractionResponseDTO | None = None
        self.generate_response: ReadingInteractionResponseDTO | None = None
        self.refresh_response: ReadingInteractionResponseDTO | None = None

    def read_analysis_artifact(self, **kwargs) -> ReadingInteractionResponseDTO:  # noqa: ANN003
        self.read_calls.append(dict(kwargs))
        if self.read_error is not None:
            raise self.read_error
        if self.read_response is not None:
            return self.read_response
        return self._response(status="not_generated", payload={})

    def generate_analysis_artifact(self, **kwargs) -> ReadingInteractionResponseDTO:  # noqa: ANN003
        self.generate_calls.append(dict(kwargs))
        if self.generate_error is not None:
            raise self.generate_error
        if self.generate_response is not None:
            return self.generate_response
        return self._response(
            status="completed",
            payload={
                "summary": "Generated insight",
                "reasoning": "The passage links assumptions to conclusions.",
                "explanation": "It reads the target as an argument.",
            },
            metadata={
                "artifact_id": "analysis::doc-1::section::section-1",
                "output_schema_version": "analysis_interaction_v1",
                "prompt_instruction_version": kwargs.get("prompt_instruction_version"),
            },
            artifact_id="analysis::doc-1::section::section-1",
        )

    def refresh_analysis_artifact(self, **kwargs) -> ReadingInteractionResponseDTO:  # noqa: ANN003
        self.refresh_calls.append(dict(kwargs))
        if self.refresh_error is not None:
            raise self.refresh_error
        if self.refresh_response is not None:
            return self.refresh_response
        return self._response(
            status="completed",
            payload={
                "summary": "Refreshed insight",
                "reasoning": "Refresh rebuilt the interpretation.",
                "interpretation": "The target emphasizes changed assumptions.",
                "explanation": "It replaces the current artifact.",
                "key_points": ["assumptions", "conclusions"],
            },
            metadata={
                "artifact_id": "analysis::doc-1::section::section-1",
                "output_schema_version": "analysis_interaction_v1",
                "prompt_instruction_version": kwargs.get("prompt_instruction_version"),
            },
            artifact_id="analysis::doc-1::section::section-1",
        )

    @staticmethod
    def _response(
        *,
        status: str,
        payload: dict[str, object],
        metadata: dict[str, object] | None = None,
        artifact_id: str | None = None,
        reason: str | None = None,
    ) -> ReadingInteractionResponseDTO:
        return ReadingInteractionResponseDTO(
            target=ReadingInteractionTargetDTO(
                doc_name="book.json",
                document_id="doc-1",
                document_title="Document One",
                target_level="section",
                target_id="section-1",
                chapter_id="chapter-1",
                section_id="section-1",
                title="Section One",
            ),
            interaction_type="analysis",
            status=status,
            payload=payload,
            metadata={} if metadata is None else metadata,
            reason=reason,
            artifact_id=artifact_id,
        )


class _ReadOnlyPoisonSectionTaskCoordinator:
    def __init__(self) -> None:
        self.read_calls: list[dict[str, object]] = []

    def read_analysis_artifact(self, **kwargs) -> ReadingInteractionResponseDTO:  # noqa: ANN003
        self.read_calls.append(dict(kwargs))
        return _FakeSectionTaskCoordinator._response(status="not_generated", payload={})

    def __getattr__(self, name: str) -> object:
        raise AssertionError(
            f"read route attempted forbidden coordinator access: {name}"
        )


def _section_target_payload() -> dict[str, object]:
    return {
        "target": {
            "doc_name": " book.json ",
            "target_type": "section",
            "chapter_id": " chapter-1 ",
            "section_id": " section-1 ",
        }
    }


def test_analysis_read_route_does_not_generate() -> None:
    original_coordinator = main.section_task_coordinator
    fake = _FakeSectionTaskCoordinator()
    main.section_task_coordinator = fake
    try:
        client = TestClient(main.app)
        response = client.post(
            "/documents/reading-interactions/insight/read",
            json=_section_target_payload(),
        )
    finally:
        main.section_task_coordinator = original_coordinator

    _assert(response.status_code == 200, f"unexpected status: {response.text}")
    payload = response.json()
    _assert(payload["envelope"]["status"] == "not_generated", "missing read should be stable")
    _assert(payload["payload"] is None, "not_generated response should not include payload")
    _assert(len(fake.read_calls) == 1, "read route should dispatch to app read")
    _assert(fake.generate_calls == [], "read route must not dispatch generate")
    _assert(fake.refresh_calls == [], "read route must not dispatch refresh")
    _assert(
        fake.read_calls[0]["doc_name"] == "book.json",
        "schema-normalized doc_name should reach coordinator",
    )
    _assert(
        fake.read_calls[0]["target_level"] == "section",
        "target_type should map to coordinator target_level",
    )


def test_analysis_read_route_has_no_costly_or_mutating_side_effects() -> None:
    original_coordinator = main.section_task_coordinator
    poison = _ReadOnlyPoisonSectionTaskCoordinator()
    main.section_task_coordinator = poison
    try:
        client = TestClient(main.app)
        response = client.post(
            "/documents/reading-interactions/insight/read",
            json=_section_target_payload(),
        )
    finally:
        main.section_task_coordinator = original_coordinator

    _assert(response.status_code == 200, f"unexpected status: {response.text}")
    payload = response.json()
    _assert(
        payload["envelope"]["status"] == "not_generated",
        "absent artifact reads should return a stable not_generated state",
    )
    _assert(payload["payload"] is None, "missing reads should not synthesize payloads")
    _assert(
        len(poison.read_calls) == 1,
        "read route should only dispatch through read_analysis_artifact",
    )
    _assert(
        poison.read_calls[0]["doc_name"] == "book.json",
        "read-only dispatch should still receive normalized target identity",
    )


def test_analysis_generate_route_maps_completed_payload() -> None:
    original_coordinator = main.section_task_coordinator
    fake = _FakeSectionTaskCoordinator()
    main.section_task_coordinator = fake
    try:
        client = TestClient(main.app)
        response = client.post(
            "/documents/reading-interactions/insight/generate",
            json={
                **_section_target_payload(),
                "prompt_instruction_version": " analysis_prompt_v1 ",
            },
        )
    finally:
        main.section_task_coordinator = original_coordinator

    _assert(response.status_code == 200, f"unexpected status: {response.text}")
    payload = response.json()
    _assert(payload["envelope"]["interaction_type"] == "analysis", "type should map")
    _assert(payload["envelope"]["status"] == "completed", "status should map")
    _assert(
        payload["envelope"]["artifact_id"] == "analysis::doc-1::section::section-1",
        "artifact id should map",
    )
    _assert(
        payload["envelope"]["schema_version"] == "analysis_interaction_v1",
        "schema version should map from metadata",
    )
    _assert(
        payload["envelope"]["prompt_instruction_version"] == "analysis_prompt_v1",
        "prompt version should be schema-normalized",
    )
    _assert(payload["payload"]["summary"] == "Generated insight", "summary should map")
    _assert(
        payload["payload"]["interpretation"]
        == "The passage links assumptions to conclusions.",
        "service reasoning should provide interpretation fallback",
    )
    _assert(len(fake.generate_calls) == 1, "generate route should dispatch generate")
    _assert(
        fake.generate_calls[0]["prompt_instruction_version"] == "analysis_prompt_v1",
        "prompt version should reach app orchestration",
    )


def test_analysis_refresh_route_dispatches_explicit_refresh() -> None:
    original_coordinator = main.section_task_coordinator
    fake = _FakeSectionTaskCoordinator()
    main.section_task_coordinator = fake
    try:
        client = TestClient(main.app)
        response = client.post(
            "/documents/reading-interactions/insight/refresh",
            json={
                **_section_target_payload(),
                "prompt_instruction_version": " analysis_prompt_v2 ",
            },
        )
    finally:
        main.section_task_coordinator = original_coordinator

    _assert(response.status_code == 200, f"unexpected status: {response.text}")
    payload = response.json()
    _assert(payload["payload"]["summary"] == "Refreshed insight", "refresh payload should map")
    _assert(
        payload["payload"]["interpretation"]
        == "The target emphasizes changed assumptions.",
        "explicit interpretation should be preserved",
    )
    _assert(
        payload["payload"]["key_points"] == ["assumptions", "conclusions"],
        "key points should map",
    )
    _assert(fake.generate_calls == [], "refresh route should not call generate route method")
    _assert(len(fake.refresh_calls) == 1, "refresh route should dispatch refresh")
    _assert(
        fake.refresh_calls[0]["prompt_instruction_version"] == "analysis_prompt_v2",
        "refresh prompt version should reach app orchestration",
    )


def test_analysis_routes_reject_malformed_request_before_dispatch() -> None:
    original_coordinator = main.section_task_coordinator
    fake = _FakeSectionTaskCoordinator()
    main.section_task_coordinator = fake
    try:
        client = TestClient(main.app)
        response = client.post(
            "/documents/reading-interactions/insight/read",
            json={"target": {"doc_name": "book.json", "target_type": "section"}},
        )
    finally:
        main.section_task_coordinator = original_coordinator

    _assert(response.status_code == 422, f"unexpected status: {response.text}")
    _assert(fake.read_calls == [], "malformed request should not reach app read")
    _assert(fake.generate_calls == [], "malformed request should not reach generate")
    _assert(fake.refresh_calls == [], "malformed request should not reach refresh")


def test_analysis_read_route_maps_missing_target_to_404() -> None:
    original_coordinator = main.section_task_coordinator
    fake = _FakeSectionTaskCoordinator()
    fake.read_error = ValueError("section_id 'missing-section' not found in document 'doc-1'")
    main.section_task_coordinator = fake
    try:
        client = TestClient(main.app)
        response = client.post(
            "/documents/reading-interactions/insight/read",
            json=_section_target_payload(),
        )
    finally:
        main.section_task_coordinator = original_coordinator

    _assert(response.status_code == 404, f"unexpected status: {response.text}")
    _assert(len(fake.read_calls) == 1, "well-formed missing target should reach app read")
    _assert(fake.generate_calls == [], "missing read target should not generate")


def test_analysis_generate_route_maps_recoverable_status_http_policy() -> None:
    original_coordinator = main.section_task_coordinator
    fake = _FakeSectionTaskCoordinator()
    fake.generate_response = fake._response(
        status="insufficient_content",
        payload={},
        reason="target content is too short for analysis",
    )
    main.section_task_coordinator = fake
    try:
        client = TestClient(main.app)
        response = client.post(
            "/documents/reading-interactions/insight/generate",
            json=_section_target_payload(),
        )
    finally:
        main.section_task_coordinator = original_coordinator

    _assert(response.status_code == 200, f"unexpected status: {response.text}")
    payload = response.json()
    _assert(
        payload["envelope"]["status"] == "insufficient_content",
        "recoverable terminal status should stay in envelope",
    )
    _assert(
        payload["envelope"]["reason"] == "target content is too short for analysis",
        "reason should map for insufficient content",
    )
    _assert(payload["payload"] is None, "insufficient content should not include payload")


def test_analysis_generate_route_maps_stale_target_to_409() -> None:
    original_coordinator = main.section_task_coordinator
    fake = _FakeSectionTaskCoordinator()
    fake.generate_response = fake._response(
        status="stale_target",
        payload={},
        reason="target source structure is stale",
    )
    main.section_task_coordinator = fake
    try:
        client = TestClient(main.app)
        response = client.post(
            "/documents/reading-interactions/insight/generate",
            json=_section_target_payload(),
        )
    finally:
        main.section_task_coordinator = original_coordinator

    _assert(response.status_code == 409, f"unexpected status: {response.text}")
    _assert(
        response.json()["envelope"]["status"] == "stale_target",
        "stale target status should stay in envelope",
    )


def test_analysis_generate_route_maps_generation_failed_to_502() -> None:
    original_coordinator = main.section_task_coordinator
    fake = _FakeSectionTaskCoordinator()
    fake.generate_response = fake._response(
        status="generation_failed",
        payload={},
        reason="model output was invalid JSON",
    )
    main.section_task_coordinator = fake
    try:
        client = TestClient(main.app)
        response = client.post(
            "/documents/reading-interactions/insight/generate",
            json=_section_target_payload(),
        )
    finally:
        main.section_task_coordinator = original_coordinator

    _assert(response.status_code == 502, f"unexpected status: {response.text}")
    _assert(
        response.json()["envelope"]["status"] == "generation_failed",
        "generation failure status should stay in envelope",
    )


def test_analysis_refresh_route_maps_validation_failed_to_422() -> None:
    original_coordinator = main.section_task_coordinator
    fake = _FakeSectionTaskCoordinator()
    fake.refresh_response = fake._response(
        status="validation_failed",
        payload={},
        reason="analysis payload failed response validation",
    )
    main.section_task_coordinator = fake
    try:
        client = TestClient(main.app)
        response = client.post(
            "/documents/reading-interactions/insight/refresh",
            json=_section_target_payload(),
        )
    finally:
        main.section_task_coordinator = original_coordinator

    _assert(response.status_code == 422, f"unexpected status: {response.text}")
    _assert(
        response.json()["envelope"]["status"] == "validation_failed",
        "validation failure status should stay in envelope",
    )


def main_test() -> None:
    test_analysis_read_route_does_not_generate()
    test_analysis_read_route_has_no_costly_or_mutating_side_effects()
    test_analysis_generate_route_maps_completed_payload()
    test_analysis_refresh_route_dispatches_explicit_refresh()
    test_analysis_routes_reject_malformed_request_before_dispatch()
    test_analysis_read_route_maps_missing_target_to_404()
    test_analysis_generate_route_maps_recoverable_status_http_policy()
    test_analysis_generate_route_maps_stale_target_to_409()
    test_analysis_generate_route_maps_generation_failed_to_502()
    test_analysis_refresh_route_maps_validation_failed_to_422()
    print("analysis interaction route tests passed")


if __name__ == "__main__":
    main_test()
