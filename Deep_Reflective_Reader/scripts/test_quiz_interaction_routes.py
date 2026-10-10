#!/usr/bin/env python3
"""Route regressions for the target-agnostic quiz vertical slice."""

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
        self.legacy_section_quiz_calls: list[dict[str, object]] = []
        self.legacy_chapter_quiz_calls: list[dict[str, object]] = []
        self.read_response: ReadingInteractionResponseDTO | None = None
        self.generate_response: ReadingInteractionResponseDTO | None = None
        self.refresh_response: ReadingInteractionResponseDTO | None = None

    def read_quiz_artifact(self, **kwargs) -> ReadingInteractionResponseDTO:  # noqa: ANN003
        self.read_calls.append(dict(kwargs))
        if self.read_response is not None:
            return self.read_response
        return self._response(status="not_generated", payload={})

    def generate_quiz_artifact(self, **kwargs) -> ReadingInteractionResponseDTO:  # noqa: ANN003
        self.generate_calls.append(dict(kwargs))
        if self.generate_response is not None:
            return self.generate_response
        return self._response(
            status="completed",
            payload={
                "items": [
                    {
                        "type": "short_answer",
                        "question": "What drives behavior?",
                        "answer": "Incentives",
                        "explanation": "The passage centers incentives.",
                    }
                ]
            },
            metadata={
                "artifact_id": "quiz::doc-1::section::section-1",
                "output_schema_version": "quiz_interaction_v1",
                "prompt_instruction_version": kwargs.get("prompt_instruction_version"),
                "max_items": 5,
            },
            artifact_id="quiz::doc-1::section::section-1",
        )

    def refresh_quiz_artifact(self, **kwargs) -> ReadingInteractionResponseDTO:  # noqa: ANN003
        self.refresh_calls.append(dict(kwargs))
        if self.refresh_response is not None:
            return self.refresh_response
        return self._response(
            status="completed",
            payload={
                "items": [
                    {
                        "type": "multiple_choice",
                        "question": "Which idea is central?",
                        "choices": ["Incentives", "Weather", "Typography"],
                        "answer": "Incentives",
                    },
                    {
                        "type": "true_false",
                        "question": "The passage discusses causes and effects.",
                        "answer": True,
                    },
                ]
            },
            metadata={
                "artifact_id": "quiz::doc-1::section::section-1",
                "output_schema_version": "quiz_interaction_v1",
                "prompt_instruction_version": kwargs.get("prompt_instruction_version"),
                "max_items": 5,
            },
            artifact_id="quiz::doc-1::section::section-1",
        )

    def generate_section_quiz(self, **kwargs) -> object:  # noqa: ANN003
        self.legacy_section_quiz_calls.append(dict(kwargs))
        raise AssertionError("generic quiz routes must not call legacy section quiz")

    def generate_chapter_quiz(self, **kwargs) -> object:  # noqa: ANN003
        self.legacy_chapter_quiz_calls.append(dict(kwargs))
        raise AssertionError("generic quiz routes must not call legacy chapter quiz")

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
            interaction_type="quiz",
            status=status,
            payload=payload,
            metadata={} if metadata is None else metadata,
            reason=reason,
            artifact_id=artifact_id,
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


def _install_fake(fake: _FakeSectionTaskCoordinator) -> object:
    original_coordinator = main.section_task_coordinator
    main.section_task_coordinator = fake
    return original_coordinator


def _restore_fake(original_coordinator: object) -> None:
    main.section_task_coordinator = original_coordinator


def test_quiz_read_route_returns_missing_without_generation() -> None:
    fake = _FakeSectionTaskCoordinator()
    original_coordinator = _install_fake(fake)
    try:
        client = TestClient(main.app)
        response = client.post(
            "/documents/reading-interactions/quiz/read",
            json=_section_target_payload(),
        )
    finally:
        _restore_fake(original_coordinator)

    _assert(response.status_code == 200, f"unexpected status: {response.text}")
    payload = response.json()
    _assert(payload["envelope"]["status"] == "not_generated", "missing read should map")
    _assert(payload["payload"] is None, "not_generated should not synthesize payload")
    _assert(len(fake.read_calls) == 1, "read route should dispatch to quiz read")
    _assert(fake.generate_calls == [], "read route must not dispatch generate")
    _assert(fake.refresh_calls == [], "read route must not dispatch refresh")
    _assert(fake.legacy_section_quiz_calls == [], "generic route must not call legacy section quiz")
    _assert(fake.legacy_chapter_quiz_calls == [], "generic route must not call legacy chapter quiz")
    _assert(fake.read_calls[0]["doc_name"] == "book.json", "target should normalize")


def test_quiz_generate_route_maps_completed_payload() -> None:
    fake = _FakeSectionTaskCoordinator()
    original_coordinator = _install_fake(fake)
    try:
        client = TestClient(main.app)
        response = client.post(
            "/documents/reading-interactions/quiz/generate",
            json={
                **_section_target_payload(),
                "prompt_instruction_version": " quiz_prompt_v1 ",
                "max_items": 5,
            },
        )
    finally:
        _restore_fake(original_coordinator)

    _assert(response.status_code == 200, f"unexpected status: {response.text}")
    payload = response.json()
    _assert(payload["envelope"]["interaction_type"] == "quiz", "type should map")
    _assert(payload["envelope"]["status"] == "completed", "status should map")
    _assert(
        payload["envelope"]["schema_version"] == "quiz_interaction_v1",
        "schema version should map",
    )
    _assert(
        payload["envelope"]["prompt_instruction_version"] == "quiz_prompt_v1",
        "prompt version should normalize and map",
    )
    first_item = payload["payload"]["items"][0]
    _assert(first_item["item_id"] == "q1", "route should assign stable item ids")
    _assert(first_item["item_type"] == "short_answer", "type should map")
    _assert(first_item["prompt"] == "What drives behavior?", "question should map")
    _assert(first_item["answer"] == "Incentives", "answer should map")
    _assert(payload["payload"]["max_items"] == 5, "max item policy should map")
    _assert(len(fake.generate_calls) == 1, "generate route should dispatch generate")
    _assert(
        fake.generate_calls[0]["prompt_instruction_version"] == "quiz_prompt_v1",
        "prompt version should reach app orchestration",
    )
    _assert(fake.legacy_section_quiz_calls == [], "must not call legacy section quiz")
    _assert(fake.legacy_chapter_quiz_calls == [], "must not call legacy chapter quiz")


def test_quiz_refresh_route_dispatches_explicit_refresh() -> None:
    fake = _FakeSectionTaskCoordinator()
    original_coordinator = _install_fake(fake)
    try:
        client = TestClient(main.app)
        response = client.post(
            "/documents/reading-interactions/quiz/refresh",
            json={
                **_section_target_payload(),
                "prompt_instruction_version": " quiz_prompt_v2 ",
            },
        )
    finally:
        _restore_fake(original_coordinator)

    _assert(response.status_code == 200, f"unexpected status: {response.text}")
    payload = response.json()
    items = payload["payload"]["items"]
    _assert(items[0]["item_type"] == "multiple_choice", "multiple choice should map")
    _assert(
        items[0]["options"] == ["Incentives", "Weather", "Typography"],
        "choices should map to options",
    )
    _assert(items[1]["item_type"] == "true_false", "true/false should map")
    _assert(items[1]["answer"] is True, "boolean answer should preserve")
    _assert(fake.generate_calls == [], "refresh route should not call generate method")
    _assert(len(fake.refresh_calls) == 1, "refresh route should dispatch refresh")
    _assert(
        fake.refresh_calls[0]["prompt_instruction_version"] == "quiz_prompt_v2",
        "refresh prompt version should reach app orchestration",
    )


def test_quiz_routes_reject_invalid_request_before_dispatch() -> None:
    fake = _FakeSectionTaskCoordinator()
    original_coordinator = _install_fake(fake)
    try:
        client = TestClient(main.app)
        response = client.post(
            "/documents/reading-interactions/quiz/generate",
            json={**_section_target_payload(), "max_items": 999},
        )
    finally:
        _restore_fake(original_coordinator)

    _assert(response.status_code == 422, f"unexpected status: {response.text}")
    _assert(fake.read_calls == [], "invalid request should not dispatch read")
    _assert(fake.generate_calls == [], "invalid request should not dispatch generate")
    _assert(fake.refresh_calls == [], "invalid request should not dispatch refresh")
    _assert(fake.legacy_section_quiz_calls == [], "invalid generic route must not call legacy section quiz")
    _assert(fake.legacy_chapter_quiz_calls == [], "invalid generic route must not call legacy chapter quiz")


def test_quiz_generate_route_rejects_invalid_item_shape() -> None:
    fake = _FakeSectionTaskCoordinator()
    fake.generate_response = fake._response(
        status="completed",
        payload={
            "items": [
                {
                    "type": "essay",
                    "question": "Unsupported item?",
                    "answer": "No",
                }
            ]
        },
        metadata={
            "artifact_id": "quiz::doc-1::section::section-1",
            "output_schema_version": "quiz_interaction_v1",
            "max_items": 5,
        },
        artifact_id="quiz::doc-1::section::section-1",
    )
    original_coordinator = _install_fake(fake)
    try:
        client = TestClient(main.app)
        response = client.post(
            "/documents/reading-interactions/quiz/generate",
            json=_section_target_payload(),
        )
    finally:
        _restore_fake(original_coordinator)

    _assert(response.status_code == 422, f"unexpected status: {response.text}")
    _assert(len(fake.generate_calls) == 1, "schema-valid request should reach generate")
    _assert(
        "item_type must be one of" in response.text,
        "invalid quiz item shape should surface validation detail",
    )


def main_test() -> None:
    test_quiz_read_route_returns_missing_without_generation()
    test_quiz_generate_route_maps_completed_payload()
    test_quiz_refresh_route_dispatches_explicit_refresh()
    test_quiz_routes_reject_invalid_request_before_dispatch()
    test_quiz_generate_route_rejects_invalid_item_shape()
    print("quiz interaction route tests passed")


if __name__ == "__main__":
    main_test()
