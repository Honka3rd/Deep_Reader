#!/usr/bin/env python3
"""Route regressions for critical-thinking session interactions."""

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
        self.submit_calls: list[dict[str, object]] = []
        self.retry_calls: list[dict[str, object]] = []

    def read_critical_thinking_session(self, **kwargs) -> ReadingInteractionResponseDTO:  # noqa: ANN003
        self.read_calls.append(dict(kwargs))
        if kwargs.get("session_id") == "session-1":
            return self._response(
                status="question_generated",
                payload={"question": "Which assumption matters most?"},
                metadata={
                    "output_schema_version": "critical_thinking_session_v1",
                    "prompt_instruction_version": "critical_prompt_v1",
                },
                session_id="session-1",
            )
        return self._response(status="not_generated", payload={})

    def generate_critical_thinking_question(self, **kwargs) -> ReadingInteractionResponseDTO:  # noqa: ANN003
        self.generate_calls.append(dict(kwargs))
        return self._response(
            status="question_generated",
            payload={"question": "Which assumption matters most?"},
            metadata={
                "output_schema_version": "critical_thinking_session_v1",
                "prompt_instruction_version": kwargs.get("prompt_instruction_version"),
            },
            session_id="session-1",
        )

    def submit_critical_thinking_answer(self, **kwargs) -> ReadingInteractionResponseDTO:  # noqa: ANN003
        self.submit_calls.append(dict(kwargs))
        return self._response(
            status="evaluation_failed",
            payload={
                "question": "Which assumption matters most?",
                "answer": kwargs.get("answer"),
            },
            metadata={"output_schema_version": "critical_thinking_session_v1"},
            reason="evaluation payload failed validation",
            session_id=str(kwargs.get("session_id")),
        )

    def retry_critical_thinking_evaluation(self, **kwargs) -> ReadingInteractionResponseDTO:  # noqa: ANN003
        self.retry_calls.append(dict(kwargs))
        return self._response(
            status="completed",
            payload={
                "question": "Which assumption matters most?",
                "answer": "Institutions change the incentive effect.",
                "evaluation": {
                    "feedback": "The answer identifies a relevant assumption.",
                    "strengths": "It connects assumptions to outcomes.",
                    "improvements": "It could cite the source more directly.",
                    "score": 4,
                },
            },
            metadata={"output_schema_version": "critical_thinking_session_v1"},
            session_id=str(kwargs.get("session_id")),
        )

    @staticmethod
    def _response(
        *,
        status: str,
        payload: dict[str, object],
        metadata: dict[str, object] | None = None,
        reason: str | None = None,
        session_id: str | None = None,
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
            interaction_type="critical_thinking_session",
            status=status,
            payload=payload,
            metadata={} if metadata is None else metadata,
            reason=reason,
            session_id=session_id,
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


def test_critical_read_route_returns_missing_without_generation() -> None:
    fake = _FakeSectionTaskCoordinator()
    original_coordinator = _install_fake(fake)
    try:
        client = TestClient(main.app)
        response = client.post(
            "/documents/reading-interactions/critical-thinking/read",
            json=_section_target_payload(),
        )
    finally:
        _restore_fake(original_coordinator)

    _assert(response.status_code == 200, f"unexpected status: {response.text}")
    payload = response.json()
    _assert(payload["envelope"]["status"] == "not_generated", "missing read should map")
    _assert(payload["envelope"]["session_id"] is None, "missing read should not include session")
    _assert(payload["payload"] is None, "missing read should not synthesize payload")
    _assert(len(fake.read_calls) == 1, "read route should dispatch read")
    _assert(fake.generate_calls == [], "read route must not generate a question")
    _assert(fake.submit_calls == [], "read route must not submit an answer")
    _assert(fake.retry_calls == [], "read route must not retry evaluation")


def test_critical_generate_question_route_maps_session() -> None:
    fake = _FakeSectionTaskCoordinator()
    original_coordinator = _install_fake(fake)
    try:
        client = TestClient(main.app)
        response = client.post(
            "/documents/reading-interactions/critical-thinking/generate-question",
            json={
                **_section_target_payload(),
                "prompt_instruction_version": " critical_prompt_v1 ",
            },
        )
    finally:
        _restore_fake(original_coordinator)

    _assert(response.status_code == 200, f"unexpected status: {response.text}")
    payload = response.json()
    _assert(payload["envelope"]["status"] == "question_generated", "status should map")
    _assert(payload["envelope"]["session_id"] == "session-1", "session id should map")
    _assert(
        payload["envelope"]["prompt_instruction_version"] == "critical_prompt_v1",
        "prompt version should normalize and map",
    )
    _assert(
        payload["payload"]["question"] == "Which assumption matters most?",
        "question should map",
    )
    _assert(len(fake.generate_calls) == 1, "generate route should dispatch generation")
    _assert(
        fake.generate_calls[0]["prompt_instruction_version"] == "critical_prompt_v1",
        "prompt version should reach app orchestration",
    )


def test_critical_read_route_loads_existing_session() -> None:
    fake = _FakeSectionTaskCoordinator()
    original_coordinator = _install_fake(fake)
    try:
        client = TestClient(main.app)
        response = client.post(
            "/documents/reading-interactions/critical-thinking/read",
            json={**_section_target_payload(), "session_id": " session-1 "},
        )
    finally:
        _restore_fake(original_coordinator)

    _assert(response.status_code == 200, f"unexpected status: {response.text}")
    payload = response.json()
    _assert(payload["envelope"]["status"] == "question_generated", "existing session should map")
    _assert(payload["envelope"]["session_id"] == "session-1", "session id should map")
    _assert(fake.read_calls[0]["session_id"] == "session-1", "session id should trim")


def test_critical_submit_preserves_answer_on_evaluation_failure() -> None:
    fake = _FakeSectionTaskCoordinator()
    original_coordinator = _install_fake(fake)
    try:
        client = TestClient(main.app)
        response = client.post(
            "/documents/reading-interactions/critical-thinking/submit-answer",
            json={
                **_section_target_payload(),
                "session_id": " session-1 ",
                "answer": " Institutions change the incentive effect. ",
            },
        )
    finally:
        _restore_fake(original_coordinator)

    _assert(response.status_code == 200, f"unexpected status: {response.text}")
    payload = response.json()
    _assert(payload["envelope"]["status"] == "evaluation_failed", "failure should map")
    _assert(
        payload["payload"]["submitted_answer"] == "Institutions change the incentive effect.",
        "failed evaluation should preserve submitted answer",
    )
    _assert(payload["payload"]["retry_eligible"] is True, "failure should allow retry")
    _assert(len(fake.submit_calls) == 1, "submit route should dispatch submit")
    _assert(fake.generate_calls == [], "submit must not regenerate the question")
    _assert(
        fake.submit_calls[0]["answer"] == "Institutions change the incentive effect.",
        "answer should normalize before app dispatch",
    )


def test_critical_retry_completes_without_regenerating_question() -> None:
    fake = _FakeSectionTaskCoordinator()
    original_coordinator = _install_fake(fake)
    try:
        client = TestClient(main.app)
        response = client.post(
            "/documents/reading-interactions/critical-thinking/retry-evaluation",
            json={**_section_target_payload(), "session_id": " session-1 "},
        )
    finally:
        _restore_fake(original_coordinator)

    _assert(response.status_code == 200, f"unexpected status: {response.text}")
    payload = response.json()
    _assert(payload["envelope"]["status"] == "completed", "completed retry should map")
    _assert(
        payload["payload"]["evaluation"]["feedback"]
        == "The answer identifies a relevant assumption.",
        "evaluation feedback should map",
    )
    _assert(payload["payload"]["retry_eligible"] is False, "completed retry should close retry")
    _assert(len(fake.retry_calls) == 1, "retry route should dispatch retry")
    _assert(fake.generate_calls == [], "retry must not regenerate the question")


def test_critical_submit_rejects_malformed_request_before_dispatch() -> None:
    fake = _FakeSectionTaskCoordinator()
    original_coordinator = _install_fake(fake)
    try:
        client = TestClient(main.app)
        response = client.post(
            "/documents/reading-interactions/critical-thinking/submit-answer",
            json={**_section_target_payload(), "session_id": "session-1", "answer": "  "},
        )
    finally:
        _restore_fake(original_coordinator)

    _assert(response.status_code == 422, f"unexpected status: {response.text}")
    _assert(fake.submit_calls == [], "malformed submit should not dispatch")
    _assert(fake.retry_calls == [], "malformed submit should not retry")
    _assert(fake.generate_calls == [], "malformed submit should not generate")


def main_test() -> None:
    test_critical_read_route_returns_missing_without_generation()
    test_critical_generate_question_route_maps_session()
    test_critical_read_route_loads_existing_session()
    test_critical_submit_preserves_answer_on_evaluation_failure()
    test_critical_retry_completes_without_regenerating_question()
    test_critical_submit_rejects_malformed_request_before_dispatch()
    print("critical-thinking interaction route tests passed")


if __name__ == "__main__":
    main_test()
