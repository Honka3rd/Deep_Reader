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
        self.read_response: ReadingInteractionResponseDTO | None = None
        self.generate_response: ReadingInteractionResponseDTO | None = None
        self.submit_response: ReadingInteractionResponseDTO | None = None
        self.retry_response: ReadingInteractionResponseDTO | None = None

    def read_critical_thinking_session(self, **kwargs) -> ReadingInteractionResponseDTO:  # noqa: ANN003
        self.read_calls.append(dict(kwargs))
        if self.read_response is not None:
            return self.read_response
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
        if self.generate_response is not None:
            return self.generate_response
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
        if self.submit_response is not None:
            return self.submit_response
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
        if self.retry_response is not None:
            return self.retry_response
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


def _content_block_target_payload() -> dict[str, object]:
    return {
        "target": {
            "doc_name": " book.json ",
            "target_type": "content_block",
            "chapter_id": " chapter-1 ",
            "section_id": " section-1 ",
            "task_unit_id": " task-unit-1 ",
            "content_block_id": " task-unit-1:block:0 ",
        }
    }


def _content_block_response(
    *,
    status: str,
    payload: dict[str, object],
    metadata: dict[str, object] | None = None,
    reason: str | None = None,
    session_id: str | None = "session-1",
) -> ReadingInteractionResponseDTO:
    return ReadingInteractionResponseDTO(
        target=ReadingInteractionTargetDTO(
            doc_name="book.json",
            document_id="doc-1",
            document_title="Document One",
            target_level="content_block",
            target_id="task-unit-1:block:0",
            chapter_id="chapter-1",
            section_id="section-1",
            task_unit_id="task-unit-1",
            content_block_id="task-unit-1:block:0",
            title="Unit One",
        ),
        interaction_type="critical_thinking_session",
        status=status,
        payload=payload,
        metadata={} if metadata is None else metadata,
        reason=reason,
        session_id=session_id,
    )


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


def test_critical_generic_routes_preserve_content_block_target_identity() -> None:
    fake = _FakeSectionTaskCoordinator()
    fake.read_response = _content_block_response(
        status="question_generated",
        payload={"question": "What does this block assume?"},
        metadata={"output_schema_version": "critical_thinking_session_v1"},
    )
    fake.generate_response = _content_block_response(
        status="question_generated",
        payload={"question": "What does this block assume?"},
        metadata={"output_schema_version": "critical_thinking_session_v1"},
    )
    fake.submit_response = _content_block_response(
        status="evaluation_failed",
        payload={
            "question": "What does this block assume?",
            "answer": "It assumes the selected detail changes the argument.",
        },
        metadata={"output_schema_version": "critical_thinking_session_v1"},
        reason="evaluation payload failed validation",
    )
    fake.retry_response = _content_block_response(
        status="completed",
        payload={
            "question": "What does this block assume?",
            "answer": "It assumes the selected detail changes the argument.",
            "evaluation": {
                "feedback": "The answer stays anchored to the selected block.",
                "strengths": "It identifies a local assumption.",
                "improvements": "It could quote the block directly.",
                "score": 4,
            },
        },
        metadata={"output_schema_version": "critical_thinking_session_v1"},
    )
    original_coordinator = _install_fake(fake)
    try:
        client = TestClient(main.app)
        read_response = client.post(
            "/documents/reading-interactions/critical-thinking/read",
            json={**_content_block_target_payload(), "session_id": " session-1 "},
        )
        generate_response = client.post(
            "/documents/reading-interactions/critical-thinking/generate-question",
            json=_content_block_target_payload(),
        )
        submit_response = client.post(
            "/documents/reading-interactions/critical-thinking/submit-answer",
            json={
                **_content_block_target_payload(),
                "session_id": " session-1 ",
                "answer": " It assumes the selected detail changes the argument. ",
            },
        )
        retry_response = client.post(
            "/documents/reading-interactions/critical-thinking/retry-evaluation",
            json={**_content_block_target_payload(), "session_id": " session-1 "},
        )
    finally:
        _restore_fake(original_coordinator)

    _assert(read_response.status_code == 200, f"unexpected status: {read_response.text}")
    _assert(generate_response.status_code == 200, f"unexpected status: {generate_response.text}")
    _assert(submit_response.status_code == 200, f"unexpected status: {submit_response.text}")
    _assert(retry_response.status_code == 200, f"unexpected status: {retry_response.text}")
    generated = generate_response.json()
    submitted = submit_response.json()
    retried = retry_response.json()
    _assert(
        generated["envelope"]["target"]["target_type"] == "content_block",
        "generate response should preserve content-block target type",
    )
    _assert(
        generated["envelope"]["target"]["content_block_id"]
        == "task-unit-1:block:0",
        "generate response should preserve content_block_id",
    )
    _assert(
        submitted["envelope"]["target"]["content_block_id"]
        == "task-unit-1:block:0",
        "submit response should preserve content_block_id",
    )
    _assert(
        retried["envelope"]["target"]["content_block_id"]
        == "task-unit-1:block:0",
        "retry response should preserve content_block_id",
    )
    _assert(
        fake.read_calls[0]["content_block_id"] == "task-unit-1:block:0",
        "read dispatch should include content_block_id",
    )
    _assert(
        fake.generate_calls[0]["content_block_id"] == "task-unit-1:block:0",
        "generate dispatch should include content_block_id",
    )
    _assert(
        fake.submit_calls[0]["content_block_id"] == "task-unit-1:block:0",
        "submit dispatch should include content_block_id",
    )
    _assert(
        fake.retry_calls[0]["content_block_id"] == "task-unit-1:block:0",
        "retry dispatch should include content_block_id",
    )
    _assert(
        fake.submit_calls[0]["answer"]
        == "It assumes the selected detail changes the argument.",
        "submit answer should normalize before dispatch",
    )


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
    test_critical_generic_routes_preserve_content_block_target_identity()
    test_critical_submit_preserves_answer_on_evaluation_failure()
    test_critical_retry_completes_without_regenerating_question()
    test_critical_submit_rejects_malformed_request_before_dispatch()
    print("critical-thinking interaction route tests passed")


if __name__ == "__main__":
    main_test()
