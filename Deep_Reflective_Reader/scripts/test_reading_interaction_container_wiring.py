#!/usr/bin/env python3
"""Regression tests for container-backed reading interaction wiring."""

from __future__ import annotations

from pathlib import Path
import sys

from dependency_injector import providers

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.app_DI_config import AppDIConfig  # noqa: E402
from config.container import ApplicationLookupContainer  # noqa: E402
from llm.llm_model_capabilities import (  # noqa: E402
    ENDPOINT_KIND_CHAT_COMPLETIONS,
    LLMModelCapabilities,
)
from section_tasks.reading_interaction_service_contracts import (  # noqa: E402
    ReadingInteractionRequest,
)
from section_tasks.reading_target_resolver import ResolvedReadingTarget  # noqa: E402


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


class FakeLLMProvider:
    def complete_text(self, prompt: str) -> str:
        if "items array" in prompt:
            return (
                '{"items":[{"item_id":"q1","type":"short_answer",'
                '"question":"What is the central assumption?",'
                '"answer":"That evidence supports the claim.",'
                '"explanation":"The target links evidence to a conclusion."}]}'
            )
        if "Return a strict JSON object with one non-empty string field: question" in prompt:
            return '{"question":"What assumption should the reader challenge?"}'
        if "feedback, strengths, improvements" in prompt:
            return (
                '{"feedback":"Grounded answer","strengths":"Uses the text",'
                '"improvements":"Compare an alternative explanation","score":4}'
            )
        return (
            '{"summary":"Summary","reasoning":"Reasoning",'
            '"explanation":"Explanation"}'
        )

    def get_model_capabilities(self) -> LLMModelCapabilities:
        return LLMModelCapabilities(
            model_name="fake",
            endpoint_kind=ENDPOINT_KIND_CHAT_COMPLETIONS,
            context_window_tokens=4096,
            max_output_tokens=512,
        )


def _request(interaction_type: str) -> ReadingInteractionRequest:
    target = ResolvedReadingTarget(
        document_id="doc-1",
        document_title="Document One",
        target_level="section",
        target_id="section-1",
        content="This section explains assumptions, evidence, and conclusions.",
        chapter_id="chapter-1",
        section_id="section-1",
    )
    return ReadingInteractionRequest(
        target=target,
        interaction_type=interaction_type,
        context_metadata={"context_mode": "full_target", "doc_name": "book"},
    )


def test_container_wires_quiz_and_critical_thinking_runtime_dependencies() -> None:
    container = ApplicationLookupContainer.build(AppDIConfig())
    container.llm_provider.override(providers.Object(FakeLLMProvider()))
    try:
        coordinator = container.section_task_coordinator()
        quiz_artifact = container.quiz_interaction_service().generate(_request("quiz"))
        critical_service = container.critical_thinking_session_service()
        question_session = critical_service.generate_question(
            _request("critical_thinking_session")
        )
        submitted = critical_service.submit_answer(
            question_session,
            "The assumed relation between evidence and conclusion should be tested.",
        )
        completed = critical_service.evaluate_answer(submitted)
    finally:
        container.llm_provider.reset_override()

    _assert(
        coordinator.quiz_interaction_orchestrator is not None,
        "SectionTaskCoordinator should receive quiz orchestrator from container",
    )
    _assert(
        coordinator.critical_thinking_session_service is not None,
        "SectionTaskCoordinator should receive critical-thinking service from container",
    )
    _assert(
        coordinator.critical_thinking_session_store is not None,
        "SectionTaskCoordinator should receive critical-thinking session store",
    )
    _assert(quiz_artifact.status == "completed", "quiz service should generate")
    _assert(
        question_session.status == "question_generated",
        "critical-thinking service should generate a question",
    )
    _assert(completed.status == "completed", "critical-thinking evaluation should complete")


if __name__ == "__main__":
    test_container_wires_quiz_and_critical_thinking_runtime_dependencies()
    print("reading interaction container wiring tests passed")
