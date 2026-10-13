#!/usr/bin/env python3
"""Regression tests for critical-thinking LLM prompt assembly."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from section_tasks.critical_thinking_llm_generator import (  # noqa: E402
    CriticalThinkingLLMGenerator,
)
from section_tasks.critical_thinking_session_service import (  # noqa: E402
    CRITICAL_THINKING_EVALUATION_INSTRUCTION,
)
from section_tasks.reading_interaction_prompt_context import (  # noqa: E402
    NO_EXTERNAL_RESEARCH_DEFAULT_INSTRUCTION,
)
from section_tasks.reading_interaction_service_contracts import (  # noqa: E402
    ReadingInteractionArtifact,
    ReadingInteractionRequest,
)
from section_tasks.reading_target_resolver import ResolvedReadingTarget  # noqa: E402


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


class FakeLLMProvider:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def complete_text(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return "{}"


def _target() -> ResolvedReadingTarget:
    return ResolvedReadingTarget(
        document_id="doc-1",
        document_title="许三观卖血记",
        target_level="task_unit",
        target_id="unit-1",
        content="许三观决定去卖血，这一决定体现了家庭压力和生存困境。",
        document_language="zh",
        chapter_id="chapter-1",
        section_id="section-1",
        task_unit_id="unit-1",
    )


def _submitted_session(answer: str) -> ReadingInteractionArtifact:
    return ReadingInteractionArtifact.from_target(
        target=_target(),
        interaction_type="critical_thinking_session",
        status="answer_submitted",
        payload={
            "question": "许三观的决定暴露了什么假设？",
            "answer": answer,
        },
        metadata={
            "context": {
                "context_mode": "full_target",
                "document_language": "zh",
            }
        },
    )


def test_question_prompt_uses_document_language_metadata() -> None:
    llm_provider = FakeLLMProvider()
    generator = CriticalThinkingLLMGenerator(llm_provider)
    request = ReadingInteractionRequest(
        target=_target(),
        interaction_type="critical_thinking_session",
        context_metadata={"context_mode": "full_target", "document_language": "zh"},
    )

    generator.generate_question(request, "Generate one question.")

    prompt = llm_provider.prompts[0]
    _assert(
        "Output language: Write every user-facing JSON string in Chinese (zh)." in prompt,
        "question prompt should constrain output language from request metadata",
    )


def test_evaluation_prompt_uses_session_language_metadata() -> None:
    llm_provider = FakeLLMProvider()
    generator = CriticalThinkingLLMGenerator(llm_provider)
    session = _submitted_session("它假设家庭责任高于个人身体风险。")

    generator.evaluate_answer(session, CRITICAL_THINKING_EVALUATION_INSTRUCTION)

    prompt = llm_provider.prompts[0]
    _assert(
        "Output language: Write every user-facing JSON string in Chinese (zh)." in prompt,
        "evaluation prompt should preserve session output language",
    )
    _assert(
        NO_EXTERNAL_RESEARCH_DEFAULT_INSTRUCTION in prompt,
        "evaluation prompt should forbid external research by default",
    )
    _assert(
        "evaluate information gain" in prompt,
        "evaluation prompt should include information-gain rubric",
    )
    _assert(
        "copy or repeat the question, primary source, existing insight" in prompt,
        "evaluation prompt should score low-information repetition lower",
    )
    _assert(
        "evidence-bound distinction, reframing, uncertainty reduction" in prompt,
        "evaluation prompt should reward grounded original thinking",
    )
    _assert(
        "do not add new JSON fields" in prompt,
        "evaluation prompt should preserve the current strict JSON shape",
    )


def test_evaluation_prompt_carries_low_high_information_gain_pairs() -> None:
    llm_provider = FakeLLMProvider()
    generator = CriticalThinkingLLMGenerator(llm_provider)
    low_information_answer = (
        "许三观决定去卖血，这一决定体现了家庭压力和生存困境。"
    )
    high_information_answer = (
        "这个决定不只是家庭压力的结果，还暴露了一个更危险的假设："
        "他把身体当成可以反复抵押的家庭资源。这个判断仍需要保留不确定性，"
        "因为这一段只显示了他的选择，还没有证明他完全认同这种牺牲逻辑。"
    )

    generator.evaluate_answer(
        _submitted_session(low_information_answer),
        CRITICAL_THINKING_EVALUATION_INSTRUCTION,
    )
    generator.evaluate_answer(
        _submitted_session(high_information_answer),
        CRITICAL_THINKING_EVALUATION_INSTRUCTION,
    )

    low_prompt = llm_provider.prompts[0]
    high_prompt = llm_provider.prompts[1]
    for prompt, answer in (
        (low_prompt, low_information_answer),
        (high_prompt, high_information_answer),
    ):
        _assert(answer in prompt, "evaluation prompt should include submitted answer")
        _assert(
            "answers that mostly copy or repeat the question, primary source" in prompt,
            "evaluation prompt should retain low-information guidance",
        )
        _assert(
            "evidence-bound distinction, reframing, uncertainty reduction" in prompt,
            "evaluation prompt should retain high-information guidance",
        )
        _assert(
            "do not add new JSON fields" in prompt,
            "paired examples should still preserve the current JSON contract",
        )


if __name__ == "__main__":
    test_question_prompt_uses_document_language_metadata()
    test_evaluation_prompt_uses_session_language_metadata()
    test_evaluation_prompt_carries_low_high_information_gain_pairs()
    print("critical-thinking llm generator tests passed")
