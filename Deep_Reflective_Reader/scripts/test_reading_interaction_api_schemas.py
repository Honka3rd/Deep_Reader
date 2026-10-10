#!/usr/bin/env python3
"""Schema regression tests for public reading interaction API contracts."""

from __future__ import annotations

from pathlib import Path
import sys

from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api_schemas import (  # noqa: E402
    AnalysisArtifactPayloadResponse,
    AnalysisInteractionGenerateRequest,
    AnalysisInteractionReadRequest,
    AnalysisInteractionRefreshRequest,
    AnalysisInteractionResponse,
    ArtifactAwareInteractionMetadataResponse,
    CriticalThinkingAnswerSubmitRequest,
    CriticalThinkingEvaluationRetryRequest,
    CriticalThinkingQuestionGenerateRequest,
    CriticalThinkingSessionReadRequest,
    CriticalThinkingSessionResponse,
    QuizArtifactPayloadResponse,
    QuizInteractionGenerateRequest,
    QuizInteractionReadRequest,
    QuizInteractionRefreshRequest,
    QuizInteractionResponse,
    ReadingInteractionResponseEnvelope,
    ReadingInteractionTargetRequest,
    ReadingInteractionTargetResponse,
)
from shared.artifact_target_model import ArtifactTargetLevel  # noqa: E402


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _assert_target_validation_error(payload: dict, expected_message: str) -> None:
    try:
        ReadingInteractionTargetRequest.model_validate(payload)
    except ValidationError as error:
        _assert(
            expected_message in str(error),
            f"expected {expected_message!r} in target validation error: {error}",
        )
        return
    raise AssertionError(f"expected target validation error containing {expected_message!r}")


def _assert_envelope_validation_error(payload: dict, expected_message: str) -> None:
    try:
        ReadingInteractionResponseEnvelope.model_validate(payload)
    except ValidationError as error:
        _assert(
            expected_message in str(error),
            f"expected {expected_message!r} in envelope validation error: {error}",
        )
        return
    raise AssertionError(f"expected envelope validation error containing {expected_message!r}")


def _assert_analysis_response_validation_error(
    payload: dict,
    expected_message: str,
) -> None:
    try:
        AnalysisInteractionResponse.model_validate(payload)
    except ValidationError as error:
        _assert(
            expected_message in str(error),
            f"expected {expected_message!r} in analysis validation error: {error}",
        )
        return
    raise AssertionError(f"expected analysis validation error containing {expected_message!r}")


def _assert_analysis_payload_validation_error(
    payload: dict,
    expected_message: str,
) -> None:
    try:
        AnalysisArtifactPayloadResponse.model_validate(payload)
    except ValidationError as error:
        _assert(
            expected_message in str(error),
            f"expected {expected_message!r} in analysis payload error: {error}",
        )
        return
    raise AssertionError(f"expected analysis payload error containing {expected_message!r}")


def _assert_quiz_response_validation_error(
    payload: dict,
    expected_message: str,
) -> None:
    try:
        QuizInteractionResponse.model_validate(payload)
    except ValidationError as error:
        _assert(
            expected_message in str(error),
            f"expected {expected_message!r} in quiz validation error: {error}",
        )
        return
    raise AssertionError(f"expected quiz validation error containing {expected_message!r}")


def _assert_quiz_payload_validation_error(
    payload: dict,
    expected_message: str,
) -> None:
    try:
        QuizArtifactPayloadResponse.model_validate(payload)
    except ValidationError as error:
        _assert(
            expected_message in str(error),
            f"expected {expected_message!r} in quiz payload error: {error}",
        )
        return
    raise AssertionError(f"expected quiz payload error containing {expected_message!r}")


def _assert_critical_response_validation_error(
    payload: dict,
    expected_message: str,
) -> None:
    try:
        CriticalThinkingSessionResponse.model_validate(payload)
    except ValidationError as error:
        _assert(
            expected_message in str(error),
            f"expected {expected_message!r} in critical-thinking error: {error}",
        )
        return
    raise AssertionError(
        f"expected critical-thinking validation error containing {expected_message!r}"
    )


def _target_payload(**updates: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "doc_name": "  Book One  ",
        "target_type": "section",
        "chapter_id": " chapter-1 ",
        "section_id": " section-1 ",
        "source_structure_version": 3,
        "source_hash": " source-hash ",
    }
    payload.update(updates)
    return payload


def _target_response_payload(**updates: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "doc_name": "Book One",
        "target_type": "section",
        "target_id": "section-1",
        "document_id": "doc-1",
        "document_title": "Book One",
        "chapter_id": "chapter-1",
        "section_id": "section-1",
        "title": "Section One",
    }
    payload.update(updates)
    return payload


def _analysis_envelope_payload(**updates: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "target": _target_response_payload(),
        "interaction_type": "analysis",
        "status": "completed",
        "artifact_id": "analysis-1",
        "generated_at": "2026-10-09T12:00:00Z",
        "schema_version": "analysis_interaction_v1",
        "prompt_instruction_version": "prompt_v1",
        "source_structure_version": 3,
        "source_hash": "source-hash",
    }
    payload.update(updates)
    return payload


def _analysis_payload(**updates: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "summary": "  Compact summary.  ",
        "reasoning": "  Why this matters.  ",
        "interpretation": "  The passage argues for active reading.  ",
        "explanation": "  Existing service-compatible explanation.  ",
        "key_points": ["  point one  ", "point two"],
    }
    payload.update(updates)
    return payload


def _quiz_envelope_payload(**updates: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "target": _target_response_payload(),
        "interaction_type": "quiz",
        "status": "completed",
        "artifact_id": "quiz-1",
        "generated_at": "2026-10-09T12:00:00Z",
        "schema_version": "quiz_interaction_v1",
        "prompt_instruction_version": "prompt_v1",
        "source_structure_version": 3,
        "source_hash": "source-hash",
    }
    payload.update(updates)
    return payload


def _quiz_payload(**updates: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "max_items": 3,
        "items": [
            {
                "item_id": " q1 ",
                "item_type": "short-answer",
                "prompt": " What is the core claim? ",
                "answer": " The core claim. ",
                "explanation": " Directly stated in the passage. ",
            },
            {
                "item_id": "q2",
                "item_type": "multiple_choice",
                "prompt": "Choose the best interpretation.",
                "options": [" A ", "B", "C"],
                "answer": "A",
                "explanation": "A best matches the passage.",
            },
            {
                "item_id": "q3",
                "item_type": "true_false",
                "prompt": "The passage supports active reading.",
                "answer": True,
            },
        ],
    }
    payload.update(updates)
    return payload


def _critical_envelope_payload(**updates: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "target": _target_response_payload(),
        "interaction_type": "critical_thinking_session",
        "status": "question_generated",
        "session_id": "session-1",
        "generated_at": "2026-10-09T12:00:00Z",
        "schema_version": "critical_thinking_session_v1",
        "prompt_instruction_version": "prompt_v1",
        "source_structure_version": 3,
        "source_hash": "source-hash",
    }
    payload.update(updates)
    return payload


def _critical_payload(**updates: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "question": " Which assumption would most change the argument? ",
        "submitted_answer": " The institution assumption. ",
        "evaluation": {
            "feedback": " The answer identifies a relevant assumption. ",
            "score": 4.5,
            "suggested_refinement": " Add direct textual evidence. ",
            "strengths": "Clear causal reasoning.",
            "improvements": "Quote the source more directly.",
        },
    }
    payload.update(updates)
    return payload


def test_reading_interaction_target_request_accepts_id_based_targets() -> None:
    section_target = ReadingInteractionTargetRequest.model_validate(_target_payload())

    _assert(section_target.doc_name == "Book One", "doc_name should trim")
    _assert(section_target.target_type == "section", "target type should preserve")
    _assert(section_target.chapter_id == "chapter-1", "chapter_id should trim")
    _assert(section_target.section_id == "section-1", "section_id should trim")
    _assert(section_target.source_hash == "source-hash", "source_hash should trim")

    document_target = ReadingInteractionTargetRequest.model_validate(
        {
            "doc_name": "Book One",
            "target_type": "book",
        }
    )
    _assert(
        document_target.target_type == "document",
        "book should normalize to document",
    )

    task_unit_target = ReadingInteractionTargetRequest.model_validate(
        _target_payload(
            target_type="task-unit",
            task_unit_id=" unit-1 ",
        )
    )
    _assert(
        task_unit_target.target_type == "task_unit",
        "task-unit spelling should normalize",
    )
    _assert(task_unit_target.task_unit_id == "unit-1", "task_unit_id should trim")


def test_reading_interaction_target_request_rejects_bad_shapes_and_titles() -> None:
    _assert_target_validation_error(
        _target_payload(doc_name="  "),
        "doc_name cannot be empty",
    )
    _assert_target_validation_error(
        _target_payload(target_type="chapter", chapter_id=None),
        "chapter target requires chapter_id",
    )
    _assert_target_validation_error(
        _target_payload(target_type="chapter", section_id="section-1"),
        "chapter target must not include section_id or task_unit_id",
    )
    _assert_target_validation_error(
        _target_payload(target_type="document", chapter_id="chapter-1"),
        "document target must not include child target ids",
    )
    _assert_target_validation_error(
        _target_payload(chapter_title="Chapter One"),
        "Extra inputs are not permitted",
    )


def test_reading_interaction_response_envelope_accepts_shared_metadata() -> None:
    envelope = ReadingInteractionResponseEnvelope.model_validate(
        {
            "target": _target_response_payload(),
            "interaction_type": " analysis ",
            "status": " completed ",
            "artifact_id": " artifact-1 ",
            "generated_at": "2026-10-09T12:00:00Z",
            "updated_at": "2026-10-09T12:01:00Z",
            "schema_version": " analysis_v1 ",
            "prompt_instruction_version": " prompt_v1 ",
            "source_structure_version": 3,
            "source_hash": " source-hash ",
            "artifact_context_metadata": {
                "artifact_context_mode": "referenced",
                "referenced_artifact_ids": ["quiz-1"],
                "referenced_artifact_target_levels": [ArtifactTargetLevel.TASK_UNIT],
            },
        }
    )

    _assert(envelope.interaction_type == "analysis", "interaction type should trim")
    _assert(envelope.status == "completed", "status should trim")
    _assert(envelope.artifact_id == "artifact-1", "artifact_id should trim")
    _assert(envelope.schema_version == "analysis_v1", "schema version should trim")
    _assert(
        isinstance(
            envelope.artifact_context_metadata,
            ArtifactAwareInteractionMetadataResponse,
        ),
        "artifact context metadata should validate through shared schema",
    )


def test_reading_interaction_response_envelope_accepts_missing_state() -> None:
    envelope = ReadingInteractionResponseEnvelope.model_validate(
        {
            "target": _target_response_payload(),
            "interaction_type": "quiz",
            "status": "not_generated",
        }
    )

    _assert(envelope.status == "not_generated", "missing artifact state should serialize")
    _assert(envelope.artifact_id is None, "missing state should not include artifact id")


def test_reading_interaction_response_envelope_rejects_invalid_states() -> None:
    _assert_envelope_validation_error(
        {
            "target": _target_response_payload(),
            "interaction_type": "analysis",
            "status": "question_generated",
        },
        "only valid for critical_thinking_session",
    )
    _assert_envelope_validation_error(
        {
            "target": _target_response_payload(),
            "interaction_type": "analysis",
            "status": "generation_failed",
        },
        "requires reason",
    )
    _assert_envelope_validation_error(
        {
            "target": _target_response_payload(),
            "interaction_type": "quiz",
            "status": "not_generated",
            "artifact_id": "artifact-1",
        },
        "not_generated response must not include artifact_id or session_id",
    )
    _assert_envelope_validation_error(
        {
            "target": _target_response_payload(),
            "interaction_type": "quiz",
            "status": "completed",
            "drawer_open": True,
        },
        "Extra inputs are not permitted",
    )


def test_analysis_interaction_requests_use_shared_target_schema() -> None:
    read_request = AnalysisInteractionReadRequest.model_validate(
        {
            "target": _target_payload(),
        }
    )
    _assert(read_request.target.target_type == "section", "read should keep shared target")

    generate_request = AnalysisInteractionGenerateRequest.model_validate(
        {
            "target": _target_payload(
                target_type="chapter",
                chapter_id=" chapter-1 ",
                section_id=None,
            ),
            "prompt_instruction_version": " prompt_v2 ",
        }
    )
    _assert(
        generate_request.target.target_type == "chapter",
        "generate should validate shared target",
    )
    _assert(
        generate_request.prompt_instruction_version == "prompt_v2",
        "generate prompt version should trim",
    )

    refresh_request = AnalysisInteractionRefreshRequest.model_validate(
        {
            "target": _target_payload(target_type="task_unit", task_unit_id=" unit-1 "),
            "prompt_instruction_version": "  ",
        }
    )
    _assert(
        refresh_request.target.task_unit_id == "unit-1",
        "refresh should validate task-unit target",
    )
    _assert(
        refresh_request.prompt_instruction_version is None,
        "empty refresh prompt version should normalize to none",
    )


def test_analysis_interaction_response_accepts_completed_payload() -> None:
    response = AnalysisInteractionResponse.model_validate(
        {
            "envelope": _analysis_envelope_payload(),
            "payload": _analysis_payload(),
        }
    )

    _assert(response.envelope.interaction_type == "analysis", "must use analysis envelope")
    _assert(response.envelope.status == "completed", "completed state should validate")
    _assert(response.payload is not None, "completed state should include payload")
    _assert(response.payload.summary == "Compact summary.", "summary should trim")
    _assert(response.payload.reasoning == "Why this matters.", "reasoning should trim")
    _assert(
        response.payload.interpretation == "The passage argues for active reading.",
        "interpretation should trim",
    )
    _assert(
        response.payload.key_points == ["point one", "point two"],
        "key points should trim",
    )


def test_analysis_interaction_response_accepts_missing_and_failure_states() -> None:
    missing_response = AnalysisInteractionResponse.model_validate(
        {
            "envelope": _analysis_envelope_payload(
                status="not_generated",
                artifact_id=None,
                generated_at=None,
                schema_version=None,
                prompt_instruction_version=None,
            ),
            "payload": None,
        }
    )
    _assert(
        missing_response.envelope.status == "not_generated",
        "read missing state should validate",
    )

    insufficient_response = AnalysisInteractionResponse.model_validate(
        {
            "envelope": _analysis_envelope_payload(
                status="insufficient_content",
                artifact_id=None,
                reason="target content too short",
            ),
            "payload": None,
        }
    )
    _assert(
        insufficient_response.envelope.reason == "target content too short",
        "failure reason should preserve",
    )


def test_analysis_interaction_response_rejects_invalid_shapes() -> None:
    _assert_analysis_response_validation_error(
        {
            "envelope": _analysis_envelope_payload(status="completed"),
            "payload": None,
        },
        "completed analysis response requires payload",
    )
    _assert_analysis_response_validation_error(
        {
            "envelope": _analysis_envelope_payload(
                interaction_type="quiz",
                status="completed",
            ),
            "payload": _analysis_payload(),
        },
        "analysis response envelope requires interaction_type='analysis'",
    )
    _assert_analysis_response_validation_error(
        {
            "envelope": _analysis_envelope_payload(
                status="generation_failed",
                reason="bad model output",
            ),
            "payload": _analysis_payload(),
        },
        "analysis payload is only valid when envelope status is completed",
    )
    _assert_analysis_payload_validation_error(
        _analysis_payload(summary="  "),
        "summary cannot be empty",
    )
    _assert_analysis_payload_validation_error(
        _analysis_payload(key_points=["good", "  "]),
        "key_points cannot contain empty values",
    )


def test_quiz_interaction_requests_use_shared_target_schema() -> None:
    read_request = QuizInteractionReadRequest.model_validate(
        {
            "target": _target_payload(),
        }
    )
    _assert(read_request.target.target_type == "section", "read should keep shared target")

    generate_request = QuizInteractionGenerateRequest.model_validate(
        {
            "target": _target_payload(target_type="document", chapter_id=None, section_id=None),
            "max_items": 5,
            "prompt_instruction_version": " prompt_v2 ",
        }
    )
    _assert(
        generate_request.target.target_type == "document",
        "generate should validate document target",
    )
    _assert(generate_request.max_items == 5, "generate should preserve max_items")
    _assert(
        generate_request.prompt_instruction_version == "prompt_v2",
        "generate prompt version should trim",
    )

    refresh_request = QuizInteractionRefreshRequest.model_validate(
        {
            "target": _target_payload(target_type="task_unit", task_unit_id=" unit-1 "),
            "max_items": 3,
            "prompt_instruction_version": "  ",
        }
    )
    _assert(
        refresh_request.target.task_unit_id == "unit-1",
        "refresh should validate task-unit target",
    )
    _assert(
        refresh_request.prompt_instruction_version is None,
        "empty refresh prompt version should normalize to none",
    )


def test_quiz_interaction_response_accepts_completed_payload() -> None:
    response = QuizInteractionResponse.model_validate(
        {
            "envelope": _quiz_envelope_payload(),
            "payload": _quiz_payload(),
        }
    )

    _assert(response.envelope.interaction_type == "quiz", "must use quiz envelope")
    _assert(response.payload is not None, "completed quiz should include payload")
    first_item = response.payload.items[0]
    second_item = response.payload.items[1]
    third_item = response.payload.items[2]
    _assert(first_item.item_id == "q1", "item id should trim")
    _assert(first_item.item_type == "short_answer", "item type should normalize")
    _assert(first_item.prompt == "What is the core claim?", "prompt should trim")
    _assert(first_item.answer == "The core claim.", "short answer should trim")
    _assert(second_item.options == ["A", "B", "C"], "options should trim")
    _assert(second_item.answer == "A", "multiple-choice answer should preserve")
    _assert(third_item.answer is True, "true/false answer should stay boolean")


def test_quiz_interaction_response_accepts_missing_and_failure_states() -> None:
    missing_response = QuizInteractionResponse.model_validate(
        {
            "envelope": _quiz_envelope_payload(
                status="not_generated",
                artifact_id=None,
                generated_at=None,
                schema_version=None,
                prompt_instruction_version=None,
            ),
            "payload": None,
        }
    )
    _assert(
        missing_response.envelope.status == "not_generated",
        "read missing state should validate",
    )

    insufficient_response = QuizInteractionResponse.model_validate(
        {
            "envelope": _quiz_envelope_payload(
                status="insufficient_content",
                artifact_id=None,
                reason="target content too short",
            ),
            "payload": None,
        }
    )
    _assert(
        insufficient_response.envelope.reason == "target content too short",
        "insufficient-content reason should preserve",
    )


def test_quiz_interaction_response_rejects_invalid_shapes() -> None:
    _assert_quiz_response_validation_error(
        {
            "envelope": _quiz_envelope_payload(status="completed"),
            "payload": None,
        },
        "completed quiz response requires payload",
    )
    _assert_quiz_response_validation_error(
        {
            "envelope": _quiz_envelope_payload(
                interaction_type="analysis",
                status="completed",
            ),
            "payload": _quiz_payload(),
        },
        "quiz response envelope requires interaction_type='quiz'",
    )
    _assert_quiz_response_validation_error(
        {
            "envelope": _quiz_envelope_payload(
                status="validation_failed",
                reason="invalid quiz output",
            ),
            "payload": _quiz_payload(),
        },
        "quiz payload is only valid when envelope status is completed",
    )
    _assert_quiz_payload_validation_error(
        {
            "max_items": 1,
            "items": _quiz_payload()["items"][:2],
        },
        "quiz payload contains 2 items; max_items is 1",
    )
    _assert_quiz_payload_validation_error(
        {
            "items": [
                {
                    "item_id": "q1",
                    "item_type": "essay",
                    "prompt": "Explain.",
                    "answer": "Answer.",
                }
            ],
        },
        "item_type must be one of",
    )
    _assert_quiz_payload_validation_error(
        {
            "items": [
                {
                    "item_id": "q1",
                    "item_type": "multiple_choice",
                    "prompt": "Pick one.",
                    "options": ["A", "B"],
                    "answer": "C",
                }
            ],
        },
        "multiple_choice quiz item answer must match one option",
    )
    _assert_quiz_payload_validation_error(
        {
            "items": [
                {
                    "item_id": "q1",
                    "item_type": "true_false",
                    "prompt": "True?",
                    "answer": "true",
                }
            ],
        },
        "true_false quiz item answer must be boolean",
    )


def test_critical_thinking_requests_use_shared_target_and_session_ids() -> None:
    read_request = CriticalThinkingSessionReadRequest.model_validate(
        {
            "target": _target_payload(),
            "session_id": " session-1 ",
        }
    )
    _assert(read_request.target.target_type == "section", "read should use target")
    _assert(read_request.session_id == "session-1", "session_id should trim")

    generate_request = CriticalThinkingQuestionGenerateRequest.model_validate(
        {
            "target": _target_payload(target_type="document", chapter_id=None, section_id=None),
            "prompt_instruction_version": " prompt_v2 ",
        }
    )
    _assert(
        generate_request.target.target_type == "document",
        "generate should validate document target",
    )
    _assert(
        generate_request.prompt_instruction_version == "prompt_v2",
        "prompt version should trim",
    )

    submit_request = CriticalThinkingAnswerSubmitRequest.model_validate(
        {
            "target": _target_payload(),
            "session_id": " session-1 ",
            "answer": " My answer. ",
        }
    )
    _assert(submit_request.target.target_type == "section", "submit should use target")
    _assert(submit_request.session_id == "session-1", "submit session should trim")
    _assert(submit_request.answer == "My answer.", "submitted answer should trim")

    retry_request = CriticalThinkingEvaluationRetryRequest.model_validate(
        {
            "target": _target_payload(),
            "session_id": " session-1 ",
        }
    )
    _assert(retry_request.target.target_type == "section", "retry should use target")
    _assert(retry_request.session_id == "session-1", "retry session should trim")


def test_critical_thinking_response_accepts_lifecycle_states() -> None:
    generated = CriticalThinkingSessionResponse.model_validate(
        {
            "envelope": _critical_envelope_payload(status="question_generated"),
            "payload": {
                "question": " Which assumption would most change the argument? ",
            },
        }
    )
    _assert(
        generated.payload is not None and generated.payload.question is not None,
        "generated question should validate",
    )
    _assert(
        generated.payload.question == "Which assumption would most change the argument?",
        "question should trim",
    )

    answered = CriticalThinkingSessionResponse.model_validate(
        {
            "envelope": _critical_envelope_payload(status="answer_submitted"),
            "payload": _critical_payload(evaluation=None),
        }
    )
    _assert(
        answered.payload is not None
        and answered.payload.submitted_answer == "The institution assumption.",
        "submitted answer should trim and preserve",
    )

    failed = CriticalThinkingSessionResponse.model_validate(
        {
            "envelope": _critical_envelope_payload(
                status="evaluation_failed",
                reason="invalid evaluation JSON",
            ),
            "payload": _critical_payload(evaluation=None),
        }
    )
    _assert(
        failed.payload is not None and failed.payload.retry_eligible is True,
        "evaluation_failed should mark retry eligible",
    )

    completed = CriticalThinkingSessionResponse.model_validate(
        {
            "envelope": _critical_envelope_payload(status="completed"),
            "payload": _critical_payload(),
        }
    )
    _assert(
        completed.payload is not None and completed.payload.evaluation is not None,
        "completed should include evaluation",
    )
    _assert(completed.payload.evaluation.feedback.startswith("The answer"), "feedback trims")
    _assert(completed.payload.evaluation.score == 4.5, "score should preserve")
    _assert(
        completed.payload.evaluation.suggested_refinement
        == "Add direct textual evidence.",
        "suggested refinement should trim",
    )
    _assert(
        completed.payload.retry_eligible is False,
        "completed should not be retry eligible",
    )


def test_critical_thinking_response_accepts_missing_and_failure_states() -> None:
    missing = CriticalThinkingSessionResponse.model_validate(
        {
            "envelope": _critical_envelope_payload(
                status="not_generated",
                session_id=None,
                generated_at=None,
                schema_version=None,
                prompt_instruction_version=None,
            ),
            "payload": None,
        }
    )
    _assert(missing.envelope.status == "not_generated", "missing state should validate")

    insufficient = CriticalThinkingSessionResponse.model_validate(
        {
            "envelope": _critical_envelope_payload(
                status="insufficient_content",
                session_id=None,
                reason="target content too short",
            ),
            "payload": None,
        }
    )
    _assert(
        insufficient.envelope.reason == "target content too short",
        "insufficient reason should preserve",
    )


def test_critical_thinking_response_rejects_invalid_lifecycle_shapes() -> None:
    _assert_critical_response_validation_error(
        {
            "envelope": _critical_envelope_payload(interaction_type="quiz"),
            "payload": {"question": "Question?"},
        },
        "only valid for critical_thinking_session",
    )
    _assert_critical_response_validation_error(
        {
            "envelope": _critical_envelope_payload(status="question_generated"),
            "payload": None,
        },
        "question_generated response requires question",
    )
    _assert_critical_response_validation_error(
        {
            "envelope": _critical_envelope_payload(status="answer_submitted"),
            "payload": {"question": "Question?"},
        },
        "answer_submitted response requires question and submitted_answer",
    )
    _assert_critical_response_validation_error(
        {
            "envelope": _critical_envelope_payload(
                status="evaluation_failed",
                reason="invalid evaluation",
            ),
            "payload": _critical_payload(),
        },
        "evaluation_failed response must not include evaluation",
    )
    _assert_critical_response_validation_error(
        {
            "envelope": _critical_envelope_payload(status="completed"),
            "payload": _critical_payload(evaluation=None),
        },
        "completed response requires evaluation",
    )
    _assert_critical_response_validation_error(
        {
            "envelope": _critical_envelope_payload(
                status="not_generated",
                session_id="session-1",
            ),
            "payload": None,
        },
        "not_generated response must not include artifact_id or session_id",
    )
    _assert_critical_response_validation_error(
        {
            "envelope": _critical_envelope_payload(
                status="completed",
                session_id=None,
            ),
            "payload": _critical_payload(),
        },
        "completed critical-thinking response requires session_id",
    )


if __name__ == "__main__":
    test_reading_interaction_target_request_accepts_id_based_targets()
    test_reading_interaction_target_request_rejects_bad_shapes_and_titles()
    test_reading_interaction_response_envelope_accepts_shared_metadata()
    test_reading_interaction_response_envelope_accepts_missing_state()
    test_reading_interaction_response_envelope_rejects_invalid_states()
    test_analysis_interaction_requests_use_shared_target_schema()
    test_analysis_interaction_response_accepts_completed_payload()
    test_analysis_interaction_response_accepts_missing_and_failure_states()
    test_analysis_interaction_response_rejects_invalid_shapes()
    test_quiz_interaction_requests_use_shared_target_schema()
    test_quiz_interaction_response_accepts_completed_payload()
    test_quiz_interaction_response_accepts_missing_and_failure_states()
    test_quiz_interaction_response_rejects_invalid_shapes()
    test_critical_thinking_requests_use_shared_target_and_session_ids()
    test_critical_thinking_response_accepts_lifecycle_states()
    test_critical_thinking_response_accepts_missing_and_failure_states()
    test_critical_thinking_response_rejects_invalid_lifecycle_shapes()
    print("reading interaction API schema tests passed")
