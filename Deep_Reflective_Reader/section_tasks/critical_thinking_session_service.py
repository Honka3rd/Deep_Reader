from __future__ import annotations

from collections.abc import Callable
import json
from typing import Any

from section_tasks.artifact_validity import (
    ReadingInteractionTargetValidity,
    ReadingInteractionValidityPolicy,
)
from context.artifact_aware_context import ArtifactAwareContextResult
from section_tasks.reading_interaction_service_contracts import (
    ARTIFACT_REFERENCE_METADATA_KEY,
    ReadingInteractionArtifact,
    ReadingInteractionRequest,
    build_artifact_reference_metadata,
)


CRITICAL_THINKING_OUTPUT_SCHEMA_VERSION = "critical_thinking_session_v1"
CRITICAL_THINKING_QUESTION_INSTRUCTION = (
    "You are running critical thinking training. Generate exactly one question "
    "grounded in the current reading target. The question should invite reasoning, "
    "evidence use, assumptions, implications, or counterarguments. Return a strict "
    "JSON object with one non-empty string field: question. Do not answer it."
)
CRITICAL_THINKING_EVALUATION_INSTRUCTION = (
    "You are evaluating a user's answer for critical thinking training. Assess the "
    "answer against the question and reading target. Return a strict JSON object "
    "with non-empty string fields: feedback, strengths, improvements. You may also "
    "include numeric score from 0 to 5."
)
CRITICAL_THINKING_CONTINUITY_GUIDANCE = (
    "Critical-thinking learning-continuity guidance: Treat lower-level "
    "critical-thinking sessions as prior local training signals. Do not repeat "
    "the same local question. Generate one broader question for the current target "
    "that builds toward synthesis, assumption testing, evidence comparison, "
    "implications, or counterarguments."
)
CRITICAL_THINKING_CONTINUITY_METADATA_VERSION = "critical_thinking_continuity_v1"
REQUIRED_EVALUATION_FIELDS = ("feedback", "strengths", "improvements")


class CriticalThinkingSessionService:
    """Manage a one-question critical-thinking interaction session."""

    interaction_type = "critical_thinking_session"

    def __init__(
        self,
        question_generator: Callable[
            [ReadingInteractionRequest, str],
            str | dict[str, Any],
        ],
        evaluation_generator: Callable[
            [ReadingInteractionArtifact, str],
            str | dict[str, Any],
        ],
        *,
        min_content_chars: int = 20,
        min_alnum_chars: int = 8,
        target_validity_checker: (
            Callable[[ReadingInteractionRequest], ReadingInteractionTargetValidity]
            | None
        ) = None,
        artifact_context_provider: (
            Callable[[ReadingInteractionRequest], ArtifactAwareContextResult | None]
            | None
        ) = None,
    ) -> None:
        self._question_generator = question_generator
        self._evaluation_generator = evaluation_generator
        self._artifact_context_provider = artifact_context_provider
        self._validity_policy = ReadingInteractionValidityPolicy(
            interaction_label="critical-thinking training",
            min_content_chars=min_content_chars,
            min_alnum_chars=min_alnum_chars,
            target_validity_checker=target_validity_checker,
        )

    def generate(
        self,
        request: ReadingInteractionRequest,
    ) -> ReadingInteractionArtifact:
        """Generate and validate a persisted-session-shaped question artifact."""
        return self.generate_question(request)

    def generate_question(
        self,
        request: ReadingInteractionRequest,
    ) -> ReadingInteractionArtifact:
        """Generate one question or a recoverable generation status."""
        if request.interaction_type != self.interaction_type:
            raise ValueError(
                "CriticalThinkingSessionService requires "
                "interaction_type='critical_thinking_session'"
            )

        preflight_result = self._validity_policy.preflight(request)
        if preflight_result is not None:
            return self._artifact_from_request(
                request=request,
                status=preflight_result.status,
                reason=preflight_result.reason,
            )

        generation_request = self._request_with_secondary_context(request)

        try:
            raw_output = self._question_generator(
                generation_request,
                CRITICAL_THINKING_QUESTION_INSTRUCTION,
            )
            payload = self._parse_and_validate_question(raw_output)
        except ValueError as error:
            return self._artifact_from_request(
                request=generation_request,
                status="generation_failed",
                reason=str(error),
            )

        return self._artifact_from_request(
            request=generation_request,
            status="question_generated",
            payload=payload,
        )

    def _request_with_secondary_context(
        self,
        request: ReadingInteractionRequest,
    ) -> ReadingInteractionRequest:
        if self._artifact_context_provider is None:
            return request
        artifact_context = self._artifact_context_provider(request)
        if artifact_context is None:
            return request
        secondary_context = artifact_context.context_text
        context_metadata_updates: dict[str, object] = {
            "artifact_context": artifact_context.to_metadata()
        }
        if artifact_context.abstraction_hint_applied:
            secondary_context = self._append_continuity_guidance(secondary_context)
            context_metadata_updates["critical_thinking_continuity"] = {
                "guidance_applied": True,
                "instruction_version": (
                    CRITICAL_THINKING_CONTINUITY_METADATA_VERSION
                ),
            }
        return request.with_secondary_context(
            secondary_context,
            context_metadata_updates=context_metadata_updates,
        )

    def submit_answer(
        self,
        session: ReadingInteractionArtifact,
        answer: str,
    ) -> ReadingInteractionArtifact:
        """Attach a user answer while preserving the generated question."""
        self._require_session(session)
        if session.status not in {"question_generated", "evaluation_failed"}:
            raise ValueError(
                "answer submission requires a question_generated or evaluation_failed session"
            )
        normalized_answer = answer.strip()
        if not normalized_answer:
            raise ValueError("critical-thinking answer must be non-empty")

        question = self._question_from_session(session)
        payload = {
            "question": question,
            "answer": normalized_answer,
        }
        return self._artifact_from_session(
            session=session,
            status="answer_submitted",
            payload=payload,
        )

    def evaluate_answer(
        self,
        session: ReadingInteractionArtifact,
    ) -> ReadingInteractionArtifact:
        """Evaluate a submitted answer or preserve it for retry on failure."""
        self._require_session(session)
        if session.status not in {"answer_submitted", "evaluation_failed"}:
            raise ValueError(
                "answer evaluation requires an answer_submitted or evaluation_failed session"
            )

        question = self._question_from_session(session)
        answer = self._answer_from_session(session)
        base_payload = {
            "question": question,
            "answer": answer,
        }

        try:
            raw_output = self._evaluation_generator(
                session,
                CRITICAL_THINKING_EVALUATION_INSTRUCTION,
            )
            evaluation = self._parse_and_validate_evaluation(raw_output)
        except ValueError as error:
            return self._artifact_from_session(
                session=session,
                status="evaluation_failed",
                payload=base_payload,
                reason=str(error),
            )

        return self._artifact_from_session(
            session=session,
            status="completed",
            payload={
                **base_payload,
                "evaluation": evaluation,
            },
        )

    def _artifact_from_request(
        self,
        *,
        request: ReadingInteractionRequest,
        status: str,
        payload: dict[str, object] | None = None,
        reason: str | None = None,
    ) -> ReadingInteractionArtifact:
        metadata = self._metadata(
            prompt_instruction_version=request.prompt_instruction_version,
            context_metadata=request.context_metadata,
        )
        return ReadingInteractionArtifact.from_target(
            target=request.target,
            interaction_type=self.interaction_type,
            status=status,
            payload=payload,
            metadata=metadata,
            reason=reason,
        )

    def _artifact_from_session(
        self,
        *,
        session: ReadingInteractionArtifact,
        status: str,
        payload: dict[str, object],
        reason: str | None = None,
    ) -> ReadingInteractionArtifact:
        metadata = dict(session.metadata)
        metadata["output_schema_version"] = CRITICAL_THINKING_OUTPUT_SCHEMA_VERSION
        metadata["flow"] = "question_answer_evaluation"

        return ReadingInteractionArtifact(
            interaction_type=self.interaction_type,
            status=status,
            target_level=session.target_level,
            target_id=session.target_id,
            document_id=session.document_id,
            chapter_id=session.chapter_id,
            section_id=session.section_id,
            task_unit_id=session.task_unit_id,
            payload=dict(payload),
            metadata=metadata,
            reason=reason,
        )

    @staticmethod
    def _metadata(
        *,
        prompt_instruction_version: str | None,
        context_metadata: dict[str, object],
    ) -> dict[str, object]:
        metadata: dict[str, object] = {
            "output_schema_version": CRITICAL_THINKING_OUTPUT_SCHEMA_VERSION,
            "flow": "question_answer_evaluation",
            "question_instruction": CRITICAL_THINKING_QUESTION_INSTRUCTION,
            "evaluation_instruction": CRITICAL_THINKING_EVALUATION_INSTRUCTION,
        }
        if prompt_instruction_version is not None:
            metadata["prompt_instruction_version"] = prompt_instruction_version
        if context_metadata:
            metadata["context"] = dict(context_metadata)
            artifact_reference = build_artifact_reference_metadata(context_metadata)
            if artifact_reference is not None:
                metadata[ARTIFACT_REFERENCE_METADATA_KEY] = artifact_reference
        return metadata

    def _require_session(self, session: ReadingInteractionArtifact) -> None:
        if session.interaction_type != self.interaction_type:
            raise ValueError(
                "CriticalThinkingSessionService requires a critical-thinking session"
            )

    @staticmethod
    def _question_from_session(session: ReadingInteractionArtifact) -> str:
        question = session.payload.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("critical-thinking session missing generated question")
        return question.strip()

    @staticmethod
    def _answer_from_session(session: ReadingInteractionArtifact) -> str:
        answer = session.payload.get("answer")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("critical-thinking session missing submitted answer")
        return answer.strip()

    @staticmethod
    def _parse_json_object(
        raw_output: str | dict[str, Any],
        *,
        output_name: str,
    ) -> dict[str, Any]:
        if isinstance(raw_output, str):
            try:
                parsed_output = json.loads(raw_output)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid {output_name} JSON: {error.msg}") from error
        elif isinstance(raw_output, dict):
            parsed_output = dict(raw_output)
        else:
            raise ValueError(f"{output_name} output must be a JSON object")

        if not isinstance(parsed_output, dict):
            raise ValueError(f"{output_name} output must be a JSON object")
        return parsed_output

    @staticmethod
    def _parse_and_validate_question(
        raw_output: str | dict[str, Any],
    ) -> dict[str, object]:
        parsed_output = CriticalThinkingSessionService._parse_json_object(
            raw_output,
            output_name="critical-thinking question",
        )
        question = parsed_output.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("critical-thinking output missing non-empty question")
        return {"question": question.strip()}

    @staticmethod
    def _append_continuity_guidance(secondary_context: str) -> str:
        if not secondary_context:
            return CRITICAL_THINKING_CONTINUITY_GUIDANCE
        return f"{secondary_context}\n{CRITICAL_THINKING_CONTINUITY_GUIDANCE}"

    @staticmethod
    def _parse_and_validate_evaluation(
        raw_output: str | dict[str, Any],
    ) -> dict[str, object]:
        parsed_output = CriticalThinkingSessionService._parse_json_object(
            raw_output,
            output_name="critical-thinking evaluation",
        )

        evaluation: dict[str, object] = {}
        for field_name in REQUIRED_EVALUATION_FIELDS:
            value = parsed_output.get(field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    "critical-thinking evaluation missing non-empty "
                    f"'{field_name}'"
                )
            evaluation[field_name] = value.strip()

        score = parsed_output.get("score")
        if score is not None:
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                raise ValueError("critical-thinking evaluation score must be numeric")
            if score < 0 or score > 5:
                raise ValueError("critical-thinking evaluation score must be 0..5")
            evaluation["score"] = score

        return evaluation
