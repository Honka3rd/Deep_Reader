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


QUIZ_OUTPUT_SCHEMA_VERSION = "quiz_interaction_v1"
QUIZ_DEDUPLICATION_INSTRUCTION = (
    "Quiz deduplication guidance: Treat lower-level quiz artifacts and concepts "
    "as coverage signals. Do not copy, concatenate, or repeat lower-level quiz "
    "items. Prefer synthesis, transfer, comparison, and cross-unit understanding "
    "for the current target."
)
QUIZ_DEDUPLICATION_METADATA_VERSION = "quiz_deduplication_v1"
QUIZ_TYPES = frozenset({"short_answer", "multiple_choice", "true_false"})
DEFAULT_QUIZ_MAX_ITEMS_BY_TARGET_LEVEL = {
    "task_unit": 3,
    "section": 5,
    "chapter": 10,
    "document": 25,
}


class QuizInteractionService:
    """Generate validated quiz artifacts for resolved reading targets."""

    interaction_type = "quiz"

    def __init__(
        self,
        generator: Callable[
            [ReadingInteractionRequest, int, tuple[str, ...]],
            str | dict[str, Any],
        ],
        *,
        max_items_by_target_level: dict[str, int] | None = None,
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
        self._generator = generator
        self._artifact_context_provider = artifact_context_provider
        self._max_items_by_target_level = self._validate_limits(
            max_items_by_target_level
        )
        self._validity_policy = ReadingInteractionValidityPolicy(
            interaction_label="quiz generation",
            min_content_chars=min_content_chars,
            min_alnum_chars=min_alnum_chars,
            target_validity_checker=target_validity_checker,
        )

    def generate(
        self,
        request: ReadingInteractionRequest,
    ) -> ReadingInteractionArtifact:
        """Generate a validated quiz artifact or an explicit failure status."""
        if request.interaction_type != self.interaction_type:
            raise ValueError("QuizInteractionService requires interaction_type='quiz'")

        preflight_result = self._validity_policy.preflight(request)
        if preflight_result is not None:
            return self._artifact(
                request=request,
                status=preflight_result.status,
                reason=preflight_result.reason,
            )

        generation_request = self._request_with_secondary_context(request)
        max_items = self._max_items_for_target_level(request.target.target_level)
        valid_types = tuple(sorted(QUIZ_TYPES))

        try:
            raw_output = self._generator(generation_request, max_items, valid_types)
            payload = self._parse_and_validate_output(
                raw_output=raw_output,
                max_items=max_items,
            )
            self._validate_deduplicated_output(
                payload=payload,
                secondary_context=generation_request.secondary_context,
            )
        except ValueError as error:
            return self._artifact(
                request=generation_request,
                status="generation_failed",
                reason=str(error),
            )

        return self._artifact(
            request=generation_request,
            status="completed",
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
        if artifact_context.deduplication_hint_applied:
            secondary_context = self._append_deduplication_guidance(secondary_context)
            context_metadata_updates["quiz_deduplication"] = {
                "guidance_applied": True,
                "instruction_version": QUIZ_DEDUPLICATION_METADATA_VERSION,
            }
        return request.with_secondary_context(
            secondary_context,
            context_metadata_updates=context_metadata_updates,
        )

    def _artifact(
        self,
        *,
        request: ReadingInteractionRequest,
        status: str,
        payload: dict[str, object] | None = None,
        reason: str | None = None,
    ) -> ReadingInteractionArtifact:
        metadata: dict[str, object] = {
            "output_schema_version": QUIZ_OUTPUT_SCHEMA_VERSION,
            "max_items": self._max_items_for_target_level(request.target.target_level),
            "valid_types": tuple(sorted(QUIZ_TYPES)),
        }
        if request.prompt_instruction_version is not None:
            metadata["prompt_instruction_version"] = request.prompt_instruction_version
        if request.context_metadata:
            metadata["context"] = dict(request.context_metadata)
            artifact_reference = build_artifact_reference_metadata(
                request.context_metadata
            )
            if artifact_reference is not None:
                metadata[ARTIFACT_REFERENCE_METADATA_KEY] = artifact_reference

        return ReadingInteractionArtifact.from_target(
            target=request.target,
            interaction_type=self.interaction_type,
            status=status,
            payload=payload,
            metadata=metadata,
            reason=reason,
        )

    def _max_items_for_target_level(self, target_level: str) -> int:
        normalized_target_level = "document" if target_level == "book" else target_level
        try:
            return self._max_items_by_target_level[normalized_target_level]
        except KeyError as error:
            raise ValueError(
                "quiz max item limit is missing for target level: "
                f"'{target_level}'"
            ) from error

    @staticmethod
    def _validate_limits(
        max_items_by_target_level: dict[str, int] | None,
    ) -> dict[str, int]:
        limits = dict(DEFAULT_QUIZ_MAX_ITEMS_BY_TARGET_LEVEL)
        if max_items_by_target_level is not None:
            limits.update(max_items_by_target_level)

        for target_level in DEFAULT_QUIZ_MAX_ITEMS_BY_TARGET_LEVEL:
            value = limits.get(target_level)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(
                    "quiz max item limits must be positive integers for "
                    f"target level '{target_level}'"
                )
        return limits

    @staticmethod
    def _parse_and_validate_output(
        *,
        raw_output: str | dict[str, Any],
        max_items: int,
    ) -> dict[str, object]:
        if isinstance(raw_output, str):
            try:
                parsed_output = json.loads(raw_output)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid quiz JSON: {error.msg}") from error
        elif isinstance(raw_output, dict):
            parsed_output = dict(raw_output)
        else:
            raise ValueError("quiz output must be a JSON object")

        if not isinstance(parsed_output, dict):
            raise ValueError("quiz output must be a JSON object")

        raw_items = parsed_output.get("items")
        if not isinstance(raw_items, list):
            raise ValueError("quiz output missing 'items' list")
        if not raw_items:
            raise ValueError("quiz output requires at least one item")
        if len(raw_items) > max_items:
            raise ValueError(
                f"quiz output contains {len(raw_items)} items; max is {max_items}"
            )

        return {
            "items": [
                QuizInteractionService._validate_item(raw_item, index)
                for index, raw_item in enumerate(raw_items)
            ]
        }

    @staticmethod
    def _validate_item(raw_item: Any, index: int) -> dict[str, object]:
        item_label = f"quiz item {index}"
        if not isinstance(raw_item, dict):
            raise ValueError(f"{item_label} must be an object")

        quiz_type = raw_item.get("type")
        if not isinstance(quiz_type, str) or quiz_type.strip() not in QUIZ_TYPES:
            raise ValueError(f"{item_label} has invalid quiz type")
        quiz_type = quiz_type.strip()

        question = raw_item.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"{item_label} missing non-empty question")

        normalized_item: dict[str, object] = {
            "type": quiz_type,
            "question": question.strip(),
        }

        if quiz_type == "short_answer":
            answer = raw_item.get("answer")
            if not isinstance(answer, str) or not answer.strip():
                raise ValueError(f"{item_label} missing non-empty answer")
            normalized_item["answer"] = answer.strip()
        elif quiz_type == "true_false":
            answer = raw_item.get("answer")
            if not isinstance(answer, bool):
                raise ValueError(f"{item_label} true_false answer must be boolean")
            normalized_item["answer"] = answer
        else:
            choices = raw_item.get("choices")
            if not isinstance(choices, list) or len(choices) < 2:
                raise ValueError(
                    f"{item_label} multiple_choice requires at least two choices"
                )
            normalized_choices = []
            for choice in choices:
                if not isinstance(choice, str) or not choice.strip():
                    raise ValueError(
                        f"{item_label} multiple_choice choices must be non-empty strings"
                    )
                normalized_choices.append(choice.strip())
            answer = raw_item.get("answer")
            if not isinstance(answer, str) or answer.strip() not in normalized_choices:
                raise ValueError(
                    f"{item_label} multiple_choice answer must match one choice"
                )
            normalized_item["choices"] = normalized_choices
            normalized_item["answer"] = answer.strip()

        explanation = raw_item.get("explanation")
        if isinstance(explanation, str) and explanation.strip():
            normalized_item["explanation"] = explanation.strip()

        return normalized_item

    @staticmethod
    def _append_deduplication_guidance(secondary_context: str) -> str:
        if not secondary_context:
            return QUIZ_DEDUPLICATION_INSTRUCTION
        return f"{secondary_context}\n{QUIZ_DEDUPLICATION_INSTRUCTION}"

    @staticmethod
    def _validate_deduplicated_output(
        *,
        payload: dict[str, object],
        secondary_context: str,
    ) -> None:
        avoid_exact_questions = QuizInteractionService._deduplication_focus_phrases(
            secondary_context
        )
        if not avoid_exact_questions:
            return

        raw_items = payload.get("items")
        if not isinstance(raw_items, list):
            return

        normalized_avoid = {
            QuizInteractionService._normalize_question_text(question)
            for question in avoid_exact_questions
            if QuizInteractionService._normalize_question_text(question)
        }
        for raw_item in raw_items:
            if not isinstance(raw_item, dict):
                continue
            question = raw_item.get("question")
            if not isinstance(question, str):
                continue
            normalized_question = QuizInteractionService._normalize_question_text(
                question
            )
            if normalized_question in normalized_avoid:
                raise ValueError(
                    "quiz output duplicates lower-level artifact coverage"
                )

    @staticmethod
    def _deduplication_focus_phrases(secondary_context: str) -> list[str]:
        phrases: list[str] = []
        if QUIZ_DEDUPLICATION_INSTRUCTION not in secondary_context:
            return phrases

        for line in secondary_context.splitlines():
            if "; focus=" not in line:
                continue
            for part in line.split(";"):
                part = part.strip()
                if part.startswith("focus="):
                    phrase = part.removeprefix("focus=").strip()
                    if phrase:
                        phrases.append(phrase)
        return phrases

    @staticmethod
    def _normalize_question_text(value: str) -> str:
        return " ".join(value.strip().lower().split())
