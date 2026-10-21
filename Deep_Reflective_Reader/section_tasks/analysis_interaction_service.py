from __future__ import annotations

from section_tasks.interaction_source_sufficiency import InsufficientPrimarySource, check_source_abstention

from collections.abc import Callable
import json
import re
from typing import Any
from section_tasks.interaction_policy_identity import policy_metadata
from section_tasks.interaction_output_language import request_language, validate_output_language

from section_tasks.artifact_validity import (
    ReadingInteractionTargetValidity,
    ReadingInteractionValidityPolicy,
)
from context.artifact_aware_context import ArtifactAwareContextResult
from section_tasks.reading_interaction_service_contracts import (
    ANCESTOR_SUMMARY_CONTEXT_METADATA_KEY,
    ARTIFACT_REFERENCE_METADATA_KEY,
    ReadingInteractionArtifact,
    ReadingInteractionRequest,
    build_ancestor_context_metadata,
    build_artifact_reference_metadata,
    build_external_research_metadata,
    EXTERNAL_RESEARCH_METADATA_KEY,
)


ANALYSIS_OUTPUT_SCHEMA_VERSION = "analysis_interaction_v1"
REQUIRED_ANALYSIS_FIELDS = ("summary", "reasoning", "explanation")
_LATIN_TOKEN_PATTERN = re.compile(r"[a-zA-Z][a-zA-Z'-]{2,}")
_CJK_SEQUENCE_PATTERN = re.compile(r"[\u3400-\u9fff]{2,}")
_GENERIC_PLACEHOLDERS = {
    "analysis",
    "explanation",
    "insight",
    "reasoning",
    "summary",
}
_GENERIC_UNSUPPORTED_MARKERS = (
    "important theme",
    "the passage shows",
    "the section shows",
    "the text shows",
    "this passage",
    "this section",
    "this text",
    "reveals a theme",
    "highlights the importance",
    "helps the reader understand",
)
_SOURCE_TOKEN_STOPWORDS = {
    "about",
    "after",
    "also",
    "because",
    "before",
    "chapter",
    "could",
    "document",
    "explains",
    "from",
    "into",
    "passage",
    "section",
    "should",
    "source",
    "that",
    "their",
    "there",
    "these",
    "this",
    "through",
    "with",
    "would",
}


class AnalysisInteractionService:
    """Generate validated analysis artifacts for resolved reading targets."""

    interaction_type = "analysis"

    def __init__(
        self,
        generator: Callable[[ReadingInteractionRequest], str | dict[str, Any]],
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
        self._generator = generator
        self._artifact_context_provider = artifact_context_provider
        self._validity_policy = ReadingInteractionValidityPolicy(
            interaction_label="analysis",
            min_content_chars=min_content_chars,
            min_alnum_chars=min_alnum_chars,
            require_semantic_content=True,
            target_validity_checker=target_validity_checker,
        )

    def generate(
        self,
        request: ReadingInteractionRequest,
    ) -> ReadingInteractionArtifact:
        """Generate a validated analysis artifact or an explicit failure status."""
        if request.interaction_type != self.interaction_type:
            raise ValueError("AnalysisInteractionService requires interaction_type='analysis'")

        preflight_result = self._validity_policy.preflight(request)
        if preflight_result is not None:
            return self._artifact(
                request=request,
                status=preflight_result.status,
                reason=preflight_result.reason,
            )

        generation_request = self._request_with_secondary_context(request)

        try:
            raw_output = self._generator(generation_request)
            payload = self._parse_and_validate_output(raw_output)
            validate_output_language(payload, request_language(generation_request))
            self._validate_payload_against_source(
                payload,
                generation_request.target.content,
            )
        except InsufficientPrimarySource as error:
            return self._artifact(
                request=generation_request,
                status="insufficient_content",
                reason=str(error),
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

    def preflight(
        self,
        request: ReadingInteractionRequest,
    ) -> ReadingInteractionArtifact | None:
        """Return a preflight artifact for non-readable/stale targets, else None."""
        if request.interaction_type != self.interaction_type:
            raise ValueError("AnalysisInteractionService requires interaction_type='analysis'")
        preflight_result = self._validity_policy.preflight(request)
        if preflight_result is None:
            return None
        return self._artifact(
            request=request,
            status=preflight_result.status,
            reason=preflight_result.reason,
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
        return request.with_secondary_context(
            artifact_context.context_text,
            context_metadata_updates={"artifact_context": artifact_context.to_metadata()},
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
            "output_schema_version": ANALYSIS_OUTPUT_SCHEMA_VERSION,
        }
        metadata.update(policy_metadata(request))
        if request.context_metadata:
            metadata["context"] = dict(request.context_metadata)
            artifact_reference = build_artifact_reference_metadata(
                request.context_metadata
            )
            if artifact_reference is not None:
                metadata[ARTIFACT_REFERENCE_METADATA_KEY] = artifact_reference
        ancestor_context_metadata = build_ancestor_context_metadata(request)
        if ancestor_context_metadata is not None:
            metadata[ANCESTOR_SUMMARY_CONTEXT_METADATA_KEY] = ancestor_context_metadata
        external_research_metadata = build_external_research_metadata(request)
        if external_research_metadata is not None:
            metadata[EXTERNAL_RESEARCH_METADATA_KEY] = external_research_metadata

        return ReadingInteractionArtifact.from_target(
            target=request.target,
            interaction_type=self.interaction_type,
            status=status,
            payload=payload,
            metadata=metadata,
            reason=reason,
        )

    @staticmethod
    def _parse_and_validate_output(
        raw_output: str | dict[str, Any],
    ) -> dict[str, object]:
        if isinstance(raw_output, str):
            try:
                parsed_output = AnalysisInteractionService._load_json_object_text(
                    raw_output
                )
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid analysis JSON: {error.msg}") from error
        elif isinstance(raw_output, dict):
            parsed_output = dict(raw_output)
        else:
            raise ValueError("analysis output must be a JSON object")

        if not isinstance(parsed_output, dict):
            raise ValueError("analysis output must be a JSON object")

        check_source_abstention(parsed_output)

        payload: dict[str, object] = {}
        for field_name in REQUIRED_ANALYSIS_FIELDS:
            value = parsed_output.get(field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"analysis output missing non-empty '{field_name}'")
            payload[field_name] = value.strip()

        return payload

    @staticmethod
    def _validate_payload_against_source(
        payload: dict[str, object],
        source_content: str,
    ) -> None:
        combined_output = " ".join(
            str(payload.get(field_name, "")) for field_name in REQUIRED_ANALYSIS_FIELDS
        )
        if AnalysisInteractionService._is_placeholder_payload(payload):
            raise ValueError("analysis output is placeholder-like and lacks insight")

        if AnalysisInteractionService._has_source_detail_overlap(
            source_content,
            combined_output,
        ):
            return

        if AnalysisInteractionService._looks_generic_without_source_detail(
            combined_output
        ):
            raise ValueError(
                "analysis output lacks concrete primary-source detail"
            )

    @staticmethod
    def _is_placeholder_payload(payload: dict[str, object]) -> bool:
        normalized_values = {
            re.sub(r"[^a-z]+", "", str(payload.get(field_name, "")).lower())
            for field_name in REQUIRED_ANALYSIS_FIELDS
        }
        return bool(normalized_values) and normalized_values.issubset(
            _GENERIC_PLACEHOLDERS
        )

    @staticmethod
    def _has_source_detail_overlap(source_content: str, output_text: str) -> bool:
        source_markers = AnalysisInteractionService._source_detail_markers(
            source_content
        )
        if not source_markers:
            return False
        output_markers = AnalysisInteractionService._source_detail_markers(output_text)
        return bool(source_markers.intersection(output_markers))

    @staticmethod
    def _source_detail_markers(text: str) -> set[str]:
        markers = {
            token.lower()
            for token in _LATIN_TOKEN_PATTERN.findall(text)
            if len(token) >= 4 and token.lower() not in _SOURCE_TOKEN_STOPWORDS
        }
        for sequence in _CJK_SEQUENCE_PATTERN.findall(text):
            markers.update(
                sequence[index : index + 2]
                for index in range(0, max(len(sequence) - 1, 0))
            )
        return markers

    @staticmethod
    def _looks_generic_without_source_detail(output_text: str) -> bool:
        normalized_output = " ".join(output_text.lower().split())
        return any(
            marker in normalized_output for marker in _GENERIC_UNSUPPORTED_MARKERS
        )

    @staticmethod
    def _load_json_object_text(raw_output: str) -> Any:
        output = raw_output.strip()
        if not output:
            raise json.JSONDecodeError("Expecting value", raw_output, 0)

        if output.startswith("```"):
            lines = output.splitlines()
            if len(lines) >= 3 and lines[-1].strip() == "```":
                output = "\n".join(lines[1:-1]).strip()

        try:
            return json.loads(output)
        except json.JSONDecodeError:
            object_start = output.find("{")
            if object_start < 0:
                raise
            parsed_output, _ = json.JSONDecoder().raw_decode(output[object_start:])
            return parsed_output
