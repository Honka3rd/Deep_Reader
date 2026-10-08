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


ANALYSIS_OUTPUT_SCHEMA_VERSION = "analysis_interaction_v1"
REQUIRED_ANALYSIS_FIELDS = ("summary", "reasoning", "explanation")


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

    @staticmethod
    def _parse_and_validate_output(
        raw_output: str | dict[str, Any],
    ) -> dict[str, object]:
        if isinstance(raw_output, str):
            try:
                parsed_output = json.loads(raw_output)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid analysis JSON: {error.msg}") from error
        elif isinstance(raw_output, dict):
            parsed_output = dict(raw_output)
        else:
            raise ValueError("analysis output must be a JSON object")

        if not isinstance(parsed_output, dict):
            raise ValueError("analysis output must be a JSON object")

        payload: dict[str, object] = {}
        for field_name in REQUIRED_ANALYSIS_FIELDS:
            value = parsed_output.get(field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"analysis output missing non-empty '{field_name}'")
            payload[field_name] = value.strip()

        return payload
