from __future__ import annotations

from typing import Protocol

from section_tasks.analysis_interaction_service import AnalysisInteractionService
from section_tasks.reading_interaction_service_contracts import (
    ReadingInteractionArtifact,
    ReadingInteractionRequest,
)


PERSISTED_ANALYSIS_STATUSES = frozenset({"completed", "insufficient_content"})


class AnalysisInteractionArtifactStore(Protocol):
    """Persistence boundary required by analysis read/generate orchestration."""

    def get_analysis_artifact(
        self,
        request: ReadingInteractionRequest,
    ) -> ReadingInteractionArtifact | None:
        """Return the current persisted analysis artifact for the request target."""
        ...

    def save_analysis_artifact(
        self,
        artifact: ReadingInteractionArtifact,
    ) -> ReadingInteractionArtifact:
        """Persist and return the analysis artifact."""
        ...


class AnalysisInteractionOrchestrator:
    """Split analysis artifact reads from explicit generation requests."""

    interaction_type = "analysis"

    def __init__(
        self,
        *,
        service: AnalysisInteractionService,
        artifact_store: AnalysisInteractionArtifactStore,
    ) -> None:
        self._service = service
        self._artifact_store = artifact_store

    def read(
        self,
        request: ReadingInteractionRequest,
    ) -> ReadingInteractionArtifact:
        """Read current artifact state without generating missing analysis."""
        self._validate_request(request)
        existing_artifact = self._artifact_store.get_analysis_artifact(request)
        if existing_artifact is not None:
            return existing_artifact
        return ReadingInteractionArtifact.from_target(
            target=request.target,
            interaction_type=self.interaction_type,
            status="not_generated",
        )

    def generate(
        self,
        request: ReadingInteractionRequest,
    ) -> ReadingInteractionArtifact:
        """Generate analysis only on explicit request, then persist valid outcomes."""
        self._validate_request(request)
        if not request.refresh:
            existing_artifact = self._artifact_store.get_analysis_artifact(request)
            if existing_artifact is not None:
                return existing_artifact

        generated_artifact = self._service.generate(request)
        if generated_artifact.status in PERSISTED_ANALYSIS_STATUSES:
            return self._artifact_store.save_analysis_artifact(generated_artifact)
        return generated_artifact

    def _validate_request(self, request: ReadingInteractionRequest) -> None:
        if request.interaction_type != self.interaction_type:
            raise ValueError(
                "AnalysisInteractionOrchestrator requires interaction_type='analysis'"
            )
