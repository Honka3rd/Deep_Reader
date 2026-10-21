from __future__ import annotations

from typing import Protocol
from section_tasks.interaction_policy_identity import policy_matches, server_request, stale_projection

from section_tasks.quiz_interaction_service import QuizInteractionService
from section_tasks.reading_interaction_service_contracts import (
    ReadingInteractionArtifact,
    ReadingInteractionRequest,
)


PERSISTED_QUIZ_STATUSES = frozenset({"completed", "insufficient_content"})
REUSABLE_QUIZ_STATUSES = frozenset({"completed"})


class QuizInteractionArtifactStore(Protocol):
    """Persistence boundary required by quiz read/generate orchestration."""

    def get_quiz_artifact(
        self,
        request: ReadingInteractionRequest,
    ) -> ReadingInteractionArtifact | None:
        """Return the current persisted quiz artifact for the request target."""
        ...

    def save_quiz_artifact(
        self,
        artifact: ReadingInteractionArtifact,
    ) -> ReadingInteractionArtifact:
        """Persist and return the quiz artifact."""
        ...


class QuizInteractionOrchestrator:
    """Split quiz artifact reads from explicit generation requests."""

    interaction_type = "quiz"

    def __init__(
        self,
        *,
        service: QuizInteractionService,
        artifact_store: QuizInteractionArtifactStore,
    ) -> None:
        self._service = service
        self._artifact_store = artifact_store

    def read(
        self,
        request: ReadingInteractionRequest,
        *, artifact_store=None,
    ) -> ReadingInteractionArtifact:
        """Read current artifact state without generating missing quizzes."""
        self._validate_request(request)
        request = server_request(request)
        existing_artifact = (artifact_store or self._artifact_store).get_quiz_artifact(request)
        if existing_artifact is not None:
            preflight = self._service.preflight(request)
            if preflight is not None:
                return preflight
            if existing_artifact.status == "insufficient_content":
                return ReadingInteractionArtifact.from_target(
                    target=request.target,
                    interaction_type=self.interaction_type,
                    status="not_generated",
                )
            return stale_projection(existing_artifact, request)
        return ReadingInteractionArtifact.from_target(
            target=request.target,
            interaction_type=self.interaction_type,
            status="not_generated",
        )

    def generate(
        self,
        request: ReadingInteractionRequest,
    ) -> ReadingInteractionArtifact:
        """Generate a quiz only on explicit request, then persist valid outcomes."""
        self._validate_request(request)
        request = server_request(request)
        if not request.refresh:
            existing_artifact = self._artifact_store.get_quiz_artifact(request)
            if (
                existing_artifact is not None
                and existing_artifact.status in REUSABLE_QUIZ_STATUSES
                and policy_matches(existing_artifact, request)
                and self._service.preflight(request) is None
            ):
                return existing_artifact

        generated_artifact = self._service.generate(request)
        if generated_artifact.status in PERSISTED_QUIZ_STATUSES:
            return self._artifact_store.save_quiz_artifact(generated_artifact)
        return generated_artifact

    def _validate_request(self, request: ReadingInteractionRequest) -> None:
        if request.interaction_type != self.interaction_type:
            raise ValueError(
                "QuizInteractionOrchestrator requires interaction_type='quiz'"
            )
