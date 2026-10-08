from __future__ import annotations

from typing import Protocol

from section_tasks.quiz_interaction_service import QuizInteractionService
from section_tasks.reading_interaction_service_contracts import (
    ReadingInteractionArtifact,
    ReadingInteractionRequest,
)


PERSISTED_QUIZ_STATUSES = frozenset({"completed", "insufficient_content"})


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
    ) -> ReadingInteractionArtifact:
        """Read current artifact state without generating missing quizzes."""
        self._validate_request(request)
        existing_artifact = self._artifact_store.get_quiz_artifact(request)
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
        """Generate a quiz only on explicit request, then persist valid outcomes."""
        self._validate_request(request)
        if not request.refresh:
            existing_artifact = self._artifact_store.get_quiz_artifact(request)
            if existing_artifact is not None:
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
