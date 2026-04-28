from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SummaryArtifact:
    """Persisted summary artifact metadata and payload."""

    content: str
    language: str | None = None
    generated_at: str | None = None
    source_hash: str | None = None
    prompt_version: str | None = None
    task_unit_split_mode: str | None = None
    semantic_top_k_candidates: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize summary artifact to JSON-compatible dictionary."""
        return {
            "content": self.content,
            "language": self.language,
            "generated_at": self.generated_at,
            "source_hash": self.source_hash,
            "prompt_version": self.prompt_version,
            "task_unit_split_mode": self.task_unit_split_mode,
            "semantic_top_k_candidates": self.semantic_top_k_candidates,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SummaryArtifact":
        """Deserialize summary artifact from dictionary."""
        return cls(
            content=str(data.get("content", "")),
            language=None if data.get("language") is None else str(data.get("language")),
            generated_at=(
                None if data.get("generated_at") is None else str(data.get("generated_at"))
            ),
            source_hash=(
                None if data.get("source_hash") is None else str(data.get("source_hash"))
            ),
            prompt_version=(
                None
                if data.get("prompt_version") is None
                else str(data.get("prompt_version"))
            ),
            task_unit_split_mode=(
                None
                if data.get("task_unit_split_mode") is None
                else str(data.get("task_unit_split_mode"))
            ),
            semantic_top_k_candidates=(
                None
                if data.get("semantic_top_k_candidates") is None
                else int(data.get("semantic_top_k_candidates"))
            ),
        )


@dataclass(frozen=True)
class QuizArtifact:
    """Persisted quiz artifact metadata and payload."""

    items: list[dict[str, Any]]
    language: str | None = None
    generated_at: str | None = None
    source_hash: str | None = None
    prompt_version: str | None = None
    quiz_schema_version: str | None = None
    task_unit_split_mode: str | None = None
    semantic_top_k_candidates: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize quiz artifact to JSON-compatible dictionary."""
        return {
            "items": list(self.items),
            "language": self.language,
            "generated_at": self.generated_at,
            "source_hash": self.source_hash,
            "prompt_version": self.prompt_version,
            "quiz_schema_version": self.quiz_schema_version,
            "task_unit_split_mode": self.task_unit_split_mode,
            "semantic_top_k_candidates": self.semantic_top_k_candidates,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "QuizArtifact":
        """Deserialize quiz artifact from dictionary."""
        raw_items = data.get("items", [])
        items: list[dict[str, Any]] = []
        for item in raw_items:
            if isinstance(item, dict):
                items.append(dict(item))
        return cls(
            items=items,
            language=None if data.get("language") is None else str(data.get("language")),
            generated_at=(
                None if data.get("generated_at") is None else str(data.get("generated_at"))
            ),
            source_hash=(
                None if data.get("source_hash") is None else str(data.get("source_hash"))
            ),
            prompt_version=(
                None
                if data.get("prompt_version") is None
                else str(data.get("prompt_version"))
            ),
            quiz_schema_version=(
                None
                if data.get("quiz_schema_version") is None
                else str(data.get("quiz_schema_version"))
            ),
            task_unit_split_mode=(
                None
                if data.get("task_unit_split_mode") is None
                else str(data.get("task_unit_split_mode"))
            ),
            semantic_top_k_candidates=(
                None
                if data.get("semantic_top_k_candidates") is None
                else int(data.get("semantic_top_k_candidates"))
            ),
        )


@dataclass(frozen=True)
class TaskArtifacts:
    """Section/task-unit level artifact container."""

    summary: SummaryArtifact | None = None
    quiz: QuizArtifact | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize task artifacts to dictionary."""
        return {
            "summary": None if self.summary is None else self.summary.to_dict(),
            "quiz": None if self.quiz is None else self.quiz.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "TaskArtifacts | None":
        """Deserialize optional task artifacts from dictionary."""
        if data is None:
            return None
        summary_payload = data.get("summary")
        quiz_payload = data.get("quiz")
        summary_artifact = (
            SummaryArtifact.from_dict(summary_payload)
            if isinstance(summary_payload, dict)
            else None
        )
        quiz_artifact = (
            QuizArtifact.from_dict(quiz_payload)
            if isinstance(quiz_payload, dict)
            else None
        )
        if summary_artifact is None and quiz_artifact is None:
            return None
        return cls(summary=summary_artifact, quiz=quiz_artifact)


@dataclass(frozen=True)
class DocumentTaskArtifacts:
    """Document-level artifact container reserved for chapter/layout task outputs."""

    chapter_artifacts: dict[str, TaskArtifacts] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize document-level task artifacts to dictionary."""
        return {
            "chapter_artifacts": {
                chapter_key: artifact.to_dict()
                for chapter_key, artifact in self.chapter_artifacts.items()
            },
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any] | None,
    ) -> "DocumentTaskArtifacts | None":
        """Deserialize optional document-level task artifacts from dictionary."""
        if data is None:
            return None

        chapter_payload = data.get("chapter_artifacts", {})
        chapter_artifacts: dict[str, TaskArtifacts] = {}
        if isinstance(chapter_payload, dict):
            for key, value in chapter_payload.items():
                if not isinstance(value, dict):
                    continue
                parsed = TaskArtifacts.from_dict(value)
                if parsed is not None:
                    chapter_artifacts[str(key)] = parsed

        metadata_payload = data.get("metadata", {})
        metadata = dict(metadata_payload) if isinstance(metadata_payload, dict) else {}

        if not chapter_artifacts and not metadata:
            return None
        return cls(chapter_artifacts=chapter_artifacts, metadata=metadata)
