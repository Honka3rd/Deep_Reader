from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from shared.artifact_target_model import ArtifactTargetLevel


COMMON_READING_INTERACTION_ARTIFACT_TYPES = frozenset(
    {"analysis", "quiz", "critical_thinking_session"}
)
COMMON_ARTIFACT_STATUSES = frozenset(
    {
        "not_generated",
        "completed",
        "insufficient_content",
        "generation_failed",
        "stale_target",
        "question_generated",
        "answer_submitted",
        "evaluation_failed",
    }
)
_CRITICAL_THINKING_ONLY_STATUSES = frozenset(
    {"question_generated", "answer_submitted", "evaluation_failed"}
)
_REASON_REQUIRED_STATUSES = frozenset(
    {"insufficient_content", "generation_failed", "stale_target", "evaluation_failed"}
)


def _normalize_optional_id(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


@dataclass(frozen=True)
class CommonArtifactTarget:
    """Hierarchy-aware target metadata for the common artifact entity."""

    target_level: str
    target_id: str
    document_id: str
    chapter_id: str | None = None
    section_id: str | None = None
    task_unit_id: str | None = None
    content_block_id: str | None = None

    def __post_init__(self) -> None:
        target_level = self.target_level.strip()
        if target_level == "book":
            target_level = ArtifactTargetLevel.DOCUMENT.value
        target_id = self.target_id.strip()
        document_id = self.document_id.strip()
        chapter_id = _normalize_optional_id(self.chapter_id)
        section_id = _normalize_optional_id(self.section_id)
        task_unit_id = _normalize_optional_id(self.task_unit_id)
        content_block_id = _normalize_optional_id(self.content_block_id)

        if not target_id or not document_id:
            raise ValueError("common artifact target_id and document_id are required")
        if target_level not in {level.value for level in ArtifactTargetLevel}:
            raise ValueError(f"unsupported common artifact target level: '{target_level}'")
        if target_level == ArtifactTargetLevel.CHAPTER.value and not chapter_id:
            raise ValueError("chapter artifact target requires chapter_id")
        if target_level == ArtifactTargetLevel.SECTION.value and not section_id:
            raise ValueError("section artifact target requires section_id")
        if target_level == ArtifactTargetLevel.TASK_UNIT.value and not task_unit_id:
            raise ValueError("task_unit artifact target requires task_unit_id")
        if (
            target_level == ArtifactTargetLevel.CONTENT_BLOCK.value
            and (not task_unit_id or not content_block_id)
        ):
            raise ValueError(
                "content_block artifact target requires task_unit_id and content_block_id"
            )

        object.__setattr__(self, "target_level", target_level)
        object.__setattr__(self, "target_id", target_id)
        object.__setattr__(self, "document_id", document_id)
        object.__setattr__(self, "chapter_id", chapter_id)
        object.__setattr__(self, "section_id", section_id)
        object.__setattr__(self, "task_unit_id", task_unit_id)
        object.__setattr__(self, "content_block_id", content_block_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_level": self.target_level,
            "target_id": self.target_id,
            "document_id": self.document_id,
            "chapter_id": self.chapter_id,
            "section_id": self.section_id,
            "task_unit_id": self.task_unit_id,
            "content_block_id": self.content_block_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CommonArtifactTarget":
        return cls(
            target_level=str(data["target_level"]),
            target_id=str(data["target_id"]),
            document_id=str(data["document_id"]),
            chapter_id=(
                None if data.get("chapter_id") is None else str(data.get("chapter_id"))
            ),
            section_id=(
                None if data.get("section_id") is None else str(data.get("section_id"))
            ),
            task_unit_id=(
                None
                if data.get("task_unit_id") is None
                else str(data.get("task_unit_id"))
            ),
            content_block_id=(
                None
                if data.get("content_block_id") is None
                else str(data.get("content_block_id"))
            ),
        )


@dataclass(frozen=True)
class CommonArtifact:
    """Single artifact entity shape for typed interaction outputs."""

    artifact_type: str
    status: str
    target: CommonArtifactTarget
    payload: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    reason: str | None = None
    artifact_id: str | None = None
    source_structure_version: int | None = None
    source_hash: str | None = None

    def __post_init__(self) -> None:
        artifact_type = self.artifact_type.strip()
        status = self.status.strip()
        reason = None if self.reason is None else self.reason.strip()
        source_hash = None if self.source_hash is None else self.source_hash.strip()

        if artifact_type not in COMMON_READING_INTERACTION_ARTIFACT_TYPES:
            raise ValueError(f"unsupported common artifact type: '{self.artifact_type}'")
        if status not in COMMON_ARTIFACT_STATUSES:
            raise ValueError(f"unsupported common artifact status: '{self.status}'")
        if (
            status in _CRITICAL_THINKING_ONLY_STATUSES
            and artifact_type != "critical_thinking_session"
        ):
            raise ValueError(
                f"status '{status}' is only valid for critical_thinking_session"
            )
        if status == "completed" and not self.payload:
            raise ValueError("completed common artifact requires non-empty payload")
        if status in _REASON_REQUIRED_STATUSES and not reason:
            raise ValueError(f"status '{status}' requires a reason")
        if self.source_structure_version is not None and self.source_structure_version < 1:
            raise ValueError("source_structure_version must be positive when provided")

        object.__setattr__(self, "artifact_type", artifact_type)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "payload", dict(self.payload))
        object.__setattr__(self, "metadata", dict(self.metadata))
        object.__setattr__(self, "reason", reason or None)
        object.__setattr__(self, "source_hash", source_hash or None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "artifact_type": self.artifact_type,
            "status": self.status,
            "target": self.target.to_dict(),
            "payload": dict(self.payload),
            "metadata": dict(self.metadata),
            "reason": self.reason,
            "source_structure_version": self.source_structure_version,
            "source_hash": self.source_hash,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CommonArtifact":
        return cls(
            artifact_id=(
                None if data.get("artifact_id") is None else str(data.get("artifact_id"))
            ),
            artifact_type=str(data["artifact_type"]),
            status=str(data["status"]),
            target=CommonArtifactTarget.from_dict(data["target"]),
            payload=dict(data.get("payload", {})),
            metadata=dict(data.get("metadata", {})),
            reason=None if data.get("reason") is None else str(data.get("reason")),
            source_structure_version=(
                None
                if data.get("source_structure_version") is None
                else int(data.get("source_structure_version"))
            ),
            source_hash=(
                None if data.get("source_hash") is None else str(data.get("source_hash"))
            ),
        )
