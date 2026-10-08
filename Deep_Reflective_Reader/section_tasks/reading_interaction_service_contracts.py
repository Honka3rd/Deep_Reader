from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from section_tasks.reading_target_resolver import ResolvedReadingTarget


READING_INTERACTION_TYPES = frozenset(
    {"analysis", "quiz", "critical_thinking_session"}
)
READING_INTERACTION_STATUSES = frozenset(
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
ARTIFACT_CONTEXT_METADATA_KEY = "artifact_context"
ARTIFACT_REFERENCE_METADATA_KEY = "artifact_reference"
_ARTIFACT_REFERENCE_FIELDS = (
    "artifact_context_mode",
    "primary_source_evidence_ids",
    "referenced_artifact_ids",
    "referenced_artifact_types",
    "referenced_artifact_target_levels",
    "coverage_counts",
    "deduplication_hint_applied",
    "abstraction_hint_applied",
    "artifact_context_pruned_reason",
)


def build_artifact_reference_metadata(
    context_metadata: dict[str, object],
) -> dict[str, object] | None:
    """Return persisted artifact-reference metadata without context text or payloads."""
    raw_artifact_context = context_metadata.get(ARTIFACT_CONTEXT_METADATA_KEY)
    if not isinstance(raw_artifact_context, dict):
        return None

    reference_metadata: dict[str, object] = {}
    for field_name in _ARTIFACT_REFERENCE_FIELDS:
        if field_name not in raw_artifact_context:
            continue
        value = raw_artifact_context[field_name]
        if isinstance(value, list):
            reference_metadata[field_name] = list(value)
        elif isinstance(value, dict):
            reference_metadata[field_name] = dict(value)
        else:
            reference_metadata[field_name] = value

    return reference_metadata or None


@dataclass(frozen=True)
class ReadingInteractionRequest:
    """Target-agnostic request contract for reading interaction services."""

    target: ResolvedReadingTarget
    interaction_type: str
    refresh: bool = False
    secondary_context: str = ""
    context_metadata: dict[str, object] = field(default_factory=dict)
    prompt_instruction_version: str | None = None

    def __post_init__(self) -> None:
        interaction_type = self.interaction_type.strip()
        secondary_context = self.secondary_context.strip()
        if interaction_type not in READING_INTERACTION_TYPES:
            raise ValueError(
                "unsupported reading interaction type: "
                f"'{self.interaction_type}'"
            )
        object.__setattr__(self, "interaction_type", interaction_type)
        object.__setattr__(self, "secondary_context", secondary_context)
        object.__setattr__(self, "context_metadata", dict(self.context_metadata))
        if self.prompt_instruction_version is not None:
            prompt_instruction_version = self.prompt_instruction_version.strip()
            object.__setattr__(
                self,
                "prompt_instruction_version",
                prompt_instruction_version or None,
            )

    def with_secondary_context(
        self,
        secondary_context: str,
        *,
        context_metadata_updates: dict[str, object] | None = None,
    ) -> "ReadingInteractionRequest":
        """Return a copy with secondary artifact context, preserving the source target."""
        context_metadata = dict(self.context_metadata)
        if context_metadata_updates:
            context_metadata.update(context_metadata_updates)
        return ReadingInteractionRequest(
            target=self.target,
            interaction_type=self.interaction_type,
            refresh=self.refresh,
            secondary_context=secondary_context,
            context_metadata=context_metadata,
            prompt_instruction_version=self.prompt_instruction_version,
        )


@dataclass(frozen=True)
class ReadingInteractionArtifact:
    """Validated reading interaction output without persistence ownership."""

    interaction_type: str
    status: str
    target_level: str
    target_id: str
    document_id: str
    payload: dict[str, object] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)
    chapter_id: str | None = None
    section_id: str | None = None
    task_unit_id: str | None = None
    reason: str | None = None

    @classmethod
    def from_target(
        cls,
        *,
        target: ResolvedReadingTarget,
        interaction_type: str,
        status: str,
        payload: dict[str, object] | None = None,
        metadata: dict[str, object] | None = None,
        reason: str | None = None,
    ) -> "ReadingInteractionArtifact":
        """Build and validate an artifact-shaped service result from a resolved target."""
        return cls(
            interaction_type=interaction_type,
            status=status,
            target_level=target.target_level,
            target_id=target.target_id,
            document_id=target.document_id,
            chapter_id=target.chapter_id,
            section_id=target.section_id,
            task_unit_id=target.task_unit_id,
            payload={} if payload is None else dict(payload),
            metadata={} if metadata is None else dict(metadata),
            reason=reason,
        )

    def __post_init__(self) -> None:
        interaction_type = self.interaction_type.strip()
        status = self.status.strip()
        target_level = self.target_level.strip()
        target_id = self.target_id.strip()
        document_id = self.document_id.strip()
        reason = None if self.reason is None else self.reason.strip()

        if interaction_type not in READING_INTERACTION_TYPES:
            raise ValueError(
                "unsupported reading interaction type: "
                f"'{self.interaction_type}'"
            )
        if status not in READING_INTERACTION_STATUSES:
            raise ValueError(f"unsupported reading interaction status: '{self.status}'")
        if status in _CRITICAL_THINKING_ONLY_STATUSES and (
            interaction_type != "critical_thinking_session"
        ):
            raise ValueError(
                f"status '{status}' is only valid for critical_thinking_session"
            )
        if not target_level or not target_id or not document_id:
            raise ValueError("artifact target_level, target_id, and document_id are required")
        if status == "completed" and not self.payload:
            raise ValueError("completed interaction artifact requires non-empty payload")
        if status in {
            "insufficient_content",
            "generation_failed",
            "stale_target",
            "evaluation_failed",
        }:
            if not reason:
                raise ValueError(f"status '{status}' requires a reason")

        object.__setattr__(self, "interaction_type", interaction_type)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "target_level", target_level)
        object.__setattr__(self, "target_id", target_id)
        object.__setattr__(self, "document_id", document_id)
        object.__setattr__(self, "payload", dict(self.payload))
        object.__setattr__(self, "metadata", dict(self.metadata))
        object.__setattr__(self, "reason", reason)

    def to_dict(self) -> dict[str, object]:
        """Serialize the validated artifact contract without raw target objects."""
        return {
            "interaction_type": self.interaction_type,
            "status": self.status,
            "target_level": self.target_level,
            "target_id": self.target_id,
            "document_id": self.document_id,
            "chapter_id": self.chapter_id,
            "section_id": self.section_id,
            "task_unit_id": self.task_unit_id,
            "payload": dict(self.payload),
            "metadata": dict(self.metadata),
            "reason": self.reason,
        }


class ReadingInteractionService(Protocol):
    """Service contract for target-agnostic reading interactions."""

    interaction_type: str

    def generate(
        self,
        request: ReadingInteractionRequest,
    ) -> ReadingInteractionArtifact:
        """Generate one validated interaction artifact result for a resolved target."""
        ...
