from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol

from shared.artifact_target_model import ArtifactTargetLevel


ANCESTOR_SUMMARY_SCHEMA_VERSION = "ancestor_summary_v1"
ANCESTOR_SUMMARY_CONTEXT_METADATA_KEY = "ancestor_summary_context"
ANCESTOR_SUMMARY_TARGET_LEVELS = frozenset(
    {
        ArtifactTargetLevel.TASK_UNIT.value,
        ArtifactTargetLevel.SECTION.value,
        ArtifactTargetLevel.CHAPTER.value,
    }
)
ANCESTOR_SUMMARY_READINESS_STATUSES = frozenset(
    {"reusable", "missing", "stale", "refresh_required"}
)
ANCESTOR_SUMMARY_PREPARATION_OPERATIONS = frozenset({"generate", "refresh"})
ANCESTOR_SUMMARY_FAILURE_POLICIES = frozenset({"fail_fast", "degraded"})


def _normalize_optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


def _normalize_optional_positive_int(value: int | None, field_name: str) -> int | None:
    if value is None:
        return None
    if value < 1:
        raise ValueError(f"{field_name} must be positive when provided")
    return value


@dataclass(frozen=True)
class AncestorSummaryTarget:
    """Hierarchy-aware identity for an ancestor summary artifact."""

    target_level: str
    target_id: str
    document_id: str
    chapter_id: str | None = None
    section_id: str | None = None
    task_unit_id: str | None = None
    title: str | None = None

    def __post_init__(self) -> None:
        target_level = self.target_level.strip()
        target_id = self.target_id.strip()
        document_id = self.document_id.strip()
        chapter_id = _normalize_optional_text(self.chapter_id)
        section_id = _normalize_optional_text(self.section_id)
        task_unit_id = _normalize_optional_text(self.task_unit_id)
        title = _normalize_optional_text(self.title)

        if target_level == "book":
            target_level = ArtifactTargetLevel.DOCUMENT.value
        if target_level not in ANCESTOR_SUMMARY_TARGET_LEVELS:
            raise ValueError(
                f"unsupported ancestor summary target level: '{self.target_level}'"
            )
        if not target_id or not document_id:
            raise ValueError("ancestor summary target_id and document_id are required")
        if target_level == ArtifactTargetLevel.CHAPTER.value and not chapter_id:
            raise ValueError("chapter ancestor summary target requires chapter_id")
        if target_level == ArtifactTargetLevel.SECTION.value and not section_id:
            raise ValueError("section ancestor summary target requires section_id")
        if target_level == ArtifactTargetLevel.TASK_UNIT.value and not task_unit_id:
            raise ValueError("task_unit ancestor summary target requires task_unit_id")

        object.__setattr__(self, "target_level", target_level)
        object.__setattr__(self, "target_id", target_id)
        object.__setattr__(self, "document_id", document_id)
        object.__setattr__(self, "chapter_id", chapter_id)
        object.__setattr__(self, "section_id", section_id)
        object.__setattr__(self, "task_unit_id", task_unit_id)
        object.__setattr__(self, "title", title)

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_level": self.target_level,
            "target_id": self.target_id,
            "document_id": self.document_id,
            "chapter_id": self.chapter_id,
            "section_id": self.section_id,
            "task_unit_id": self.task_unit_id,
            "title": self.title,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AncestorSummaryTarget":
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
            title=None if data.get("title") is None else str(data.get("title")),
        )


@dataclass(frozen=True)
class AncestorSummaryArtifact:
    """Compact ancestor summary used as reading-interaction prompt context."""

    target: AncestorSummaryTarget
    summary: str
    schema_version: str = ANCESTOR_SUMMARY_SCHEMA_VERSION
    prompt_instruction_version: str | None = None
    source_structure_version: int | None = None
    source_hash: str | None = None
    document_language: str | None = None
    generated_at: str | None = None
    artifact_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        summary = self.summary.strip()
        schema_version = self.schema_version.strip()
        prompt_instruction_version = _normalize_optional_text(
            self.prompt_instruction_version
        )
        source_structure_version = _normalize_optional_positive_int(
            self.source_structure_version,
            "source_structure_version",
        )
        source_hash = _normalize_optional_text(self.source_hash)
        document_language = _normalize_optional_text(self.document_language)
        generated_at = _normalize_optional_text(self.generated_at)
        artifact_id = _normalize_optional_text(self.artifact_id)

        if not summary:
            raise ValueError("ancestor summary requires non-empty summary text")
        if not schema_version:
            raise ValueError("ancestor summary schema_version is required")
        if generated_at is not None:
            try:
                datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
            except ValueError as error:
                raise ValueError("generated_at must be ISO-8601 when provided") from error

        object.__setattr__(self, "summary", summary)
        object.__setattr__(self, "schema_version", schema_version)
        object.__setattr__(
            self,
            "prompt_instruction_version",
            prompt_instruction_version,
        )
        object.__setattr__(
            self,
            "source_structure_version",
            source_structure_version,
        )
        object.__setattr__(self, "source_hash", source_hash)
        object.__setattr__(self, "document_language", document_language)
        object.__setattr__(self, "generated_at", generated_at)
        object.__setattr__(self, "artifact_id", artifact_id)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "target": self.target.to_dict(),
            "summary": self.summary,
            "schema_version": self.schema_version,
            "prompt_instruction_version": self.prompt_instruction_version,
            "source_structure_version": self.source_structure_version,
            "source_hash": self.source_hash,
            "document_language": self.document_language,
            "generated_at": self.generated_at,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AncestorSummaryArtifact":
        return cls(
            artifact_id=(
                None if data.get("artifact_id") is None else str(data.get("artifact_id"))
            ),
            target=AncestorSummaryTarget.from_dict(data["target"]),
            summary=str(data["summary"]),
            schema_version=str(
                data.get("schema_version", ANCESTOR_SUMMARY_SCHEMA_VERSION)
            ),
            prompt_instruction_version=(
                None
                if data.get("prompt_instruction_version") is None
                else str(data.get("prompt_instruction_version"))
            ),
            source_structure_version=(
                None
                if data.get("source_structure_version") is None
                else int(data.get("source_structure_version"))
            ),
            source_hash=(
                None if data.get("source_hash") is None else str(data.get("source_hash"))
            ),
            document_language=(
                None
                if data.get("document_language") is None
                else str(data.get("document_language"))
            ),
            generated_at=(
                None
                if data.get("generated_at") is None
                else str(data.get("generated_at"))
            ),
            metadata={
                str(key): value for key, value in dict(data.get("metadata", {})).items()
            },
        )

    def to_prompt_reference(self) -> dict[str, Any]:
        """Return bounded prompt/provenance data, not raw ancestor source text."""
        return {
            "artifact_id": self.artifact_id,
            "target_level": self.target.target_level,
            "target_id": self.target.target_id,
            "title": self.target.title,
            "summary": self.summary,
            "schema_version": self.schema_version,
            "prompt_instruction_version": self.prompt_instruction_version,
            "source_structure_version": self.source_structure_version,
            "source_hash": self.source_hash,
            "document_language": self.document_language,
            "generated_at": self.generated_at,
        }


@dataclass(frozen=True)
class AncestorSummaryPromptContext:
    """Ordered ancestor summary references for one reading-interaction request."""

    summaries: tuple[AncestorSummaryArtifact, ...] = field(default_factory=tuple)
    context_status: str = "complete"
    missing_target_levels: tuple[str, ...] = field(default_factory=tuple)
    stale_target_levels: tuple[str, ...] = field(default_factory=tuple)
    pruned_reason: str | None = None

    def __post_init__(self) -> None:
        summaries = tuple(self.summaries)
        context_status = self.context_status.strip()
        missing_target_levels = tuple(
            level.strip() for level in self.missing_target_levels if level.strip()
        )
        stale_target_levels = tuple(
            level.strip() for level in self.stale_target_levels if level.strip()
        )
        pruned_reason = _normalize_optional_text(self.pruned_reason)

        if context_status not in {"complete", "partial", "empty"}:
            raise ValueError(
                f"unsupported ancestor summary context status: '{self.context_status}'"
            )
        if context_status == "complete" and (missing_target_levels or stale_target_levels):
            raise ValueError("complete context cannot declare missing or stale levels")
        if context_status == "empty" and summaries:
            raise ValueError("empty context cannot include ancestor summaries")

        object.__setattr__(self, "summaries", summaries)
        object.__setattr__(self, "context_status", context_status)
        object.__setattr__(self, "missing_target_levels", missing_target_levels)
        object.__setattr__(self, "stale_target_levels", stale_target_levels)
        object.__setattr__(self, "pruned_reason", pruned_reason)

    def to_metadata(self) -> dict[str, Any]:
        return {
            ANCESTOR_SUMMARY_CONTEXT_METADATA_KEY: {
                "context_status": self.context_status,
                "summary_count": len(self.summaries),
                "summary_artifact_ids": [
                    summary.artifact_id
                    for summary in self.summaries
                    if summary.artifact_id is not None
                ],
                "summary_target_levels": [
                    summary.target.target_level for summary in self.summaries
                ],
                "summary_target_ids": [
                    summary.target.target_id for summary in self.summaries
                ],
                "missing_target_levels": list(self.missing_target_levels),
                "stale_target_levels": list(self.stale_target_levels),
                "pruned_reason": self.pruned_reason,
            }
        }

    def to_prompt_references(self) -> list[dict[str, Any]]:
        return [summary.to_prompt_reference() for summary in self.summaries]


@dataclass(frozen=True)
class AncestorSummaryRequirement:
    """Current source/version requirement for one ancestor summary lookup."""

    target: AncestorSummaryTarget
    doc_name: str | None = None
    schema_version: str = ANCESTOR_SUMMARY_SCHEMA_VERSION
    prompt_instruction_version: str | None = None
    source_structure_version: int | None = None
    source_hash: str | None = None
    document_language: str | None = None

    def __post_init__(self) -> None:
        doc_name = _normalize_optional_text(self.doc_name)
        schema_version = self.schema_version.strip()
        prompt_instruction_version = _normalize_optional_text(
            self.prompt_instruction_version
        )
        source_structure_version = _normalize_optional_positive_int(
            self.source_structure_version,
            "source_structure_version",
        )
        source_hash = _normalize_optional_text(self.source_hash)
        document_language = _normalize_optional_text(self.document_language)

        if not schema_version:
            raise ValueError("ancestor summary requirement schema_version is required")

        object.__setattr__(self, "doc_name", doc_name)
        object.__setattr__(self, "schema_version", schema_version)
        object.__setattr__(
            self,
            "prompt_instruction_version",
            prompt_instruction_version,
        )
        object.__setattr__(
            self,
            "source_structure_version",
            source_structure_version,
        )
        object.__setattr__(self, "source_hash", source_hash)
        object.__setattr__(self, "document_language", document_language)

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target.to_dict(),
            "doc_name": self.doc_name,
            "schema_version": self.schema_version,
            "prompt_instruction_version": self.prompt_instruction_version,
            "source_structure_version": self.source_structure_version,
            "source_hash": self.source_hash,
            "document_language": self.document_language,
        }


@dataclass(frozen=True)
class AncestorSummaryReadinessResult:
    """Reusable/missing/stale decision for one ancestor summary requirement."""

    status: str
    requirement: AncestorSummaryRequirement
    summary: AncestorSummaryArtifact | None = None
    reason: str | None = None
    stale_fields: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        status = self.status.strip()
        reason = _normalize_optional_text(self.reason)
        stale_fields = tuple(field_name.strip() for field_name in self.stale_fields)

        if status not in ANCESTOR_SUMMARY_READINESS_STATUSES:
            raise ValueError(f"unsupported ancestor summary readiness status: '{status}'")
        if status == "reusable":
            if self.summary is None:
                raise ValueError("reusable ancestor summary result requires a summary")
            if reason or stale_fields:
                raise ValueError("reusable ancestor summary result cannot be stale")
        else:
            if not reason:
                raise ValueError(f"status '{status}' requires a reason")
        if status == "stale" and not stale_fields:
            raise ValueError("stale ancestor summary result requires stale_fields")
        if status != "stale" and stale_fields:
            raise ValueError("only stale ancestor summary results may declare stale_fields")

        object.__setattr__(self, "status", status)
        object.__setattr__(self, "reason", reason)
        object.__setattr__(self, "stale_fields", stale_fields)

    @property
    def is_reusable(self) -> bool:
        return self.status == "reusable"

    def to_metadata(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "target_level": self.requirement.target.target_level,
            "target_id": self.requirement.target.target_id,
            "summary_artifact_id": (
                None if self.summary is None else self.summary.artifact_id
            ),
            "reason": self.reason,
            "stale_fields": list(self.stale_fields),
            "schema_version": self.requirement.schema_version,
            "prompt_instruction_version": self.requirement.prompt_instruction_version,
            "source_structure_version": self.requirement.source_structure_version,
            "source_hash": self.requirement.source_hash,
            "document_language": self.requirement.document_language,
        }


def evaluate_ancestor_summary_readiness(
    requirement: AncestorSummaryRequirement,
    existing_summary: AncestorSummaryArtifact | None,
    *,
    refresh: bool = False,
) -> AncestorSummaryReadinessResult:
    """Return a pure readiness decision without generation, persistence, or UI state."""
    if existing_summary is None:
        return AncestorSummaryReadinessResult(
            status="missing",
            requirement=requirement,
            reason="ancestor summary is missing",
        )
    if refresh:
        return AncestorSummaryReadinessResult(
            status="refresh_required",
            requirement=requirement,
            summary=existing_summary,
            reason="explicit refresh requested",
        )

    stale_fields: list[str] = []
    if existing_summary.target.to_dict() != requirement.target.to_dict():
        stale_fields.append("target")
    if existing_summary.schema_version != requirement.schema_version:
        stale_fields.append("schema_version")
    if (
        requirement.prompt_instruction_version is not None
        and existing_summary.prompt_instruction_version
        != requirement.prompt_instruction_version
    ):
        stale_fields.append("prompt_instruction_version")
    if (
        requirement.source_structure_version is not None
        and existing_summary.source_structure_version
        != requirement.source_structure_version
    ):
        stale_fields.append("source_structure_version")
    if (
        requirement.source_hash is not None
        and existing_summary.source_hash != requirement.source_hash
    ):
        stale_fields.append("source_hash")
    if (
        requirement.document_language is not None
        and existing_summary.document_language != requirement.document_language
    ):
        stale_fields.append("document_language")

    if stale_fields:
        return AncestorSummaryReadinessResult(
            status="stale",
            requirement=requirement,
            summary=existing_summary,
            reason="ancestor summary does not match current requirement",
            stale_fields=tuple(stale_fields),
        )

    return AncestorSummaryReadinessResult(
        status="reusable",
        requirement=requirement,
        summary=existing_summary,
    )


class AncestorSummaryReader(Protocol):
    """Read-only boundary for current ancestor summaries."""

    def get_ancestor_summary(
        self,
        requirement: AncestorSummaryRequirement,
    ) -> AncestorSummaryArtifact | None:
        """Return the current summary for the requirement target, if present."""
        ...


class AncestorSummaryGenerator(Protocol):
    """Generation boundary for one required ancestor summary."""

    def generate_ancestor_summary(
        self,
        requirement: AncestorSummaryRequirement,
    ) -> AncestorSummaryArtifact:
        """Generate a compact ancestor summary for the requirement target."""
        ...


class AncestorSummaryPreparationError(ValueError):
    """Raised when ancestor context cannot be prepared without degrading context."""

    def __init__(
        self,
        message: str,
        *,
        requirement: AncestorSummaryRequirement,
        readiness: AncestorSummaryReadinessResult,
        cause: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.requirement = requirement
        self.readiness = readiness
        self.cause = cause


@dataclass(frozen=True)
class AncestorSummaryPreparationFailure:
    """Bounded provenance for an ancestor summary preparation failure."""

    requirement: AncestorSummaryRequirement
    readiness_status: str
    reason: str
    error_type: str | None = None

    def __post_init__(self) -> None:
        readiness_status = self.readiness_status.strip()
        reason = self.reason.strip()
        error_type = _normalize_optional_text(self.error_type)
        if readiness_status not in ANCESTOR_SUMMARY_READINESS_STATUSES:
            raise ValueError(
                "unsupported ancestor summary preparation failure readiness status: "
                f"'{self.readiness_status}'"
            )
        if not reason:
            raise ValueError("ancestor summary preparation failure requires a reason")
        object.__setattr__(self, "readiness_status", readiness_status)
        object.__setattr__(self, "reason", reason)
        object.__setattr__(self, "error_type", error_type)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "target_level": self.requirement.target.target_level,
            "target_id": self.requirement.target.target_id,
            "readiness_status": self.readiness_status,
            "reason": self.reason,
            "error_type": self.error_type,
        }


@dataclass(frozen=True)
class AncestorSummaryPreparationResult:
    """Prepared ancestor context for one explicit interaction generation request."""

    prompt_context: AncestorSummaryPromptContext
    readiness_results: tuple[AncestorSummaryReadinessResult, ...]
    generated_summaries: tuple[AncestorSummaryArtifact, ...] = field(default_factory=tuple)
    preparation_failures: tuple[AncestorSummaryPreparationFailure, ...] = (
        field(default_factory=tuple)
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "readiness_results", tuple(self.readiness_results))
        object.__setattr__(self, "generated_summaries", tuple(self.generated_summaries))
        object.__setattr__(
            self,
            "preparation_failures",
            tuple(self.preparation_failures),
        )

    def to_metadata(self) -> dict[str, Any]:
        metadata = self.prompt_context.to_metadata()
        metadata[ANCESTOR_SUMMARY_CONTEXT_METADATA_KEY].update(
            {
                "readiness": [
                    result.to_metadata() for result in self.readiness_results
                ],
                "generated_summary_count": len(self.generated_summaries),
                "generated_summary_target_levels": [
                    summary.target.target_level for summary in self.generated_summaries
                ],
                "generated_summary_target_ids": [
                    summary.target.target_id for summary in self.generated_summaries
                ],
                "preparation_failure_count": len(self.preparation_failures),
                "preparation_failures": [
                    failure.to_metadata() for failure in self.preparation_failures
                ],
            }
        )
        return metadata


class AncestorSummaryPreparationService:
    """Prepare ancestor summaries only for explicit generate/refresh operations."""

    def __init__(
        self,
        *,
        reader: AncestorSummaryReader,
        generator: AncestorSummaryGenerator,
    ) -> None:
        self._reader = reader
        self._generator = generator

    def prepare(
        self,
        requirements: tuple[AncestorSummaryRequirement, ...],
        *,
        operation: str,
        failure_policy: str = "fail_fast",
    ) -> AncestorSummaryPreparationResult:
        """Return prepared context; reject read-like operations before any side effects."""
        operation = operation.strip()
        if operation not in ANCESTOR_SUMMARY_PREPARATION_OPERATIONS:
            raise ValueError(
                "ancestor summary preparation is only allowed for explicit "
                "generate or refresh operations"
            )
        failure_policy = failure_policy.strip()
        if failure_policy not in ANCESTOR_SUMMARY_FAILURE_POLICIES:
            raise ValueError(
                "ancestor summary failure_policy must be 'fail_fast' or 'degraded'"
            )

        requirements = tuple(requirements)
        if not requirements:
            return AncestorSummaryPreparationResult(
                prompt_context=AncestorSummaryPromptContext(context_status="empty"),
                readiness_results=(),
            )

        refresh = operation == "refresh"
        readiness_results: list[AncestorSummaryReadinessResult] = []
        prepared_summaries: list[AncestorSummaryArtifact] = []
        generated_summaries: list[AncestorSummaryArtifact] = []
        preparation_failures: list[AncestorSummaryPreparationFailure] = []

        for requirement in requirements:
            existing_summary = self._reader.get_ancestor_summary(requirement)
            readiness = evaluate_ancestor_summary_readiness(
                requirement,
                existing_summary,
                refresh=refresh,
            )
            readiness_results.append(readiness)

            if readiness.is_reusable:
                if readiness.summary is None:
                    raise ValueError("reusable ancestor summary is missing summary")
                prepared_summaries.append(readiness.summary)
                continue

            try:
                generated_summary = self._generator.generate_ancestor_summary(
                    requirement
                )
            except Exception as error:
                failure = AncestorSummaryPreparationFailure(
                    requirement=requirement,
                    readiness_status=readiness.status,
                    reason=str(error) or "ancestor summary generation failed",
                    error_type=type(error).__name__,
                )
                if failure_policy == "fail_fast":
                    raise AncestorSummaryPreparationError(
                        "ancestor summary preparation failed: "
                        f"{failure.reason}",
                        requirement=requirement,
                        readiness=readiness,
                        cause=error,
                    ) from error
                preparation_failures.append(failure)
                continue

            generated_readiness = evaluate_ancestor_summary_readiness(
                requirement,
                generated_summary,
            )
            if not generated_readiness.is_reusable:
                failure = AncestorSummaryPreparationFailure(
                    requirement=requirement,
                    readiness_status=generated_readiness.status,
                    reason=(
                        generated_readiness.reason
                        or "generated ancestor summary is not reusable"
                    ),
                    error_type=None,
                )
                if failure_policy == "fail_fast":
                    raise AncestorSummaryPreparationError(
                        "generated ancestor summary does not satisfy current "
                        "requirement",
                        requirement=requirement,
                        readiness=generated_readiness,
                    )
                preparation_failures.append(failure)
                continue
            prepared_summaries.append(generated_summary)
            generated_summaries.append(generated_summary)

        context_status = "complete"
        missing_target_levels: tuple[str, ...] = ()
        stale_target_levels: tuple[str, ...] = ()
        pruned_reason = None
        if preparation_failures:
            context_status = "partial"
            missing_target_levels = tuple(
                failure.requirement.target.target_level
                for failure in preparation_failures
                if failure.readiness_status in {"missing", "refresh_required"}
            )
            stale_target_levels = tuple(
                failure.requirement.target.target_level
                for failure in preparation_failures
                if failure.readiness_status == "stale"
            )
            pruned_reason = "ancestor_summary_preparation_failed"

        return AncestorSummaryPreparationResult(
            prompt_context=AncestorSummaryPromptContext(
                summaries=tuple(prepared_summaries),
                context_status=context_status,
                missing_target_levels=missing_target_levels,
                stale_target_levels=stale_target_levels,
                pruned_reason=pruned_reason,
            ),
            readiness_results=tuple(readiness_results),
            generated_summaries=tuple(generated_summaries),
            preparation_failures=tuple(preparation_failures),
        )
