from dataclasses import dataclass, field
from enum import Enum
from typing import Any

MetricValue = int | float


class SectionTaskMode(str, Enum):
    """Section-to-task-unit relationship mode for section-first UI rendering."""

    DIRECT = "direct"
    SPLIT = "split"
    MERGED = "merged"


@dataclass(frozen=True)
class TaskUnitDTO:
    """Frontend-safe task-unit metadata without large content payload."""

    unit_id: str
    title: str | None
    container_title: str | None
    source_section_ids: list[str]
    is_fallback_generated: bool
    artifacts: "ArtifactAvailabilityDTO | None" = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize task-unit metadata into JSON-friendly dictionary."""
        return {
            "unit_id": self.unit_id,
            "title": self.title,
            "container_title": self.container_title,
            "source_section_ids": list(self.source_section_ids),
            "is_fallback_generated": self.is_fallback_generated,
            "artifacts": (
                None
                if self.artifacts is None
                else self.artifacts.to_dict()
            ),
        }


@dataclass(frozen=True)
class ArtifactAvailabilityDTO:
    """Lightweight artifact availability metadata for UI cache awareness."""

    has_summary: bool = False
    has_quiz: bool = False
    summary_cache_valid: bool | None = None
    quiz_cache_valid: bool | None = None
    summary_invalid_reason: str | None = None
    quiz_invalid_reason: str | None = None
    summary_generated_at: str | None = None
    quiz_generated_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize artifact availability into JSON-friendly dictionary."""
        return {
            "has_summary": self.has_summary,
            "has_quiz": self.has_quiz,
            "summary_cache_valid": self.summary_cache_valid,
            "quiz_cache_valid": self.quiz_cache_valid,
            "summary_invalid_reason": self.summary_invalid_reason,
            "quiz_invalid_reason": self.quiz_invalid_reason,
            "summary_generated_at": self.summary_generated_at,
            "quiz_generated_at": self.quiz_generated_at,
        }


@dataclass(frozen=True)
class DocumentTaskLayoutSectionDTO:
    """Section-first layout node with embedded task-unit metadata."""

    section_id: str
    title: str | None
    container_title: str | None
    section_role: str | None
    parent_chapter_id: str | None
    section_kind: str | None
    is_implicit_section: bool
    task_mode: SectionTaskMode
    task_units: list[TaskUnitDTO]
    artifacts: ArtifactAvailabilityDTO | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize section layout node into JSON-friendly dictionary."""
        return {
            "section_id": self.section_id,
            "title": self.title,
            "container_title": self.container_title,
            "section_role": self.section_role,
            "parent_chapter_id": self.parent_chapter_id,
            "section_kind": self.section_kind,
            "is_implicit_section": self.is_implicit_section,
            "task_mode": self.task_mode.value,
            "task_units": [task_unit.to_dict() for task_unit in self.task_units],
            "artifacts": (
                None
                if self.artifacts is None
                else self.artifacts.to_dict()
            ),
        }


@dataclass(frozen=True)
class DocumentTaskLayoutChapterDTO:
    """Hierarchy-first chapter node for document -> chapter -> section -> task_unit layout."""

    chapter_id: str
    title: str | None
    level: int
    chapter_role: str | None
    sections: list[DocumentTaskLayoutSectionDTO]
    artifacts: ArtifactAvailabilityDTO | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize chapter layout node into JSON-friendly dictionary."""
        return {
            "chapter_id": self.chapter_id,
            "title": self.title,
            "level": self.level,
            "chapter_role": self.chapter_role,
            "sections": [section.to_dict() for section in self.sections],
            "artifacts": (
                None
                if self.artifacts is None
                else self.artifacts.to_dict()
            ),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class EnhancedParseRecommendationDTO:
    """Frontend-safe enhanced parser recommendation payload."""

    should_recommend: bool
    score: int
    reasons: list[str]
    metrics: dict[str, MetricValue]

    def to_dict(self) -> dict[str, Any]:
        """Serialize recommendation payload into JSON-friendly dictionary."""
        return {
            "should_recommend": self.should_recommend,
            "score": self.score,
            "reasons": list(self.reasons),
            "metrics": dict(self.metrics),
        }


@dataclass(frozen=True)
class ProfileStructureDiagnosticsDTO:
    """Lightweight mixed-source diagnostics for task-layout observability.

    Source-of-truth split:
    - parser/post shape and title risk fields come from persisted profile metadata.
    - task-unit availability/coverage fields come from *current* layout sections.
    This DTO is projection-only and must not be treated as persisted profile state.
    """

    parser_metadata_shape: str | None = None
    post_actual_structure_shape: str | None = None
    title_uniqueness_risk: str | None = None
    title_target_requires_id: bool = False
    task_unit_stats_available: bool = False
    task_unit_section_coverage: float | None = None
    parser_post_shape_mismatch: bool = False
    enhanced_parse_hint: str | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize profile diagnostics into JSON-friendly dictionary."""
        return {
            "parser_metadata_shape": self.parser_metadata_shape,
            "post_actual_structure_shape": self.post_actual_structure_shape,
            "title_uniqueness_risk": self.title_uniqueness_risk,
            "title_target_requires_id": self.title_target_requires_id,
            "task_unit_stats_available": self.task_unit_stats_available,
            "task_unit_section_coverage": self.task_unit_section_coverage,
            "parser_post_shape_mismatch": self.parser_post_shape_mismatch,
            "enhanced_parse_hint": self.enhanced_parse_hint,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class DocumentTaskLayout:
    """Frontend-consumable layout with hierarchy-first chapters tree.

    `sections` / `task_units` / `chapter_artifacts` are internal transitional
    DTO fields for backward-compatible tests/mappers. Public REST response is
    chapters-first and should not rely on those top-level legacy mirrors.
    """

    document_id: str
    title: str
    language: str | None
    chapters: list[DocumentTaskLayoutChapterDTO]
    sections: list[DocumentTaskLayoutSectionDTO] = field(default_factory=list)
    task_units: list[TaskUnitDTO] = field(default_factory=list)
    chapter_artifacts: dict[str, ArtifactAvailabilityDTO] = field(default_factory=dict)
    enhanced_parse_recommendation: EnhancedParseRecommendationDTO | None = None
    profile_diagnostics: ProfileStructureDiagnosticsDTO | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize layout into JSON-friendly dictionary."""
        return {
            "document_id": self.document_id,
            "title": self.title,
            "language": self.language,
            "chapters": [chapter.to_dict() for chapter in self.chapters],
            "sections": [section.to_dict() for section in self.sections],
            "task_units": [task_unit.to_dict() for task_unit in self.task_units],
            "chapter_artifacts": {
                chapter_key: artifact.to_dict()
                for chapter_key, artifact in self.chapter_artifacts.items()
            },
            "enhanced_parse_recommendation": (
                None
                if self.enhanced_parse_recommendation is None
                else self.enhanced_parse_recommendation.to_dict()
            ),
            "profile_diagnostics": (
                None
                if self.profile_diagnostics is None
                else self.profile_diagnostics.to_dict()
            ),
        }
