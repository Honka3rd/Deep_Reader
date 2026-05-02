from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DocumentProfile:
    """Document profile data used for prompt conditioning."""
    topic: str
    summary: str
    document_language: str
    structure_profile: "DocumentStructureProfile | None" = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "topic": self.topic,
            "summary": self.summary,
            "document_language": self.document_language,
        }
        if self.structure_profile is not None:
            payload["structure_profile"] = self.structure_profile.to_dict()
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DocumentProfile":
        structure_profile_payload = payload.get("structure_profile")
        structure_profile = (
            None
            if not isinstance(structure_profile_payload, dict)
            else DocumentStructureProfile.from_dict(structure_profile_payload)
        )
        return cls(
            topic=str(payload.get("topic") or "").strip(),
            summary=str(payload.get("summary") or "").strip(),
            document_language=str(payload.get("document_language") or "").strip(),
            structure_profile=structure_profile,
        )


@dataclass(frozen=True)
class StructureHeadingPattern:
    exists: bool
    pattern_type: str | None = None
    description: str | None = None
    examples: list[str] = field(default_factory=list)
    confidence: float | None = None
    suggested_regex: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "exists": self.exists,
            "pattern_type": self.pattern_type,
            "description": self.description,
            "examples": list(self.examples),
            "confidence": self.confidence,
            "suggested_regex": self.suggested_regex,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "StructureHeadingPattern":
        return cls(
            exists=bool(payload.get("exists")),
            pattern_type=_optional_text(payload.get("pattern_type")),
            description=_optional_text(payload.get("description")),
            examples=_string_list(payload.get("examples")),
            confidence=_optional_confidence(payload.get("confidence")),
            suggested_regex=_optional_text(payload.get("suggested_regex")),
        )


@dataclass(frozen=True)
class StructureHeadingPatterns:
    chapter: StructureHeadingPattern | None = None
    section: StructureHeadingPattern | None = None
    front_matter: StructureHeadingPattern | None = None
    appendix: StructureHeadingPattern | None = None
    back_matter: StructureHeadingPattern | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "chapter": None if self.chapter is None else self.chapter.to_dict(),
            "section": None if self.section is None else self.section.to_dict(),
            "front_matter": (
                None if self.front_matter is None else self.front_matter.to_dict()
            ),
            "appendix": None if self.appendix is None else self.appendix.to_dict(),
            "back_matter": None if self.back_matter is None else self.back_matter.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "StructureHeadingPatterns":
        return cls(
            chapter=_optional_heading_pattern(payload.get("chapter")),
            section=_optional_heading_pattern(payload.get("section")),
            front_matter=_optional_heading_pattern(payload.get("front_matter")),
            appendix=_optional_heading_pattern(payload.get("appendix")),
            back_matter=_optional_heading_pattern(payload.get("back_matter")),
        )


@dataclass(frozen=True)
class StructureRegionHint:
    exists: bool
    count_estimate: int | None = None
    examples: list[str] = field(default_factory=list)
    confidence: float | None = None
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "exists": self.exists,
            "count_estimate": self.count_estimate,
            "examples": list(self.examples),
            "confidence": self.confidence,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "StructureRegionHint":
        return cls(
            exists=bool(payload.get("exists")),
            count_estimate=_optional_int(payload.get("count_estimate")),
            examples=_string_list(payload.get("examples")),
            confidence=_optional_confidence(payload.get("confidence")),
            notes=_optional_text(payload.get("notes")),
        )


@dataclass(frozen=True)
class StructureSpecialRegions:
    toc: StructureRegionHint | None = None
    front_matter: StructureRegionHint | None = None
    appendix: StructureRegionHint | None = None
    back_matter: StructureRegionHint | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "toc": None if self.toc is None else self.toc.to_dict(),
            "front_matter": (
                None if self.front_matter is None else self.front_matter.to_dict()
            ),
            "appendix": None if self.appendix is None else self.appendix.to_dict(),
            "back_matter": None if self.back_matter is None else self.back_matter.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "StructureSpecialRegions":
        return cls(
            toc=_optional_region_hint(payload.get("toc")),
            front_matter=_optional_region_hint(payload.get("front_matter")),
            appendix=_optional_region_hint(payload.get("appendix")),
            back_matter=_optional_region_hint(payload.get("back_matter")),
        )


@dataclass(frozen=True)
class StructureQualityHints:
    quality_label: str | None = None
    likely_single_blob: bool = False
    likely_over_fragmented: bool = False
    likely_chapter_only: bool = False
    likely_chapter_section: bool = False
    likely_essay: bool = False
    likely_ocr_noisy: bool = False
    confidence: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "quality_label": self.quality_label,
            "likely_single_blob": self.likely_single_blob,
            "likely_over_fragmented": self.likely_over_fragmented,
            "likely_chapter_only": self.likely_chapter_only,
            "likely_chapter_section": self.likely_chapter_section,
            "likely_essay": self.likely_essay,
            "likely_ocr_noisy": self.likely_ocr_noisy,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "StructureQualityHints":
        return cls(
            quality_label=_optional_text(payload.get("quality_label")),
            likely_single_blob=bool(payload.get("likely_single_blob")),
            likely_over_fragmented=bool(payload.get("likely_over_fragmented")),
            likely_chapter_only=bool(payload.get("likely_chapter_only")),
            likely_chapter_section=bool(payload.get("likely_chapter_section")),
            likely_essay=bool(payload.get("likely_essay")),
            likely_ocr_noisy=bool(payload.get("likely_ocr_noisy")),
            confidence=_optional_confidence(payload.get("confidence")),
        )


@dataclass(frozen=True)
class StructureRecommendedStrategy:
    structured_parser_mode: str | None = None
    task_unit_split_mode: str | None = None
    semantic_top_k_candidates: int | None = None
    needs_enhanced_parse: bool = False
    needs_manual_review: bool = False
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "structured_parser_mode": self.structured_parser_mode,
            "task_unit_split_mode": self.task_unit_split_mode,
            "semantic_top_k_candidates": self.semantic_top_k_candidates,
            "needs_enhanced_parse": self.needs_enhanced_parse,
            "needs_manual_review": self.needs_manual_review,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "StructureRecommendedStrategy":
        return cls(
            structured_parser_mode=_optional_text(payload.get("structured_parser_mode")),
            task_unit_split_mode=_optional_text(payload.get("task_unit_split_mode")),
            semantic_top_k_candidates=_optional_int(payload.get("semantic_top_k_candidates")),
            needs_enhanced_parse=bool(payload.get("needs_enhanced_parse")),
            needs_manual_review=bool(payload.get("needs_manual_review")),
            reason=_optional_text(payload.get("reason")),
        )


@dataclass(frozen=True)
class DocumentStructureProfile:
    profile_version: str = "structure_profile_v1"
    generated_by: str = "llm_profile_builder"
    document_structure_type: str | None = None
    confidence: float | None = None
    heading_patterns: StructureHeadingPatterns | None = None
    special_regions: StructureSpecialRegions | None = None
    quality_hints: StructureQualityHints | None = None
    recommended_strategy: StructureRecommendedStrategy | None = None
    risks: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_version": self.profile_version,
            "generated_by": self.generated_by,
            "document_structure_type": self.document_structure_type,
            "confidence": self.confidence,
            "heading_patterns": (
                None if self.heading_patterns is None else self.heading_patterns.to_dict()
            ),
            "special_regions": (
                None if self.special_regions is None else self.special_regions.to_dict()
            ),
            "quality_hints": (
                None if self.quality_hints is None else self.quality_hints.to_dict()
            ),
            "recommended_strategy": (
                None
                if self.recommended_strategy is None
                else self.recommended_strategy.to_dict()
            ),
            "risks": list(self.risks),
            "evidence": list(self.evidence),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DocumentStructureProfile":
        return cls(
            profile_version=(
                _optional_text(payload.get("profile_version")) or "structure_profile_v1"
            ),
            generated_by=_optional_text(payload.get("generated_by")) or "llm_profile_builder",
            document_structure_type=_optional_text(payload.get("document_structure_type")),
            confidence=_optional_confidence(payload.get("confidence")),
            heading_patterns=(
                None
                if not isinstance(payload.get("heading_patterns"), dict)
                else StructureHeadingPatterns.from_dict(payload["heading_patterns"])
            ),
            special_regions=(
                None
                if not isinstance(payload.get("special_regions"), dict)
                else StructureSpecialRegions.from_dict(payload["special_regions"])
            ),
            quality_hints=(
                None
                if not isinstance(payload.get("quality_hints"), dict)
                else StructureQualityHints.from_dict(payload["quality_hints"])
            ),
            recommended_strategy=(
                None
                if not isinstance(payload.get("recommended_strategy"), dict)
                else StructureRecommendedStrategy.from_dict(payload["recommended_strategy"])
            ),
            risks=_string_list(payload.get("risks")),
            evidence=_string_list(payload.get("evidence")),
        )


def _optional_text(value: Any) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


def _optional_confidence(value: Any) -> float | None:
    if value is None:
        return None
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return None
    if confidence < 0.0 or confidence > 1.0:
        return None
    return confidence


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        normalized = _optional_text(item)
        if normalized is not None:
            result.append(normalized)
    return result


def _optional_heading_pattern(value: Any) -> StructureHeadingPattern | None:
    if not isinstance(value, dict):
        return None
    return StructureHeadingPattern.from_dict(value)


def _optional_region_hint(value: Any) -> StructureRegionHint | None:
    if not isinstance(value, dict):
        return None
    return StructureRegionHint.from_dict(value)
