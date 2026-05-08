from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from language.language_code import LanguageCode, LanguageCodeResolver


class ScriptSystem(StrEnum):
    LATIN = "latin"
    SIMPLIFIED_CHINESE = "simplified_chinese"
    TRADITIONAL_CHINESE = "traditional_chinese"
    JAPANESE = "japanese"
    KOREAN = "korean"
    CYRILLIC = "cyrillic"
    ARABIC = "arabic"
    DEVANAGARI = "devanagari"
    THAI = "thai"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class TextForm(StrEnum):
    NOVEL = "novel"
    ESSAY = "essay"
    ACADEMIC_PAPER = "academic_paper"
    TECHNICAL_DOCUMENT = "technical_document"
    FINANCIAL_REPORT = "financial_report"
    LEGAL_DOCUMENT = "legal_document"
    NEWS_ARTICLE = "news_article"
    MANUAL = "manual"
    DIALOGUE_SCRIPT = "dialogue_script"
    POETRY = "poetry"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class DiscourseMode(StrEnum):
    NARRATIVE = "narrative"
    EXPOSITORY = "expository"
    ARGUMENTATIVE = "argumentative"
    INSTRUCTIONAL = "instructional"
    DIALOGUE = "dialogue"
    REFERENCE = "reference"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class LineBreakQuality(StrEnum):
    PARAGRAPH_LIKE = "paragraph_like"
    HARD_WRAPPED = "hard_wrapped"
    LINE_PER_SENTENCE = "line_per_sentence"
    BROKEN_OCR = "broken_ocr"
    MINIMAL_LINE_BREAKS = "minimal_line_breaks"
    UNKNOWN = "unknown"


class OCRNoiseLevel(StrEnum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    UNKNOWN = "unknown"


class DialogueDensity(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    UNKNOWN = "unknown"


class LikelihoodLevel(StrEnum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    UNKNOWN = "unknown"


class DocumentStructureShape(StrEnum):
    CHAPTER_ONLY = "chapter_only"
    PART_CHAPTER = "part_chapter"
    CHAPTER_SECTION = "chapter_section"
    ESSAY_SECTIONS = "essay_sections"
    FLAT_LONG_TEXT = "flat_long_text"
    OVER_FRAGMENTED = "over_fragmented"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class HeadingStyle(StrEnum):
    CHINESE_CHAPTER_NUMBERS = "chinese_chapter_numbers"
    ENGLISH_CHAPTER_WORDS = "english_chapter_words"
    ROMAN_NUMERAL_PARTS = "roman_numeral_parts"
    NUMBERED_DECIMAL_SECTIONS = "numbered_decimal_sections"
    NUMBERED_CHINESE_POINTS = "numbered_chinese_points"
    PLAIN_TITLE_HEADINGS = "plain_title_headings"
    NONE = "none"
    MIXED = "mixed"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ParserRelevantMetadata:
    metadata_version: str = "parser_metadata_v1"
    script_system: ScriptSystem | None = None
    text_form: TextForm | None = None
    discourse_mode: DiscourseMode | None = None
    line_break_quality: LineBreakQuality | None = None
    ocr_noise_level: OCRNoiseLevel | None = None
    dialogue_density: DialogueDensity | None = None
    toc_likelihood: LikelihoodLevel | None = None
    front_matter_likelihood: LikelihoodLevel | None = None
    terminal_region_likelihood: LikelihoodLevel | None = None
    document_structure_shape: DocumentStructureShape | None = None
    likely_heading_style: HeadingStyle | None = None
    title_uniqueness_risk: LikelihoodLevel | None = None
    confidence: float | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata_version": self.metadata_version,
            "script_system": _enum_value(self.script_system),
            "text_form": _enum_value(self.text_form),
            "discourse_mode": _enum_value(self.discourse_mode),
            "line_break_quality": _enum_value(self.line_break_quality),
            "ocr_noise_level": _enum_value(self.ocr_noise_level),
            "dialogue_density": _enum_value(self.dialogue_density),
            "toc_likelihood": _enum_value(self.toc_likelihood),
            "front_matter_likelihood": _enum_value(self.front_matter_likelihood),
            "terminal_region_likelihood": _enum_value(self.terminal_region_likelihood),
            "document_structure_shape": _enum_value(self.document_structure_shape),
            "likely_heading_style": _enum_value(self.likely_heading_style),
            "title_uniqueness_risk": _enum_value(self.title_uniqueness_risk),
            "confidence": self.confidence,
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ParserRelevantMetadata":
        return cls(
            metadata_version=(
                _optional_text(payload.get("metadata_version")) or "parser_metadata_v1"
            ),
            script_system=_optional_enum(
                payload.get("script_system"),
                ScriptSystem,
                ScriptSystem.UNKNOWN,
            ),
            text_form=_optional_enum(
                payload.get("text_form"),
                TextForm,
                TextForm.UNKNOWN,
            ),
            discourse_mode=_optional_enum(
                payload.get("discourse_mode"),
                DiscourseMode,
                DiscourseMode.UNKNOWN,
            ),
            line_break_quality=_optional_enum(
                payload.get("line_break_quality"),
                LineBreakQuality,
                LineBreakQuality.UNKNOWN,
            ),
            ocr_noise_level=_optional_enum(
                payload.get("ocr_noise_level"),
                OCRNoiseLevel,
                OCRNoiseLevel.UNKNOWN,
            ),
            dialogue_density=_optional_enum(
                payload.get("dialogue_density"),
                DialogueDensity,
                DialogueDensity.UNKNOWN,
            ),
            toc_likelihood=_optional_enum(
                payload.get("toc_likelihood"),
                LikelihoodLevel,
                LikelihoodLevel.UNKNOWN,
            ),
            front_matter_likelihood=_optional_enum(
                payload.get("front_matter_likelihood"),
                LikelihoodLevel,
                LikelihoodLevel.UNKNOWN,
            ),
            terminal_region_likelihood=_optional_enum(
                payload.get("terminal_region_likelihood"),
                LikelihoodLevel,
                LikelihoodLevel.UNKNOWN,
            ),
            document_structure_shape=_optional_enum(
                payload.get("document_structure_shape"),
                DocumentStructureShape,
                DocumentStructureShape.UNKNOWN,
            ),
            likely_heading_style=_optional_enum(
                payload.get("likely_heading_style"),
                HeadingStyle,
                HeadingStyle.UNKNOWN,
            ),
            title_uniqueness_risk=_optional_enum(
                payload.get("title_uniqueness_risk"),
                LikelihoodLevel,
                LikelihoodLevel.UNKNOWN,
            ),
            confidence=_optional_confidence(payload.get("confidence")),
            notes=_string_list(payload.get("notes")),
        )


@dataclass(frozen=True)
class PostStructureMetadata:
    metadata_version: str = "post_structure_metadata_v1"
    chapter_count: int = 0
    section_count: int = 0
    task_unit_count: int = 0
    front_matter_chapter_count: int = 0
    main_body_chapter_count: int = 0
    appendix_chapter_count: int = 0
    back_matter_chapter_count: int = 0
    unknown_region_chapter_count: int = 0
    implicit_section_count: int = 0
    explicit_section_count: int = 0
    duplicate_chapter_titles: list[str] = field(default_factory=list)
    duplicate_section_titles: list[str] = field(default_factory=list)
    repeated_local_chapter_titles: list[str] = field(default_factory=list)
    title_uniqueness_risk: LikelihoodLevel | None = None
    actual_structure_shape: DocumentStructureShape | None = None
    title_coverage: float | None = None
    avg_sections_per_chapter: float | None = None
    avg_task_units_per_section: float | None = None
    max_section_char_length: int | None = None
    avg_section_char_length: float | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata_version": self.metadata_version,
            "chapter_count": self.chapter_count,
            "section_count": self.section_count,
            "task_unit_count": self.task_unit_count,
            "front_matter_chapter_count": self.front_matter_chapter_count,
            "main_body_chapter_count": self.main_body_chapter_count,
            "appendix_chapter_count": self.appendix_chapter_count,
            "back_matter_chapter_count": self.back_matter_chapter_count,
            "unknown_region_chapter_count": self.unknown_region_chapter_count,
            "implicit_section_count": self.implicit_section_count,
            "explicit_section_count": self.explicit_section_count,
            "duplicate_chapter_titles": list(self.duplicate_chapter_titles),
            "duplicate_section_titles": list(self.duplicate_section_titles),
            "repeated_local_chapter_titles": list(self.repeated_local_chapter_titles),
            "title_uniqueness_risk": _enum_value(self.title_uniqueness_risk),
            "actual_structure_shape": _enum_value(self.actual_structure_shape),
            "title_coverage": self.title_coverage,
            "avg_sections_per_chapter": self.avg_sections_per_chapter,
            "avg_task_units_per_section": self.avg_task_units_per_section,
            "max_section_char_length": self.max_section_char_length,
            "avg_section_char_length": self.avg_section_char_length,
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PostStructureMetadata":
        return cls(
            metadata_version=(
                _optional_text(payload.get("metadata_version"))
                or "post_structure_metadata_v1"
            ),
            chapter_count=max(0, _optional_int(payload.get("chapter_count")) or 0),
            section_count=max(0, _optional_int(payload.get("section_count")) or 0),
            task_unit_count=max(0, _optional_int(payload.get("task_unit_count")) or 0),
            front_matter_chapter_count=max(
                0, _optional_int(payload.get("front_matter_chapter_count")) or 0
            ),
            main_body_chapter_count=max(
                0, _optional_int(payload.get("main_body_chapter_count")) or 0
            ),
            appendix_chapter_count=max(
                0, _optional_int(payload.get("appendix_chapter_count")) or 0
            ),
            back_matter_chapter_count=max(
                0, _optional_int(payload.get("back_matter_chapter_count")) or 0
            ),
            unknown_region_chapter_count=max(
                0, _optional_int(payload.get("unknown_region_chapter_count")) or 0
            ),
            implicit_section_count=max(
                0, _optional_int(payload.get("implicit_section_count")) or 0
            ),
            explicit_section_count=max(
                0, _optional_int(payload.get("explicit_section_count")) or 0
            ),
            duplicate_chapter_titles=_string_list(payload.get("duplicate_chapter_titles")),
            duplicate_section_titles=_string_list(payload.get("duplicate_section_titles")),
            repeated_local_chapter_titles=_string_list(
                payload.get("repeated_local_chapter_titles")
            ),
            title_uniqueness_risk=_optional_enum(
                payload.get("title_uniqueness_risk"),
                LikelihoodLevel,
                LikelihoodLevel.UNKNOWN,
            ),
            actual_structure_shape=_optional_enum(
                payload.get("actual_structure_shape"),
                DocumentStructureShape,
                DocumentStructureShape.UNKNOWN,
            ),
            title_coverage=_optional_float(payload.get("title_coverage")),
            avg_sections_per_chapter=_optional_float(
                payload.get("avg_sections_per_chapter")
            ),
            avg_task_units_per_section=_optional_float(
                payload.get("avg_task_units_per_section")
            ),
            max_section_char_length=_optional_int(payload.get("max_section_char_length")),
            avg_section_char_length=_optional_float(payload.get("avg_section_char_length")),
            notes=_string_list(payload.get("notes")),
        )


@dataclass(frozen=True)
class DocumentProfile:
    """Document profile data used for prompt conditioning."""

    topic: str
    summary: str
    document_language: LanguageCode
    parser_metadata: ParserRelevantMetadata | None = None
    post_structure_metadata: PostStructureMetadata | None = None
    # legacy compatibility only
    structure_profile: "DocumentStructureProfile | None" = None

    @property
    def document_language_code(self) -> str:
        """Return canonical language code string used by prompt/log/JSON paths."""
        return self.document_language.value

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "topic": self.topic,
            "summary": self.summary,
            "document_language": self.document_language_code,
        }
        if self.parser_metadata is not None:
            payload["parser_metadata"] = self.parser_metadata.to_dict()
        if self.post_structure_metadata is not None:
            payload["post_structure_metadata"] = self.post_structure_metadata.to_dict()
        if self.structure_profile is not None:
            payload["structure_profile"] = self.structure_profile.to_dict()
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DocumentProfile":
        parser_metadata_payload = payload.get("parser_metadata")
        post_structure_metadata_payload = payload.get("post_structure_metadata")
        structure_profile_payload = payload.get("structure_profile")
        parser_metadata = (
            None
            if not isinstance(parser_metadata_payload, dict)
            else ParserRelevantMetadata.from_dict(parser_metadata_payload)
        )
        post_structure_metadata = (
            None
            if not isinstance(post_structure_metadata_payload, dict)
            else PostStructureMetadata.from_dict(post_structure_metadata_payload)
        )
        structure_profile = (
            None
            if not isinstance(structure_profile_payload, dict)
            else DocumentStructureProfile.from_dict(structure_profile_payload)
        )
        return cls(
            topic=str(payload.get("topic") or "").strip(),
            summary=str(payload.get("summary") or "").strip(),
            document_language=LanguageCodeResolver.resolve(payload.get("document_language")),
            parser_metadata=parser_metadata,
            post_structure_metadata=post_structure_metadata,
            structure_profile=structure_profile,
        )


# -----------------------------------------------------------------------------
# Legacy compatibility model: keep old structure_profile schema loadable.
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class StructureCharRange:
    start_char: int | None = None
    end_char: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_char": self.start_char,
            "end_char": self.end_char,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "StructureCharRange":
        return cls(
            start_char=_optional_int(payload.get("start_char")),
            end_char=_optional_int(payload.get("end_char")),
        )


@dataclass(frozen=True)
class StructureRegionHints:
    exists: bool = False
    ranges: list[StructureCharRange] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "exists": self.exists,
            "ranges": [char_range.to_dict() for char_range in self.ranges],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "StructureRegionHints":
        ranges_payload = payload.get("ranges")
        ranges: list[StructureCharRange] = []
        if isinstance(ranges_payload, list):
            for item in ranges_payload:
                if isinstance(item, dict):
                    ranges.append(StructureCharRange.from_dict(item))
        return cls(
            exists=bool(payload.get("exists")),
            ranges=ranges,
        )


@dataclass(frozen=True)
class StructureRegions:
    front_matter: StructureRegionHints | None = None
    back_matter: StructureRegionHints | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "front_matter": (
                None if self.front_matter is None else self.front_matter.to_dict()
            ),
            "back_matter": (
                None if self.back_matter is None else self.back_matter.to_dict()
            ),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "StructureRegions":
        return cls(
            front_matter=_optional_region_hints(payload.get("front_matter")),
            back_matter=_optional_region_hints(payload.get("back_matter")),
        )


@dataclass(frozen=True)
class StructureHeadingRule:
    enabled: bool = False
    keywords: list[str] = field(default_factory=list)
    regex_candidates: list[str] = field(default_factory=list)
    positions: list[int] = field(default_factory=list)
    line_anchor_window: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "keywords": list(self.keywords),
            "regex_candidates": list(self.regex_candidates),
            "positions": list(self.positions),
            "line_anchor_window": self.line_anchor_window,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "StructureHeadingRule":
        return cls(
            enabled=bool(payload.get("enabled")),
            keywords=_string_list(payload.get("keywords")),
            regex_candidates=_string_list(payload.get("regex_candidates")),
            positions=_int_list(payload.get("positions")),
            line_anchor_window=max(
                0,
                _optional_int(payload.get("line_anchor_window")) or 0,
            ),
        )


@dataclass(frozen=True)
class StructureHeadingRules:
    chapter: StructureHeadingRule | None = None
    section: StructureHeadingRule | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "chapter": None if self.chapter is None else self.chapter.to_dict(),
            "section": None if self.section is None else self.section.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "StructureHeadingRules":
        return cls(
            chapter=_optional_heading_rule(payload.get("chapter")),
            section=_optional_heading_rule(payload.get("section")),
        )


@dataclass(frozen=True)
class StructureSplitPolicyHint:
    prefer_heading_boundaries: bool = True
    prefer_paragraph_boundaries: bool = True
    allow_single_newline_as_paragraph: bool = False
    fallback_mode: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "prefer_heading_boundaries": self.prefer_heading_boundaries,
            "prefer_paragraph_boundaries": self.prefer_paragraph_boundaries,
            "allow_single_newline_as_paragraph": self.allow_single_newline_as_paragraph,
            "fallback_mode": self.fallback_mode,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "StructureSplitPolicyHint":
        return cls(
            prefer_heading_boundaries=bool(payload.get("prefer_heading_boundaries", True)),
            prefer_paragraph_boundaries=bool(payload.get("prefer_paragraph_boundaries", True)),
            allow_single_newline_as_paragraph=bool(
                payload.get("allow_single_newline_as_paragraph", False)
            ),
            fallback_mode=_optional_text(payload.get("fallback_mode")),
        )


@dataclass(frozen=True)
class DocumentStructureProfile:
    profile_version: str = "parser_hints_v1"
    document_language: str | None = None
    structure_type: str | None = None
    structure_level_count: int | None = None
    parser_mode_hint: str | None = None
    regions: StructureRegions | None = None
    heading_rules: StructureHeadingRules | None = None
    split_policy_hint: StructureSplitPolicyHint | None = None
    confidence: float | None = None

    @property
    def document_structure_type(self) -> str | None:
        return self.structure_type

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_version": self.profile_version,
            "document_language": self.document_language,
            "structure_type": self.structure_type,
            "structure_level_count": self.structure_level_count,
            "parser_mode_hint": self.parser_mode_hint,
            "regions": None if self.regions is None else self.regions.to_dict(),
            "heading_rules": (
                None if self.heading_rules is None else self.heading_rules.to_dict()
            ),
            "split_policy_hint": (
                None
                if self.split_policy_hint is None
                else self.split_policy_hint.to_dict()
            ),
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DocumentStructureProfile":
        profile_version = _optional_text(payload.get("profile_version")) or "parser_hints_v1"
        document_language = _optional_text(payload.get("document_language"))
        structure_type = _optional_text(payload.get("structure_type"))
        if structure_type is None:
            structure_type = _optional_text(payload.get("document_structure_type"))
        structure_level_count = _optional_structure_level_count(
            payload.get("structure_level_count")
        )
        parser_mode_hint = _optional_text(payload.get("parser_mode_hint"))
        if parser_mode_hint is None:
            parser_mode_hint = _extract_legacy_parser_mode_hint(payload)
        regions = _optional_regions(payload.get("regions"))
        if regions is None:
            regions = _regions_from_legacy_payload(payload.get("special_regions"))
        heading_rules = _optional_heading_rules(payload.get("heading_rules"))
        if heading_rules is None:
            heading_rules = _heading_rules_from_legacy_payload(payload.get("heading_patterns"))
        split_policy_hint = _optional_split_policy_hint(payload.get("split_policy_hint"))
        if split_policy_hint is None:
            split_policy_hint = _split_policy_from_legacy_payload(
                payload.get("recommended_strategy")
            )
        return cls(
            profile_version=profile_version,
            document_language=document_language,
            structure_type=structure_type,
            structure_level_count=structure_level_count,
            parser_mode_hint=parser_mode_hint,
            regions=regions,
            heading_rules=heading_rules,
            split_policy_hint=split_policy_hint,
            confidence=_optional_confidence(payload.get("confidence")),
        )


def _optional_text(value: Any) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


def _enum_value(value: StrEnum | None) -> str | None:
    if value is None:
        return None
    return value.value


def _optional_enum(
    value: Any,
    enum_type: type[StrEnum],
    unknown_value: StrEnum | None = None,
) -> StrEnum | None:
    normalized = _optional_text(value)
    if normalized is None:
        return None
    try:
        return enum_type(normalized)
    except ValueError:
        return unknown_value


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


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_structure_level_count(value: Any) -> int | None:
    normalized = _optional_int(value)
    if normalized is None:
        return None
    if normalized < 0:
        return 0
    if normalized > 2:
        return 2
    return normalized


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        normalized = _optional_text(item)
        if normalized is not None:
            result.append(normalized)
    return result


def _int_list(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    result: list[int] = []
    for item in value:
        normalized = _optional_int(item)
        if normalized is not None:
            result.append(normalized)
    return result


def _optional_region_hints(value: Any) -> StructureRegionHints | None:
    if not isinstance(value, dict):
        return None
    return StructureRegionHints.from_dict(value)


def _optional_regions(value: Any) -> StructureRegions | None:
    if not isinstance(value, dict):
        return None
    return StructureRegions.from_dict(value)


def _optional_heading_rule(value: Any) -> StructureHeadingRule | None:
    if not isinstance(value, dict):
        return None
    return StructureHeadingRule.from_dict(value)


def _optional_heading_rules(value: Any) -> StructureHeadingRules | None:
    if not isinstance(value, dict):
        return None
    return StructureHeadingRules.from_dict(value)


def _optional_split_policy_hint(value: Any) -> StructureSplitPolicyHint | None:
    if not isinstance(value, dict):
        return None
    return StructureSplitPolicyHint.from_dict(value)


def _legacy_region_exists(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    return bool(payload.get("exists"))


def _regions_from_legacy_payload(payload: Any) -> StructureRegions | None:
    if not isinstance(payload, dict):
        return None
    front_payload = payload.get("front_matter")
    appendix_payload = payload.get("appendix")
    back_payload = payload.get("back_matter")
    front = StructureRegionHints(exists=_legacy_region_exists(front_payload), ranges=[])
    back = StructureRegionHints(
        exists=_legacy_region_exists(back_payload)
        or _legacy_region_exists(appendix_payload),
        ranges=[],
    )
    if not front.exists and not back.exists:
        return None
    return StructureRegions(front_matter=front, back_matter=back)


def _legacy_heading_rule(payload: Any) -> StructureHeadingRule | None:
    if not isinstance(payload, dict):
        return None
    regex_candidates: list[str] = []
    suggested_regex = _optional_text(payload.get("suggested_regex"))
    if suggested_regex is not None:
        regex_candidates.append(suggested_regex)
    return StructureHeadingRule(
        enabled=bool(payload.get("exists")),
        keywords=[],
        regex_candidates=regex_candidates,
        positions=[],
        line_anchor_window=0,
    )


def _heading_rules_from_legacy_payload(payload: Any) -> StructureHeadingRules | None:
    if not isinstance(payload, dict):
        return None
    chapter = _legacy_heading_rule(payload.get("chapter"))
    section = _legacy_heading_rule(payload.get("section"))
    if chapter is None and section is None:
        return None
    return StructureHeadingRules(chapter=chapter, section=section)


def _extract_legacy_parser_mode_hint(payload: dict[str, Any]) -> str | None:
    recommended_strategy = payload.get("recommended_strategy")
    if not isinstance(recommended_strategy, dict):
        return None
    return _optional_text(recommended_strategy.get("structured_parser_mode"))


def _split_policy_from_legacy_payload(payload: Any) -> StructureSplitPolicyHint | None:
    if not isinstance(payload, dict):
        return None
    fallback_mode = _optional_text(payload.get("task_unit_split_mode"))
    if fallback_mode is None and not bool(payload.get("needs_enhanced_parse")):
        return None
    return StructureSplitPolicyHint(
        prefer_heading_boundaries=True,
        prefer_paragraph_boundaries=True,
        allow_single_newline_as_paragraph=False,
        fallback_mode=fallback_mode,
    )
