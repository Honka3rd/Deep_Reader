from dataclasses import dataclass
from typing import TypeAlias

from document_structure.structured_document import StructuredDocument, StructuredSection

MetricValue: TypeAlias = int | float


@dataclass(frozen=True)
class EnhancedParseTriggerDecision:
    """Deterministic recommendation decision for trying enhanced parser."""

    should_recommend: bool
    score: int
    reasons: list[str]
    metrics: dict[str, MetricValue]


class EnhancedParseTriggerEvaluator:
    """Score-based evaluator for enhanced-parser recommendation."""

    _SCORE_SINGLE_FALLBACK_SECTION = 5
    _SCORE_LOW_TITLE_COVERAGE = 3
    _SCORE_ABNORMAL_SECTION_STRUCTURE = 3
    _SCORE_HIGH_AFFECTED_SECTION_RATIO = 1
    _SCORE_HIGH_FALLBACK_TASK_UNIT_RATIO = 1

    def __init__(
        self,
        min_section_count: int = 3,
        max_section_count: int = 220,
        min_title_coverage: float = 0.55,
        min_sections_for_ratio_signal: int = 8,
        min_units_for_ratio_signal: int = 8,
        max_affected_section_ratio: float = 0.55,
        max_fallback_task_unit_ratio: float = 0.55,
        min_avg_chars_per_section: int = 180,
        max_avg_chars_per_section: int = 8000,
        min_raw_chars_for_structure_density_signal: int = 2000,
        recommend_score_threshold: int = 4,
    ):
        self.min_section_count = max(1, int(min_section_count))
        self.max_section_count = max(self.min_section_count, int(max_section_count))
        self.min_title_coverage = float(min_title_coverage)
        self.min_sections_for_ratio_signal = max(1, int(min_sections_for_ratio_signal))
        self.min_units_for_ratio_signal = max(1, int(min_units_for_ratio_signal))
        self.max_affected_section_ratio = float(max_affected_section_ratio)
        self.max_fallback_task_unit_ratio = float(max_fallback_task_unit_ratio)
        self.min_avg_chars_per_section = max(1, int(min_avg_chars_per_section))
        self.max_avg_chars_per_section = max(
            self.min_avg_chars_per_section,
            int(max_avg_chars_per_section),
        )
        self.min_raw_chars_for_structure_density_signal = max(
            1,
            int(min_raw_chars_for_structure_density_signal),
        )
        self.recommend_score_threshold = max(1, int(recommend_score_threshold))

    def evaluate(
        self,
        *,
        structured_document: StructuredDocument,
        affected_section_ratio: float,
        fallback_task_unit_ratio: float,
        total_task_units: int,
    ) -> EnhancedParseTriggerDecision:
        """Evaluate whether enhanced parser should be recommended."""
        reasons: list[str] = []
        score = 0
        sections = structured_document.sections
        total_sections = len(sections)
        raw_text = structured_document.raw_text
        raw_text_length = len(raw_text)
        safe_affected_section_ratio = self._normalize_ratio(affected_section_ratio)
        safe_fallback_task_unit_ratio = self._normalize_ratio(fallback_task_unit_ratio)
        avg_chars_per_section = (
            raw_text_length / total_sections if total_sections > 0 else float(raw_text_length)
        )
        sections_per_10k_chars = (
            (total_sections * 10_000.0) / raw_text_length if raw_text_length > 0 else 0.0
        )
        fallback_section_like_count = sum(
            1
            for section in sections
            if self._is_fallback_like_section(
                section=section,
                raw_text=raw_text,
                raw_text_length=raw_text_length,
            )
        )

        if total_sections == 1 and fallback_section_like_count == 1:
            reasons.append("single_fallback_section")
            score += self._SCORE_SINGLE_FALLBACK_SECTION

        title_coverage = self._compute_title_coverage(sections)
        if (
            total_sections >= self.min_section_count
            and title_coverage < self.min_title_coverage
        ):
            reasons.append("low_title_coverage")
            score += self._SCORE_LOW_TITLE_COVERAGE

        abnormal_count_structure = (
            total_sections < self.min_section_count
            or total_sections > self.max_section_count
        )
        abnormal_density_structure = (
            raw_text_length >= self.min_raw_chars_for_structure_density_signal
            and total_sections >= self.min_section_count
            and (
                avg_chars_per_section < self.min_avg_chars_per_section
                or avg_chars_per_section > self.max_avg_chars_per_section
            )
        )
        if abnormal_count_structure or abnormal_density_structure:
            reasons.append("abnormal_section_structure")
            score += self._SCORE_ABNORMAL_SECTION_STRUCTURE

        if (
            total_sections >= self.min_sections_for_ratio_signal
            and safe_affected_section_ratio > self.max_affected_section_ratio
        ):
            reasons.append("high_affected_section_ratio")
            score += self._SCORE_HIGH_AFFECTED_SECTION_RATIO

        if (
            total_task_units >= self.min_units_for_ratio_signal
            and safe_fallback_task_unit_ratio > self.max_fallback_task_unit_ratio
        ):
            reasons.append("high_fallback_task_unit_ratio")
            score += self._SCORE_HIGH_FALLBACK_TASK_UNIT_RATIO

        metrics: dict[str, MetricValue] = {
            "total_sections": total_sections,
            "total_task_units": max(0, int(total_task_units)),
            "title_coverage": title_coverage,
            "affected_section_ratio": safe_affected_section_ratio,
            "fallback_task_unit_ratio": safe_fallback_task_unit_ratio,
            "raw_text_length": raw_text_length,
            "avg_chars_per_section": avg_chars_per_section,
            "sections_per_10k_chars": sections_per_10k_chars,
            "fallback_section_like_count": fallback_section_like_count,
        }

        return EnhancedParseTriggerDecision(
            should_recommend=score >= self.recommend_score_threshold,
            score=score,
            reasons=reasons,
            metrics=metrics,
        )

    @staticmethod
    def _compute_title_coverage(sections: list[StructuredSection]) -> float:
        """Compute ratio of sections with non-empty title."""
        if not sections:
            return 1.0
        titled_sections = sum(1 for section in sections if (section.title or "").strip())
        return titled_sections / len(sections)

    @staticmethod
    def _is_fallback_like_section(
        *,
        section: StructuredSection,
        raw_text: str,
        raw_text_length: int,
    ) -> bool:
        """Heuristic check for fallback-like section output shape."""
        if (section.title or "").strip():
            return False
        if (section.container_title or "").strip():
            return False
        if section.char_start != 0 or section.char_end != raw_text_length:
            return False
        normalized_section_content = (section.content or "").strip()
        normalized_raw_text = (raw_text or "").strip()
        if not normalized_raw_text:
            return not normalized_section_content
        return normalized_section_content == normalized_raw_text

    @staticmethod
    def _normalize_ratio(value: float) -> float:
        """Normalize ratio into [0.0, 1.0] range for deterministic evaluation."""
        try:
            normalized = float(value)
        except (TypeError, ValueError):
            return 0.0
        if normalized < 0.0:
            return 0.0
        if normalized > 1.0:
            return 1.0
        return normalized
