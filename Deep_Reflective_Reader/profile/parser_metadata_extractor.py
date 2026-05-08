from __future__ import annotations

import re
from dataclasses import replace

from document_structure.document_structure_language_registry import (
    DocumentStructureLanguageRegistry,
)
from language.language_code import LanguageCode, LanguageCodeResolver
from language.language_discourse_registry import LanguageDiscourseRegistry
from language.language_script_registry import LanguageScriptRegistry
from profile.document_profile import (
    DialogueDensity,
    DiscourseMode,
    DocumentStructureShape,
    HeadingStyle,
    LikelihoodLevel,
    LineBreakQuality,
    OCRNoiseLevel,
    ParserRelevantMetadata,
    ScriptSystem,
    TextForm,
)


class ParserMetadataExtractor:
    """Deterministic extractor for high-confidence parser-relevant metadata."""

    def __init__(
        self,
        script_registry: LanguageScriptRegistry | None = None,
        structure_registry: DocumentStructureLanguageRegistry | None = None,
        discourse_registry: LanguageDiscourseRegistry | None = None,
    ):
        self._script_registry = script_registry or LanguageScriptRegistry()
        self._structure_registry = structure_registry or DocumentStructureLanguageRegistry()
        self._discourse_registry = discourse_registry or LanguageDiscourseRegistry()

    def extract(
        self,
        *,
        text: str,
        document_language: LanguageCode | str,
    ) -> ParserRelevantMetadata:
        normalized_text = text or ""
        resolved_language = LanguageCodeResolver.resolve(document_language)
        script_system = self._detect_script_system(
            text=normalized_text,
            document_language=resolved_language,
        )
        line_break_quality = self._detect_line_break_quality(normalized_text)
        ocr_noise_level = self._detect_ocr_noise_level(normalized_text)
        dialogue_density = self._detect_dialogue_density(
            text=normalized_text,
            document_language=resolved_language,
        )
        toc_likelihood = self._detect_toc_likelihood(
            text=normalized_text,
            document_language=resolved_language,
        )
        front_matter_likelihood = self._detect_front_matter_likelihood(
            text=normalized_text,
            document_language=resolved_language,
        )
        terminal_region_likelihood = self._detect_terminal_region_likelihood(
            text=normalized_text,
            document_language=resolved_language,
        )

        return ParserRelevantMetadata(
            script_system=script_system,
            line_break_quality=line_break_quality,
            ocr_noise_level=ocr_noise_level,
            dialogue_density=dialogue_density,
            toc_likelihood=toc_likelihood,
            front_matter_likelihood=front_matter_likelihood,
            terminal_region_likelihood=terminal_region_likelihood,
            text_form=TextForm.UNKNOWN,
            discourse_mode=DiscourseMode.UNKNOWN,
            document_structure_shape=DocumentStructureShape.UNKNOWN,
            likely_heading_style=HeadingStyle.UNKNOWN,
            title_uniqueness_risk=LikelihoodLevel.UNKNOWN,
            confidence=None,
            notes=[],
        )

    def merge_classification(
        self,
        *,
        base_metadata: ParserRelevantMetadata,
        text_form: TextForm | None = None,
        discourse_mode: DiscourseMode | None = None,
        document_structure_shape: DocumentStructureShape | None = None,
        likely_heading_style: HeadingStyle | None = None,
        title_uniqueness_risk: LikelihoodLevel | None = None,
        confidence: float | None = None,
        notes: list[str] | None = None,
    ) -> ParserRelevantMetadata:
        merged_notes = list(base_metadata.notes)
        if notes:
            merged_notes.extend(note for note in notes if isinstance(note, str) and note.strip())
        return replace(
            base_metadata,
            text_form=text_form or base_metadata.text_form,
            discourse_mode=discourse_mode or base_metadata.discourse_mode,
            document_structure_shape=(
                document_structure_shape or base_metadata.document_structure_shape
            ),
            likely_heading_style=(
                likely_heading_style or base_metadata.likely_heading_style
            ),
            title_uniqueness_risk=(
                title_uniqueness_risk or base_metadata.title_uniqueness_risk
            ),
            confidence=confidence if confidence is not None else base_metadata.confidence,
            notes=merged_notes,
        )

    def _detect_script_system(
        self,
        *,
        text: str,
        document_language: LanguageCode | str,
    ) -> ScriptSystem:
        return self._script_registry.detect_script_system(
            text=text,
            language=document_language,
        )

    def _detect_line_break_quality(self, text: str) -> LineBreakQuality:
        if not text:
            return LineBreakQuality.UNKNOWN
        lines = text.splitlines()
        if len(lines) <= 2:
            return LineBreakQuality.MINIMAL_LINE_BREAKS
        non_empty = [line for line in lines if line.strip()]
        if not non_empty:
            return LineBreakQuality.UNKNOWN
        empty_ratio = (len(lines) - len(non_empty)) / max(1, len(lines))
        avg_len = sum(len(line.strip()) for line in non_empty) / len(non_empty)
        short_ratio = (
            sum(1 for line in non_empty if len(line.strip()) <= 26)
            / len(non_empty)
        )
        sentence_line_ratio = (
            sum(
                1
                for line in non_empty
                if line.strip().endswith(("。", "！", "？", ".", "!", "?"))
            )
            / len(non_empty)
        )
        if empty_ratio >= 0.10 and avg_len >= 35:
            return LineBreakQuality.PARAGRAPH_LIKE
        if short_ratio >= 0.60 and avg_len <= 38 and empty_ratio <= 0.06:
            return LineBreakQuality.HARD_WRAPPED
        if sentence_line_ratio >= 0.70 and avg_len <= 40:
            return LineBreakQuality.LINE_PER_SENTENCE
        if "\n" not in text:
            return LineBreakQuality.MINIMAL_LINE_BREAKS
        if self._detect_ocr_noise_level(text) in {OCRNoiseLevel.HIGH, OCRNoiseLevel.MEDIUM} and short_ratio > 0.7:
            return LineBreakQuality.BROKEN_OCR
        return LineBreakQuality.UNKNOWN

    def _detect_ocr_noise_level(self, text: str) -> OCRNoiseLevel:
        if not text:
            return OCRNoiseLevel.UNKNOWN
        sample = text[:12000]
        replacement_count = sample.count("�")
        weird_symbol_count = sum(
            1
            for char in sample
            if ord(char) < 32 and char not in {"\n", "\r", "\t"}
        )
        orphan_space_ratio = sample.count("  ") / max(1, len(sample))
        punctuation_count = sum(
            1
            for char in sample
            if char in "。！？；,.!?;:：，、"
        )
        punctuation_ratio = punctuation_count / max(1, len(sample))
        noise_score = 0.0
        noise_score += min(1.0, replacement_count / 20)
        noise_score += min(1.0, weird_symbol_count / 30)
        noise_score += min(1.0, orphan_space_ratio * 10)
        if punctuation_ratio < 0.002:
            noise_score += 0.5
        if noise_score >= 1.8:
            return OCRNoiseLevel.HIGH
        if noise_score >= 1.0:
            return OCRNoiseLevel.MEDIUM
        if noise_score >= 0.35:
            return OCRNoiseLevel.LOW
        return OCRNoiseLevel.NONE

    def _detect_dialogue_density(
        self,
        text: str,
        document_language: LanguageCode | str,
    ) -> DialogueDensity:
        if not text:
            return DialogueDensity.UNKNOWN
        dialogue_cues = self._discourse_registry.get_dialogue_cues(document_language)
        quote_chars = tuple(dialogue_cues.quote_chars)
        sample = text[:12000]
        non_empty_lines = [line.strip() for line in sample.splitlines() if line.strip()]
        line_count = max(1, len(non_empty_lines))
        quote_count = sum(1 for char in sample if char in quote_chars)
        dash_dialogue_lines = self._count_dash_dialogue_lines(
            lines=non_empty_lines,
            dash_prefixes=dialogue_cues.dialogue_dash_prefixes,
        )
        quote_like_short_lines = self._count_quote_like_short_lines(
            lines=non_empty_lines,
            quote_chars=quote_chars,
        )
        speech_verb_lines = self._count_speech_verb_lines(
            lines=non_empty_lines,
            speech_verb_hints=dialogue_cues.speech_verb_hints,
        )
        quote_char_density = quote_count / max(1, len(sample))
        dash_ratio = dash_dialogue_lines / line_count
        quote_short_ratio = quote_like_short_lines / line_count

        # Conservative rule: quote chars alone cannot escalate to HIGH.
        if (
            dash_dialogue_lines >= 4
            or dash_ratio >= 0.18
            or (
                quote_like_short_lines >= 8
                and quote_short_ratio >= 0.25
                and speech_verb_lines >= 3
            )
            or (
                dash_dialogue_lines >= 2
                and quote_like_short_lines >= 3
            )
        ):
            return DialogueDensity.HIGH
        if (
            dash_dialogue_lines >= 1
            or quote_like_short_lines >= 3
            or quote_short_ratio >= 0.10
            or speech_verb_lines >= 2
            or quote_char_density >= 0.012
        ):
            return DialogueDensity.MEDIUM
        return DialogueDensity.LOW

    def _count_dash_dialogue_lines(
        self,
        *,
        lines: list[str],
        dash_prefixes: tuple[str, ...],
    ) -> int:
        return sum(
            1
            for line in lines
            if (
                line.startswith(dash_prefixes)
                and len(line) <= 50
                and line.endswith(("。", "！", "？", ".", "!", "?"))
            )
        )

    def _count_quote_like_short_lines(
        self,
        *,
        lines: list[str],
        quote_chars: tuple[str, ...],
    ) -> int:
        quote_like_count = 0
        for line in lines:
            if len(line) > 90:
                continue
            quote_hits = sum(1 for char in line if char in quote_chars)
            starts_with_quote = line.startswith(quote_chars)
            if starts_with_quote and quote_hits >= 2:
                quote_like_count += 1
                continue
            if quote_hits >= 4 and len(line) <= 60:
                quote_like_count += 1
        return quote_like_count

    def _count_speech_verb_lines(
        self,
        *,
        lines: list[str],
        speech_verb_hints: tuple[str, ...],
    ) -> int:
        if not speech_verb_hints:
            return 0
        count = 0
        for line in lines:
            lowered = f" {line.lower()} "
            if any(hint in lowered for hint in speech_verb_hints):
                count += 1
        return count

    def _detect_toc_likelihood(
        self,
        *,
        text: str,
        document_language: LanguageCode | str,
    ) -> LikelihoodLevel:
        head = text[:5000].lower()
        toc_markers = self._structure_registry.get_toc_markers(document_language)
        hits = sum(1 for marker in toc_markers if marker and marker.lower() in head)
        dotted_leaders = len(re.findall(r"\.{3,}\s*\d{1,4}", head))
        if hits >= 2 or dotted_leaders >= 4:
            return LikelihoodLevel.HIGH
        if hits >= 1 or dotted_leaders >= 2:
            return LikelihoodLevel.MEDIUM
        if dotted_leaders == 1:
            return LikelihoodLevel.LOW
        return LikelihoodLevel.NONE

    def _detect_front_matter_likelihood(
        self,
        *,
        text: str,
        document_language: LanguageCode | str,
    ) -> LikelihoodLevel:
        head = text[:8000].lower()
        markers = self._structure_registry.get_front_matter_markers(document_language)
        hits = sum(1 for marker in markers if marker and marker.lower() in head)
        if hits >= 3:
            return LikelihoodLevel.HIGH
        if hits == 2:
            return LikelihoodLevel.MEDIUM
        if hits == 1:
            return LikelihoodLevel.LOW
        return LikelihoodLevel.NONE

    def _detect_terminal_region_likelihood(
        self,
        *,
        text: str,
        document_language: LanguageCode | str,
    ) -> LikelihoodLevel:
        tail = text[-9000:].lower()
        appendix_markers = self._structure_registry.get_appendix_markers(document_language)
        back_matter_markers = self._structure_registry.get_back_matter_markers(
            document_language
        )
        markers = appendix_markers + back_matter_markers
        hits = sum(1 for marker in markers if marker and marker.lower() in tail)
        if hits >= 3:
            return LikelihoodLevel.HIGH
        if hits == 2:
            return LikelihoodLevel.MEDIUM
        if hits == 1:
            return LikelihoodLevel.LOW
        return LikelihoodLevel.NONE
