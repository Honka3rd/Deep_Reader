import re
from dataclasses import dataclass

from document_structure.document_structure_language_registry import (
    DocumentStructureLanguageRegistry,
)
from document_structure.structured_document import StructuredSection
from language.language_code import LanguageCode


@dataclass(frozen=True)
class _LineInfo:
    """Line text plus absolute char range in original raw text."""

    line: str
    stripped: str
    char_start: int
    char_end: int


@dataclass(frozen=True)
class _HeadingCandidate:
    """Internal heading candidate with absolute start offset."""

    title: str
    char_start: int


class SectionSplitter:
    """Heuristic splitter from raw text to flat structured sections."""

    _NON_TEXT_DECORATION_PATTERN = re.compile(r"^[-_=~*#\s]+$")
    _CJK_PATTERN = re.compile(r"[\u4e00-\u9fff]")
    _LATIN_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")
    _SENTENCE_END_PUNCTUATION_PATTERN = re.compile(r"[。！？!?;；,.，:]$")
    _WHITESPACE_COLLAPSE_PATTERN = re.compile(r"\s+")

    def __init__(
        self,
        language_registry: DocumentStructureLanguageRegistry | None = None,
    ):
        """Initialize splitter with language-rule registry."""
        self.language_registry = language_registry or DocumentStructureLanguageRegistry()

    def split(
        self,
        raw_text: str,
        language: LanguageCode = LanguageCode.UNKNOWN,
    ) -> list[StructuredSection]:
        """Split raw text into flat structured sections with char ranges.

        Raises:
            ValueError: when ``language`` is ``LanguageCode.UNKNOWN``.
        """
        self._validate_language(language)
        if not raw_text:
            return [self._single_fallback_section(raw_text)]

        lines = self._build_line_infos(raw_text)
        strong_headings = self._detect_strong_headings(lines, language)
        if strong_headings:
            return self._build_sections_from_headings(raw_text, strong_headings)

        weak_headings = self._detect_weak_headings(lines, language)
        if len(weak_headings) >= 2:
            return self._build_sections_from_headings(raw_text, weak_headings)

        return [self._single_fallback_section(raw_text)]

    def _validate_language(self, language: LanguageCode) -> None:
        """Validate language before section splitting."""
        if language == LanguageCode.UNKNOWN:
            raise ValueError(
                "bad_request: unsupported document structure language 'unknown'"
            )

    @staticmethod
    def _build_line_infos(raw_text: str) -> list[_LineInfo]:
        """Build line records with absolute char offsets."""
        line_infos: list[_LineInfo] = []
        cursor = 0
        for line in raw_text.splitlines(keepends=True):
            start = cursor
            end = cursor + len(line)
            cursor = end
            line_infos.append(
                _LineInfo(
                    line=line,
                    stripped=line.strip(),
                    char_start=start,
                    char_end=end,
                )
            )

        if not line_infos:
            line_infos.append(
                _LineInfo(
                    line=raw_text,
                    stripped=raw_text.strip(),
                    char_start=0,
                    char_end=len(raw_text),
                )
            )
        return line_infos

    def _detect_strong_headings(
        self,
        line_infos: list[_LineInfo],
        language: LanguageCode,
    ) -> list[_HeadingCandidate]:
        """Detect strong headings from language-registry regex patterns."""
        patterns = self.language_registry.get_strong_heading_patterns(language)
        candidates: list[_HeadingCandidate] = []
        for info in line_infos:
            if not info.stripped:
                continue
            if any(pattern.match(info.stripped) for pattern in patterns):
                candidates.append(
                    _HeadingCandidate(
                        title=info.stripped,
                        char_start=info.char_start,
                    )
                )
        return candidates

    def _detect_weak_headings(
        self,
        line_infos: list[_LineInfo],
        language: LanguageCode,
    ) -> list[_HeadingCandidate]:
        """Detect standalone short-line headings as weak candidates."""
        patterns = self.language_registry.get_strong_heading_patterns(language)
        weak_aliases = self.language_registry.get_weak_heading_aliases(language)
        weak_signals = self.language_registry.get_weak_heading_signals(language)
        weak_tokens = tuple(dict.fromkeys((*weak_aliases, *weak_signals)))
        candidates: list[_HeadingCandidate] = []
        for idx, info in enumerate(line_infos):
            if not info.stripped:
                continue
            if any(pattern.match(info.stripped) for pattern in patterns):
                continue

            has_weak_signal = self._contains_weak_signal(
                text=info.stripped,
                weak_signals=weak_tokens,
            )

            max_line_length = 60 if has_weak_signal else 40
            max_cjk_length = 30 if has_weak_signal else 20
            max_latin_tokens = 12 if has_weak_signal else 8

            if len(info.stripped) > max_line_length:
                continue
            if self._NON_TEXT_DECORATION_PATTERN.match(info.stripped):
                continue
            if self._SENTENCE_END_PUNCTUATION_PATTERN.search(info.stripped):
                continue

            prev_blank = idx == 0 or not line_infos[idx - 1].stripped
            next_blank = idx == len(line_infos) - 1 or not line_infos[idx + 1].stripped
            if not (prev_blank and next_blank):
                continue

            if self._CJK_PATTERN.search(info.stripped):
                if len(info.stripped) > max_cjk_length:
                    continue
            else:
                token_count = len(self._LATIN_TOKEN_PATTERN.findall(info.stripped))
                if token_count == 0 or token_count > max_latin_tokens:
                    continue

            candidates.append(
                _HeadingCandidate(
                    title=info.stripped,
                    char_start=info.char_start,
                )
            )
        return candidates

    @classmethod
    def _contains_weak_signal(
        cls,
        *,
        text: str,
        weak_signals: tuple[str, ...],
    ) -> bool:
        """Check whether line text contains one weak heading alias/signal."""
        if not weak_signals:
            return False

        normalized = cls._WHITESPACE_COLLAPSE_PATTERN.sub(" ", text.strip().lower())
        for signal in weak_signals:
            signal_normalized = cls._WHITESPACE_COLLAPSE_PATTERN.sub(
                " ",
                signal.strip().lower(),
            )
            if not signal_normalized:
                continue
            if signal_normalized in normalized:
                return True
        return False

    @staticmethod
    def _single_fallback_section(raw_text: str) -> StructuredSection:
        """Build fallback section when no headings are detected."""
        return StructuredSection(
            section_id="section-0",
            section_index=0,
            title=None,
            level=1,
            content=raw_text,
            char_start=0,
            char_end=len(raw_text),
        )

    @staticmethod
    def _build_section(
        *,
        section_index: int,
        title: str | None,
        raw_text: str,
        char_start: int,
        char_end: int,
    ) -> StructuredSection:
        """Build one structured section from text slice offsets."""
        return StructuredSection(
            section_id=f"section-{section_index}",
            section_index=section_index,
            title=title,
            level=1,
            content=raw_text[char_start:char_end],
            char_start=char_start,
            char_end=char_end,
        )

    def _build_sections_from_headings(
        self,
        raw_text: str,
        headings: list[_HeadingCandidate],
    ) -> list[StructuredSection]:
        """Build contiguous sections from heading boundary starts."""
        ordered_headings = sorted(headings, key=lambda item: item.char_start)
        sections: list[StructuredSection] = []

        first_heading_start = ordered_headings[0].char_start
        if first_heading_start > 0:
            sections.append(
                self._build_section(
                    section_index=0,
                    title=None,
                    raw_text=raw_text,
                    char_start=0,
                    char_end=first_heading_start,
                )
            )

        for heading_idx, heading in enumerate(ordered_headings):
            char_start = heading.char_start
            if heading_idx + 1 < len(ordered_headings):
                char_end = ordered_headings[heading_idx + 1].char_start
            else:
                char_end = len(raw_text)

            section_index = len(sections)
            sections.append(
                self._build_section(
                    section_index=section_index,
                    title=heading.title,
                    raw_text=raw_text,
                    char_start=char_start,
                    char_end=char_end,
                )
            )

        if not sections:
            return [self._single_fallback_section(raw_text)]
        return sections
