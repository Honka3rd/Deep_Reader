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


@dataclass(frozen=True)
class _HeadingPrecedenceResult:
    """Post-processed heading list plus optional section container metadata."""

    headings: list[_HeadingCandidate]
    container_title_by_start: dict[int, str]


class SectionSplitter:
    """Heuristic splitter from raw text to flat structured sections."""

    _NON_TEXT_DECORATION_PATTERN = re.compile(r"^[-_=~*#\s]+$")
    _CJK_PATTERN = re.compile(r"[\u4e00-\u9fff]")
    _LATIN_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")
    _SENTENCE_END_PUNCTUATION_PATTERN = re.compile(r"[。！？!?;；,.，:]$")
    _WHITESPACE_COLLAPSE_PATTERN = re.compile(r"\s+")
    _BODY_PROSE_TERMINAL_PATTERN = re.compile(r"[.!?。！？]$")
    _TOC_SCAN_LINES = 180
    _TOC_MIN_HEADING_COUNT = 3

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
        main_body_start = self._find_main_body_start(
            raw_text=raw_text,
            line_infos=lines,
            language=language,
        )
        body_lines = [info for info in lines if info.char_start >= main_body_start]
        if not body_lines:
            return [self._single_fallback_section(raw_text)]

        strong_headings = self._detect_strong_headings(body_lines, language)
        if strong_headings:
            precedence_result = self._apply_heading_precedence(
                headings=strong_headings,
                line_infos=body_lines,
                language=language,
            )
            strong_headings = precedence_result.headings
            if not strong_headings:
                return [self._single_fallback_section(raw_text, char_start=main_body_start)]

            region_start = self._adjust_region_start_after_precedence(
                region_start=main_body_start,
                headings=strong_headings,
                line_infos=body_lines,
            )
            return self._build_sections_from_headings(
                raw_text,
                strong_headings,
                region_start=region_start,
                container_title_by_start=precedence_result.container_title_by_start,
            )

        weak_headings = self._detect_weak_headings(body_lines, language)
        if len(weak_headings) >= 2:
            return self._build_sections_from_headings(
                raw_text,
                weak_headings,
                region_start=main_body_start,
            )

        return [self._single_fallback_section(raw_text, char_start=main_body_start)]

    def _find_main_body_start(
        self,
        *,
        raw_text: str,
        line_infos: list[_LineInfo],
        language: LanguageCode,
    ) -> int:
        """Find conservative main-body start offset to avoid TOC/front-matter pollution."""
        if not raw_text:
            return 0

        strong_patterns = self.language_registry.get_strong_heading_patterns(language)
        if not strong_patterns:
            return 0
        toc_markers = self.language_registry.get_toc_markers(language)
        front_matter_markers = self.language_registry.get_front_matter_markers(language)
        body_start_heading_hints = self.language_registry.get_body_start_heading_hints(language)

        # Focus search near the beginning to avoid trimming legitimate middle sections.
        early_char_limit = max(1500, int(len(raw_text) * 0.35))
        contents_index = self._find_marker_index(
            line_infos=line_infos,
            early_char_limit=early_char_limit,
            markers=toc_markers,
        )
        if contents_index is None:
            # Fallback: some texts label front matter but omit explicit TOC marker.
            contents_index = self._find_marker_index(
                line_infos=line_infos,
                early_char_limit=early_char_limit,
                markers=front_matter_markers,
            )
        if contents_index is None:
            return 0

        toc_window_end = min(len(line_infos), contents_index + 1 + self._TOC_SCAN_LINES)
        for idx in range(contents_index + 1, toc_window_end):
            stripped = line_infos[idx].stripped
            if not stripped:
                continue
            if self._looks_like_body_prose(stripped):
                toc_window_end = idx
                break

        toc_heading_titles = self._collect_heading_titles(
            line_infos[contents_index + 1:toc_window_end],
            strong_patterns,
            body_start_heading_hints,
        )
        if len(toc_heading_titles) < self._TOC_MIN_HEADING_COUNT:
            return 0

        toc_heading_set = set(toc_heading_titles)
        seen_heading_titles: set[str] = set()
        for idx in range(contents_index + 1, len(line_infos)):
            stripped = line_infos[idx].stripped
            if not stripped:
                continue
            if not self._is_strong_heading(stripped, strong_patterns):
                continue

            normalized_title = self._normalize_heading_title(stripped)
            if (
                body_start_heading_hints
                and not self._contains_heading_hint(normalized_title, body_start_heading_hints)
            ):
                continue
            has_seen_before = normalized_title in seen_heading_titles
            seen_heading_titles.add(normalized_title)

            if not has_seen_before:
                continue
            if normalized_title not in toc_heading_set:
                continue
            if self._has_prose_after(
                line_infos=line_infos,
                heading_index=idx,
                strong_patterns=strong_patterns,
            ):
                return line_infos[idx].char_start

        # Conservative fallback: keep original text untouched if body start is uncertain.
        return 0

    def _find_marker_index(
        self,
        *,
        line_infos: list[_LineInfo],
        early_char_limit: int,
        markers: tuple[str, ...],
    ) -> int | None:
        """Find language-specific marker in the early document area."""
        if not markers:
            return None

        normalized_markers = {
            self._normalize_heading_title(marker)
            for marker in markers
            if marker.strip()
        }
        if not normalized_markers:
            return None

        for idx, info in enumerate(line_infos):
            if info.char_start > early_char_limit:
                break
            normalized_line = self._normalize_heading_title(info.stripped)
            if normalized_line in normalized_markers:
                return idx
        return None

    @staticmethod
    def _is_strong_heading(
        stripped_text: str,
        strong_patterns: tuple[re.Pattern[str], ...],
    ) -> bool:
        """Check whether one stripped line matches any strong heading pattern."""
        return any(pattern.match(stripped_text) for pattern in strong_patterns)

    def _collect_heading_titles(
        self,
        line_infos: list[_LineInfo],
        strong_patterns: tuple[re.Pattern[str], ...],
        body_start_heading_hints: tuple[str, ...],
    ) -> list[str]:
        """Collect normalized strong heading titles from a TOC-like line window."""
        titles: list[str] = []
        for info in line_infos:
            stripped = info.stripped
            if not stripped:
                continue
            if not self._is_strong_heading(stripped, strong_patterns):
                continue
            normalized_title = self._normalize_heading_title(stripped)
            if (
                body_start_heading_hints
                and not self._contains_heading_hint(normalized_title, body_start_heading_hints)
            ):
                continue
            titles.append(normalized_title)
        return titles

    @classmethod
    def _normalize_heading_title(cls, heading: str) -> str:
        """Normalize heading text for duplicate matching between TOC and body."""
        normalized = cls._WHITESPACE_COLLAPSE_PATTERN.sub(" ", heading.strip().lower())
        return normalized.rstrip(".:：;；")

    @classmethod
    def _contains_heading_hint(
        cls,
        normalized_heading: str,
        hints: tuple[str, ...],
    ) -> bool:
        """Check whether normalized heading text contains one body-start heading hint."""
        for hint in hints:
            normalized_hint = cls._normalize_heading_title(hint)
            if not normalized_hint:
                continue
            if normalized_hint in normalized_heading:
                return True
        return False

    def _has_prose_after(
        self,
        *,
        line_infos: list[_LineInfo],
        heading_index: int,
        strong_patterns: tuple[re.Pattern[str], ...],
    ) -> bool:
        """Check if a heading is followed by likely narrative prose, not another TOC list."""
        for idx in range(heading_index + 1, min(len(line_infos), heading_index + 4)):
            stripped = line_infos[idx].stripped
            if not stripped:
                continue
            if self._is_strong_heading(stripped, strong_patterns):
                continue
            if self._looks_like_body_prose(stripped):
                return True
        return False

    @classmethod
    def _looks_like_body_prose(cls, stripped: str) -> bool:
        """Heuristic prose-line detector used to anchor body start after TOC."""
        if len(stripped) < 60:
            return False
        if not cls._BODY_PROSE_TERMINAL_PATTERN.search(stripped):
            return False
        if cls._CJK_PATTERN.search(stripped):
            return len(stripped) >= 25
        return len(cls._LATIN_TOKEN_PATTERN.findall(stripped)) >= 10

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

    def _apply_heading_precedence(
        self,
        *,
        headings: list[_HeadingCandidate],
        line_infos: list[_LineInfo],
        language: LanguageCode,
    ) -> _HeadingPrecedenceResult:
        """Drop low-value part boundaries and preserve part context for following chapter."""
        if len(headings) < 2:
            return _HeadingPrecedenceResult(headings=headings, container_title_by_start={})

        part_hints = self.language_registry.get_part_heading_hints(language)
        chapter_hints = self.language_registry.get_chapter_heading_hints(language)
        if not part_hints or not chapter_hints:
            return _HeadingPrecedenceResult(headings=headings, container_title_by_start={})

        ordered = sorted(headings, key=lambda item: item.char_start)
        index_by_start = {
            info.char_start: idx
            for idx, info in enumerate(line_infos)
        }
        filtered: list[_HeadingCandidate] = []
        skipped_indices: set[int] = set()
        container_title_by_start: dict[int, str] = {}

        for idx, heading in enumerate(ordered):
            next_heading = ordered[idx + 1] if idx + 1 < len(ordered) else None
            if next_heading is None:
                continue

            normalized_title = self._normalize_heading_title(heading.title)
            normalized_next = self._normalize_heading_title(next_heading.title)
            if not self._contains_heading_hint(normalized_title, part_hints):
                continue
            if not self._contains_heading_hint(normalized_next, chapter_hints):
                continue

            current_line_idx = index_by_start.get(heading.char_start)
            next_line_idx = index_by_start.get(next_heading.char_start)
            if current_line_idx is None or next_line_idx is None:
                continue

            if self._has_meaningful_content_between_lines(
                line_infos=line_infos,
                start_line_idx=current_line_idx + 1,
                end_line_idx=next_line_idx,
            ):
                continue

            # Skip shell part heading, but keep its title as chapter container metadata.
            container_title_by_start[next_heading.char_start] = heading.title
            skipped_indices.add(idx)

        for idx, heading in enumerate(ordered):
            if idx in skipped_indices:
                continue
            filtered.append(heading)
        return _HeadingPrecedenceResult(
            headings=filtered,
            container_title_by_start=container_title_by_start,
        )

    def _adjust_region_start_after_precedence(
        self,
        *,
        region_start: int,
        headings: list[_HeadingCandidate],
        line_infos: list[_LineInfo],
    ) -> int:
        """Avoid generating an empty pre-heading section after precedence filtering."""
        if not headings:
            return region_start

        first_heading_start = min(item.char_start for item in headings)
        if first_heading_start <= region_start:
            return region_start

        index_by_start = {
            info.char_start: idx
            for idx, info in enumerate(line_infos)
        }
        first_idx = index_by_start.get(first_heading_start)
        if first_idx is None:
            return region_start

        if not self._has_meaningful_content_between_lines(
            line_infos=line_infos,
            start_line_idx=0,
            end_line_idx=first_idx,
        ):
            return first_heading_start
        return region_start

    def _has_meaningful_content_between_lines(
        self,
        *,
        line_infos: list[_LineInfo],
        start_line_idx: int,
        end_line_idx: int,
    ) -> bool:
        """Return whether line slice contains meaningful prose-like content."""
        for idx in range(start_line_idx, max(start_line_idx, end_line_idx)):
            stripped = line_infos[idx].stripped
            if not stripped:
                continue
            if self._NON_TEXT_DECORATION_PATTERN.match(stripped):
                continue
            if self._looks_like_body_prose(stripped):
                return True

            if self._CJK_PATTERN.search(stripped):
                if len(stripped) >= 20:
                    return True
            else:
                if len(stripped) >= 40:
                    return True
                if len(self._LATIN_TOKEN_PATTERN.findall(stripped)) >= 8:
                    return True
        return False

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
    def _single_fallback_section(
        raw_text: str,
        *,
        char_start: int = 0,
        char_end: int | None = None,
    ) -> StructuredSection:
        """Build fallback section when no headings are detected."""
        final_end = len(raw_text) if char_end is None else char_end
        return StructuredSection(
            section_id="section-0",
            section_index=0,
            title=None,
            level=1,
            content=raw_text[char_start:final_end],
            char_start=char_start,
            char_end=final_end,
        )

    @staticmethod
    def _build_section(
        *,
        section_index: int,
        title: str | None,
        raw_text: str,
        char_start: int,
        char_end: int,
        container_title: str | None = None,
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
            container_title=container_title,
        )

    def _build_sections_from_headings(
        self,
        raw_text: str,
        headings: list[_HeadingCandidate],
        *,
        region_start: int = 0,
        container_title_by_start: dict[int, str] | None = None,
    ) -> list[StructuredSection]:
        """Build contiguous sections from heading boundary starts."""
        ordered_headings = sorted(headings, key=lambda item: item.char_start)
        sections: list[StructuredSection] = []
        container_map = container_title_by_start or {}

        first_heading_start = ordered_headings[0].char_start
        if first_heading_start > region_start:
            sections.append(
                self._build_section(
                    section_index=0,
                    title=None,
                    raw_text=raw_text,
                    char_start=region_start,
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
                    container_title=container_map.get(heading.char_start),
                )
            )

        if not sections:
            return [self._single_fallback_section(raw_text, char_start=region_start)]
        return sections
