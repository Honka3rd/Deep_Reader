import re

from document_structure.abstract_section_splitter import AbstractSectionSplitter
from document_structure.document_structure_language_registry import (
    DocumentStructureLanguageRegistry,
)
from document_structure.section_splitter_dto import (
    HeadingCandidate,
    HeadingPrecedenceResult,
    LineInfo,
)
from document_structure.section_role import SectionRole
from document_structure.structured_document import StructuredSection
from language.language_code import LanguageCode


class CommonSectionSplitter(AbstractSectionSplitter):
    """Heuristic/common splitter from raw text to flat structured sections."""

    _NON_TEXT_DECORATION_PATTERN = re.compile(r"^[-_=~*#\s]+$")
    _CJK_PATTERN = re.compile(r"[\u4e00-\u9fff]")
    _LATIN_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")
    _SENTENCE_END_PUNCTUATION_PATTERN = re.compile(r"[。！？!?;；,.，:]$")
    _WHITESPACE_COLLAPSE_PATTERN = re.compile(r"\s+")
    _BODY_PROSE_TERMINAL_PATTERN = re.compile(r"[.!?。！？]$")
    _MARKER_PREFIX_SUFFIX_PUNCTUATION = frozenset((":", "：", "-", "–", "—", "(", "（", "[", "【"))
    _TOC_SCAN_LINES = 180
    _TOC_MIN_HEADING_COUNT = 3
    _MAIN_BODY_EARLY_REGION_RATIO = 0.35
    _FRONT_MATTER_EARLY_REGION_RATIO = 0.45
    _BACK_REGION_RATIO = 0.60
    _SECTION_ROLE_TOC = SectionRole.TOC
    _SECTION_ROLE_FRONT_MATTER = SectionRole.FRONT_MATTER
    _SECTION_ROLE_MAIN_BODY = SectionRole.MAIN_BODY
    _SECTION_ROLE_APPENDIX = SectionRole.APPENDIX
    _SECTION_ROLE_BACK_MATTER = SectionRole.BACK_MATTER

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
            return [
                self._single_fallback_section(
                    raw_text,
                    section_role=self._SECTION_ROLE_MAIN_BODY,
                )
            ]

        lines = self._build_line_infos(raw_text)
        main_body_start = self._find_main_body_start(
            raw_text=raw_text,
            line_infos=lines,
            language=language,
        )
        body_lines = [info for info in lines if info.char_start >= main_body_start]
        if not body_lines:
            return [
                self._single_fallback_section(
                    raw_text,
                    section_role=self._SECTION_ROLE_MAIN_BODY,
                )
            ]

        strong_headings = self._detect_strong_headings(body_lines, language)
        if strong_headings:
            precedence_result = self._apply_heading_precedence(
                headings=strong_headings,
                line_infos=body_lines,
                language=language,
            )
            strong_headings = precedence_result.headings
            if not strong_headings:
                return [
                    self._single_fallback_section(
                        raw_text,
                        char_start=main_body_start,
                        section_role=self._SECTION_ROLE_MAIN_BODY,
                    )
                ]

            region_start = self._adjust_region_start_after_precedence(
                region_start=main_body_start,
                headings=strong_headings,
                line_infos=body_lines,
            )
            section_role_by_start = self._build_section_role_by_start(
                headings=strong_headings,
                language=language,
                raw_text_length=len(raw_text),
            )
            return self._build_sections_from_headings(
                raw_text,
                strong_headings,
                region_start=region_start,
                container_title_by_start=precedence_result.container_title_by_start,
                section_role_by_start=section_role_by_start,
            )

        weak_headings = self._detect_weak_headings(body_lines, language)
        if len(weak_headings) >= 2:
            section_role_by_start = self._build_section_role_by_start(
                headings=weak_headings,
                language=language,
                raw_text_length=len(raw_text),
            )
            return self._build_sections_from_headings(
                raw_text,
                weak_headings,
                region_start=main_body_start,
                section_role_by_start=section_role_by_start,
            )

        return [
            self._single_fallback_section(
                raw_text,
                char_start=main_body_start,
                section_role=self._SECTION_ROLE_MAIN_BODY,
            )
        ]

    def _find_main_body_start(
        self,
        *,
        raw_text: str,
        line_infos: list[LineInfo],
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
        early_char_limit = max(
            1500,
            int(len(raw_text) * self._MAIN_BODY_EARLY_REGION_RATIO),
        )
        toc_marker_index = self._find_marker_index(
            line_infos=line_infos,
            early_char_limit=early_char_limit,
            markers=toc_markers,
        )
        if toc_marker_index is not None:
            toc_body_start = self._find_body_start_after_toc(
                line_infos=line_infos,
                toc_marker_index=toc_marker_index,
                strong_patterns=strong_patterns,
                body_start_heading_hints=body_start_heading_hints,
            )
            if toc_body_start is not None:
                return toc_body_start

        front_marker_index = self._find_marker_index(
            line_infos=line_infos,
            early_char_limit=max(
                early_char_limit,
                int(len(raw_text) * self._FRONT_MATTER_EARLY_REGION_RATIO),
            ),
            markers=front_matter_markers,
        )
        if front_marker_index is not None:
            front_body_start = self._find_body_start_after_front_matter(
                line_infos=line_infos,
                front_marker_index=front_marker_index,
                strong_patterns=strong_patterns,
                body_start_heading_hints=body_start_heading_hints,
                front_matter_markers=front_matter_markers,
                toc_markers=toc_markers,
            )
            if front_body_start is not None:
                return front_body_start

        # Conservative fallback: keep original text untouched if body start is uncertain.
        return 0

    def _find_body_start_after_toc(
        self,
        *,
        line_infos: list[LineInfo],
        toc_marker_index: int,
        strong_patterns: tuple[re.Pattern[str], ...],
        body_start_heading_hints: tuple[str, ...],
    ) -> int | None:
        """Find main-body start by matching TOC heading list and duplicated body headings."""
        toc_window_end = min(len(line_infos), toc_marker_index + 1 + self._TOC_SCAN_LINES)
        for idx in range(toc_marker_index + 1, toc_window_end):
            stripped = line_infos[idx].stripped
            if not stripped:
                continue
            if self._looks_like_body_prose(stripped):
                toc_window_end = idx
                break

        toc_heading_titles = self._collect_heading_titles(
            line_infos[toc_marker_index + 1:toc_window_end],
            strong_patterns,
            body_start_heading_hints,
        )
        if len(toc_heading_titles) < self._TOC_MIN_HEADING_COUNT:
            return None

        toc_heading_set = set(toc_heading_titles)
        seen_heading_titles: set[str] = set()
        for idx in range(toc_marker_index + 1, len(line_infos)):
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
        return None

    def _find_body_start_after_front_matter(
        self,
        *,
        line_infos: list[LineInfo],
        front_marker_index: int,
        strong_patterns: tuple[re.Pattern[str], ...],
        body_start_heading_hints: tuple[str, ...],
        front_matter_markers: tuple[str, ...],
        toc_markers: tuple[str, ...],
    ) -> int | None:
        """Find first likely real body heading after front-matter area."""
        normalized_front_markers = tuple(
            self._normalize_heading_title(marker)
            for marker in front_matter_markers
            if marker.strip()
        )
        normalized_toc_markers = tuple(
            self._normalize_heading_title(marker)
            for marker in toc_markers
            if marker.strip()
        )
        scan_end = min(len(line_infos), front_marker_index + 1 + (self._TOC_SCAN_LINES * 2))
        for idx in range(front_marker_index + 1, scan_end):
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
            if self._contains_heading_hint(normalized_title, normalized_front_markers):
                continue
            if self._contains_heading_hint(normalized_title, normalized_toc_markers):
                continue
            if self._has_prose_after(
                line_infos=line_infos,
                heading_index=idx,
                strong_patterns=strong_patterns,
            ):
                return line_infos[idx].char_start
        return None

    def _find_marker_index(
        self,
        *,
        line_infos: list[LineInfo],
        early_char_limit: int,
        markers: tuple[str, ...],
    ) -> int | None:
        """Find language-specific marker in the early document area."""
        if not markers:
            return None

        normalized_markers = tuple(
            self._normalize_heading_title(marker)
            for marker in markers
            if marker.strip()
        )
        if not normalized_markers:
            return None

        for idx, info in enumerate(line_infos):
            if info.char_start > early_char_limit:
                break
            normalized_line = self._normalize_heading_title(info.stripped)
            if any(
                self._line_matches_marker(
                    normalized_line=normalized_line,
                    raw_line=info.stripped,
                    normalized_marker=marker,
                )
                for marker in normalized_markers
            ):
                return idx
        return None

    @classmethod
    def _line_matches_marker(
        cls,
        *,
        normalized_line: str,
        raw_line: str,
        normalized_marker: str,
    ) -> bool:
        """Match one marker against one line with marker-type-aware policy."""
        if not normalized_line or not normalized_marker:
            return False

        # Tier 1: exact normalized line match is always accepted.
        if normalized_line == normalized_marker:
            return True

        # Tier 2: only heading-like lines can use tolerant/prefix matching.
        if not cls._is_heading_like_marker_line(raw_line):
            return False

        if cls._is_short_cjk_marker(normalized_marker):
            return cls._matches_short_cjk_marker(
                normalized_line=normalized_line,
                normalized_marker=normalized_marker,
            )

        if " " in normalized_marker:
            return cls._matches_heading_like_prefix(
                normalized_line=normalized_line,
                normalized_marker=normalized_marker,
            )

        if cls._looks_latin_word(normalized_marker):
            return cls._matches_latin_word_marker(
                normalized_line=normalized_line,
                normalized_marker=normalized_marker,
            )

        return cls._matches_heading_like_prefix(
            normalized_line=normalized_line,
            normalized_marker=normalized_marker,
        )

    @classmethod
    def _is_heading_like_marker_line(cls, raw_line: str) -> bool:
        """Return whether line shape looks like heading instead of body prose."""
        stripped = raw_line.strip()
        if not stripped:
            return False
        if cls._looks_like_body_prose(stripped):
            return False

        if cls._CJK_PATTERN.search(stripped):
            return len(stripped) <= 32

        token_count = len(cls._LATIN_TOKEN_PATTERN.findall(stripped))
        if token_count == 0:
            return False
        return token_count <= 12 and len(stripped) <= 90

    @classmethod
    def _matches_heading_like_prefix(
        cls,
        *,
        normalized_line: str,
        normalized_marker: str,
    ) -> bool:
        """Match marker at heading prefix with conservative suffix tolerance."""
        if not normalized_line.startswith(normalized_marker):
            return False
        suffix = normalized_line[len(normalized_marker):].strip()
        if not suffix:
            return True

        first_char = suffix[0]
        if first_char in cls._MARKER_PREFIX_SUFFIX_PUNCTUATION:
            return True

        suffix_token_count = len(cls._LATIN_TOKEN_PATTERN.findall(suffix))
        if cls._CJK_PATTERN.search(suffix):
            return len(suffix) <= 8
        return cls._is_label_like_suffix_tokens(
            suffix=suffix,
            token_count=suffix_token_count,
        )

    @classmethod
    def _matches_short_cjk_marker(
        cls,
        *,
        normalized_line: str,
        normalized_marker: str,
    ) -> bool:
        """Use stricter matching for short CJK markers (e.g. '序')."""
        if normalized_line == normalized_marker:
            return True
        if not normalized_line.startswith(normalized_marker):
            return False

        suffix = normalized_line[len(normalized_marker):].strip()
        if not suffix:
            return True
        if suffix[0] in cls._MARKER_PREFIX_SUFFIX_PUNCTUATION:
            return True

        # Allow tiny heading variants like "序章", "序言", but avoid long sentence matches.
        return len(suffix) <= 2

    @classmethod
    def _matches_latin_word_marker(
        cls,
        *,
        normalized_line: str,
        normalized_marker: str,
    ) -> bool:
        """Match one Latin single-word marker as heading token/prefix, not body substring."""
        if normalized_line == normalized_marker:
            return True
        if not normalized_line.startswith(normalized_marker):
            return False

        suffix = normalized_line[len(normalized_marker):].strip()
        if not suffix:
            return True
        if suffix[0] in cls._MARKER_PREFIX_SUFFIX_PUNCTUATION:
            return True
        suffix_token_count = len(cls._LATIN_TOKEN_PATTERN.findall(suffix))
        return cls._is_label_like_suffix_tokens(
            suffix=suffix,
            token_count=suffix_token_count,
        )

    @classmethod
    def _is_short_cjk_marker(cls, normalized_marker: str) -> bool:
        """Return whether marker is a short CJK marker needing strict policy."""
        return bool(cls._CJK_PATTERN.search(normalized_marker)) and len(normalized_marker) <= 2

    @staticmethod
    def _looks_latin_word(normalized_marker: str) -> bool:
        """Return whether marker is a Latin single-word style token."""
        return bool(re.fullmatch(r"[a-z0-9]+", normalized_marker))

    @staticmethod
    def _is_label_like_suffix_tokens(*, suffix: str, token_count: int) -> bool:
        """Allow short heading suffix labels like 'A', 'II', '2', but block prose-like suffixes."""
        if token_count <= 0 or token_count > 3 or len(suffix) > 20:
            return False
        suffix_tokens = re.findall(r"[a-z0-9]+", suffix)
        if len(suffix_tokens) != token_count:
            return False
        return all(
            re.fullmatch(r"(?:[a-z]|[ivxlcdm]+|\d+)", token) is not None
            for token in suffix_tokens
        )

    @staticmethod
    def _is_strong_heading(
        stripped_text: str,
        strong_patterns: tuple[re.Pattern[str], ...],
    ) -> bool:
        """Check whether one stripped line matches any strong heading pattern."""
        return any(pattern.match(stripped_text) for pattern in strong_patterns)

    def _collect_heading_titles(
        self,
        line_infos: list[LineInfo],
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
        """Check whether heading text matches one hint with heading-aware policy."""
        for hint in hints:
            normalized_hint = cls._normalize_heading_title(hint)
            if not normalized_hint:
                continue
            if cls._matches_heading_hint(
                normalized_heading=normalized_heading,
                normalized_hint=normalized_hint,
            ):
                return True
        return False

    @classmethod
    def _matches_heading_hint(
        cls,
        *,
        normalized_heading: str,
        normalized_hint: str,
    ) -> bool:
        """Match one heading hint with exact/prefix/token-aware safeguards."""
        if not normalized_heading or not normalized_hint:
            return False
        if cls._matches_exact_heading_hint(
            normalized_heading=normalized_heading,
            normalized_hint=normalized_hint,
        ):
            return True

        if cls._is_short_cjk_heading_hint(normalized_hint):
            return cls._matches_short_cjk_heading_hint(
                normalized_heading=normalized_heading,
                normalized_hint=normalized_hint,
            )

        if " " in normalized_hint:
            return cls._matches_heading_hint_prefix(
                normalized_heading=normalized_heading,
                normalized_hint=normalized_hint,
            )

        if cls._looks_latin_word(normalized_hint):
            return cls._matches_latin_word_heading_hint(
                normalized_heading=normalized_heading,
                normalized_hint=normalized_hint,
            )

        return cls._matches_heading_hint_prefix(
            normalized_heading=normalized_heading,
            normalized_hint=normalized_hint,
        )

    @staticmethod
    def _matches_exact_heading_hint(
        *,
        normalized_heading: str,
        normalized_hint: str,
    ) -> bool:
        """Return True when heading and hint are exactly equal after normalization."""
        return normalized_heading == normalized_hint

    @classmethod
    def _matches_heading_hint_prefix(
        cls,
        *,
        normalized_heading: str,
        normalized_hint: str,
    ) -> bool:
        """Allow heading-prefix hint matching with conservative suffix checks."""
        if not normalized_heading.startswith(normalized_hint):
            return False
        suffix = normalized_heading[len(normalized_hint):].strip()
        if not suffix:
            return True

        first_char = suffix[0]
        if first_char in cls._MARKER_PREFIX_SUFFIX_PUNCTUATION:
            return True
        if cls._CJK_PATTERN.search(suffix):
            return len(suffix) <= 8

        token_count = len(cls._LATIN_TOKEN_PATTERN.findall(suffix))
        return cls._is_label_like_suffix_tokens(
            suffix=suffix,
            token_count=token_count,
        )

    @classmethod
    def _matches_short_cjk_heading_hint(
        cls,
        *,
        normalized_heading: str,
        normalized_hint: str,
    ) -> bool:
        """Use strict policy for short CJK hints to avoid substring pollution."""
        if normalized_heading == normalized_hint:
            return True
        if not normalized_heading.startswith(normalized_hint):
            return False

        suffix = normalized_heading[len(normalized_hint):].strip()
        if not suffix:
            return True
        if suffix[0] in cls._MARKER_PREFIX_SUFFIX_PUNCTUATION:
            return True
        return len(suffix) <= 2

    @classmethod
    def _matches_latin_word_heading_hint(
        cls,
        *,
        normalized_heading: str,
        normalized_hint: str,
    ) -> bool:
        """Allow Latin single-word hint only as heading prefix/token boundary."""
        if not normalized_heading.startswith(normalized_hint):
            return False
        suffix = normalized_heading[len(normalized_hint):].strip()
        if not suffix:
            return True
        if suffix[0] in cls._MARKER_PREFIX_SUFFIX_PUNCTUATION:
            return True
        token_count = len(cls._LATIN_TOKEN_PATTERN.findall(suffix))
        return cls._is_label_like_suffix_tokens(
            suffix=suffix,
            token_count=token_count,
        )

    @classmethod
    def _is_short_cjk_heading_hint(cls, normalized_hint: str) -> bool:
        """Return whether this hint is a short CJK token (e.g. '序')."""
        return bool(cls._CJK_PATTERN.search(normalized_hint)) and len(normalized_hint) <= 2

    def _has_prose_after(
        self,
        *,
        line_infos: list[LineInfo],
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
        if cls._CJK_PATTERN.search(stripped):
            if len(stripped) < 25:
                return False
            if not cls._BODY_PROSE_TERMINAL_PATTERN.search(stripped):
                return False
            return len(stripped) >= 25
        if len(stripped) < 60:
            return False
        if not cls._BODY_PROSE_TERMINAL_PATTERN.search(stripped):
            return False
        return len(cls._LATIN_TOKEN_PATTERN.findall(stripped)) >= 10

    def _validate_language(self, language: LanguageCode) -> None:
        """Validate language before section splitting."""
        if language == LanguageCode.UNKNOWN:
            raise ValueError(
                "bad_request: unsupported document structure language 'unknown'"
            )

    @staticmethod
    def _build_line_infos(raw_text: str) -> list[LineInfo]:
        """Build line records with absolute char offsets."""
        line_infos: list[LineInfo] = []
        cursor = 0
        for line in raw_text.splitlines(keepends=True):
            start = cursor
            end = cursor + len(line)
            cursor = end
            line_infos.append(
                LineInfo(
                    line=line,
                    stripped=line.strip(),
                    char_start=start,
                    char_end=end,
                )
            )

        if not line_infos:
            line_infos.append(
                LineInfo(
                    line=raw_text,
                    stripped=raw_text.strip(),
                    char_start=0,
                    char_end=len(raw_text),
                )
            )
        return line_infos

    def _detect_strong_headings(
        self,
        line_infos: list[LineInfo],
        language: LanguageCode,
    ) -> list[HeadingCandidate]:
        """Detect strong headings from language-registry regex patterns."""
        patterns = self.language_registry.get_strong_heading_patterns(language)
        part_hints = self.language_registry.get_part_heading_hints(language)
        chapter_hints = self.language_registry.get_chapter_heading_hints(language)
        candidates: list[HeadingCandidate] = []
        for info in line_infos:
            if not info.stripped:
                continue
            if (
                any(pattern.match(info.stripped) for pattern in patterns)
                or self._is_region_marker_heading(info.stripped, language)
            ):
                candidates.append(
                    HeadingCandidate(
                        title=info.stripped,
                        char_start=info.char_start,
                        level=self._infer_heading_level(
                            heading_text=info.stripped,
                            part_hints=part_hints,
                            chapter_hints=chapter_hints,
                        ),
                    )
                )
        return candidates

    def _infer_heading_level(
        self,
        *,
        heading_text: str,
        part_hints: tuple[str, ...],
        chapter_hints: tuple[str, ...],
    ) -> int:
        """Infer heading level with canonical policy: Part=1, Chapter=2."""
        normalized_title = self._normalize_heading_title(heading_text)
        is_part_heading = bool(part_hints) and self._contains_level_hint(
            normalized_title,
            part_hints,
        )
        is_chapter_heading = bool(chapter_hints) and self._contains_level_hint(
            normalized_title,
            chapter_hints,
        )

        if is_chapter_heading:
            return 2
        if is_part_heading:
            return 1
        return 1

    @classmethod
    def _contains_level_hint(
        cls,
        normalized_heading: str,
        hints: tuple[str, ...],
    ) -> bool:
        """Match level hints with heading-prefix semantics (Part/Chapter hierarchy use only)."""
        for hint in hints:
            normalized_hint = cls._normalize_heading_title(hint)
            if not normalized_hint:
                continue
            if normalized_heading == normalized_hint:
                return True
            if normalized_heading.startswith(normalized_hint):
                suffix = normalized_heading[len(normalized_hint):]
                if not suffix:
                    return True
                # Accept common heading continuations like space + number/word.
                if suffix[:1].isspace():
                    return True
                if suffix[0] in cls._MARKER_PREFIX_SUFFIX_PUNCTUATION:
                    return True
        return False

    def _is_region_marker_heading(self, stripped_text: str, language: LanguageCode) -> bool:
        """Treat standalone region markers as strong headings to improve region awareness."""
        if len(stripped_text) > 60:
            return False
        if self._SENTENCE_END_PUNCTUATION_PATTERN.search(stripped_text):
            return False
        normalized_title = self._normalize_heading_title(stripped_text)
        marker_groups = (
            self.language_registry.get_toc_markers(language),
            self.language_registry.get_front_matter_markers(language),
            self.language_registry.get_appendix_markers(language),
            self.language_registry.get_back_matter_markers(language),
        )
        for markers in marker_groups:
            if self._contains_heading_hint(normalized_title, markers):
                return True
        return False

    def _apply_heading_precedence(
        self,
        *,
        headings: list[HeadingCandidate],
        line_infos: list[LineInfo],
        language: LanguageCode,
    ) -> HeadingPrecedenceResult:
        """Drop low-value part boundaries and preserve part context for following chapter."""
        if len(headings) < 2:
            return HeadingPrecedenceResult(headings=headings, container_title_by_start={})

        part_hints = self.language_registry.get_part_heading_hints(language)
        chapter_hints = self.language_registry.get_chapter_heading_hints(language)
        if not part_hints or not chapter_hints:
            return HeadingPrecedenceResult(headings=headings, container_title_by_start={})

        ordered = sorted(headings, key=lambda item: item.char_start)
        index_by_start = {
            info.char_start: idx
            for idx, info in enumerate(line_infos)
        }
        filtered: list[HeadingCandidate] = []
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
        return HeadingPrecedenceResult(
            headings=filtered,
            container_title_by_start=container_title_by_start,
        )

    def _adjust_region_start_after_precedence(
        self,
        *,
        region_start: int,
        headings: list[HeadingCandidate],
        line_infos: list[LineInfo],
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
        line_infos: list[LineInfo],
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
        line_infos: list[LineInfo],
        language: LanguageCode,
    ) -> list[HeadingCandidate]:
        """Detect standalone short-line headings as weak candidates."""
        patterns = self.language_registry.get_strong_heading_patterns(language)
        weak_aliases = self.language_registry.get_weak_heading_aliases(language)
        weak_signals = self.language_registry.get_weak_heading_signals(language)
        weak_tokens = tuple(dict.fromkeys((*weak_aliases, *weak_signals)))
        candidates: list[HeadingCandidate] = []
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
                HeadingCandidate(
                    title=info.stripped,
                    char_start=info.char_start,
                    level=1,
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
        section_role: SectionRole | None = None,
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
            section_role=section_role,
        )

    @staticmethod
    def _build_section(
        *,
        section_index: int,
        title: str | None,
        raw_text: str,
        char_start: int,
        char_end: int,
        level: int = 1,
        container_title: str | None = None,
        section_role: SectionRole | None = None,
    ) -> StructuredSection:
        """Build one structured section from text slice offsets."""
        return StructuredSection(
            section_id=f"section-{section_index}",
            section_index=section_index,
            title=title,
            level=max(1, int(level)),
            content=raw_text[char_start:char_end],
            char_start=char_start,
            char_end=char_end,
            container_title=container_title,
            section_role=section_role,
        )

    def _build_sections_from_headings(
        self,
        raw_text: str,
        headings: list[HeadingCandidate],
        *,
        region_start: int = 0,
        container_title_by_start: dict[int, str] | None = None,
        section_role_by_start: dict[int, SectionRole] | None = None,
    ) -> list[StructuredSection]:
        """Build contiguous sections from heading boundary starts."""
        ordered_headings = sorted(headings, key=lambda item: item.char_start)
        sections: list[StructuredSection] = []
        container_map = container_title_by_start or {}
        section_role_map = section_role_by_start or {}
        current_container_title: str | None = None
        current_section_role = self._SECTION_ROLE_MAIN_BODY

        first_heading_start = ordered_headings[0].char_start
        if first_heading_start > region_start:
            sections.append(
                self._build_section(
                    section_index=0,
                    title=None,
                    raw_text=raw_text,
                    char_start=region_start,
                    char_end=first_heading_start,
                    section_role=current_section_role,
                )
            )

        for heading_idx, heading in enumerate(ordered_headings):
            if heading.char_start in container_map:
                # Carry the part/container context forward for following chapter sections
                # until the next container boundary is encountered.
                current_container_title = container_map[heading.char_start]
            if heading.char_start in section_role_map:
                current_section_role = section_role_map[heading.char_start]

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
                    level=heading.level,
                    container_title=current_container_title,
                    section_role=current_section_role,
                )
            )

        if not sections:
            return [
                self._single_fallback_section(
                    raw_text,
                    char_start=region_start,
                    section_role=self._SECTION_ROLE_MAIN_BODY,
                )
            ]
        return sections

    def _build_section_role_by_start(
        self,
        *,
        headings: list[HeadingCandidate],
        language: LanguageCode,
        raw_text_length: int,
    ) -> dict[int, SectionRole]:
        """Classify heading boundaries into region roles for downstream task consumption."""
        if raw_text_length <= 0:
            return {}

        toc_markers = self.language_registry.get_toc_markers(language)
        front_markers = self.language_registry.get_front_matter_markers(language)
        appendix_markers = self.language_registry.get_appendix_markers(language)
        back_markers = self.language_registry.get_back_matter_markers(language)
        role_by_start: dict[int, SectionRole] = {}

        for heading in sorted(headings, key=lambda item: item.char_start):
            position_ratio = heading.char_start / raw_text_length
            normalized_title = self._normalize_heading_title(heading.title)

            if (
                position_ratio <= self._FRONT_MATTER_EARLY_REGION_RATIO
                and self._contains_heading_hint(normalized_title, toc_markers)
            ):
                role_by_start[heading.char_start] = self._SECTION_ROLE_TOC
                continue
            if (
                position_ratio <= self._FRONT_MATTER_EARLY_REGION_RATIO
                and self._contains_heading_hint(normalized_title, front_markers)
            ):
                role_by_start[heading.char_start] = self._SECTION_ROLE_FRONT_MATTER
                continue
            if (
                position_ratio >= self._BACK_REGION_RATIO
                and self._contains_heading_hint(normalized_title, appendix_markers)
            ):
                role_by_start[heading.char_start] = self._SECTION_ROLE_APPENDIX
                continue
            if (
                position_ratio >= self._BACK_REGION_RATIO
                and self._contains_heading_hint(normalized_title, back_markers)
            ):
                role_by_start[heading.char_start] = self._SECTION_ROLE_BACK_MATTER
                continue
            role_by_start[heading.char_start] = self._SECTION_ROLE_MAIN_BODY
        return role_by_start
