from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum

from document_structure.text_normalization import normalize_ocr_text
from doc_loaders.pdf_page_evidence import PdfPageLayoutEvidence


class TocShape(StrEnum):
    UNKNOWN = "unknown"
    FLAT_CHAPTER = "flat_chapter"
    CHAPTER_SECTION = "chapter_section"
    DEEP_HIERARCHY = "deep_hierarchy"


@dataclass(frozen=True)
class TocEntry:
    """One deterministic TOC candidate extracted from source text."""

    title: str
    level: int
    page_number: int | None
    char_start: int
    char_end: int


@dataclass(frozen=True)
class TocDetectionResult:
    """TOC evidence and whether it is safe to use for structure splitting."""

    detected: bool
    usable: bool
    confidence: float
    shape: TocShape
    entries: list[TocEntry] = field(default_factory=list)
    toc_char_start: int | None = None
    toc_char_end: int | None = None
    candidate_page_indices: list[int] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class TocPageCandidate:
    page_index: int
    score: float
    writing_mode: str
    reading_order: str
    evidence: list[str] = field(default_factory=list)


class TableOfContentsDetector:
    """Detect conservative TOC evidence without using metadata or an LLM."""

    _MARKER_PATTERN = re.compile(
        r"^(?:table\s+of\s+contents|contents|目錄|目录|目次)$",
        re.IGNORECASE,
    )
    _PAGE_SUFFIX_PATTERN = re.compile(r"(?:\.{2,}\s*|\s{2,}|\t+)(\d{1,4})\s*$")
    _NUMBER_PREFIX_PATTERN = re.compile(
        r"^(?:(?:chapter|chap\.)\s+)?(?:\d+(?:\.\d+)*|[IVXLCDM]+)[\s.:、-]+",
        re.IGNORECASE,
    )
    _MAX_SCAN_CHARS = 30_000
    _MAX_SCAN_LINES = 180

    def detect(self, raw_text: str) -> TocDetectionResult:
        """Return TOC evidence from the early source region."""
        text = raw_text or ""
        if not text.strip():
            return TocDetectionResult(False, False, 0.0, TocShape.UNKNOWN)

        lines = self._line_records(text)
        early_lines = [line for line in lines if line[1] <= self._MAX_SCAN_CHARS]
        marker_index = next(
            (index for index, (_, _, _, value) in enumerate(early_lines) if self._is_marker(value)),
            None,
        )
        if marker_index is not None:
            candidate_lines = early_lines[
                marker_index + 1 : marker_index + 1 + self._MAX_SCAN_LINES
            ]
            region_start = early_lines[marker_index][1]
        else:
            candidate_lines = early_lines[: self._MAX_SCAN_LINES]
            region_start = candidate_lines[0][1] if candidate_lines else 0

        entries = self._extract_entries(candidate_lines)
        anchored_entries = [entry for entry in entries if entry.page_number is not None]
        if len(anchored_entries) >= 3:
            entries = anchored_entries
        if len(entries) < 3:
            title_list_entries = self._extract_title_list_entries(candidate_lines)
            if len(title_list_entries) >= 4:
                return TocDetectionResult(
                    detected=True,
                    usable=False,
                    confidence=0.3,
                    shape=self._classify_shape(title_list_entries),
                    entries=title_list_entries,
                    toc_char_start=region_start,
                    toc_char_end=max(entry.char_end for entry in title_list_entries),
                    reasons=[
                        "toc_like_title_list",
                        "missing_page_anchors",
                        "toc_split_rejected_conservative_fallback",
                    ],
                )
            return TocDetectionResult(
                detected=False,
                usable=False,
                confidence=0.0,
                shape=TocShape.UNKNOWN,
                reasons=["insufficient_toc_entries"],
            )

        body_start = max(entry.char_end for entry in entries)
        matched_count = sum(
            1 for entry in entries if self._title_occurs_after(text, entry.title, body_start)
        )
        page_count = sum(entry.page_number is not None for entry in entries)
        level_count = sum(entry.level > 1 for entry in entries)
        marker_bonus = 0.25 if marker_index is not None else 0.0
        page_ratio = page_count / len(entries)
        match_ratio = matched_count / len(entries)
        confidence = min(
            1.0,
            0.25
            + marker_bonus
            + min(0.35, page_ratio * 0.35)
            + min(0.30, match_ratio * 0.30)
            + (0.10 if level_count else 0.0),
        )
        reasons: list[str] = []
        if marker_index is not None:
            reasons.append("toc_marker")
        if page_count:
            reasons.append("page_number_suffixes")
        if matched_count:
            reasons.append("body_title_matches")
        if level_count:
            reasons.append("nested_entry_indentation_or_numbering")
        if page_ratio < 0.6:
            reasons.append("missing_page_anchors")
        if match_ratio < 0.5:
            reasons.append("insufficient_body_title_matches")

        shape = self._classify_shape(entries)
        usable = (
            len(entries) >= 3
            and page_ratio >= 0.6
            and match_ratio >= 0.5
            and confidence >= 0.65
        )
        if usable:
            reasons.append("toc_split_authorized")
        else:
            reasons.append("toc_split_rejected_conservative_fallback")

        return TocDetectionResult(
            detected=True,
            usable=usable,
            confidence=round(confidence, 4),
            shape=shape,
            entries=entries,
            toc_char_start=region_start,
            toc_char_end=body_start,
            reasons=reasons,
        )

    def detect_page_candidates(
        self,
        pages: list[PdfPageLayoutEvidence],
    ) -> list[TocPageCandidate]:
        """Rank pages as TOC candidates using layout evidence, not metadata labels."""
        candidates: list[TocPageCandidate] = []
        total_pages = max(1, len(pages))
        for page in pages:
            relative_position = page.page_index / total_pages
            score = 0.0
            reasons: list[str] = []
            if relative_position <= 0.15:
                score += 0.25
                reasons.append("early_document_page")
            if page.writing_mode == "vertical":
                score += 0.2
                reasons.append("vertical_layout")
            if page.reading_order == "right_to_left":
                score += 0.1
                reasons.append("right_to_left_columns")
            if page.column_count >= 3:
                score += 0.2
                reasons.append("multiple_text_columns")
            if page.image_count and page.analysis_stage == "coordinate_ocr":
                score += 0.05
                reasons.append("coordinate_ocr_available")
            if _looks_like_toc_title_list(page.ocr_text):
                score += 0.2
                reasons.append("short_title_density")
            if any(term in normalize_ocr_text(page.ocr_text).lower() for term in ("目录", "目次", "tableofcontents")):
                score += 0.2
                reasons.append("toc_marker")
            if score >= 0.45:
                candidates.append(
                    TocPageCandidate(
                        page_index=page.page_index,
                        score=round(min(1.0, score), 4),
                        writing_mode=page.writing_mode,
                        reading_order=page.reading_order,
                        evidence=reasons,
                    )
                )
        return candidates

    def detect_with_page_evidence(
        self,
        raw_text: str,
        pages: list[PdfPageLayoutEvidence],
    ) -> TocDetectionResult:
        """Combine text evidence with page layout candidates without forcing a split."""
        result = self.detect(raw_text)
        candidates = self.detect_page_candidates(pages)
        if not candidates:
            return result
        candidate_page_indices = [candidate.page_index for candidate in candidates]
        reasons = list(result.reasons)
        reasons.append("page_layout_toc_candidates")
        if any(candidate.writing_mode == "vertical" for candidate in candidates):
            reasons.append("vertical_toc_candidate")
        if any(candidate.reading_order == "right_to_left" for candidate in candidates):
            reasons.append("right_to_left_toc_candidate")
        return TocDetectionResult(
            detected=True,
            usable=result.usable,
            confidence=max(result.confidence, max(candidate.score for candidate in candidates)),
            shape=result.shape,
            entries=result.entries,
            toc_char_start=result.toc_char_start,
            toc_char_end=result.toc_char_end,
            candidate_page_indices=candidate_page_indices,
            reasons=reasons,
        )

    def validate_for_projection(
        self,
        result: TocDetectionResult,
        pages: list[PdfPageLayoutEvidence],
    ) -> TocDetectionResult:
        """Authorize TOC projection only after deterministic global checks."""
        reasons = list(result.reasons)
        entries = result.entries
        page_numbers = [entry.page_number for entry in entries]
        valid_page_numbers = [number for number in page_numbers if number is not None]
        monotonic = all(
            left <= right
            for left, right in zip(valid_page_numbers, valid_page_numbers[1:])
        )
        page_count = len(pages)
        page_range_ok = bool(valid_page_numbers) and all(
            1 <= number <= page_count + 20 for number in valid_page_numbers
        )
        candidate_scores = self.detect_page_candidates(pages)
        candidate_pages = {candidate.page_index for candidate in candidate_scores}
        has_candidate_group = bool(candidate_pages) and any(
            right - left == 1
            for left, right in zip(sorted(candidate_pages), sorted(candidate_pages)[1:])
        )
        if not result.detected:
            reasons.append("toc_projection_rejected_not_detected")
        if len(entries) < 3:
            reasons.append("toc_projection_rejected_insufficient_entries")
        if len(valid_page_numbers) / max(1, len(entries)) < 0.6:
            reasons.append("toc_projection_rejected_missing_page_numbers")
        if not monotonic:
            reasons.append("toc_projection_rejected_non_monotonic_pages")
        if not page_range_ok:
            reasons.append("toc_projection_rejected_page_range")
        if not has_candidate_group:
            reasons.append("toc_projection_rejected_missing_page_group")
        usable = (
            result.usable
            and len(entries) >= 3
            and len(valid_page_numbers) / max(1, len(entries)) >= 0.6
            and monotonic
            and page_range_ok
            and has_candidate_group
        )
        reasons.append(
            "toc_projection_authorized" if usable else "toc_projection_rejected_global_validation"
        )
        return TocDetectionResult(
            detected=result.detected,
            usable=usable,
            confidence=result.confidence,
            shape=result.shape,
            entries=entries,
            toc_char_start=result.toc_char_start,
            toc_char_end=result.toc_char_end,
            candidate_page_indices=result.candidate_page_indices,
            reasons=reasons,
        )

    @staticmethod
    def _line_records(text: str) -> list[tuple[int, int, int, str]]:
        records: list[tuple[int, int, int, str]] = []
        for match in re.finditer(r"[^\n\r]+", text):
            value = match.group(0).strip()
            if value:
                records.append((len(records), match.start(), match.end(), value))
        return records

    @classmethod
    def _extract_entries(
        cls,
        lines: list[tuple[int, int, int, str]],
    ) -> list[TocEntry]:
        entries: list[TocEntry] = []
        for _, start, end, value in lines:
            page_match = cls._PAGE_SUFFIX_PATTERN.search(value)
            title = value[: page_match.start()].strip() if page_match else value
            if not title or len(title) > 180 or len(title.split()) > 24:
                continue
            if not page_match and not cls._looks_like_numbered_entry(value):
                continue
            level = cls._entry_level(value, title)
            entries.append(
                TocEntry(
                    title=cls._normalize_title(title),
                    level=level,
                    page_number=(None if page_match is None else int(page_match.group(1))),
                    char_start=start,
                    char_end=end,
                )
            )
        return entries

    @classmethod
    def _extract_title_list_entries(
        cls,
        lines: list[tuple[int, int, int, str]],
    ) -> list[TocEntry]:
        """Extract a short-title run when OCR removed TOC page-number columns."""
        entries: list[TocEntry] = []
        for _, start, end, value in lines[:24]:
            normalized = cls._normalize_title(value)
            if not normalized or normalized in {"/", "|", "-"}:
                continue
            if len(normalized) > 80 or len(normalized.split()) > 12:
                break
            if re.search(r"[。！？!?]$", normalized):
                break
            entries.append(
                TocEntry(
                    title=normalized,
                    level=1,
                    page_number=None,
                    char_start=start,
                    char_end=end,
                )
            )
        return entries

    @classmethod
    def _looks_like_numbered_entry(cls, value: str) -> bool:
        return bool(cls._NUMBER_PREFIX_PATTERN.match(value))

    @classmethod
    def _entry_level(cls, value: str, title: str) -> int:
        indentation = len(value) - len(value.lstrip())
        number_match = re.match(r"^\s*\d+(?:\.(\d+))+", value)
        if number_match:
            return max(1, value.split()[0].count(".") + 1)
        if cls._PAGE_SUFFIX_PATTERN.search(value):
            return 2 if indentation >= 2 else 1
        return 2 if indentation >= 2 else 1 if title == value else 2

    @staticmethod
    def _normalize_title(value: str) -> str:
        return normalize_ocr_text(value).strip(" .\t")

    @classmethod
    def _is_marker(cls, value: str) -> bool:
        return bool(cls._MARKER_PATTERN.match(normalize_ocr_text(value)))

    @classmethod
    def _title_occurs_after(cls, text: str, title: str, offset: int) -> bool:
        normalized_title = cls._normalize_title(title).lower()
        if not normalized_title:
            return False
        tail = normalize_ocr_text(text[offset:]).lower()
        return normalized_title in tail

    @staticmethod
    def _classify_shape(entries: list[TocEntry]) -> TocShape:
        if any(entry.level >= 3 for entry in entries):
            return TocShape.DEEP_HIERARCHY
        if any(entry.level == 2 for entry in entries):
            return TocShape.CHAPTER_SECTION
        return TocShape.FLAT_CHAPTER


def _looks_like_toc_title_list(value: str) -> bool:
    lines = [line.strip() for line in (value or "").splitlines() if line.strip()]
    short_lines = [line for line in lines if 1 <= len(line) <= 24]
    return len(short_lines) >= 5 and len(short_lines) / max(1, len(lines)) >= 0.5
