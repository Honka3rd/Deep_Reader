from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum

from document_structure.text_normalization import normalize_ocr_text
from doc_loaders.pdf_page_evidence import PdfOcrWordEvidence, PdfPageLayoutEvidence


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


@dataclass(frozen=True)
class TocPageCandidateGroup:
    """One contiguous layout-stable TOC candidate page group."""

    start_page_index: int
    end_page_index: int
    page_indices: list[int] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class TocRegionBox:
    left: int
    top: int
    right: int
    bottom: int


@dataclass(frozen=True)
class TocReconstructedEntry:
    """One TOC title/page pair reconstructed from OCR geometry."""

    page_index: int
    title: str
    page_number: int | None
    level: int
    title_box: TocRegionBox | None
    page_number_box: TocRegionBox | None
    confidence: float
    evidence: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class TocNormalizedToken:
    """OCR token in deterministic logical reading order with original coordinates."""

    page_index: int
    logical_index: int
    group_index: int
    text: str
    raw_text: str
    box: TocRegionBox
    confidence: float
    writing_mode: str
    reading_order: str
    rotation: str
    order_hypothesis: str


@dataclass(frozen=True)
class TocNormalizedPair:
    """One normalized title/page pair used only for TOC detection and audit."""

    page_index: int
    title: str
    page_number: int | None
    page_number_raw_text: str
    page_number_system: str
    title_box: TocRegionBox | None
    page_number_box: TocRegionBox | None
    raw_ocr: str
    writing_mode: str
    reading_order: str
    rotation: str
    order_hypothesis: str
    confidence: float
    confidence_breakdown: dict[str, float] = field(default_factory=dict)
    evidence: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class TocOcrQualityGateResult:
    """Quality gate separating TOC-looking pages from split-usable OCR entries."""

    page_index: int
    detected: bool
    usable_for_splitting: bool
    confidence: float
    entry_count: int
    title_completeness: float
    page_number_coverage: float
    geometric_pairing_coverage: float
    ordering_consistency: float
    body_title_recall: float
    reasons: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class TocPageNumberRegion:
    """Region-scoped TOC page number evidence normalized for anchor validation."""

    page_index: int
    raw_text: str
    normalized_text: str
    value: int | None
    numeral_system: str
    box: TocRegionBox | None
    confidence: float
    evidence: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class TocPageNumberOffsetValidationResult:
    """Global printed-page to source-page offset validation result."""

    valid: bool
    selected_offset: int | None
    candidate_offsets: list[int] = field(default_factory=list)
    matched_entry_count: int = 0
    page_number_count: int = 0
    reasons: list[str] = field(default_factory=list)


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
    _DOTTED_LEADER_PATTERN = re.compile(r"(?:\.{2,}|…{1,})\s*(?:\d{1,4}|[①-⑳])\s*$")
    _CIRCLED_OR_BOXED_NUMBER_PATTERN = re.compile(
        r"(?:[①-⑳]|[\[(【]\s*\d{1,4}\s*[\])】])"
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
            if page.writing_mode in {"horizontal", "unknown"}:
                score += 0.05
                reasons.append("horizontal_layout")
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
            if self._looks_like_dotted_leader_lines(page.ocr_text):
                score += 0.2
                reasons.append("dotted_leader_evidence")
            if self._has_page_number_evidence(page.ocr_text):
                score += 0.15
                reasons.append("page_number_evidence")
            if self._has_separated_page_number_column(page.ocr_text):
                score += 0.15
                reasons.append("separated_page_number_column")
            if self._has_circled_or_boxed_page_numbers(page.ocr_text):
                score += 0.1
                reasons.append("circled_or_boxed_page_number")
            if self._has_ocr_fragmented_title_density(page.ocr_text):
                score += 0.1
                reasons.append("ocr_fragmented_title_density")
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

    def group_page_candidates(
        self,
        pages: list[PdfPageLayoutEvidence],
    ) -> list[TocPageCandidateGroup]:
        """Group adjacent TOC candidate pages while terminating on layout breaks."""
        candidates = {
            candidate.page_index: candidate
            for candidate in self.detect_page_candidates(pages)
        }
        groups: list[TocPageCandidateGroup] = []
        active_indices: list[int] = []
        active_evidence: list[str] = []
        active_writing_mode: str | None = None
        active_reading_order: str | None = None
        previous_page_index: int | None = None

        def flush() -> None:
            nonlocal active_indices, active_evidence, active_writing_mode, active_reading_order
            if len(active_indices) >= 2:
                groups.append(
                    TocPageCandidateGroup(
                        start_page_index=active_indices[0],
                        end_page_index=active_indices[-1],
                        page_indices=list(active_indices),
                        evidence=list(dict.fromkeys(active_evidence)),
                    )
                )
            active_indices = []
            active_evidence = []
            active_writing_mode = None
            active_reading_order = None

        for page in sorted(pages, key=lambda item: item.page_index):
            candidate = candidates.get(page.page_index)
            if candidate is None:
                flush()
                previous_page_index = page.page_index
                continue
            layout_matches = (
                active_writing_mode in {None, candidate.writing_mode}
                and active_reading_order in {None, candidate.reading_order}
            )
            adjacent = (
                previous_page_index is None
                or page.page_index - previous_page_index == 1
            )
            if not layout_matches or not adjacent:
                flush()
            if not active_indices:
                active_writing_mode = candidate.writing_mode
                active_reading_order = candidate.reading_order
                active_evidence.extend(["toc_group_start", *candidate.evidence])
            else:
                active_evidence.extend(["toc_group_continuation", *candidate.evidence])
            active_indices.append(page.page_index)
            previous_page_index = page.page_index

        flush()
        return groups

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

    def reconstruct_page_entries(
        self,
        page: PdfPageLayoutEvidence,
    ) -> list[TocReconstructedEntry]:
        """Reconstruct TOC entries from OCR coordinates without authorizing splitting."""
        return [
            TocReconstructedEntry(
                page_index=pair.page_index,
                title=pair.title,
                page_number=pair.page_number,
                level=1,
                title_box=pair.title_box,
                page_number_box=pair.page_number_box,
                confidence=pair.confidence,
                evidence=list(pair.evidence),
            )
            for pair in self.reconstruct_normalized_page_pairs(page)
        ]

    def normalize_page_reading_order(
        self,
        page: PdfPageLayoutEvidence,
    ) -> list[TocNormalizedToken]:
        """Return OCR tokens in logical reading order while retaining coordinates."""
        if not page.ocr_words:
            return []
        grouped_words = (
            _group_words_by_vertical_column(page.ocr_words)
            if page.writing_mode == "vertical"
            else _group_words_by_horizontal_row(page.ocr_words)
        )
        order_hypothesis = _order_hypothesis_for_page(page)
        tokens: list[TocNormalizedToken] = []
        logical_index = 0
        for group_index, group in enumerate(grouped_words):
            ordered = _order_words_in_group(page, group)
            for word in ordered:
                tokens.append(
                    TocNormalizedToken(
                        page_index=page.page_index,
                        logical_index=logical_index,
                        group_index=group_index,
                        text=normalize_ocr_text(word.text),
                        raw_text=word.text,
                        box=TocRegionBox(
                            left=word.left,
                            top=word.top,
                            right=word.left + word.width,
                            bottom=word.top + word.height,
                        ),
                        confidence=round(word.confidence / 100.0, 4),
                        writing_mode=page.writing_mode,
                        reading_order=page.reading_order,
                        rotation=page.orientation,
                        order_hypothesis=order_hypothesis,
                    )
                )
                logical_index += 1
        return tokens

    def reconstruct_normalized_page_pairs(
        self,
        page: PdfPageLayoutEvidence,
    ) -> list[TocNormalizedPair]:
        """Expose normalized TOC title/page pairs without changing parser authority."""
        if not page.ocr_words:
            return []
        if page.writing_mode == "vertical":
            return self._reconstruct_vertical_pairs(page)
        return self._reconstruct_horizontal_pairs(page)

    def evaluate_toc_ocr_quality(
        self,
        page: PdfPageLayoutEvidence,
        body_text: str = "",
    ) -> TocOcrQualityGateResult:
        """Gate TOC OCR quality without inferring missing page anchors."""
        candidates = self.detect_page_candidates([page])
        pairs = self.reconstruct_normalized_page_pairs(page)
        detected = bool(candidates or pairs)
        entry_count = len(pairs)
        reasons: list[str] = []
        if detected:
            reasons.append("toc_like_page_detected")
        else:
            reasons.append("toc_quality_rejected_not_detected")
        if not pairs:
            reasons.append("toc_quality_rejected_no_reconstructed_entries")

        title_count = sum(1 for pair in pairs if pair.title.strip())
        page_number_count = sum(pair.page_number is not None for pair in pairs)
        geometric_pair_count = sum(
            pair.title_box is not None and pair.page_number_box is not None
            for pair in pairs
        )
        title_completeness = title_count / max(1, entry_count)
        page_number_coverage = page_number_count / max(1, entry_count)
        geometric_pairing_coverage = geometric_pair_count / max(1, entry_count)
        ordering_consistency = self._page_number_ordering_consistency(pairs)
        body_title_recall = self._body_title_recall(pairs, body_text)

        if title_completeness < 0.8:
            reasons.append("toc_quality_rejected_incomplete_titles")
        if page_number_coverage < 0.8:
            reasons.append("toc_quality_rejected_missing_page_numbers")
        if geometric_pairing_coverage < 0.8:
            reasons.append("toc_quality_rejected_missing_geometric_pairs")
        if ordering_consistency < 1.0:
            reasons.append("toc_quality_rejected_page_order")
        if body_text and body_title_recall < 0.5:
            reasons.append("toc_quality_rejected_low_body_title_recall")
        if not body_text:
            reasons.append("toc_quality_rejected_missing_body_text")
        ambiguous_orientation = page.orientation not in {
            "portrait",
            "landscape",
            "upright",
        } or "ambiguous_orientation" in page.evidence
        if ambiguous_orientation:
            reasons.append("toc_quality_rejected_ambiguous_orientation")

        confidence = min(
            1.0,
            title_completeness * 0.25
            + page_number_coverage * 0.25
            + geometric_pairing_coverage * 0.20
            + ordering_consistency * 0.15
            + body_title_recall * 0.15,
        )
        usable = (
            detected
            and entry_count >= 3
            and title_completeness >= 0.8
            and page_number_coverage >= 0.8
            and geometric_pairing_coverage >= 0.8
            and ordering_consistency == 1.0
            and body_title_recall >= 0.5
            and body_text.strip() != ""
            and not ambiguous_orientation
        )
        reasons.append(
            "toc_quality_usable_for_splitting"
            if usable
            else "toc_quality_not_usable_for_splitting"
        )
        return TocOcrQualityGateResult(
            page_index=page.page_index,
            detected=detected,
            usable_for_splitting=usable,
            confidence=round(confidence, 4),
            entry_count=entry_count,
            title_completeness=round(title_completeness, 4),
            page_number_coverage=round(page_number_coverage, 4),
            geometric_pairing_coverage=round(geometric_pairing_coverage, 4),
            ordering_consistency=round(ordering_consistency, 4),
            body_title_recall=round(body_title_recall, 4),
            reasons=reasons,
        )

    def recognize_page_number_regions(
        self,
        page: PdfPageLayoutEvidence,
    ) -> list[TocPageNumberRegion]:
        """Normalize only bounded page-number regions from reconstructed TOC pairs."""
        regions: list[TocPageNumberRegion] = []
        for pair in self.reconstruct_normalized_page_pairs(page):
            value, system, recognition_evidence = self._recognize_page_number_value(
                pair.page_number_raw_text
            )
            evidence = ["region_scoped_page_number", *recognition_evidence, *pair.evidence]
            regions.append(
                TocPageNumberRegion(
                    page_index=pair.page_index,
                    raw_text=pair.page_number_raw_text,
                    normalized_text=normalize_ocr_text(pair.page_number_raw_text),
                    value=value,
                    numeral_system=system,
                    box=pair.page_number_box,
                    confidence=pair.confidence,
                    evidence=evidence,
                )
            )
        return regions

    def validate_page_number_anchor_offsets(
        self,
        pairs: list[TocNormalizedPair],
        page_text_by_index: dict[int, str],
    ) -> TocPageNumberOffsetValidationResult:
        """Validate one explicit printed-page to source-page offset hypothesis."""
        reasons: list[str] = []
        anchored_pairs = [pair for pair in pairs if pair.page_number is not None]
        page_number_count = len(anchored_pairs)
        if page_number_count < 2:
            return TocPageNumberOffsetValidationResult(
                valid=False,
                selected_offset=None,
                page_number_count=page_number_count,
                reasons=["insufficient_page_number_regions"],
            )
        page_numbers = [pair.page_number for pair in anchored_pairs if pair.page_number is not None]
        monotonic = all(left <= right for left, right in zip(page_numbers, page_numbers[1:]))
        if not monotonic:
            reasons.append("non_monotonic_page_numbers")
        if not page_text_by_index:
            reasons.append("missing_page_text_for_offset_validation")
        page_indices = sorted(page_text_by_index)
        min_page_index = page_indices[0] if page_indices else 0
        max_page_index = page_indices[-1] if page_indices else -1
        min_number = min(page_numbers)
        max_number = max(page_numbers)
        candidate_offsets = [
            offset
            for offset in range(min_page_index - max_number, max_page_index - min_number + 1)
            if all(number + offset in page_text_by_index for number in page_numbers)
        ]
        if not candidate_offsets:
            reasons.append("page_anchor_out_of_range")

        scored_offsets: dict[int, int] = {}
        for offset in candidate_offsets:
            matched = 0
            for pair in anchored_pairs:
                if pair.page_number is None:
                    continue
                page_text = page_text_by_index.get(pair.page_number + offset, "")
                title = self._normalize_title(pair.title).lower()
                if title and title in normalize_ocr_text(page_text).lower():
                    matched += 1
            scored_offsets[offset] = matched

        max_matches = max(scored_offsets.values(), default=0)
        best_offsets = [
            offset for offset, count in scored_offsets.items() if count == max_matches and count >= 2
        ]
        if max_matches < 2:
            reasons.append("insufficient_body_title_matches")
        if len(best_offsets) > 1:
            reasons.append("ambiguous_page_number_offset")

        selected_offset = best_offsets[0] if len(best_offsets) == 1 and monotonic else None
        valid = selected_offset is not None and not reasons
        if valid:
            reasons.append("page_number_offset_validated")
        else:
            reasons.append("page_number_offset_rejected")
        return TocPageNumberOffsetValidationResult(
            valid=valid,
            selected_offset=selected_offset,
            candidate_offsets=candidate_offsets,
            matched_entry_count=max_matches,
            page_number_count=page_number_count,
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
        candidate_groups = self.group_page_candidates(pages)
        has_candidate_group = bool(candidate_groups)
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
        else:
            reasons.extend(
                f"toc_projection_page_group:{group.start_page_index}-{group.end_page_index}"
                for group in candidate_groups
            )
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

    @classmethod
    def _reconstruct_horizontal_pairs(
        cls,
        page: PdfPageLayoutEvidence,
    ) -> list[TocNormalizedPair]:
        pairs: list[TocNormalizedPair] = []
        for row in _group_words_by_horizontal_row(page.ocr_words):
            ordered = _order_words_in_group(page, row)
            page_word = next(
                (word for word in reversed(ordered) if cls._parse_page_number(word.text) is not None),
                None,
            )
            if page_word is None:
                continue
            title_words = [
                word
                for word in ordered
                if word.left < page_word.left
                and not cls._is_leader_token(word.text)
                and cls._parse_page_number(word.text) is None
            ]
            title = cls._normalize_reconstructed_title(title_words)
            if not title:
                continue
            confidence = min(1.0, 0.45 + min(len(title_words) * 0.08, 0.25))
            evidence = ["geometry_horizontal_row", "page_number_region"]
            if any(cls._is_leader_token(word.text) for word in ordered):
                confidence = min(1.0, confidence + 0.15)
                evidence.append("leader_line_endpoint")
            if title_words and cls._same_row(title_words[-1], page_word):
                confidence = min(1.0, confidence + 0.1)
                evidence.append("aligned_geometry")
            confidence_breakdown = {
                "base": 0.45,
                "title_token_coverage": min(len(title_words) * 0.08, 0.25),
                "leader_line_endpoint": (
                    0.15 if any(cls._is_leader_token(word.text) for word in ordered) else 0.0
                ),
                "aligned_geometry": 0.1 if title_words and cls._same_row(title_words[-1], page_word) else 0.0,
            }
            pairs.append(
                TocNormalizedPair(
                    page_index=page.page_index,
                    title=title,
                    page_number=cls._parse_page_number(page_word.text),
                    page_number_raw_text=page_word.text,
                    page_number_system=cls._recognize_page_number_value(page_word.text)[1],
                    title_box=_box_for_words(title_words),
                    page_number_box=_box_for_words([page_word]),
                    raw_ocr=" ".join(word.text for word in ordered),
                    writing_mode=page.writing_mode,
                    reading_order=page.reading_order,
                    rotation=page.orientation,
                    order_hypothesis=_order_hypothesis_for_page(page),
                    confidence=round(confidence, 4),
                    confidence_breakdown=confidence_breakdown,
                    evidence=evidence,
                )
            )
        return pairs

    @classmethod
    def _reconstruct_vertical_pairs(
        cls,
        page: PdfPageLayoutEvidence,
    ) -> list[TocNormalizedPair]:
        pairs: list[TocNormalizedPair] = []
        for column in _group_words_by_vertical_column(page.ocr_words):
            ordered = _order_words_in_group(page, column)
            page_word = next(
                (word for word in reversed(ordered) if cls._parse_page_number(word.text) is not None),
                None,
            )
            if page_word is None:
                continue
            title_words = [
                word
                for word in ordered
                if word.top < page_word.top
                and not cls._is_leader_token(word.text)
                and cls._parse_page_number(word.text) is None
            ]
            title = cls._normalize_reconstructed_title(title_words)
            if not title:
                continue
            confidence = min(1.0, 0.5 + min(len(title_words) * 0.06, 0.3))
            evidence = ["geometry_vertical_column", "page_number_region"]
            if page.reading_order == "right_to_left":
                confidence = min(1.0, confidence + 0.1)
                evidence.append("right_to_left_column_order")
            confidence_breakdown = {
                "base": 0.5,
                "title_token_coverage": min(len(title_words) * 0.06, 0.3),
                "right_to_left_column_order": 0.1 if page.reading_order == "right_to_left" else 0.0,
            }
            pairs.append(
                TocNormalizedPair(
                    page_index=page.page_index,
                    title=title,
                    page_number=cls._parse_page_number(page_word.text),
                    page_number_raw_text=page_word.text,
                    page_number_system=cls._recognize_page_number_value(page_word.text)[1],
                    title_box=_box_for_words(title_words),
                    page_number_box=_box_for_words([page_word]),
                    raw_ocr="".join(word.text for word in ordered),
                    writing_mode=page.writing_mode,
                    reading_order=page.reading_order,
                    rotation=page.orientation,
                    order_hypothesis=_order_hypothesis_for_page(page),
                    confidence=round(confidence, 4),
                    confidence_breakdown=confidence_breakdown,
                    evidence=evidence,
                )
            )
        return pairs

    @classmethod
    def _normalize_reconstructed_title(cls, words: list[PdfOcrWordEvidence]) -> str:
        title = "".join(word.text for word in words)
        return cls._normalize_title(title)

    @classmethod
    def _parse_page_number(cls, value: str) -> int | None:
        return cls._recognize_page_number_value(value)[0]

    @classmethod
    def _recognize_page_number_value(cls, value: str) -> tuple[int | None, str, list[str]]:
        raw_compact = re.sub(r"\s+", "", value or "")
        normalized = normalize_ocr_text(value).strip()
        if not normalized:
            return None, "unknown", []
        compact = re.sub(r"\s+", "", normalized)
        circled = {
            "①": 1,
            "②": 2,
            "③": 3,
            "④": 4,
            "⑤": 5,
            "⑥": 6,
            "⑦": 7,
            "⑧": 8,
            "⑨": 9,
            "⑩": 10,
            "⑪": 11,
            "⑫": 12,
            "⑬": 13,
            "⑭": 14,
            "⑮": 15,
            "⑯": 16,
            "⑰": 17,
            "⑱": 18,
            "⑲": 19,
            "⑳": 20,
        }
        if raw_compact in circled:
            return circled[raw_compact], "circled", ["circled_page_number"]
        boxed = re.fullmatch(r"[\[(【]\s*(\d{1,4})\s*[\])】]", normalized)
        if boxed:
            return int(boxed.group(1)), "boxed_arabic", ["boxed_page_number"]
        if re.fullmatch(r"(?:\d\s*){1,4}", normalized):
            evidence = ["arabic_page_number"]
            if compact != normalized:
                evidence.append("fragmented_page_number")
            return int(compact), "arabic", evidence
        roman_value = _parse_roman_page_number(compact)
        if roman_value is not None:
            return roman_value, "roman", ["roman_page_number"]
        chinese_value = _parse_chinese_page_number(compact)
        if chinese_value is not None:
            return chinese_value, "chinese", ["chinese_page_number"]
        match = re.search(r"\d{1,4}", compact)
        if match:
            return int(match.group(0)), "arabic_embedded", ["arabic_page_number"]
        return None, "unknown", []

    @staticmethod
    def _is_leader_token(value: str) -> bool:
        return bool(re.fullmatch(r"[.·・…＿_-]{2,}", value.strip()))

    @staticmethod
    def _same_row(left: PdfOcrWordEvidence, right: PdfOcrWordEvidence) -> bool:
        left_center = left.top + left.height / 2
        right_center = right.top + right.height / 2
        return abs(left_center - right_center) <= max(left.height, right.height, 12)

    @staticmethod
    def _page_number_ordering_consistency(pairs: list[TocNormalizedPair]) -> float:
        numbers = [pair.page_number for pair in pairs if pair.page_number is not None]
        if len(numbers) < 2:
            return 0.0
        return 1.0 if all(left <= right for left, right in zip(numbers, numbers[1:])) else 0.0

    @classmethod
    def _body_title_recall(
        cls,
        pairs: list[TocNormalizedPair],
        body_text: str,
    ) -> float:
        if not pairs or not body_text.strip():
            return 0.0
        body = normalize_ocr_text(body_text).lower()
        matched = 0
        for pair in pairs:
            title = cls._normalize_title(pair.title).lower()
            if title and title in body:
                matched += 1
        return matched / len(pairs)

    @classmethod
    def _looks_like_dotted_leader_lines(cls, value: str) -> bool:
        lines = [line.strip() for line in (value or "").splitlines() if line.strip()]
        if len(lines) < 3:
            return False
        leader_lines = [
            line for line in lines if cls._DOTTED_LEADER_PATTERN.search(line)
        ]
        return len(leader_lines) >= 3

    @classmethod
    def _has_page_number_evidence(cls, value: str) -> bool:
        lines = [line.strip() for line in (value or "").splitlines() if line.strip()]
        suffix_count = sum(1 for line in lines if cls._PAGE_SUFFIX_PATTERN.search(line))
        circled_count = sum(
            1 for line in lines if cls._CIRCLED_OR_BOXED_NUMBER_PATTERN.search(line)
        )
        numeric_only_count = sum(1 for line in lines if re.fullmatch(r"\d{1,4}", line))
        return suffix_count + circled_count + numeric_only_count >= 3

    @classmethod
    def _has_separated_page_number_column(cls, value: str) -> bool:
        lines = [line.strip() for line in (value or "").splitlines() if line.strip()]
        title_like_lines = [
            line
            for line in lines
            if not re.fullmatch(r"\d{1,4}|[①-⑳]", line)
            and 1 <= len(normalize_ocr_text(line)) <= 40
        ]
        number_like_lines = [
            line
            for line in lines
            if re.fullmatch(r"\d{1,4}|[①-⑳]", line)
        ]
        return len(title_like_lines) >= 3 and len(number_like_lines) >= 3

    @classmethod
    def _has_circled_or_boxed_page_numbers(cls, value: str) -> bool:
        return len(cls._CIRCLED_OR_BOXED_NUMBER_PATTERN.findall(value or "")) >= 2

    @staticmethod
    def _has_ocr_fragmented_title_density(value: str) -> bool:
        lines = [line.strip() for line in (value or "").splitlines() if line.strip()]
        if len(lines) < 6:
            return False
        short_alpha_or_cjk_lines = [
            line
            for line in lines
            if 1 <= len(normalize_ocr_text(line)) <= 8
            and not re.fullmatch(r"\d{1,4}|[①-⑳]", line)
        ]
        return len(short_alpha_or_cjk_lines) / len(lines) >= 0.6


def _looks_like_toc_title_list(value: str) -> bool:
    lines = [line.strip() for line in (value or "").splitlines() if line.strip()]
    short_lines = [line for line in lines if 1 <= len(line) <= 24]
    return len(short_lines) >= 5 and len(short_lines) / max(1, len(lines)) >= 0.5


def _group_words_by_horizontal_row(
    words: list[PdfOcrWordEvidence],
) -> list[list[PdfOcrWordEvidence]]:
    keyed: dict[tuple[int | None, int | None, int | None], list[PdfOcrWordEvidence]] = {}
    for word in words:
        key = (word.block_num, word.par_num, word.line_num)
        if all(value is not None for value in key):
            keyed.setdefault(key, []).append(word)
    if keyed:
        return [group for group in keyed.values() if len(group) >= 2]

    rows: list[list[PdfOcrWordEvidence]] = []
    for word in sorted(words, key=lambda item: (item.top, item.left)):
        center = word.top + word.height / 2
        for row in rows:
            row_center = sum(item.top + item.height / 2 for item in row) / len(row)
            if abs(center - row_center) <= max(word.height, 14):
                row.append(word)
                break
        else:
            rows.append([word])
    return [row for row in rows if len(row) >= 2]


def _group_words_by_vertical_column(
    words: list[PdfOcrWordEvidence],
) -> list[list[PdfOcrWordEvidence]]:
    columns: list[list[PdfOcrWordEvidence]] = []
    for word in sorted(words, key=lambda item: item.left + item.width / 2, reverse=True):
        center = word.left + word.width / 2
        for column in columns:
            column_center = sum(item.left + item.width / 2 for item in column) / len(column)
            if abs(center - column_center) <= max(word.width, 18):
                column.append(word)
                break
        else:
            columns.append([word])
    return [column for column in columns if len(column) >= 2]


def _order_words_in_group(
    page: PdfPageLayoutEvidence,
    words: list[PdfOcrWordEvidence],
) -> list[PdfOcrWordEvidence]:
    if page.writing_mode == "vertical":
        return sorted(words, key=lambda word: (word.top, word.left))
    if page.reading_order == "right_to_left":
        return sorted(words, key=lambda word: (-word.left, word.top))
    return sorted(words, key=lambda word: (word.left, word.top))


def _order_hypothesis_for_page(page: PdfPageLayoutEvidence) -> str:
    if page.writing_mode == "vertical" and page.reading_order == "right_to_left":
        return "vertical_rtl_columns_top_to_bottom"
    if page.writing_mode == "vertical":
        return "vertical_columns_top_to_bottom"
    if page.reading_order == "right_to_left":
        return "horizontal_rtl_rows"
    return "horizontal_ltr_rows"


def _parse_roman_page_number(value: str) -> int | None:
    if not value or not re.fullmatch(r"[IVXLCDMivxlcdm]{1,12}", value):
        return None
    roman_values = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    total = 0
    previous = 0
    for character in reversed(value.upper()):
        current = roman_values[character]
        if current < previous:
            total -= current
        else:
            total += current
            previous = current
    return total if 0 < total <= 3999 else None


def _parse_chinese_page_number(value: str) -> int | None:
    if not value or not re.fullmatch(r"[零〇一二三四五六七八九十百千两兩]+", value):
        return None
    digits = {
        "零": 0,
        "〇": 0,
        "一": 1,
        "二": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
        "两": 2,
        "兩": 2,
    }
    units = {"十": 10, "百": 100, "千": 1000}
    if all(character in digits for character in value):
        return int("".join(str(digits[character]) for character in value))
    total = 0
    current = 0
    for character in value:
        if character in digits:
            current = digits[character]
            continue
        unit = units[character]
        total += (current or 1) * unit
        current = 0
    total += current
    return total if total > 0 else None


def _box_for_words(words: list[PdfOcrWordEvidence]) -> TocRegionBox | None:
    if not words:
        return None
    return TocRegionBox(
        left=min(word.left for word in words),
        top=min(word.top for word in words),
        right=max(word.left + word.width for word in words),
        bottom=max(word.top + word.height for word in words),
    )
