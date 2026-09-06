from document_structure.abstract_section_splitter import AbstractSectionSplitter
from document_structure.section_splitter import CommonSectionSplitter
from document_structure.section_splitter_selector import (
    SectionSplitterMode,
    SectionSplitterSelector,
)
import re

from document_structure.section_role import SectionRole
from document_structure.structured_document import (
    StructuredChapter,
    StructuredDocument,
    StructuredSection,
)
from document_structure.structured_hierarchy_builder import (
    build_document_hierarchy_from_sections,
)
from document_structure.toc_detector import TableOfContentsDetector
from doc_loaders.pdf_page_evidence import PdfPageLayoutEvidence, PdfPageTextBoundary
from doc_loaders.pdf_outline import PdfOutlineEntry, PdfOutlineResult
from document_structure.text_normalization import normalize_ocr_text
from language.language_code import LanguageCode


class StructuredDocumentBuilder:
    """Build one StructuredDocument from metadata + raw text + language."""

    def __init__(
        self,
        section_splitter: AbstractSectionSplitter | None = None,
        section_splitter_selector: SectionSplitterSelector | None = None,
        toc_detector: TableOfContentsDetector | None = None,
    ):
        """Initialize builder with injected splitter dependency."""
        self.section_splitter = section_splitter or CommonSectionSplitter()
        self.section_splitter_selector = section_splitter_selector
        self.toc_detector = toc_detector or TableOfContentsDetector()

    def build(
        self,
        document_id: str,
        title: str,
        raw_text: str,
        language: LanguageCode,
        source_path: str | None = None,
        parser_mode: SectionSplitterMode | str = SectionSplitterMode.COMMON,
        page_evidence: list[PdfPageLayoutEvidence] | None = None,
        page_boundaries: list[PdfPageTextBoundary] | None = None,
        outline: PdfOutlineResult | None = None,
    ) -> StructuredDocument:
        """Build a structured document with fallback on split errors."""
        try:
            resolved_mode = SectionSplitterMode.resolve(parser_mode)
            outline_sections = (
                self._project_outline_sections(raw_text, outline, page_boundaries)
                if outline is not None and outline.usable
                else []
            )
            if outline_sections:
                sections = outline_sections
                parse_provenance = self._build_parse_provenance(
                    requested_parser_mode=resolved_mode.value,
                    effective_parser_mode="native_pdf_outline",
                    fallback_used=False,
                    fallback_reason=None,
                    source="validated_pdf_outline",
                )
            elif self.section_splitter_selector is not None:
                sections, parse_provenance = (
                    self.section_splitter_selector.split_with_provenance(
                        raw_text=raw_text,
                        language=language,
                        mode=resolved_mode,
                    )
                )
            else:
                sections = self.section_splitter.split(
                    raw_text=raw_text,
                    language=language,
                )
                parse_provenance = self._build_parse_provenance(
                    requested_parser_mode=SectionSplitterMode.COMMON.value,
                    effective_parser_mode=SectionSplitterMode.COMMON.value,
                    fallback_used=False,
                    fallback_reason=None,
                    source="common_section_splitter",
                )
            if outline is not None:
                parse_provenance["outline"] = outline.to_dict()
            if outline_sections:
                toc_result = None
            else:
                toc_result = (
                    self.toc_detector.detect_with_page_evidence(raw_text, page_evidence)
                    if page_evidence is not None
                    else self.toc_detector.detect(raw_text)
                )
                if page_evidence is not None:
                    toc_result = self.toc_detector.validate_for_projection(
                        toc_result,
                        page_evidence,
                    )
            if toc_result is not None:
                parse_provenance["toc_detection"] = self._serialize_toc_detection(toc_result)
            projected_sections = (
                self._project_toc_sections(raw_text, toc_result, page_boundaries)
                if toc_result is not None and toc_result.usable
                else []
            )
            if projected_sections:
                sections = projected_sections
                parse_provenance["effective_parser"] = "validated_toc_projection"
            if not sections:
                return self._build_fallback_document(
                    document_id=document_id,
                    title=title,
                    source_path=source_path,
                    language=language,
                    raw_text=raw_text,
                    error_code="empty_sections_result",
                    error_message="SectionSplitter returned no sections.",
                    parse_provenance=self._build_parse_provenance(
                        requested_parser_mode=resolved_mode.value,
                        effective_parser_mode="fallback_document",
                        fallback_used=True,
                        fallback_reason="empty_sections_result",
                        source="structured_document_builder",
                    ),
                )

            document = StructuredDocument(
                document_id=document_id,
                title=title,
                source_path=source_path,
                language=language.value,
                raw_text=raw_text,
                sections=sections,
                structure_error_code=None,
                structure_error_message=None,
                parse_provenance=parse_provenance,
            )
            return build_document_hierarchy_from_sections(document)
        except ValueError as error:
            resolved_mode = SectionSplitterMode.resolve(parser_mode)
            return self._build_fallback_document(
                document_id=document_id,
                title=title,
                source_path=source_path,
                language=language,
                raw_text=raw_text,
                error_code=self._map_value_error_code(error),
                error_message=str(error),
                parse_provenance=self._build_parse_provenance(
                    requested_parser_mode=resolved_mode.value,
                    effective_parser_mode="fallback_document",
                    fallback_used=True,
                    fallback_reason=self._map_value_error_code(error),
                    source="structured_document_builder",
                ),
            )
        except Exception as error:
            resolved_mode = SectionSplitterMode.resolve(parser_mode)
            return self._build_fallback_document(
                document_id=document_id,
                title=title,
                source_path=source_path,
                language=language,
                raw_text=raw_text,
                error_code="section_split_unexpected_error",
                error_message=str(error),
                parse_provenance=self._build_parse_provenance(
                    requested_parser_mode=resolved_mode.value,
                    effective_parser_mode="fallback_document",
                    fallback_used=True,
                    fallback_reason="section_split_unexpected_error",
                    source="structured_document_builder",
                ),
            )

    @staticmethod
    def _map_value_error_code(error: ValueError) -> str:
        """Map expected ValueError messages to stable structure error codes."""
        message = str(error).lower()
        if "unsupported document structure language" in message:
            return "unsupported_structure_language"
        return "section_split_value_error"

    @staticmethod
    def _build_fallback_document(
        *,
        document_id: str,
        title: str,
        source_path: str | None,
        language: LanguageCode,
        raw_text: str,
        error_code: str,
        error_message: str,
        parse_provenance: dict[str, object] | None = None,
    ) -> StructuredDocument:
        """Build fallback document with one full-text section."""
        fallback_section = StructuredSection(
            section_id="section-0",
            section_index=0,
            title=None,
            level=1,
            content=raw_text,
            char_start=0,
            char_end=len(raw_text),
            section_role=SectionRole.MAIN_BODY,
        )
        fallback_document = StructuredDocument(
            document_id=document_id,
            title=title,
            source_path=source_path,
            language=language.value,
            raw_text=raw_text,
            sections=[fallback_section],
            structure_error_code=error_code,
            structure_error_message=error_message,
            parse_provenance=(
                dict(parse_provenance)
                if parse_provenance is not None
                else StructuredDocumentBuilder._build_parse_provenance(
                    requested_parser_mode=SectionSplitterMode.COMMON.value,
                    effective_parser_mode="fallback_document",
                    fallback_used=True,
                    fallback_reason=error_code,
                    source="structured_document_builder",
                )
            ),
        )
        return build_document_hierarchy_from_sections(fallback_document)

    @staticmethod
    def _build_parse_provenance(
        *,
        requested_parser_mode: str,
        effective_parser_mode: str,
        fallback_used: bool,
        fallback_reason: str | None,
        source: str,
    ) -> dict[str, object]:
        return {
            "requested_parser_mode": requested_parser_mode,
            "effective_parser_mode": effective_parser_mode,
            "fallback_used": fallback_used,
            "fallback_reason": fallback_reason,
            "source": source,
        }

    @staticmethod
    def _project_outline_sections(
        raw_text: str,
        outline: PdfOutlineResult,
        page_boundaries: list[PdfPageTextBoundary] | None,
    ) -> list[StructuredSection]:
        """Project a validated native Outline into page-backed section ranges."""
        if not page_boundaries or not outline.entries:
            return []
        separator_text = "\n\n" if "\n\n" in raw_text else "\n"
        boundary_texts = [boundary.text for boundary in page_boundaries if boundary.text]
        if separator_text.join(boundary_texts) != raw_text:
            return []
        page_by_index = {boundary.page_index: boundary for boundary in page_boundaries}
        starts: list[tuple[PdfOutlineEntry, int]] = []
        for entry in outline.entries:
            if entry.page_index is None or entry.page_index not in page_by_index:
                return []
            boundary = page_by_index[entry.page_index]
            normalized_title = normalize_ocr_text(entry.title).strip()
            page_text = normalize_ocr_text(boundary.text)
            relative = boundary.text.find(entry.title)
            if relative < 0:
                relative = page_text.find(normalized_title)
            start = boundary.char_start if relative < 0 else boundary.char_start + relative
            starts.append((entry, start))
        if not starts:
            return []
        top_level = [(entry, start) for entry, start in starts if entry.level == 1]
        if len(top_level) < 2:
            return []
        sections: list[StructuredSection] = []
        section_index = 0
        for top_index, (chapter, chapter_start) in enumerate(top_level):
            chapter_end = top_level[top_index + 1][1] if top_index + 1 < len(top_level) else len(raw_text)
            children = [
                (entry, start)
                for entry, start in starts
                if entry.level > chapter.level and chapter_start <= start < chapter_end
            ]
            ranges = (
                [(chapter, chapter_start)]
                if not children
                else [(chapter, chapter_start), *children]
            )
            for child_index, (entry, start) in enumerate(ranges):
                end = (
                    ranges[child_index + 1][1]
                    if child_index + 1 < len(ranges)
                    else chapter_end
                )
                if end <= start:
                    return []
                sections.append(
                    StructuredSection(
                        section_id=f"outline-section-{section_index}",
                        section_index=section_index,
                        title=entry.title,
                        level=1 if child_index == 0 else 2,
                        content=raw_text[start:end],
                        char_start=start,
                        char_end=end,
                        section_kind=("toc_chapter" if child_index == 0 else "toc_subsection"),
                        section_role=SectionRole.MAIN_BODY,
                    )
                )
                section_index += 1
        return sections

    @staticmethod
    def _serialize_toc_detection(result: object) -> dict[str, object]:
        """Keep TOC evidence JSON-safe without making it structure authority."""
        return {
            "detected": bool(getattr(result, "detected", False)),
            "usable": bool(getattr(result, "usable", False)),
            "confidence": float(getattr(result, "confidence", 0.0)),
            "shape": str(getattr(getattr(result, "shape", None), "value", "unknown")),
            "entry_count": len(getattr(result, "entries", []) or []),
            "toc_char_start": getattr(result, "toc_char_start", None),
            "toc_char_end": getattr(result, "toc_char_end", None),
            "candidate_page_indices": list(
                getattr(result, "candidate_page_indices", []) or []
            ),
            "reasons": list(getattr(result, "reasons", []) or []),
        }

    @staticmethod
    def _project_toc_sections(
        raw_text: str,
        toc_result: object,
        page_boundaries: list[PdfPageTextBoundary] | None,
    ) -> list[StructuredSection]:
        """Project validated TOC entries to section ranges; reject partial matches."""
        entries = list(getattr(toc_result, "entries", []) or [])
        toc_end = getattr(toc_result, "toc_char_end", None)
        if len(entries) < 3 or toc_end is None or not page_boundaries:
            return []
        canonical_text = "\n\n".join(boundary.text for boundary in page_boundaries)
        if canonical_text != raw_text:
            return []
        page_starts = {boundary.page_index: boundary.char_start for boundary in page_boundaries}
        page_ends = {boundary.page_index: boundary.char_end for boundary in page_boundaries}
        lines = [
            match
            for match in re.finditer(r"[^\n\r]+", raw_text)
            if match.start() >= toc_end
        ]
        matches: list[tuple[object, int, int]] = []
        cursor = 0
        for entry in entries:
            title = normalize_ocr_text(str(getattr(entry, "title", ""))).strip(" .\t")
            found = None
            for line in lines[cursor:]:
                if normalize_ocr_text(line.group(0)).strip(" .\t") == title:
                    found = line
                    cursor = lines.index(line, cursor) + 1
                    break
            if found is None:
                continue
            matches.append((entry, found.start(), found.end()))
        if len(matches) / max(1, len(entries)) < 0.6:
            return []
        # Require each matched title to live on the page predicted by its printed number.
        page_offsets: list[int] = []
        for entry, start, _ in matches:
            page_number = getattr(entry, "page_number", None)
            if page_number is None:
                continue
            actual_page = next(
                (index for index, (page_start, page_end) in enumerate(zip(page_starts.values(), page_ends.values())) if page_start <= start <= page_end),
                None,
            )
            if actual_page is not None:
                page_offsets.append(actual_page - int(page_number))
        if not page_offsets or len(set(page_offsets)) != 1:
            return []
        page_offset = page_offsets[0]
        if any(
            int(getattr(entry, "page_number", 0)) + page_offset not in page_starts
            for entry, _, _ in matches
            if getattr(entry, "page_number", None) is not None
        ):
            return []
        sections: list[StructuredSection] = []
        for index, (entry, start, heading_end) in enumerate(matches):
            end = matches[index + 1][1] if index + 1 < len(matches) else len(raw_text)
            sections.append(
                StructuredSection(
                    section_id=f"toc-section-{index}",
                    section_index=index,
                    title=normalize_ocr_text(raw_text[start:heading_end]).strip(),
                    level=max(1, int(getattr(entry, "level", 1))),
                    content=raw_text[start:end],
                    char_start=start,
                    char_end=end,
                    section_kind=(
                        "toc_chapter"
                        if max(1, int(getattr(entry, "level", 1))) == 1
                        else "toc_subsection"
                    ),
                    section_role=SectionRole.MAIN_BODY,
                )
            )
        return sections
