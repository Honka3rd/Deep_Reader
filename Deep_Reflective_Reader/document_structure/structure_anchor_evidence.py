from __future__ import annotations

from dataclasses import dataclass, field

from doc_loaders.pdf_page_evidence import PdfPageTextBoundary
from document_structure.structured_document import (
    StructuredChapter,
    StructuredDocument,
    StructuredSection,
)


@dataclass(frozen=True)
class StructureAnchorEvidence:
    """Lightweight parsed anchor evidence for TOC edit prefill."""

    target_type: str
    target_id: str
    title: str | None
    anchor_type: str | None
    status: str
    reason: str | None = None
    char_start: int | None = None
    char_end: int | None = None
    page_start_index: int | None = None
    page_end_index: int | None = None
    page_start_label: str | None = None
    page_end_label: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "target_type": self.target_type,
            "target_id": self.target_id,
            "title": self.title,
            "anchor_type": self.anchor_type,
            "status": self.status,
            "reason": self.reason,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "page_start_index": self.page_start_index,
            "page_end_index": self.page_end_index,
            "page_start_label": self.page_start_label,
            "page_end_label": self.page_end_label,
        }


@dataclass(frozen=True)
class ChapterStructureAnchorEvidence:
    chapter: StructureAnchorEvidence
    sections: list[StructureAnchorEvidence] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "chapter": self.chapter.to_dict(),
            "sections": [section.to_dict() for section in self.sections],
        }


@dataclass(frozen=True)
class DocumentStructureAnchorEvidence:
    document_id: str
    chapters: list[ChapterStructureAnchorEvidence] = field(default_factory=list)
    evidence_schema_version: int = 1

    def to_dict(self) -> dict[str, object]:
        return {
            "document_id": self.document_id,
            "evidence_schema_version": self.evidence_schema_version,
            "chapters": [chapter.to_dict() for chapter in self.chapters],
        }


def project_structure_anchor_evidence(
    document: StructuredDocument,
    *,
    page_boundaries: list[PdfPageTextBoundary] | None = None,
) -> DocumentStructureAnchorEvidence:
    """Project existing hierarchy spans into read-only anchor evidence."""
    boundaries = list(page_boundaries or [])
    chapters: list[ChapterStructureAnchorEvidence] = []
    for chapter in document.chapters:
        chapter_span = _chapter_char_span(chapter)
        chapters.append(
            ChapterStructureAnchorEvidence(
                chapter=_evidence_from_span(
                    target_type="chapter",
                    target_id=chapter.chapter_id,
                    title=chapter.title,
                    span=chapter_span,
                    page_boundaries=boundaries,
                ),
                sections=[
                    _evidence_from_span(
                        target_type="section",
                        target_id=section.section_id,
                        title=section.title,
                        span=(section.char_start, section.char_end),
                        page_boundaries=boundaries,
                    )
                    for section in chapter.sections
                ],
            )
        )
    return DocumentStructureAnchorEvidence(
        document_id=document.document_id,
        chapters=chapters,
    )


def _chapter_char_span(chapter: StructuredChapter) -> tuple[int, int] | None:
    spans = [
        (section.char_start, section.char_end)
        for section in chapter.sections
        if section.char_start >= 0 and section.char_end > section.char_start
    ]
    if not spans:
        return None
    return min(start for start, _ in spans), max(end for _, end in spans)


def _evidence_from_span(
    *,
    target_type: str,
    target_id: str,
    title: str | None,
    span: tuple[int, int] | None,
    page_boundaries: list[PdfPageTextBoundary],
) -> StructureAnchorEvidence:
    if span is None:
        return _unavailable(
            target_type=target_type,
            target_id=target_id,
            title=title,
            reason="missing_char_span",
        )

    char_start, char_end = span
    if char_start < 0 or char_end <= char_start:
        return _unavailable(
            target_type=target_type,
            target_id=target_id,
            title=title,
            reason="invalid_char_span",
        )

    page_range = _resolve_page_range(
        char_start=char_start,
        char_end=char_end,
        page_boundaries=page_boundaries,
    )
    if page_range is not None:
        first_page, last_page = page_range
        return StructureAnchorEvidence(
            target_type=target_type,
            target_id=target_id,
            title=title,
            anchor_type="page_range",
            status="available",
            char_start=char_start,
            char_end=char_end,
            page_start_index=first_page.page_index,
            page_end_index=last_page.page_index,
            page_start_label=first_page.page_label,
            page_end_label=last_page.page_label,
        )

    return StructureAnchorEvidence(
        target_type=target_type,
        target_id=target_id,
        title=title,
        anchor_type="char_range",
        status="available",
        reason="page_boundaries_unavailable",
        char_start=char_start,
        char_end=char_end,
    )


def _resolve_page_range(
    *,
    char_start: int,
    char_end: int,
    page_boundaries: list[PdfPageTextBoundary],
) -> tuple[PdfPageTextBoundary, PdfPageTextBoundary] | None:
    if not page_boundaries:
        return None
    overlapping = [
        boundary
        for boundary in page_boundaries
        if boundary.char_start < char_end and char_start < boundary.char_end
    ]
    if not overlapping:
        return None
    first_page = min(overlapping, key=lambda boundary: boundary.page_index)
    last_page = max(overlapping, key=lambda boundary: boundary.page_index)
    if first_page.char_start > char_start or last_page.char_end < char_end:
        return None
    return first_page, last_page


def _unavailable(
    *,
    target_type: str,
    target_id: str,
    title: str | None,
    reason: str,
) -> StructureAnchorEvidence:
    return StructureAnchorEvidence(
        target_type=target_type,
        target_id=target_id,
        title=title,
        anchor_type=None,
        status="unavailable",
        reason=reason,
    )


__all__ = [
    "ChapterStructureAnchorEvidence",
    "DocumentStructureAnchorEvidence",
    "StructureAnchorEvidence",
    "project_structure_anchor_evidence",
]
