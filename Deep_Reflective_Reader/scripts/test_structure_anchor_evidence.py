#!/usr/bin/env python3
"""Regression tests for existing hierarchy anchor evidence projection."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from doc_loaders.pdf_page_evidence import PdfPageTextBoundary  # noqa: E402
from document_structure.section_role import SectionRole  # noqa: E402
from document_structure.structure_anchor_evidence import (  # noqa: E402
    project_structure_anchor_evidence,
)
from document_structure.structured_document import (  # noqa: E402
    StructuredChapter,
    StructuredDocument,
    StructuredSection,
)


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _section(
    *,
    section_id: str,
    title: str,
    char_start: int,
    char_end: int,
    chapter_id: str = "chapter-1",
) -> StructuredSection:
    return StructuredSection(
        section_id=section_id,
        section_index=0,
        title=title,
        level=2,
        content="",
        char_start=char_start,
        char_end=char_end,
        container_title="Chapter One",
        section_role=SectionRole.MAIN_BODY,
        parent_chapter_id=chapter_id,
    )


def _document(sections: list[StructuredSection]) -> StructuredDocument:
    return StructuredDocument(
        document_id="doc-1",
        title="Doc",
        source_path=None,
        language="en",
        raw_text="",
        sections=[],
        chapters=[
            StructuredChapter(
                chapter_id="chapter-1",
                title="Chapter One",
                level=1,
                chapter_role=None,
                sections=sections,
            )
        ],
        structure_nodes=[],
    )


def test_prefers_page_range_when_page_boundaries_cover_existing_spans() -> None:
    document = _document(
        [
            _section(section_id="section-1", title="Section One", char_start=0, char_end=20),
            _section(section_id="section-2", title="Section Two", char_start=22, char_end=45),
        ]
    )
    page_boundaries = [
        PdfPageTextBoundary(
            page_index=0,
            char_start=0,
            char_end=20,
            text="page one",
            page_label="1",
        ),
        PdfPageTextBoundary(
            page_index=1,
            char_start=22,
            char_end=45,
            text="page two",
            page_label="2",
        ),
    ]

    evidence = project_structure_anchor_evidence(
        document,
        page_boundaries=page_boundaries,
    )

    chapter = evidence.chapters[0].chapter
    _assert(chapter.anchor_type == "page_range", "chapter should prefer page_range")
    _assert(chapter.page_start_index == 0, "chapter page start should be projected")
    _assert(chapter.page_end_index == 1, "chapter page end should be projected")
    _assert(chapter.char_start == 0 and chapter.char_end == 45, "chapter char span should remain")
    sections = evidence.chapters[0].sections
    _assert(
        [section.anchor_type for section in sections] == ["page_range", "page_range"],
        "sections should prefer page_range",
    )
    _assert(
        [section.page_start_label for section in sections] == ["1", "2"],
        "page labels should be projected when available",
    )


def test_falls_back_to_char_range_without_page_boundaries() -> None:
    document = _document(
        [_section(section_id="section-1", title="Section One", char_start=5, char_end=25)]
    )

    evidence = project_structure_anchor_evidence(document)

    chapter = evidence.chapters[0].chapter
    section = evidence.chapters[0].sections[0]
    _assert(chapter.anchor_type == "char_range", "chapter should use char fallback")
    _assert(section.anchor_type == "char_range", "section should use char fallback")
    _assert(
        section.reason == "page_boundaries_unavailable",
        "char fallback should report why page evidence is absent",
    )


def test_invalid_existing_span_is_explicitly_unavailable() -> None:
    document = _document(
        [_section(section_id="section-1", title="Broken", char_start=40, char_end=40)]
    )

    evidence = project_structure_anchor_evidence(document)

    chapter = evidence.chapters[0].chapter
    section = evidence.chapters[0].sections[0]
    _assert(chapter.status == "unavailable", "chapter with no valid sections is unavailable")
    _assert(chapter.reason == "missing_char_span", "chapter reason should be explicit")
    _assert(section.status == "unavailable", "invalid section span is unavailable")
    _assert(section.reason == "invalid_char_span", "section reason should be explicit")


if __name__ == "__main__":
    test_prefers_page_range_when_page_boundaries_cover_existing_spans()
    test_falls_back_to_char_range_without_page_boundaries()
    test_invalid_existing_span_is_explicitly_unavailable()
    print("structure anchor evidence tests passed")
