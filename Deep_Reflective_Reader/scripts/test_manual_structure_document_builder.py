#!/usr/bin/env python3
"""Regression tests for manual-structure StructuredDocument draft building."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from document_structure.manual_structure_document_builder import (  # noqa: E402
    build_manual_structure_document_draft,
)
from document_structure.manual_structure_projection import (  # noqa: E402
    ManualStructureAnchor,
    ManualStructureEntry,
)
from doc_loaders.pdf_page_evidence import PdfPageTextBoundary  # noqa: E402


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _codes(result) -> set[str]:
    return {issue.code for issue in result.issues}


def test_builds_hierarchy_only_structured_document_from_char_ranges() -> None:
    raw_text = (
        "Chapter One\n"
        "Section One body.\n"
        "Section Two body.\n"
        "Chapter Two\n"
        "Final body."
    )
    chapter_two_start = raw_text.index("Chapter Two")
    result = build_manual_structure_document_draft(
        document_id="manual-doc",
        title="Manual Doc",
        source_path="data/raw/manual.txt",
        language="en",
        raw_text=raw_text,
        source_hash="raw-hash",
        entries=[
            ManualStructureEntry(
                title="Chapter One",
                level=1,
                anchor=ManualStructureAnchor(
                    anchor_type="char_range",
                    char_start=0,
                    char_end=chapter_two_start,
                ),
            ),
            ManualStructureEntry(
                title="Section One",
                level=2,
                anchor=ManualStructureAnchor(
                    anchor_type="char_range",
                    char_start=raw_text.index("Section One"),
                    char_end=raw_text.index("Section Two"),
                ),
            ),
            ManualStructureEntry(
                title="Section Two",
                level=2,
                anchor=ManualStructureAnchor(
                    anchor_type="char_range",
                    char_start=raw_text.index("Section Two"),
                    char_end=chapter_two_start,
                ),
            ),
            ManualStructureEntry(
                title="Chapter Two",
                level=1,
                anchor=ManualStructureAnchor(
                    anchor_type="char_range",
                    char_start=chapter_two_start,
                ),
            ),
        ],
    )

    _assert(result.valid, f"draft build should pass: {result.issues}")
    document = result.document
    _assert(document is not None, "valid draft should include a document")
    _assert(document.sections == [], "root sections must not be populated")
    _assert(document.structure_nodes == [], "structure_nodes must not be populated")
    _assert(len(document.chapters) == 2, "two chapters should be built")
    _assert(
        [section.title for section in document.chapters[0].sections]
        == ["Section One", "Section Two"],
        "chapter one sections should preserve manual order",
    )
    _assert(
        document.chapters[1].sections[0].is_implicit_section,
        "chapter-only entry should become an implicit same-name section",
    )
    _assert(
        document.chapters[1].sections[0].content == raw_text[chapter_two_start:],
        "open final chapter range should extend to raw text end",
    )
    payload = document.to_dict()
    _assert("sections" not in payload, "default serialization must omit root sections")
    _assert(
        "structure_nodes" not in payload,
        "default serialization must omit structure_nodes",
    )
    provenance = payload["parse_provenance"]
    _assert(
        provenance["effective_parser_mode"] == "manual_structure_projection",
        "manual projection should be recorded as effective parser mode",
    )
    _assert(
        provenance["manual_structure"]["source_hash"] == "raw-hash",
        "source hash should be carried as provenance only",
    )


def test_rejects_page_range_without_page_boundaries() -> None:
    result = build_manual_structure_document_draft(
        document_id="manual-doc",
        title="Manual Doc",
        raw_text="Chapter\nBody",
        entries=[
            ManualStructureEntry(
                title="Chapter",
                level=1,
                anchor=ManualStructureAnchor(
                    anchor_type="page_range",
                    page_start_index=0,
                ),
            )
        ],
    )

    _assert(not result.valid, "page range draft should fail without page boundaries")
    _assert(result.document is None, "invalid draft must not include a document")
    _assert(
        "page_boundaries_required" in _codes(result),
        "missing page evidence should be explicit",
    )


def test_builds_hierarchy_from_page_ranges_with_page_boundaries() -> None:
    page_1 = "Chapter One\nSection One body."
    page_2 = "Section Two\nSection Two body."
    page_3 = "Chapter Two\nFinal body."
    raw_text = "\n\n".join([page_1, page_2, page_3])
    page_2_start = len(page_1) + 2
    page_3_start = page_2_start + len(page_2) + 2
    page_boundaries = [
        PdfPageTextBoundary(
            page_index=0,
            char_start=0,
            char_end=len(page_1),
            text=page_1,
            page_label="1",
        ),
        PdfPageTextBoundary(
            page_index=1,
            char_start=page_2_start,
            char_end=page_2_start + len(page_2),
            text=page_2,
            page_label="2",
        ),
        PdfPageTextBoundary(
            page_index=2,
            char_start=page_3_start,
            char_end=len(raw_text),
            text=page_3,
            page_label="3",
        ),
    ]

    result = build_manual_structure_document_draft(
        document_id="manual-doc",
        title="Manual Doc",
        raw_text=raw_text,
        source_hash="raw-hash",
        page_boundaries=page_boundaries,
        entries=[
            ManualStructureEntry(
                title="Chapter One",
                level=1,
                anchor=ManualStructureAnchor(
                    anchor_type="page_range",
                    page_start_index=0,
                    page_end_index=1,
                ),
            ),
            ManualStructureEntry(
                title="Section One",
                level=2,
                anchor=ManualStructureAnchor(
                    anchor_type="page_range",
                    page_start_index=0,
                    page_end_index=0,
                ),
            ),
            ManualStructureEntry(
                title="Section Two",
                level=2,
                anchor=ManualStructureAnchor(
                    anchor_type="page_range",
                    page_start_index=1,
                    page_end_index=1,
                ),
            ),
            ManualStructureEntry(
                title="Chapter Two",
                level=1,
                anchor=ManualStructureAnchor(
                    anchor_type="page_range",
                    page_start_index=2,
                ),
            ),
        ],
    )

    _assert(result.valid, f"page_range draft build should pass: {result.issues}")
    document = result.document
    _assert(document is not None, "valid page_range draft should include a document")
    _assert(document.sections == [], "page_range draft must not populate root sections")
    _assert(document.structure_nodes == [], "page_range draft must not populate structure_nodes")
    _assert(len(document.chapters) == 2, "two chapters should be built")
    _assert(
        document.chapters[0].sections[0].content == page_1,
        "section one should be sliced through page boundary offsets",
    )
    _assert(
        document.chapters[0].sections[1].content == page_2,
        "section two should be sliced through page boundary offsets",
    )
    _assert(
        document.chapters[1].sections[0].content == page_3,
        "open final page_range should extend to raw text end",
    )
    _assert(
        document.parse_provenance["manual_structure"]["anchor_type"] == "page_range",
        "manual provenance should record page_range source anchors",
    )


def test_rejects_ambiguous_page_boundary_evidence() -> None:
    page_1 = "Chapter One\nBody one."
    page_2 = "Chapter Two\nBody two."
    raw_text = f"{page_1}\n\n{page_2}"
    result = build_manual_structure_document_draft(
        document_id="manual-doc",
        title="Manual Doc",
        raw_text=raw_text,
        page_boundaries=[
            PdfPageTextBoundary(
                page_index=0,
                char_start=0,
                char_end=len(page_1),
                text=page_1,
            ),
            PdfPageTextBoundary(
                page_index=0,
                char_start=len(page_1) + 2,
                char_end=len(raw_text),
                text=page_2,
            ),
        ],
        entries=[
            ManualStructureEntry(
                title="Chapter One",
                level=1,
                anchor=ManualStructureAnchor(
                    anchor_type="page_range",
                    page_start_index=0,
                    page_end_index=0,
                ),
            )
        ],
    )

    _assert(not result.valid, "duplicate page boundaries should fail")
    _assert(result.document is None, "ambiguous page evidence must not build a document")
    _assert(
        "ambiguous_page_boundary" in _codes(result),
        "ambiguous page evidence should use an explicit issue code",
    )


def test_rejects_out_of_bounds_char_range() -> None:
    result = build_manual_structure_document_draft(
        document_id="manual-doc",
        title="Manual Doc",
        raw_text="short",
        entries=[
            ManualStructureEntry(
                title="Chapter",
                level=1,
                anchor=ManualStructureAnchor(
                    anchor_type="char_range",
                    char_start=0,
                    char_end=99,
                ),
            )
        ],
    )

    _assert(not result.valid, "out-of-bounds draft should fail")
    _assert(result.document is None, "invalid range must not build a document")
    _assert(
        "out_of_range_anchor" in _codes(result),
        "out-of-range code should be explicit",
    )


if __name__ == "__main__":
    test_builds_hierarchy_only_structured_document_from_char_ranges()
    test_rejects_page_range_without_page_boundaries()
    test_builds_hierarchy_from_page_ranges_with_page_boundaries()
    test_rejects_ambiguous_page_boundary_evidence()
    test_rejects_out_of_bounds_char_range()
    print("manual structure document builder tests passed")
