#!/usr/bin/env python3
"""Regression tests for deterministic manual-structure projection."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from document_structure.manual_structure_projection import (  # noqa: E402
    ManualStructureAnchor,
    ManualStructureEntry,
    project_manual_structure_plan,
)


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _codes(result) -> set[str]:
    return {issue.code for issue in result.issues}


def test_projects_valid_chapter_section_plan() -> None:
    result = project_manual_structure_plan(
        [
            ManualStructureEntry(
                title="  Chapter One ",
                level=1,
                anchor=ManualStructureAnchor(
                    anchor_type="char-range",
                    char_start=0,
                    char_end=200,
                ),
                external_id=" ch-1 ",
            ),
            ManualStructureEntry(
                title="Section One",
                level=2,
                anchor=ManualStructureAnchor(
                    anchor_type="char_range",
                    char_start=10,
                    char_end=90,
                ),
            ),
            ManualStructureEntry(
                title="Section Two",
                level=2,
                anchor=ManualStructureAnchor(
                    anchor_type="page_range",
                    page_start_index=5,
                    page_end_index=7,
                ),
            ),
        ],
        source_hash=" raw-hash ",
    )

    _assert(result.valid, f"plan should be valid: {result.issues}")
    _assert(result.source_hash == "raw-hash", "source hash should be normalized")
    _assert(len(result.normalized_entries) == 3, "all entries should be normalized")
    _assert(
        result.normalized_entries[0].anchor.anchor_type == "char_range",
        "anchor type should normalize hyphen to underscore",
    )
    _assert(
        result.normalized_entries[0].external_id == "ch-1",
        "external id should be normalized",
    )
    _assert(len(result.preview_chapters) == 1, "one chapter should be previewed")
    _assert(
        [section.title for section in result.preview_chapters[0].sections]
        == ["Section One", "Section Two"],
        "section preview should preserve order",
    )


def test_chapter_only_plan_creates_same_name_preview_section() -> None:
    result = project_manual_structure_plan(
        [
            ManualStructureEntry(
                title="Chapter Only",
                level=1,
                anchor=ManualStructureAnchor(
                    anchor_type="page_range",
                    page_start_index=1,
                    page_end_index=3,
                ),
            )
        ]
    )

    _assert(result.valid, f"chapter-only plan should be valid: {result.issues}")
    sections = result.preview_chapters[0].sections
    _assert(len(sections) == 1, "chapter-only plan should preview one implicit section")
    _assert(sections[0].title == "Chapter Only", "implicit section should reuse chapter title")


def test_rejects_invalid_shape_and_anchors() -> None:
    orphan = project_manual_structure_plan(
        [
            ManualStructureEntry(
                title="Orphan",
                level=2,
                anchor=ManualStructureAnchor(anchor_type="char_range", char_start=0),
            )
        ]
    )
    _assert(not orphan.valid, "orphan section should fail")
    _assert("invalid_level_sequence" in _codes(orphan), "orphan should report level sequence")

    too_deep = project_manual_structure_plan(
        [
            ManualStructureEntry(
                title="Too Deep",
                level=3,
                anchor=ManualStructureAnchor(anchor_type="char_range", char_start=0),
            )
        ]
    )
    _assert(not too_deep.valid, "level > 2 should fail")
    _assert("unsupported_depth" in _codes(too_deep), "depth error should be explicit")

    mixed_anchor = project_manual_structure_plan(
        [
            ManualStructureEntry(
                title="Mixed",
                level=1,
                anchor=ManualStructureAnchor(
                    anchor_type="char_range",
                    char_start=0,
                    page_start_index=1,
                ),
            )
        ]
    )
    _assert(not mixed_anchor.valid, "mixed anchor fields should fail")
    _assert("malformed_payload" in _codes(mixed_anchor), "mixed anchor should be malformed")

    unsupported_anchor = project_manual_structure_plan(
        [
            ManualStructureEntry(
                title="Bad Anchor",
                level=1,
                anchor=ManualStructureAnchor(anchor_type="line_range"),
            )
        ]
    )
    _assert(not unsupported_anchor.valid, "unsupported anchor should fail")
    _assert(
        "unsupported_anchor_type" in _codes(unsupported_anchor),
        "unsupported anchor code should be explicit",
    )


def test_rejects_overlapping_sibling_ranges() -> None:
    char_result = project_manual_structure_plan(
        [
            ManualStructureEntry(
                title="Chapter One",
                level=1,
                anchor=ManualStructureAnchor(
                    anchor_type="char_range",
                    char_start=0,
                    char_end=100,
                ),
            ),
            ManualStructureEntry(
                title="Chapter Two",
                level=1,
                anchor=ManualStructureAnchor(
                    anchor_type="char_range",
                    char_start=90,
                    char_end=150,
                ),
            ),
        ]
    )
    _assert(not char_result.valid, "overlapping chapter char ranges should fail")
    _assert("overlapping_range" in _codes(char_result), "char overlap should be explicit")

    page_result = project_manual_structure_plan(
        [
            ManualStructureEntry(
                title="Chapter One",
                level=1,
                anchor=ManualStructureAnchor(
                    anchor_type="page_range",
                    page_start_index=1,
                    page_end_index=3,
                ),
            ),
            ManualStructureEntry(
                title="Chapter Two",
                level=1,
                anchor=ManualStructureAnchor(
                    anchor_type="page_range",
                    page_start_index=3,
                    page_end_index=4,
                ),
            ),
        ]
    )
    _assert(not page_result.valid, "overlapping chapter page ranges should fail")
    _assert("overlapping_range" in _codes(page_result), "page overlap should be explicit")


if __name__ == "__main__":
    test_projects_valid_chapter_section_plan()
    test_chapter_only_plan_creates_same_name_preview_section()
    test_rejects_invalid_shape_and_anchors()
    test_rejects_overlapping_sibling_ranges()
    print("manual structure projection tests passed")
