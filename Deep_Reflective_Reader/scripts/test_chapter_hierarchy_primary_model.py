#!/usr/bin/env python3
"""Tests for chapters-as-primary hierarchy helpers."""

from __future__ import annotations

import json

from document_structure.document_hierarchy_index import (
    assert_chapter_hierarchy_consistency,
    build_section_index_from_chapters,
    flatten_sections_from_chapters,
    get_effective_sections,
    validate_chapter_hierarchy_consistency,
    with_legacy_sections_synced_from_chapters,
)
from document_structure.section_role import SectionRole
from document_structure.structured_document import (
    StructuredChapter,
    StructuredDocument,
    StructuredSection,
)
from shared.task_unit_model import TaskUnit


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _make_section(
    section_id: str,
    title: str,
    level: int = 2,
    role: SectionRole | None = SectionRole.MAIN_BODY,
) -> StructuredSection:
    section_index = 0
    if "-" in section_id:
        maybe_index = section_id.split("-")[-1]
        if maybe_index.isdigit():
            section_index = int(maybe_index)
    return StructuredSection(
        section_id=section_id,
        section_index=section_index,
        title=title,
        level=level,
        content=f"{title}-content",
        char_start=0,
        char_end=len(f"{title}-content"),
        section_role=role,
        task_units=[
            TaskUnit(
                unit_id=f"task-unit-{section_id}",
                title=title,
                container_title=None,
                content=f"{title}-unit",
                source_section_ids=[section_id],
                is_fallback_generated=False,
                parent_section_id=section_id,
            )
        ],
        parent_chapter_id=None,
    )


def test_flatten_sections_from_chapters() -> None:
    s1 = _make_section("section-1", "第一章")
    s2 = _make_section("section-2", "1.1")
    s3 = _make_section("section-3", "第二章")
    c1 = StructuredChapter(
        chapter_id="chapter-0",
        title="第一章",
        level=2,
        chapter_role="main_body",
        sections=[
            StructuredSection(**{**s1.__dict__, "parent_chapter_id": "chapter-0"}),
            StructuredSection(**{**s2.__dict__, "parent_chapter_id": "chapter-0"}),
        ],
    )
    c2 = StructuredChapter(
        chapter_id="chapter-1",
        title="第二章",
        level=2,
        chapter_role="main_body",
        sections=[
            StructuredSection(**{**s3.__dict__, "parent_chapter_id": "chapter-1"}),
        ],
    )
    doc = StructuredDocument(
        document_id="doc-a",
        title="DocA",
        source_path=None,
        language="zh",
        raw_text="x",
        sections=[],
        chapters=[c1, c2],
    )

    flattened = flatten_sections_from_chapters(doc)
    _assert(
        [section.section_id for section in flattened] == ["section-1", "section-2", "section-3"],
        "flatten order should follow chapter and section order",
    )


def test_get_effective_sections() -> None:
    legacy = _make_section("section-legacy", "Legacy")
    csec = StructuredSection(**{**_make_section("section-1", "第一章").__dict__, "parent_chapter_id": "chapter-0"})
    chapter = StructuredChapter(
        chapter_id="chapter-0",
        title="第一章",
        level=2,
        chapter_role="main_body",
        sections=[csec],
    )
    doc_with_chapters = StructuredDocument(
        document_id="doc-b",
        title="DocB",
        source_path=None,
        language="zh",
        raw_text="x",
        sections=[legacy],
        chapters=[chapter],
    )
    effective = get_effective_sections(doc_with_chapters)
    _assert(
        [section.section_id for section in effective] == ["section-1"],
        "when chapters exist, effective sections should come from hierarchy",
    )

    doc_legacy_only = StructuredDocument(
        document_id="doc-b2",
        title="DocB2",
        source_path=None,
        language="zh",
        raw_text="x",
        sections=[legacy],
        chapters=[],
    )
    effective_fallback = get_effective_sections(doc_legacy_only)
    _assert(
        effective_fallback == [],
        "when chapters are empty, effective sections should be empty in pure hierarchy mode",
    )


def test_consistency_valid() -> None:
    section = StructuredSection(**{**_make_section("section-1", "第一章").__dict__, "parent_chapter_id": "chapter-0"})
    chapter = StructuredChapter(
        chapter_id="chapter-0",
        title="第一章",
        level=2,
        chapter_role="main_body",
        sections=[section],
    )
    doc = StructuredDocument(
        document_id="doc-c",
        title="DocC",
        source_path=None,
        language="zh",
        raw_text="x",
        sections=[],
        chapters=[chapter],
    )
    warnings = validate_chapter_hierarchy_consistency(doc)
    _assert(warnings == [], "valid hierarchy should produce no warnings")
    assert_chapter_hierarchy_consistency(doc)


def test_duplicate_section_id_invalid() -> None:
    sec_a = StructuredSection(**{**_make_section("section-1", "第一章").__dict__, "parent_chapter_id": "chapter-0"})
    sec_b = StructuredSection(**{**_make_section("section-1", "第二章").__dict__, "parent_chapter_id": "chapter-1"})
    doc = StructuredDocument(
        document_id="doc-d",
        title="DocD",
        source_path=None,
        language="zh",
        raw_text="x",
        sections=[],
        chapters=[
            StructuredChapter("chapter-0", "第一章", 2, "main_body", [sec_a]),
            StructuredChapter("chapter-1", "第二章", 2, "main_body", [sec_b]),
        ],
    )
    warnings = validate_chapter_hierarchy_consistency(doc)
    _assert(
        any("duplicate_hierarchy_section_id:section-1" in warning for warning in warnings),
        "validator should catch duplicate hierarchy section id",
    )
    try:
        assert_chapter_hierarchy_consistency(doc)
    except ValueError as error:
        _assert("duplicate_hierarchy_section_id:section-1" in str(error), "assert version should raise")
    else:
        raise AssertionError("assert_chapter_hierarchy_consistency should raise on duplicate section id")


def test_parent_mismatch_invalid() -> None:
    bad_section = StructuredSection(
        **{
            **_make_section("section-1", "第一章").__dict__,
            "parent_chapter_id": "chapter-99",
        }
    )
    doc = StructuredDocument(
        document_id="doc-e",
        title="DocE",
        source_path=None,
        language="zh",
        raw_text="x",
        sections=[bad_section],
        chapters=[StructuredChapter("chapter-0", "第一章", 2, "main_body", [bad_section])],
    )
    warnings = validate_chapter_hierarchy_consistency(doc)
    _assert(
        any("section_parent_mismatch:section-1" in warning for warning in warnings),
        "validator should catch section parent mismatch",
    )


def test_task_unit_parent_mismatch_invalid() -> None:
    section = _make_section("section-1", "第一章")
    wrong_unit = TaskUnit(
        unit_id="task-unit-x",
        title="u",
        container_title=None,
        content="x",
        source_section_ids=["section-1"],
        is_fallback_generated=False,
        parent_section_id="section-999",
    )
    section = StructuredSection(**{**section.__dict__, "task_units": [wrong_unit], "parent_chapter_id": "chapter-0"})
    doc = StructuredDocument(
        document_id="doc-f",
        title="DocF",
        source_path=None,
        language="zh",
        raw_text="x",
        sections=[section],
        chapters=[StructuredChapter("chapter-0", "第一章", 2, "main_body", [section])],
    )
    warnings = validate_chapter_hierarchy_consistency(doc)
    _assert(
        any("task_unit_parent_section_mismatch:task-unit-x" in warning for warning in warnings),
        "validator should catch task unit parent mismatch",
    )


def test_sync_legacy_sections_from_chapters() -> None:
    stale = _make_section("section-stale", "Stale")
    h1 = StructuredSection(**{**_make_section("section-1", "第一章").__dict__, "parent_chapter_id": "chapter-0"})
    h2 = StructuredSection(**{**_make_section("section-2", "第二章").__dict__, "parent_chapter_id": "chapter-1"})
    doc = StructuredDocument(
        document_id="doc-g",
        title="DocG",
        source_path=None,
        language="zh",
        raw_text="x",
        sections=[stale],
        chapters=[
            StructuredChapter("chapter-0", "第一章", 2, "main_body", [h1]),
            StructuredChapter("chapter-1", "第二章", 2, "main_body", [h2]),
        ],
    )
    synced = with_legacy_sections_synced_from_chapters(doc)
    _assert(
        [section.section_id for section in synced.sections] == ["section-1", "section-2"],
        "legacy sections should be replaced by flattened chapter sections",
    )
    _assert(
        [chapter.chapter_id for chapter in synced.chapters] == ["chapter-0", "chapter-1"],
        "sync should not modify chapters",
    )


def test_structure_nodes_backward_compatibility() -> None:
    payload = {
        "document_id": "legacy-doc",
        "title": "Legacy",
        "source_path": None,
        "language": "zh",
        "raw_text": "第一章\n正文",
        "sections": [
            {
                "section_id": "section-0",
                "section_index": 0,
                "title": "第一章",
                "level": 2,
                "content": "第一章正文",
                "char_start": 0,
                "char_end": 5,
                "section_role": "main_body",
            }
        ],
        "structure_nodes": [
            {
                "node_id": "node::section-0",
                "node_type": "chapter",
                "title": "第一章",
                "level": 2,
                "content": "第一章正文",
                "char_start": 0,
                "char_end": 5,
                "section_role": "main_body",
                "children": [],
                "task_units": [],
                "task_artifacts": None,
                "source_section_ids": ["section-0"],
            }
        ],
    }
    doc = StructuredDocument.from_dict(payload)
    _assert(len(doc.structure_nodes) == 1, "legacy structure_nodes should still load")
    _assert(
        get_effective_sections(doc) == [],
        "effective sections should come only from chapters, not root sections/structure_nodes",
    )


def main() -> None:
    test_flatten_sections_from_chapters()
    test_get_effective_sections()
    test_consistency_valid()
    test_duplicate_section_id_invalid()
    test_parent_mismatch_invalid()
    test_task_unit_parent_mismatch_invalid()
    test_sync_legacy_sections_from_chapters()
    test_structure_nodes_backward_compatibility()
    # Also verify strict duplicate detection helper.
    duplicate_doc = StructuredDocument(
        document_id="doc-h",
        title="DocH",
        source_path=None,
        language="zh",
        raw_text="x",
        sections=[],
        chapters=[
            StructuredChapter(
                chapter_id="chapter-0",
                title="第一章",
                level=2,
                chapter_role="main_body",
                sections=[StructuredSection(**{**_make_section("section-1", "第一章").__dict__, "parent_chapter_id": "chapter-0"})],
            ),
            StructuredChapter(
                chapter_id="chapter-1",
                title="第二章",
                level=2,
                chapter_role="main_body",
                sections=[StructuredSection(**{**_make_section("section-1", "第二章").__dict__, "parent_chapter_id": "chapter-1"})],
            ),
        ],
    )
    try:
        build_section_index_from_chapters(duplicate_doc)
    except ValueError:
        pass
    else:
        raise AssertionError("build_section_index_from_chapters should raise on duplicate section id")

    print(
        json.dumps(
            {
                "status": "ok",
                "tests": [
                    "flatten_sections_from_chapters",
                    "get_effective_sections_hierarchy_only",
                    "consistency_valid",
                    "duplicate_section_id_invalid",
                    "parent_mismatch_invalid",
                    "task_unit_parent_mismatch_invalid",
                    "sync_legacy_sections_from_chapters",
                    "structure_nodes_backward_compatibility",
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
