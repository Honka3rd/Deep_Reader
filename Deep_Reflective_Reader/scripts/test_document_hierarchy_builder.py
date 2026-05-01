#!/usr/bin/env python3
"""Smoke tests for document->chapter->section->task-unit hierarchy builder."""

from __future__ import annotations

import json

from document_structure.section_role import SectionRole
from document_structure.structured_document import (
    StructuredChapter,
    StructuredDocument,
    StructuredSection,
)
from document_structure.structured_hierarchy_builder import DocumentHierarchyBuilder
from shared.task_artifacts import TaskArtifacts
from shared.task_unit_model import TaskUnit


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _task_unit(unit_id: str, content: str, section_id: str) -> TaskUnit:
    return TaskUnit(
        unit_id=unit_id,
        title=None,
        container_title=None,
        content=content,
        source_section_ids=[section_id],
        is_fallback_generated=False,
    )


def _section(
    *,
    section_id: str,
    section_index: int,
    title: str | None,
    level: int,
    role: SectionRole | None,
    content: str,
    task_units: list[TaskUnit],
) -> StructuredSection:
    return StructuredSection(
        section_id=section_id,
        section_index=section_index,
        title=title,
        level=level,
        content=content,
        char_start=0,
        char_end=len(content),
        section_role=role,
        task_units=task_units,
    )


def test_chapter_only_chinese_novel() -> None:
    sections = [
        _section(
            section_id="section-0",
            section_index=0,
            title="前言",
            level=1,
            role=SectionRole.FRONT_MATTER,
            content="前置內容",
            task_units=[_task_unit("task-unit-0", "前置內容", "section-0")],
        ),
        _section(
            section_id="section-1",
            section_index=1,
            title="第一章",
            level=2,
            role=SectionRole.MAIN_BODY,
            content="第一章正文",
            task_units=[_task_unit("task-unit-1", "第一章正文", "section-1")],
        ),
        _section(
            section_id="section-2",
            section_index=2,
            title="第二章",
            level=2,
            role=SectionRole.MAIN_BODY,
            content="第二章正文",
            task_units=[_task_unit("task-unit-2", "第二章正文", "section-2")],
        ),
        _section(
            section_id="section-3",
            section_index=3,
            title="第三章",
            level=2,
            role=SectionRole.MAIN_BODY,
            content="第三章正文",
            task_units=[_task_unit("task-unit-3", "第三章正文", "section-3")],
        ),
    ]
    doc = StructuredDocument(
        document_id="doc-zh-novel",
        title="Novel",
        source_path=None,
        language="zh",
        raw_text="\n".join(section.content for section in sections),
        sections=sections,
    )

    built = DocumentHierarchyBuilder().build(doc)
    _assert(len(built.chapters) == 4, "chapter-only novel should create one front-matter chapter plus 3 body chapters")
    _assert(built.chapters[0].chapter_role == "front_matter", "front matter should be materialized into hierarchy")
    _assert(
        [section.section_id for section in built.chapters[0].sections] == ["section-0"],
        "front matter chapter should retain original section id",
    )

    for chapter, expected_section_id in zip(
        built.chapters[1:],
        ["section-1", "section-2", "section-3"],
        strict=True,
    ):
        _assert(len(chapter.sections) == 1, "chapter-only should map to one implicit section")
        nested = chapter.sections[0]
        _assert(nested.section_id == expected_section_id, "chapter nested section id should match flat section id")
        _assert(nested.section_kind == "chapter_body", "chapter-only nested section should be chapter_body")
        _assert(nested.is_implicit_section is True, "chapter-only nested section should be implicit")
        _assert(len(nested.task_units) == 1, "nested task units should be preserved")

    _assert(
        built.sections == [],
        "hierarchy build should no longer persist legacy flat sections mirror",
    )


def test_chapter_with_subsections() -> None:
    sections = [
        _section(
            section_id="section-0",
            section_index=0,
            title="Chapter 1",
            level=2,
            role=SectionRole.MAIN_BODY,
            content="chapter intro",
            task_units=[_task_unit("task-unit-0", "chapter intro", "section-0")],
        ),
        _section(
            section_id="section-1",
            section_index=1,
            title="1.1 Background",
            level=3,
            role=SectionRole.MAIN_BODY,
            content="background",
            task_units=[_task_unit("task-unit-1", "background", "section-1")],
        ),
        _section(
            section_id="section-2",
            section_index=2,
            title="1.2 Method",
            level=3,
            role=SectionRole.MAIN_BODY,
            content="method",
            task_units=[_task_unit("task-unit-2", "method", "section-2")],
        ),
        _section(
            section_id="section-3",
            section_index=3,
            title="Chapter 2",
            level=2,
            role=SectionRole.MAIN_BODY,
            content="chapter 2 body",
            task_units=[_task_unit("task-unit-3", "chapter 2 body", "section-3")],
        ),
    ]
    doc = StructuredDocument(
        document_id="doc-en-subsections",
        title="With Subsections",
        source_path=None,
        language="en",
        raw_text="\n".join(section.content for section in sections),
        sections=sections,
    )
    built = DocumentHierarchyBuilder().build(doc)

    _assert(len(built.chapters) == 2, "should derive two chapters")
    chapter_1 = built.chapters[0]
    _assert(len(chapter_1.sections) == 3, "chapter 1 should include body + 2 subsections")
    _assert(chapter_1.sections[0].section_kind == "chapter_body", "first section should be chapter_body")
    _assert(chapter_1.sections[0].is_implicit_section is False, "chapter body should be explicit when subsections exist")
    _assert(chapter_1.sections[1].section_kind == "subsection", "1.1 should be subsection")
    _assert(chapter_1.sections[2].section_kind == "subsection", "1.2 should be subsection")
    _assert(len(built.chapters[1].sections) == 1, "chapter 2 should have one body section")


def test_old_json_compatibility() -> None:
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
    }
    doc = StructuredDocument.from_dict(payload)
    _assert(doc.chapters == [], "legacy payload without chapters should load chapters as empty list")


def test_nested_roundtrip() -> None:
    nested_section = _section(
        section_id="section-10",
        section_index=10,
        title="第一章",
        level=2,
        role=SectionRole.MAIN_BODY,
        content="正文",
        task_units=[
            TaskUnit(
                unit_id="task-unit-10",
                title="u",
                container_title=None,
                content="正文",
                source_section_ids=["section-10"],
                is_fallback_generated=False,
                parent_section_id="section-10",
                task_artifacts=TaskArtifacts(),
            )
        ],
    )
    chapter = StructuredChapter(
        chapter_id="chapter-0",
        title="第一章",
        level=2,
        chapter_role="main_body",
        sections=[nested_section],
        task_artifacts=TaskArtifacts(),
        metadata={"legacy_chapter_key": "chapter::第一章"},
    )
    doc = StructuredDocument(
        document_id="roundtrip-doc",
        title="Roundtrip",
        source_path=None,
        language="zh",
        raw_text="正文",
        chapters=[chapter],
    )

    restored = StructuredDocument.from_json(doc.to_json())
    _assert(len(restored.chapters) == 1, "chapter should survive roundtrip")
    _assert(len(restored.chapters[0].sections) == 1, "nested section should survive roundtrip")
    _assert(len(restored.chapters[0].sections[0].task_units) == 1, "nested task unit should survive roundtrip")
    _assert(
        restored.chapters[0].sections[0].task_units[0].parent_section_id == "section-10",
        "task-unit parent field should survive roundtrip",
    )


def test_no_id_drift_and_no_fake_sections() -> None:
    sections = [
        _section(
            section_id="section-0",
            section_index=0,
            title="Chapter 1",
            level=2,
            role=SectionRole.MAIN_BODY,
            content="body",
            task_units=[_task_unit("task-unit-0", "body", "section-0")],
        ),
        _section(
            section_id="section-1",
            section_index=1,
            title="1.1 Background",
            level=3,
            role=SectionRole.MAIN_BODY,
            content="sub",
            task_units=[_task_unit("task-unit-1", "sub", "section-1")],
        ),
    ]
    doc = StructuredDocument(
        document_id="id-doc",
        title="IDs",
        source_path=None,
        language="en",
        raw_text="body\nsub",
        sections=sections,
    )
    built = DocumentHierarchyBuilder().build(doc)

    source_by_id = {section.section_id: section for section in sections}
    seen_task_unit_ids: set[str] = set()

    for chapter in built.chapters:
        for nested_section in chapter.sections:
            _assert(
                nested_section.section_id in source_by_id,
                "hierarchy section must map to an original source section",
            )
            source_section = source_by_id[nested_section.section_id]
            _assert(
                nested_section.section_id == source_section.section_id,
                "section id must not drift between source and nested",
            )
            nested_unit_ids = [unit.unit_id for unit in nested_section.task_units]
            source_unit_ids = [unit.unit_id for unit in source_section.task_units]
            _assert(
                nested_unit_ids == source_unit_ids,
                "task-unit ids must not drift between source and nested",
            )
            for unit in nested_section.task_units:
                _assert(unit.unit_id not in seen_task_unit_ids, "task-unit id must stay globally unique")
                seen_task_unit_ids.add(unit.unit_id)

    _assert(built.sections == [], "built document should not keep legacy flat sections mirror")


def main() -> None:
    test_chapter_only_chinese_novel()
    test_chapter_with_subsections()
    test_old_json_compatibility()
    test_nested_roundtrip()
    test_no_id_drift_and_no_fake_sections()
    print(
        json.dumps(
            {
                "status": "ok",
                "tests": [
                    "chapter_only_chinese_novel",
                    "chapter_with_subsections",
                    "old_json_compatibility",
                    "nested_roundtrip",
                    "no_id_drift_and_no_fake_sections",
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
