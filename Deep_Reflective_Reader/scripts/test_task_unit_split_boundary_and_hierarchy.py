#!/usr/bin/env python3
"""Boundary-quality + hierarchy regression checks for structured document pipeline."""

from __future__ import annotations

import json
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from document_structure.section_role import SectionRole
from document_structure.document_hierarchy_index import get_effective_sections
from document_structure.structured_document import StructuredDocument, StructuredSection
from document_structure.structured_hierarchy_builder import (
    build_document_hierarchy_from_sections,
)
from section_tasks.heuristic_task_unit_split_resolver import (
    HeuristicTaskUnitSplitResolver,
)
from section_tasks.task_unit_split_mode import TaskUnitSplitMode


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _looks_sentence_like_end(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    endings = (
        "。",
        "！",
        "？",
        "；",
        "……",
        "。\"",
        "。'",
        "。”",
        "。’",
        "。”",
        "。’",
        "。）",
        "。”",
        "！”",
        "？”",
        ".",
        "!",
        "?",
        ";",
        ".\"",
        "!\"",
        "?\"",
        ".”",
        "!”",
        "?”",
    )
    return stripped.endswith(endings)


def test_ocr_line_wrap_is_not_paragraph_boundary() -> None:
    resolver = HeuristicTaskUnitSplitResolver()
    text = "第一行\n第二行\n第三行"
    idx_single = text.index("\n") + 1
    idx_double = len("第一段\n\n")
    _assert(
        resolver._is_paragraph_boundary(text=text, index=idx_single) is False,
        "single newline must not be treated as paragraph boundary",
    )
    _assert(
        resolver._is_paragraph_boundary(text="第一段\n\n第二段", index=idx_double) is True,
        "blank-line newline must be treated as paragraph boundary",
    )


def test_hard_fallback_warning_emitted() -> None:
    resolver = HeuristicTaskUnitSplitResolver(
        split_mode=TaskUnitSplitMode.SEMANTIC_SAFE,
        semantic_boundary_scorer=None,
    )
    section = StructuredSection(
        section_id="section-hard-cut",
        section_index=0,
        title="No Punctuation",
        level=1,
        content=("alpha " * 500).strip(),
        char_start=0,
        char_end=len(("alpha " * 500).strip()),
        section_role=SectionRole.MAIN_BODY,
    )
    stdout = StringIO()
    with redirect_stdout(stdout):
        units = resolver.split_section(
            section=section,
            section_index=0,
            task_unit_min_chars=200,
            task_unit_max_chars=400,
            semantic_top_k_candidates=3,
        )
    _assert(len(units) > 1, "hard fallback fixture should still split into multiple units")
    logs = stdout.getvalue()
    _assert(
        "HeuristicTaskUnitSplitResolver#hard_cut_warning" in logs,
        "hard fallback warning log should be emitted when no structural boundary exists",
    )
    _assert(
        "section_id=section-hard-cut" in logs,
        "hard fallback warning should include section_id for observability",
    )


def test_semantic_safe_ignores_line_wrap_boundary_when_sentence_end_exists() -> None:
    resolver = HeuristicTaskUnitSplitResolver(
        split_mode=TaskUnitSplitMode.SEMANTIC_SAFE,
        semantic_boundary_scorer=None,
    )
    text = (
        "前文描述" * 80
        + "。"
        + "许玉兰也看到了许\n三观，她先是瞟了他一眼，然后继续说了很多很多话。"
        + ("后文延伸" * 120)
    )
    units = resolver.split_section(
        section=StructuredSection(
            section_id="section-line-wrap",
            section_index=0,
            title="第三章",
            level=2,
            content=text,
            char_start=0,
            char_end=len(text),
            section_role=SectionRole.MAIN_BODY,
        ),
        section_index=0,
        task_unit_min_chars=300,
        task_unit_max_chars=550,
        semantic_top_k_candidates=3,
    )
    _assert(len(units) >= 2, "line-wrap fixture should split into multiple units")
    first_tail = units[0].content[-20:].replace("\n", "")
    _assert(
        "看到了许" not in first_tail,
        f"semantic_safe should not cut on OCR line-wrap between words, got tail={first_tail!r}",
    )
    _assert(
        _looks_sentence_like_end(units[0].content),
        "semantic_safe first unit should end near sentence/paragraph boundary",
    )


def test_section_7_no_mid_sentence_cut() -> dict[str, object]:
    structured_path = Path("data/structured/许三观卖血记.structured.json")
    _assert(structured_path.exists(), f"structured file not found: {structured_path}")

    payload = json.loads(structured_path.read_text(encoding="utf-8"))
    document = StructuredDocument.from_dict(payload)
    target = next(
        (section for section in get_effective_sections(document) if (section.title or "").strip() == "第七章"),
        None,
    )
    _assert(target is not None, "expected chapter section '第七章' not found")

    resolver = HeuristicTaskUnitSplitResolver(
        split_mode=TaskUnitSplitMode.SEMANTIC_SAFE,
        semantic_boundary_scorer=None,
    )
    units = resolver.split_section(
        section=target,
        section_index=target.section_index,
        task_unit_min_chars=500,
        task_unit_max_chars=1300,
        semantic_top_k_candidates=3,
    )
    _assert(len(units) >= 2, "第七章 should be split into multiple task units")

    bad_unit_ends: list[str] = []
    for unit in units[:-1]:
        if not _looks_sentence_like_end(unit.content):
            preview = unit.content[-32:].replace("\n", " ")
            bad_unit_ends.append(preview)

    _assert(
        not bad_unit_ends,
        f"semantic_safe split should avoid obvious mid-sentence cuts, bad tails={bad_unit_ends}",
    )

    return {
        "section_id": target.section_id,
        "title": target.title,
        "task_unit_count": len(units),
        "unit_lengths": [len(unit.content) for unit in units],
        "unit_tail_preview": [
            unit.content[-40:].replace("\n", " ")
            for unit in units
        ],
    }


def test_hierarchy_flat_chapters_are_leaf_nodes() -> None:
    sections = [
        StructuredSection(
            section_id="section-0",
            section_index=0,
            title="第一章",
            level=2,
            content="第一章正文。",
            char_start=0,
            char_end=6,
            section_role=SectionRole.MAIN_BODY,
        ),
        StructuredSection(
            section_id="section-1",
            section_index=1,
            title="第二章",
            level=2,
            content="第二章正文。",
            char_start=7,
            char_end=13,
            section_role=SectionRole.MAIN_BODY,
        ),
    ]
    doc = StructuredDocument(
        document_id="doc-hierarchy-flat",
        title="Flat Chapter Doc",
        source_path=None,
        language="zh",
        raw_text="\n".join(section.content for section in sections),
        sections=sections,
    )
    with_hierarchy = build_document_hierarchy_from_sections(doc)
    _assert(len(with_hierarchy.chapters) == 2, "should derive two hierarchy chapters")
    _assert(
        all(chapter.chapter_role == SectionRole.MAIN_BODY.value for chapter in with_hierarchy.chapters),
        "flat chapter headings should become main-body chapters",
    )
    _assert(
        all(len(chapter.sections) == 1 for chapter in with_hierarchy.chapters),
        "flat chapter hierarchy should not force fake subsection children",
    )


def test_hierarchy_chapter_with_subsections() -> None:
    sections = [
        StructuredSection(
            section_id="section-0",
            section_index=0,
            title="Chapter 1",
            level=2,
            content="chapter root",
            char_start=0,
            char_end=20,
            section_role=SectionRole.MAIN_BODY,
        ),
        StructuredSection(
            section_id="section-1",
            section_index=1,
            title="1.1 Background",
            level=3,
            content="background text",
            char_start=21,
            char_end=50,
            section_role=SectionRole.MAIN_BODY,
        ),
        StructuredSection(
            section_id="section-2",
            section_index=2,
            title="1.2 Method",
            level=3,
            content="method text",
            char_start=51,
            char_end=80,
            section_role=SectionRole.MAIN_BODY,
        ),
    ]
    doc = StructuredDocument(
        document_id="doc-hierarchy-sub",
        title="Chapter With Subsections",
        source_path=None,
        language="en",
        raw_text="\n".join(section.content for section in sections),
        sections=sections,
    )
    with_hierarchy = build_document_hierarchy_from_sections(doc)
    root = with_hierarchy.chapters[0]
    _assert(root.chapter_role == SectionRole.MAIN_BODY.value, "root chapter should be main body")
    _assert(len(root.sections) == 3, "chapter should include chapter_body plus subsections")
    _assert(
        root.sections[1].section_kind == "subsection" and root.sections[2].section_kind == "subsection",
        "subsections should be stored as subsection sections under chapter",
    )


def test_hierarchy_front_matter() -> None:
    sections = [
        StructuredSection(
            section_id="section-0",
            section_index=0,
            title="前言",
            level=1,
            content="前言內容",
            char_start=0,
            char_end=10,
            section_role=SectionRole.FRONT_MATTER,
        ),
        StructuredSection(
            section_id="section-1",
            section_index=1,
            title="第一章",
            level=2,
            content="正文內容",
            char_start=11,
            char_end=20,
            section_role=SectionRole.MAIN_BODY,
        ),
    ]
    doc = StructuredDocument(
        document_id="doc-front",
        title="Front Matter Doc",
        source_path=None,
        language="zh",
        raw_text="\n".join(section.content for section in sections),
        sections=sections,
    )
    with_hierarchy = build_document_hierarchy_from_sections(doc)
    _assert(
        with_hierarchy.chapters[0].chapter_role == SectionRole.FRONT_MATTER.value,
        "front matter chapter should keep front_matter role",
    )
    _assert(
        with_hierarchy.chapters[1].chapter_role == SectionRole.MAIN_BODY.value,
        "main chapter should keep main_body role",
    )


def test_backward_compatibility_load_without_structure_nodes() -> None:
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
                "content": "第一章\n正文",
                "char_start": 0,
                "char_end": 6,
                "container_title": None,
                "section_role": "main_body",
            }
        ],
    }
    try:
        StructuredDocument.from_dict(payload)
    except ValueError:
        pass
    else:
        raise AssertionError(
            "normal from_dict should fail-fast for legacy sections-only payloads"
        )

    doc = StructuredDocument.from_legacy_dict_for_migration(payload)
    _assert(doc.structure_nodes == [], "legacy migration loader should allow missing structure_nodes")
    _assert(len(doc.sections) == 1, "legacy migration loader should preserve legacy sections")


def main() -> None:
    test_ocr_line_wrap_is_not_paragraph_boundary()
    test_hard_fallback_warning_emitted()
    test_semantic_safe_ignores_line_wrap_boundary_when_sentence_end_exists()
    section_7_snapshot = test_section_7_no_mid_sentence_cut()
    test_hierarchy_flat_chapters_are_leaf_nodes()
    test_hierarchy_chapter_with_subsections()
    test_hierarchy_front_matter()
    test_backward_compatibility_load_without_structure_nodes()

    print(
        json.dumps(
            {
                "status": "ok",
                "section_7_boundary_regression": section_7_snapshot,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
