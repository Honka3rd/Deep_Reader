#!/usr/bin/env python3
"""Minimal region-awareness checks for CommonSectionSplitter."""

from __future__ import annotations

import json

from document_structure.section_splitter import CommonSectionSplitter
from language.language_code import LanguageCode


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _serialize_sections(sections: list[object]) -> list[dict[str, object]]:
    return [
        {
            "section_id": section.section_id,
            "title": section.title,
            "section_role": section.section_role,
            "char_start": section.char_start,
            "char_end": section.char_end,
        }
        for section in sections
    ]


def test_toc_plus_chapter_one(splitter: CommonSectionSplitter) -> None:
    raw_text = (
        "Table of Contents\n"
        "Chapter One\n"
        "Chapter Two\n"
        "Chapter Three\n\n"
        "Chapter One\n"
        "This is the first chapter body sentence with enough words to pass prose detection.\n"
        "Another chapter body sentence with clear narrative continuity.\n\n"
        "Chapter Two\n"
        "This is the second chapter body sentence with enough words for structure parsing.\n"
    )
    sections = splitter.split(raw_text=raw_text, language=LanguageCode.EN)
    _assert(bool(sections), "toc+chapter case should produce sections")
    _assert(
        (sections[0].title or "").lower().startswith("chapter"),
        "toc should not become first main section title",
    )


def test_preface_plus_chapter_one(splitter: CommonSectionSplitter) -> None:
    raw_text = (
        "Preface\n"
        "This preface introduces the reading context and has enough prose to be detected properly.\n"
        "It should not be confused with chapter main body segmentation.\n\n"
        "Chapter 1\n"
        "This chapter starts the real body with narrative prose and should be the first body section.\n"
    )
    sections = splitter.split(raw_text=raw_text, language=LanguageCode.EN)
    _assert(bool(sections), "preface+chapter case should produce sections")
    _assert(
        (sections[0].title or "").lower().startswith("chapter"),
        "front matter should not be treated as first body chapter",
    )


def test_chapter_plus_appendix(splitter: CommonSectionSplitter) -> None:
    raw_text = (
        "Chapter 1\n"
        "Main chapter prose starts here and contains enough words for normal section detection.\n\n"
        "Chapter 2\n"
        "Main chapter prose continues here and still belongs to body content.\n\n"
        "Appendix A\n"
        "Supplementary appendix content appears at the end with reference-style notes.\n"
    )
    sections = splitter.split(raw_text=raw_text, language=LanguageCode.EN)
    appendix_sections = [
        section
        for section in sections
        if section.title is not None and "appendix" in section.title.lower()
    ]
    _assert(appendix_sections, "appendix heading should be detected as section")
    _assert(
        appendix_sections[0].section_role == "appendix",
        "appendix section should be tagged with section_role=appendix",
    )


def test_chapter_plus_afterword_zh(splitter: CommonSectionSplitter) -> None:
    raw_text = (
        "目录\n"
        "第一章\n"
        "第二章\n\n"
        "第一章\n"
        "这一章的正文内容足够长并且以完整句号结束，确保被识别为正文段落。\n\n"
        "第二章\n"
        "第二章同样包含完整叙述句子并且具有足够长度用于正文检测。\n\n"
        "後記\n"
        "這一段是書後記內容，應當被標記為 back matter，而不是普通正文章節。\n"
    )
    sections = splitter.split(raw_text=raw_text, language=LanguageCode.ZH)
    _assert(bool(sections), "zh toc+back matter case should produce sections")
    _assert(
        all((section.title or "") != "目录" for section in sections),
        "toc marker should not pollute body sections in zh case",
    )
    back_sections = [section for section in sections if section.title == "後記"]
    _assert(back_sections, "afterword heading should be parsed as one section")
    _assert(
        back_sections[0].section_role == "back_matter",
        "afterword section should be tagged with section_role=back_matter",
    )


def test_marker_true_hits(splitter: CommonSectionSplitter) -> None:
    """Validate expected markers can be recognized as heading markers."""
    cases = [
        ("Contents", "contents", True),
        ("Table of Contents", "table of contents", True),
        ("目录", "目录", True),
        ("前言", "前言", True),
        ("後記", "後記", True),
        ("Appendix A", "appendix", True),
        ("Preface:", "preface", True),
        ("後記：", "後記", True),
        ("目录（修订版）", "目录", True),
    ]
    for line, marker, expected in cases:
        matched = splitter._line_matches_marker(
            normalized_line=splitter._normalize_heading_title(line),
            raw_line=line,
            normalized_marker=splitter._normalize_heading_title(marker),
        )
        _assert(
            matched == expected,
            f"marker true-hit mismatch: line={line!r}, marker={marker!r}, got={matched}",
        )


def test_marker_false_hits_in_body(splitter: CommonSectionSplitter) -> None:
    """Validate prose lines mentioning marker words are not treated as marker headings."""
    cases = [
        (
            "In this chapter we discuss appendix methods and follow-up analysis in detail.",
            "appendix",
        ),
        (
            "这一句正文提到后记和附录的概念，但它不是标题。",
            "后记",
        ),
        (
            "這一句正文只是提到序這個字，不應該觸發前言標記。",
            "序",
        ),
    ]
    for line, marker in cases:
        matched = splitter._line_matches_marker(
            normalized_line=splitter._normalize_heading_title(line),
            raw_line=line,
            normalized_marker=splitter._normalize_heading_title(marker),
        )
        _assert(
            matched is False,
            f"marker false-hit mismatch: line={line!r}, marker={marker!r}, got={matched}",
        )


def main() -> None:
    splitter = CommonSectionSplitter()
    test_toc_plus_chapter_one(splitter)
    test_preface_plus_chapter_one(splitter)
    test_chapter_plus_appendix(splitter)
    test_chapter_plus_afterword_zh(splitter)
    test_marker_true_hits(splitter)
    test_marker_false_hits_in_body(splitter)

    sample = splitter.split(
        raw_text=(
            "Chapter 1\n"
            "Body content line with enough length to be recognized as main body narrative.\n\n"
            "Afterword\n"
            "Final reflections with end matter semantics.\n"
        ),
        language=LanguageCode.EN,
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "sample_sections": _serialize_sections(sample),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
