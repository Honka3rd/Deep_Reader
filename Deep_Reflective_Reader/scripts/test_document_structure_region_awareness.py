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


def main() -> None:
    splitter = CommonSectionSplitter()
    test_toc_plus_chapter_one(splitter)
    test_preface_plus_chapter_one(splitter)
    test_chapter_plus_appendix(splitter)
    test_chapter_plus_afterword_zh(splitter)

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
