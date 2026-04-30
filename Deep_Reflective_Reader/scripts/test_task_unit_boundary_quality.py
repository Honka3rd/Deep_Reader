#!/usr/bin/env python3
"""Boundary-cluster quality checks for task-unit splitting."""

from __future__ import annotations

import json
from contextlib import redirect_stdout
from io import StringIO

from document_structure.structured_document import StructuredSection
from language.language_code import LanguageCode
from section_tasks.heuristic_task_unit_split_resolver import HeuristicTaskUnitSplitResolver
from section_tasks.task_unit_boundary_language_registry import TaskUnitBoundaryLanguageRegistry
from section_tasks.task_unit_split_mode import TaskUnitSplitMode


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _make_section(content: str, section_id: str = "section-test") -> StructuredSection:
    return StructuredSection(
        section_id=section_id,
        section_index=0,
        title="Boundary Test",
        level=1,
        content=content,
        char_start=0,
        char_end=len(content),
    )


def _split(
    *,
    content: str,
    mode: TaskUnitSplitMode,
    language: LanguageCode,
    min_chars: int = 280,
    max_chars: int = 560,
) -> list[str]:
    resolver = HeuristicTaskUnitSplitResolver(
        split_mode=mode,
        semantic_boundary_scorer=None,
    )
    units = resolver.split_section(
        section=_make_section(content),
        section_index=0,
        task_unit_min_chars=min_chars,
        task_unit_max_chars=max_chars,
        semantic_top_k_candidates=3,
        language_code=language,
    )
    return [unit.content for unit in units]


def _assert_no_orphan_closing_start(chunks: list[str], language: LanguageCode) -> None:
    config = TaskUnitBoundaryLanguageRegistry.get_config(language)
    closings = config.closing_quote_chars | config.closing_bracket_chars
    for chunk in chunks[1:]:
        stripped = chunk.lstrip()
        if not stripped:
            continue
        _assert(
            stripped[0] not in closings,
            f"unexpected orphan closing punctuation at chunk start: {stripped[:20]!r}",
        )


def test_chinese_closing_quote() -> dict[str, object]:
    prefix = "甲" * 520
    core = "他说：\"我的哥哥会把你的哥哥揍扁。\"于是两个孩子开始商量要不要继续。"
    suffix = "乙" * 520
    chunks = _split(
        content=prefix + core + suffix,
        mode=TaskUnitSplitMode.SEMANTIC_SAFE,
        language=LanguageCode.ZH,
        max_chars=532,
        min_chars=260,
    )
    _assert(len(chunks) >= 2, "expected multiple chunks for chinese quote case")
    _assert_no_orphan_closing_start(chunks, LanguageCode.ZH)
    _assert("。\"" in chunks[0], "first chunk should keep chinese terminal cluster together")
    _assert(chunks[1].lstrip().startswith("于是"), "next sentence should stay in next chunk")
    return {"first_tail": chunks[0][-40:], "second_head": chunks[1][:40]}


def test_chinese_ocr_line_wrap_quote() -> dict[str, object]:
    prefix = "甲" * 520
    core = "他说：\"我的哥哥会把你的哥哥揍扁。\n\"于是两个孩子开始商量要不要继续。"
    suffix = "乙" * 520
    chunks = _split(
        content=prefix + core + suffix,
        mode=TaskUnitSplitMode.SEMANTIC_SAFE,
        language=LanguageCode.ZH,
        max_chars=532,
        min_chars=260,
    )
    _assert(len(chunks) >= 2, "expected multiple chunks for OCR line-wrap quote case")
    _assert_no_orphan_closing_start(chunks, LanguageCode.ZH)
    _assert("。\n\"" in chunks[0], "first chunk should absorb cross-line closing quote")
    _assert(chunks[1].lstrip().startswith("于是"), "next sentence should stay in next chunk")
    return {"first_tail": chunks[0][-40:], "second_head": chunks[1][:40]}


def test_japanese_closing_quote() -> dict[str, object]:
    prefix = "あ" * 520
    core = "彼は言った。\u300cもう終わりだ。\u300dそして歩き出した。"
    suffix = "い" * 520
    chunks = _split(
        content=prefix + core + suffix,
        mode=TaskUnitSplitMode.SEMANTIC_SAFE,
        language=LanguageCode.JA,
        max_chars=540,
        min_chars=260,
    )
    _assert(len(chunks) >= 2, "expected multiple chunks for japanese quote case")
    _assert_no_orphan_closing_start(chunks, LanguageCode.JA)
    _assert("。\u300d" in chunks[0], "first chunk should include japanese closing quote cluster")
    return {"first_tail": chunks[0][-40:], "second_head": chunks[1][:40]}


def test_english_quote() -> dict[str, object]:
    prefix = "A" * 520
    core = "He said, \"It is over.\" Then he left quickly without looking back."
    suffix = "B" * 520
    chunks = _split(
        content=prefix + core + suffix,
        mode=TaskUnitSplitMode.SEMANTIC_SAFE,
        language=LanguageCode.EN,
        max_chars=540,
        min_chars=260,
    )
    _assert(len(chunks) >= 2, "expected multiple chunks for english quote case")
    _assert_no_orphan_closing_start(chunks, LanguageCode.EN)
    _assert(".\"" in chunks[0], "first chunk should include english closing quote cluster")
    return {"first_tail": chunks[0][-40:], "second_head": chunks[1][:40]}


def test_bracket_case() -> dict[str, object]:
    prefix = "A" * 520
    core = "This was final.) Then he left quickly to avoid further discussion."
    suffix = "B" * 520
    chunks = _split(
        content=prefix + core + suffix,
        mode=TaskUnitSplitMode.SEMANTIC_SAFE,
        language=LanguageCode.EN,
        max_chars=540,
        min_chars=260,
    )
    _assert(len(chunks) >= 2, "expected multiple chunks for bracket case")
    _assert_no_orphan_closing_start(chunks, LanguageCode.EN)
    _assert(".)" in chunks[0], "first chunk should include terminal+bracket cluster")
    return {"first_tail": chunks[0][-40:], "second_head": chunks[1][:40]}


def test_no_absorb_next_sentence() -> dict[str, object]:
    prefix = "甲" * 520
    core = "他说：\"结束了。\"于是他们离开。"
    suffix = "乙" * 520
    chunks = _split(
        content=prefix + core + suffix,
        mode=TaskUnitSplitMode.SEMANTIC_SAFE,
        language=LanguageCode.ZH,
        max_chars=532,
        min_chars=260,
    )
    _assert(len(chunks) >= 2, "expected multiple chunks for no-over-absorb case")
    _assert(chunks[0].rstrip().endswith("。\""), "should stop at terminal cluster")
    _assert(chunks[1].lstrip().startswith("于是"), "next sentence text should stay in next chunk")
    return {"first_tail": chunks[0][-30:], "second_head": chunks[1][:30]}


def test_full_languagecode_registry_coverage() -> dict[str, object]:
    coverage: dict[str, bool] = {}
    for language in LanguageCode:
        config = TaskUnitBoundaryLanguageRegistry.get_config(language)
        coverage[language.value] = bool(config.sentence_terminal_chars)
        _assert(config.paragraph_separators, f"missing paragraph separators for {language.value}")
    _assert(all(coverage.values()), "all language codes should have boundary config")
    return coverage


def test_chapter7_regression_snippet() -> dict[str, object]:
    prefix = "前文" * 260
    core = "我的两个哥哥会把你的两个哥哥揍扁。\"于是两个孩子开始商量，在地上画起了圈圈。"
    suffix = "后文" * 260
    chunks = _split(
        content=prefix + core + suffix,
        mode=TaskUnitSplitMode.SEMANTIC_SAFE,
        language=LanguageCode.ZH,
        max_chars=540,
        min_chars=260,
    )
    _assert(len(chunks) >= 2, "expected multiple chunks for chapter-7 regression snippet")
    _assert_no_orphan_closing_start(chunks, LanguageCode.ZH)
    _assert(not chunks[1].lstrip().startswith('"'), "next chunk should not start with orphan quote")
    _assert(chunks[0].rstrip().endswith('。"'), "boundary should not split between 。 and closing quote")
    return {"first_tail": chunks[0][-40:], "second_head": chunks[1][:40]}


def test_no_punctuation_source_quality_fallback() -> dict[str, object]:
    text = ("这是一段非常长的文本没有任何标点只是一直往下延伸可能来自OCR损坏" * 120)
    resolver = HeuristicTaskUnitSplitResolver(
        split_mode=TaskUnitSplitMode.SEMANTIC_SAFE,
        semantic_boundary_scorer=None,
    )
    buffer = StringIO()
    with redirect_stdout(buffer):
        units = resolver.split_section(
            section=_make_section(text, section_id="source-quality-no-punct"),
            section_index=0,
            task_unit_min_chars=300,
            task_unit_max_chars=560,
            semantic_top_k_candidates=3,
            language_code=LanguageCode.ZH,
        )
    logs = buffer.getvalue()
    _assert(len(units) >= 2, "no-punctuation text should still be split")
    _assert(
        "no_reliable_sentence_or_paragraph_boundary" in logs,
        "source-quality fallback reason should be logged",
    )
    return {"unit_count": len(units), "log_hint": "no_reliable_sentence_or_paragraph_boundary"}


def test_no_punctuation_with_paragraphs() -> dict[str, object]:
    paragraph = "这是一段没有句号但语义上完整的段落" * 20
    text = f"{paragraph}\n\n{paragraph}\n\n{paragraph}"
    resolver = HeuristicTaskUnitSplitResolver(
        split_mode=TaskUnitSplitMode.SEMANTIC_SAFE,
        semantic_boundary_scorer=None,
    )
    buffer = StringIO()
    with redirect_stdout(buffer):
        units = resolver.split_section(
            section=_make_section(text, section_id="no-punct-with-paragraphs"),
            section_index=0,
            task_unit_min_chars=260,
            task_unit_max_chars=520,
            semantic_top_k_candidates=3,
            language_code=LanguageCode.ZH,
        )
    logs = buffer.getvalue()
    _assert(len(units) >= 2, "paragraph-separated text should split")
    _assert(
        "no_reliable_sentence_or_paragraph_boundary" not in logs,
        "paragraph boundary should avoid source-quality hard fallback",
    )
    return {"unit_count": len(units)}


def test_single_newline_not_paragraph() -> dict[str, object]:
    text = ("没有标点只有单行换行" * 20 + "\n") * 80
    resolver = HeuristicTaskUnitSplitResolver(
        split_mode=TaskUnitSplitMode.SEMANTIC_SAFE,
        semantic_boundary_scorer=None,
    )
    buffer = StringIO()
    with redirect_stdout(buffer):
        units = resolver.split_section(
            section=_make_section(text, section_id="single-newline-no-punct"),
            section_index=0,
            task_unit_min_chars=280,
            task_unit_max_chars=560,
            semantic_top_k_candidates=3,
            language_code=LanguageCode.ZH,
        )
    logs = buffer.getvalue()
    _assert(len(units) >= 2, "single-newline text should still split")
    _assert(
        "no_reliable_sentence_or_paragraph_boundary" in logs,
        "single newline should not be treated as strong paragraph boundary",
    )
    return {"unit_count": len(units), "log_hint": "no_reliable_sentence_or_paragraph_boundary"}


def test_progressive_cluster_normalization() -> dict[str, object]:
    prefix = "A" * 520
    core = "He said, \"It is over.\" Then they negotiated a little longer."
    suffix = "B" * 520
    chunks = _split(
        content=prefix + core + suffix,
        mode=TaskUnitSplitMode.PROGRESSIVE,
        language=LanguageCode.EN,
        max_chars=540,
        min_chars=260,
    )
    _assert(len(chunks) >= 2, "progressive should split into multiple units")
    _assert_no_orphan_closing_start(chunks, LanguageCode.EN)
    _assert(chunks[0].rstrip().endswith('."'), "progressive should also preserve terminal cluster")
    return {"first_tail": chunks[0][-40:], "second_head": chunks[1][:40]}


def main() -> None:
    results = {
        "zh_closing_quote": test_chinese_closing_quote(),
        "zh_ocr_line_wrap_quote": test_chinese_ocr_line_wrap_quote(),
        "ja_closing_quote": test_japanese_closing_quote(),
        "en_quote": test_english_quote(),
        "bracket_case": test_bracket_case(),
        "no_absorb_next_sentence": test_no_absorb_next_sentence(),
        "languagecode_coverage": test_full_languagecode_registry_coverage(),
        "chapter7_regression_snippet": test_chapter7_regression_snippet(),
        "source_quality_fallback": test_no_punctuation_source_quality_fallback(),
        "no_punct_with_paragraphs": test_no_punctuation_with_paragraphs(),
        "single_newline_not_paragraph": test_single_newline_not_paragraph(),
        "progressive_cluster_normalization": test_progressive_cluster_normalization(),
    }
    print(json.dumps({"status": "ok", "results": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
