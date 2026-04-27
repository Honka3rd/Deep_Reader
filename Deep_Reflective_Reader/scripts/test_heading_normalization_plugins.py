#!/usr/bin/env python3
"""Smoke tests for heading normalization plugin architecture."""

from __future__ import annotations

import json

from document_structure.heading_normalization.chinese_chapter_ocr_normalization_plugin import (
    ChineseChapterOcrNormalizationPlugin,
)
from document_structure.heading_normalization.heading_normalization_executor import (
    HeadingNormalizationExecutor,
)
from document_structure.heading_normalization.heading_normalization_plugin_factory import (
    HeadingNormalizationPluginFactory,
)
from document_structure.section_splitter import CommonSectionSplitter
from language.language_code import LanguageCode


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def test_factory_language_mapping() -> None:
    factory = HeadingNormalizationPluginFactory()
    zh_plugin_names = [plugin.name for plugin in factory.get_plugins(LanguageCode.ZH)]
    en_plugin_names = [plugin.name for plugin in factory.get_plugins(LanguageCode.EN)]
    _assert(
        "chinese_chapter_ocr_normalization" in zh_plugin_names,
        "zh plugin list must include chinese chapter OCR plugin",
    )
    _assert(
        "chinese_chapter_ocr_normalization" not in en_plugin_names,
        "non-zh plugin list must not include chinese chapter OCR plugin",
    )


def test_executor_plugin_exception_isolated() -> None:
    class BrokenPlugin:
        name = "broken"

        @staticmethod
        def normalize(heading: str, language: LanguageCode) -> str:
            _ = (heading, language)
            raise RuntimeError("boom")

    class SuffixPlugin:
        name = "suffix"

        @staticmethod
        def normalize(heading: str, language: LanguageCode) -> str:
            _ = language
            return f"{heading}#ok"

    executor = HeadingNormalizationExecutor()
    normalized = executor.normalize(
        heading="第十章",
        language=LanguageCode.ZH,
        plugins=(BrokenPlugin(), SuffixPlugin()),
    )
    _assert(
        normalized == "第十章#ok",
        "executor should skip broken plugin and continue remaining plugins",
    )


def test_chinese_ocr_fixture_core_case() -> None:
    raw_text = (
        "第十章\n"
        "正文 A。\n\n"
        "第十—章\n"
        "正文 B。\n\n"
        "第十二章\n"
        "正文 C。\n"
    )
    splitter = CommonSectionSplitter()
    sections = splitter.split(raw_text=raw_text, language=LanguageCode.ZH)
    titles = [section.title for section in sections if section.title]
    _assert(
        "第十一章" in titles,
        f"ocr normalized chapter heading not detected, titles={titles}",
    )
    _assert(
        titles[:3] == ["第十章", "第十一章", "第十二章"],
        f"unexpected chapter sequence: {titles[:3]}",
    )


def test_chinese_ocr_variants() -> None:
    variants = ("—", "-", "－", "–", "─")
    splitter = CommonSectionSplitter()
    for variant in variants:
        raw_text = (
            "第十章\n"
            "正文 A。\n\n"
            f"第十{variant}章\n"
            "正文 B。\n\n"
            "第十二章\n"
            "正文 C。\n"
        )
        sections = splitter.split(raw_text=raw_text, language=LanguageCode.ZH)
        titles = [section.title for section in sections if section.title]
        _assert(
            "第十一章" in titles,
            f"variant should normalize to 第十一章: variant={variant!r} titles={titles}",
        )


def test_normal_heading_not_broken() -> None:
    plugin = ChineseChapterOcrNormalizationPlugin()
    cases = ("第一章", "第十章", "第二十九章")
    for heading in cases:
        _assert(
            plugin.normalize(heading, LanguageCode.ZH) == heading,
            f"normal heading should stay unchanged: heading={heading}",
        )


def test_non_zh_not_polluted() -> None:
    plugin = ChineseChapterOcrNormalizationPlugin()
    heading = "Chapter - 2"
    _assert(
        plugin.normalize(heading, LanguageCode.EN) == heading,
        "non-zh heading must not be changed by zh OCR plugin",
    )


def test_opening_front_matter_fixture() -> None:
    raw_text = (
        "各版本前自序\n"
        "一、中文版自序\n"
        "前置內容說明，這段文字仍屬於前言部分。\n\n"
        "第一章\n"
        "正文內容從這裡開始，應當是 main_body。\n"
    )
    splitter = CommonSectionSplitter()
    sections = splitter.split(raw_text=raw_text, language=LanguageCode.ZH)
    _assert(bool(sections), "opening front matter fixture should produce sections")
    _assert(
        sections[0].title == "各版本前自序",
        f"first section title should be opening front matter heading, got={sections[0].title!r}",
    )
    _assert(
        sections[0].section_role is not None
        and sections[0].section_role.value == "front_matter",
        f"first section role should be front_matter, got={sections[0].section_role!r}",
    )
    first_main_body = next(
        (section for section in sections if section.title == "第一章"),
        None,
    )
    _assert(first_main_body is not None, "chapter one should be parsed")
    _assert(
        first_main_body.section_role is not None
        and first_main_body.section_role.value == "main_body",
        f"chapter one role should be main_body, got={first_main_body.section_role!r}",
    )


def main() -> None:
    test_factory_language_mapping()
    test_executor_plugin_exception_isolated()
    test_chinese_ocr_fixture_core_case()
    test_chinese_ocr_variants()
    test_normal_heading_not_broken()
    test_non_zh_not_polluted()
    test_opening_front_matter_fixture()

    print(
        json.dumps(
            {
                "status": "ok",
                "tests": [
                    "factory_language_mapping",
                    "executor_plugin_exception_isolated",
                    "chinese_ocr_fixture_core_case",
                    "chinese_ocr_variants",
                    "normal_heading_not_broken",
                    "non_zh_not_polluted",
                    "opening_front_matter_fixture",
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
