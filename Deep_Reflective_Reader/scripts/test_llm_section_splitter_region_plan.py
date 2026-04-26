#!/usr/bin/env python3
"""Smoke test: LLMSectionSplitter region-aware structure plan + local apply."""

from __future__ import annotations

import json

from document_structure.llm_section_splitter import LLMSectionSplitter
from document_structure.section_splitter import CommonSectionSplitter
from language.language_code import LanguageCode
from llm.llm_model_capabilities import ENDPOINT_KIND_RESPONSES, LLMModelCapabilities
from llm.llm_provider import LLMProvider


class _FakeLLMProvider(LLMProvider):
    """Deterministic fake provider for splitter smoke tests."""

    def __init__(
        self,
        payload: dict[str, object],
        *,
        max_input_tokens: int = 8_000,
    ):
        self._payload = payload
        self._max_input_tokens = max_input_tokens

    def complete_text(self, prompt: str) -> str:
        _ = prompt
        return json.dumps(self._payload, ensure_ascii=False)

    def get_model_capabilities(self) -> LLMModelCapabilities:
        return LLMModelCapabilities(
            model_name="fake-llm",
            endpoint_kind=ENDPOINT_KIND_RESPONSES,
            max_input_tokens=self._max_input_tokens,
            max_output_tokens=2_000,
        )


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    raw_text = (
        "Table of Contents\n"
        "Preface\n"
        "Chapter 1\n"
        "Chapter 2\n"
        "Appendix A\n"
        "Afterword\n\n"
        "Preface\n"
        "This preface introduces context for readers before the main chapter flow.\n\n"
        "Chapter 1\n"
        "Main body chapter one starts with a complete prose paragraph for testing.\n\n"
        "Chapter 2\n"
        "Main body chapter two continues with enough narrative content to be valid.\n\n"
        "Appendix A\n"
        "Supplementary material appears after main chapters and should be appendix role.\n\n"
        "Afterword\n"
        "Closing reflections should be marked as back matter, not main body.\n"
    )
    plan_payload = {
        "parser_mode": "llm_enhanced",
        "sections": [
            {
                "title": "Table of Contents",
                "level": 1,
                "section_role": "toc",
                "container_title": None,
                "start_anchor_text": "Table of Contents",
                "anchor_match_mode": "exact",
                "anchor_occurrence": 1,
            },
            {
                "title": "Preface",
                "level": 1,
                "section_role": "front_matter",
                "container_title": None,
                "start_anchor_text": "Preface",
                "anchor_match_mode": "exact",
                "anchor_occurrence": 2,
            },
            {
                "title": "Chapter 1",
                "level": 2,
                "section_role": "main_body",
                "container_title": "Part I",
                "start_anchor_text": "Chapter 1",
                "anchor_match_mode": "exact",
                "anchor_occurrence": 2,
            },
            {
                "title": "Chapter 2",
                "level": 2,
                "section_role": "main_body",
                "container_title": "Part I",
                "start_anchor_text": "Chapter 2",
                "anchor_match_mode": "exact",
                "anchor_occurrence": 2,
            },
            {
                "title": "Appendix A",
                "level": 1,
                "section_role": "appendix",
                "container_title": None,
                "start_anchor_text": "Appendix A",
                "anchor_match_mode": "exact",
                "anchor_occurrence": 2,
            },
            {
                "title": "Afterword",
                "level": 1,
                "section_role": "back_matter",
                "container_title": None,
                "start_anchor_text": "Afterword",
                "anchor_match_mode": "exact",
                "anchor_occurrence": 2,
            },
        ],
    }

    preview_probe_text = "A" * 120_000
    small_cap_splitter = LLMSectionSplitter(
        llm_provider=_FakeLLMProvider(plan_payload, max_input_tokens=4_000),
        common_splitter=CommonSectionSplitter(),
    )
    large_cap_splitter = LLMSectionSplitter(
        llm_provider=_FakeLLMProvider(plan_payload, max_input_tokens=128_000),
        common_splitter=CommonSectionSplitter(),
    )
    small_budget = small_cap_splitter._compute_preview_char_budget(
        raw_text_length=len(preview_probe_text)
    )
    large_budget = large_cap_splitter._compute_preview_char_budget(
        raw_text_length=len(preview_probe_text)
    )
    _assert(
        large_budget > small_budget,
        f"preview budget should adapt to capability: small={small_budget}, large={large_budget}",
    )
    _assert(
        large_budget > 16_000,
        f"large capability preview budget should exceed old fixed cap: {large_budget}",
    )

    splitter = LLMSectionSplitter(
        llm_provider=_FakeLLMProvider(plan_payload),
        common_splitter=CommonSectionSplitter(),
    )
    sections = splitter.split(raw_text=raw_text, language=LanguageCode.EN)
    _assert(len(sections) >= 6, "expected at least 6 sections from region-aware plan")

    role_by_title = {
        (section.title or ""): (None if section.section_role is None else section.section_role.value)
        for section in sections
    }
    _assert(role_by_title.get("Table of Contents") == "toc", "toc role mismatch")
    _assert(role_by_title.get("Preface") == "front_matter", "front_matter role mismatch")
    _assert(role_by_title.get("Appendix A") == "appendix", "appendix role mismatch")
    _assert(role_by_title.get("Afterword") == "back_matter", "back_matter role mismatch")
    _assert(role_by_title.get("Chapter 1") == "main_body", "chapter1 role mismatch")
    _assert(role_by_title.get("Chapter 2") == "main_body", "chapter2 role mismatch")

    chapter_levels = {section.title: section.level for section in sections if section.title in {"Chapter 1", "Chapter 2"}}
    _assert(chapter_levels.get("Chapter 1") == 2, "chapter1 level should be preserved from plan")
    _assert(chapter_levels.get("Chapter 2") == 2, "chapter2 level should be preserved from plan")

    print(
        json.dumps(
            {
                "status": "ok",
                "preview_budget_small_cap": small_budget,
                "preview_budget_large_cap": large_budget,
                "section_count": len(sections),
                "sections": [
                    {
                        "section_id": section.section_id,
                        "title": section.title,
                        "level": section.level,
                        "section_role": (
                            None if section.section_role is None else section.section_role.value
                        ),
                        "container_title": section.container_title,
                    }
                    for section in sections
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
