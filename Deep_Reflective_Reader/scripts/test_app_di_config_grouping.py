#!/usr/bin/env python3
"""Config grouping regression tests for AppDIConfig."""

from __future__ import annotations

from config.app_DI_config import AppDIConfig
from config.container import ApplicationLookupContainer


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def test_grouped_views_match_flat_defaults() -> None:
    config = AppDIConfig()

    retrieval = config.retrieval_token_budget()
    _assert(
        retrieval.target_max_input_tokens == config.target_max_input_tokens,
        "retrieval.target_max_input_tokens mismatch",
    )
    _assert(
        retrieval.target_max_context_tokens == config.target_max_context_tokens,
        "retrieval.target_max_context_tokens mismatch",
    )

    prompt_text_normalization = config.prompt_text_normalization()
    _assert(
        prompt_text_normalization.default_excerpt_chars
        == config.profile_prompt_default_excerpt_chars,
        "profile default excerpt mismatch",
    )
    profile_policy = config.profile_prompt_policy()
    _assert(
        profile_policy.summary_prompt_overhead_tokens
        == config.profile_summary_prompt_overhead_tokens,
        "profile summary overhead mismatch",
    )

    task_unit = config.task_unit_split_policy()
    _assert(
        task_unit.semantic_top_k_candidates == config.task_unit_semantic_top_k_candidates,
        "task unit semantic top_k mismatch",
    )

    enhanced = config.enhanced_parse_policy()
    _assert(
        enhanced.recommend_score_threshold == config.enhanced_parse_recommend_score_threshold,
        "enhanced parse recommend threshold mismatch",
    )

    llm_section_preview = config.llm_section_preview_policy()
    _assert(
        llm_section_preview.prompt_overhead_tokens
        == config.llm_section_preview_prompt_overhead_tokens,
        "llm section preview overhead mismatch",
    )


def test_to_container_config_dict_contains_required_keys() -> None:
    config = AppDIConfig()
    config_map = config.to_container_config_dict()

    required_keys = [
        "llm_model",
        "target_max_input_tokens",
        "profile_prompt_default_excerpt_chars",
        "profile_summary_prompt_overhead_tokens",
        "prompt_text_normalization",
        "profile_prompt_policy",
        "llm_section_preview_policy",
        "question_scope_llm_fallback_enabled",
        "task_unit_semantic_top_k_candidates",
        "enhanced_parse_recommend_score_threshold",
    ]
    for key in required_keys:
        _assert(key in config_map, f"missing key in container config map: {key}")


def test_container_build_uses_grouped_config_map() -> None:
    app_config = AppDIConfig(
        profile_prompt_default_excerpt_chars=22_000,
        profile_summary_prompt_overhead_tokens=1_234,
    )
    container = ApplicationLookupContainer.build(app_config)
    builder = container.document_profile_builder()
    _assert(
        builder.prompt_text_normalization.default_excerpt_chars == 22_000,
        "container did not propagate profile default excerpt chars",
    )
    _assert(
        builder.profile_prompt_policy.summary_prompt_overhead_tokens == 1_234,
        "container did not propagate profile summary overhead tokens",
    )


if __name__ == "__main__":
    test_grouped_views_match_flat_defaults()
    test_to_container_config_dict_contains_required_keys()
    test_container_build_uses_grouped_config_map()
    print("test_app_di_config_grouping: ok")
