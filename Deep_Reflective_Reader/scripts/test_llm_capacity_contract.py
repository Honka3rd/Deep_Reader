"""Offline capacity migration: independent input vs shared context and QA adapter."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from config.token_budget_resolver import resolve_effective_token_budgets
from document_structure.plan_prompt_budget import enforce_prompt_budget
from llm.llm_model_capabilities import LLMModelCapabilities
from llm.openai_llm_provider import OpenAILLMProvider, OpenAIModelName


class CapacityContractTests(unittest.TestCase):
    def test_context_input_and_combined_constraints(self):
        for context, independent, expected in ((10000, None, 9000),
                                               (None, 8000, 8000),
                                               (10000, 8000, 8000),
                                               (10000, 9500, 9000)):
            with self.subTest(context=context, independent=independent):
                cap = LLMModelCapabilities("fixture", "responses", context, 2000, independent)
                self.assertEqual(cap.input_token_limit(1000), expected)
                self.assertEqual(cap.combined_token_limit(1000) - 1000, expected)

    def test_invalid_and_unknown_capacities(self):
        for context, output, independent in ((None, 1000, None), (True, 1000, None),
                                              (0, 1000, None), (10000, False, None),
                                              (10000, 1000, -1), (1.5, 1000, None)):
            with self.subTest(context=context, output=output, independent=independent):
                with self.assertRaises(ValueError):
                    LLMModelCapabilities("fixture", "responses", context, output, independent)
        cap = LLMModelCapabilities("fixture", "responses", 10000, 2000)
        for reserve in (-1, True, 2001, "500"):
            with self.assertRaises(ValueError):
                cap.input_token_limit(reserve)
        self.assertEqual(LLMModelCapabilities("fixture", "responses", 1, 2).input_token_limit(2), 0)

    def test_positional_compatibility_and_explicit_keyword_migration(self):
        cap = LLMModelCapabilities("fixture", "responses", 10000, 2000)
        self.assertEqual(cap.context_window_tokens, 10000)
        self.assertIsNone(cap.max_input_tokens)
        # Old misleading keyword-only construction must fail, not silently reinterpret it.
        with self.assertRaises(TypeError):
            LLMModelCapabilities(model_name="fixture", endpoint_kind="responses",
                                 max_input_tokens=10000, max_output_tokens=2000)

    def test_verified_model_registry(self):
        for name in OpenAIModelName:
            cap = OpenAILLMProvider._resolve_model_capabilities(name)
            expected = (1047576, 32768) if name.value.startswith("gpt-4.1") else (128000, 16384)
            self.assertEqual((cap.context_window_tokens, cap.max_output_tokens), expected)
            self.assertIsNone(cap.max_input_tokens)

    def test_provider_reserve_and_request_match(self):
        key = Mock()
        key.get.return_value = "fixture-not-a-key"
        with patch("llm.openai_llm_provider.OpenAIResponses") as client:
            client.return_value.complete.return_value.text = " result "
            provider = OpenAILLMProvider(key)
            self.assertEqual(provider.effective_output_budget, 3276)
            self.assertEqual(provider.complete_text("fixture"), "result")
            client.return_value.complete.assert_called_once_with("fixture", max_output_tokens=3276)
            self.assertEqual(provider.get_model_capabilities().input_token_limit(3276), 1044300)

    def test_qa_combined_budget_adapter_deducts_once(self):
        for context, independent, expected in ((10000, None, 9000), (None, 8000, 8000)):
            cap = LLMModelCapabilities("fixture", "responses", context, 2000, independent)
            budget = resolve_effective_token_budgets(cap, 50000, 1000, 50000, 1, 1)
            # TokenBudgetManager owns the downstream output subtraction in QA.
            self.assertEqual(budget.effective_input_budget - budget.effective_output_budget, expected)
            self.assertEqual(budget.capability_context_window_tokens, context)
            self.assertEqual(budget.capability_max_input_tokens, independent)
        fallback = resolve_effective_token_budgets(None, 3200, 500, 1500)
        self.assertTrue(fallback.fallback_used)
        self.assertEqual(fallback.effective_input_budget, 3200)

    def test_enhancement_independent_input_not_reduced_by_output(self):
        cap = LLMModelCapabilities("fixture", "responses", None, 2000, 8000)
        provider = SimpleNamespace(get_model_capabilities=lambda: cap, effective_output_budget=1000)
        decision = enforce_prompt_budget("x" * 7600, provider, transport_overhead_tokens=0)
        self.assertEqual(decision["available_input"], 7600)
        self.assertEqual(decision["safety_margin"], 400)

    def test_normalizer_reserves_output_from_context_only(self):
        provider = object.__new__(OpenAILLMProvider)
        provider.effective_output_budget = 1000
        for context, independent, expected in ((10000, None, 8900), (None, 8000, 7900)):
            provider.model_capabilities = LLMModelCapabilities("fixture", "responses", context, 2000, independent)
            result = provider.normalize_input_text_for_prompt(
                "x" * 20000, prompt_overhead_tokens=100, token_utilization_ratio=1,
                default_excerpt_chars=100, min_excerpt_chars=1,
                max_excerpt_chars_hard_cap=20000, chars_per_token=1)
            self.assertEqual(len(result), expected)

    def test_configurable_margin_and_qa_target_compatibility(self):
        cap = LLMModelCapabilities("fixture", "responses", 10000, 2000)
        provider = SimpleNamespace(get_model_capabilities=lambda: cap, effective_output_budget=1000)
        decision = enforce_prompt_budget("x", provider, safety_margin_ratio=0.1, minimum_margin_tokens=10)
        self.assertEqual(decision["available_input"], 8100)
        for ratio in (-1, 1, True, float("nan")):
            with self.assertRaises(ValueError):
                enforce_prompt_budget("x", provider, safety_margin_ratio=ratio)
        budget = resolve_effective_token_budgets(cap, 3200, 500, 1500)
        self.assertEqual((budget.effective_input_budget, budget.effective_output_budget,
                          budget.effective_context_budget), (2000, 500, 1500))


if __name__ == "__main__":
    unittest.main()
