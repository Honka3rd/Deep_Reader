"""Offline full-request budget and zero-call rejection regressions."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from document_structure.plan_prompt_budget import enforce_prompt_budget, EnhancementPromptBudgetError
from document_structure.llm_section_splitter import LLMSectionSplitter
from document_structure.structured_document_builder import StructuredDocumentBuilder
from language.language_code import LanguageCode
from test_plan_prompt_metrics import fixture_prompt


def provider(input_limit=10000, output_limit=1000):
    return SimpleNamespace(get_model_capabilities=lambda: SimpleNamespace(
        context_window_tokens=input_limit, max_output_tokens=output_limit), complete_text=Mock())


class PromptBudgetTests(unittest.TestCase):
    def test_local_tokenizer_and_failing_counter_fallback(self):
        p = provider()
        p.count_input_tokens = lambda text: 8500
        result = enforce_prompt_budget("x" * 20000, p, transport_overhead_tokens=0)
        self.assertEqual(result["input_estimate"], 8500)
        self.assertIn("local_tokenizer", result["measurement"])
        for counter in (Mock(side_effect=ValueError("PRIVATE")), lambda _: None, lambda _: True):
            p.count_input_tokens = counter
            with self.assertRaisesRegex(EnhancementPromptBudgetError, "budget_exceeded"):
                enforce_prompt_budget("x" * 20000, p)

    def test_openai_local_counter_without_api_or_download(self):
        from unittest.mock import patch
        from llm.openai_llm_provider import OpenAILLMProvider
        p = object.__new__(OpenAILLMProvider)
        p.model_capabilities = SimpleNamespace(model_name="fixture-model")
        encoding = Mock()
        encoding.encode.return_value = [1, 2, 3]
        with patch("tiktoken.encoding_for_model", return_value=encoding) as lookup:
            self.assertEqual(p.count_input_tokens("source"), 3)
            lookup.assert_called_once_with("fixture-model")
            encoding.encode.assert_called_once_with("source", disallowed_special=())
        with patch("tiktoken.encoding_for_model", side_effect=KeyError("unsupported")):
            self.assertIsNone(p.count_input_tokens("source"))

    def test_exact_boundary_and_one_byte_over(self):
        p = provider()
        result = enforce_prompt_budget("x" * 8294, p)
        self.assertEqual(result["available_input"], 8550)
        self.assertEqual(result["safety_margin"], 450)
        with self.assertRaisesRegex(EnhancementPromptBudgetError, "budget_exceeded"):
            enforce_prompt_budget("x" * 8295, p)
        p.complete_text.assert_not_called()

    def test_configured_reserve_and_utf8_fallback(self):
        p = provider()
        p.effective_output_budget = 500
        result = enforce_prompt_budget("中" * 2900, p)
        self.assertEqual(result["input_estimate"], 8700)
        self.assertEqual(result["output_reserve"], 500)
        self.assertIn("estimate", result["measurement"])

    def test_missing_or_invalid_limits_fail_closed(self):
        for p in (SimpleNamespace(), provider(None), provider(True), provider(-1), provider(10000, 0)):
            with self.subTest(p=p), self.assertRaisesRegex(EnhancementPromptBudgetError, "budget_unavailable"):
                enforce_prompt_budget("text", p)
        for reserve in (0, -1, 1001, "100", True):
            p = provider()
            p.effective_output_budget = reserve
            with self.subTest(reserve=reserve), self.assertRaisesRegex(EnhancementPromptBudgetError, "output_budget_invalid"):
                enforce_prompt_budget("text", p)

    def test_fixed_overhead_counts_and_tiny_source_cannot_bypass_gate(self):
        p = provider(2000, 500)
        with self.assertRaisesRegex(EnhancementPromptBudgetError, "budget_exceeded"):
            enforce_prompt_budget(fixture_prompt("Short source."), p)
        p.complete_text.assert_not_called()

    def test_both_planner_entrypoints_reject_without_call_or_retry(self):
        for method in ("build_candidate_plan", "split_candidate_with_provenance"):
            p = provider(2000, 500)
            splitter = LLMSectionSplitter(llm_provider=p)
            splitter._compute_preview_char_budget = lambda **_: 120000
            with self.subTest(method=method), self.assertRaisesRegex(EnhancementPromptBudgetError, "budget_exceeded"):
                getattr(splitter, method)(raw_text="A readable source paragraph.", language=LanguageCode.EN)
            p.complete_text.assert_not_called()

    def test_budget_logs_are_safe_and_builder_preserves_reason(self):
        with self.assertLogs("document_structure.plan_prompt_budget", level="WARNING") as logs:
            with self.assertRaises(EnhancementPromptBudgetError) as caught:
                enforce_prompt_budget("PRIVATE_SENTINEL", provider(1, 1))
        self.assertNotIn("PRIVATE_SENTINEL", " ".join(logs.output))
        self.assertEqual(StructuredDocumentBuilder._map_value_error_code(caught.exception),
                         "enhancement_prompt_budget_exceeded")


if __name__ == "__main__":
    unittest.main()
