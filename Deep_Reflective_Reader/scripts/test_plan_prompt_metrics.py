"""Offline serialization baseline; no provider or persistence calls."""
import json
import unittest
from unittest.mock import Mock

from document_structure.structure_plan_contract import source_preview, legacy_plan_prompt as plan_prompt, EnhancementIntent
from document_structure.common_parse_policy import TocHint
from document_structure.plan_prompt_metrics import prompt_metrics, log_prompt_metrics
from document_structure.llm_section_splitter import LLMSectionSplitter
from language.language_code import LanguageCode


def fixture_prompt(text):
    return plan_prompt(preview=source_preview(text, 120000), language="zh",
                       intent=EnhancementIntent.RESTORE_SOURCE, toc_hint=TocHint.AUTO)


class PromptMetricsTests(unittest.TestCase):
    def test_common_context_is_bounded_and_advisory(self):
        sections = [{"title": "x" * 1000, "char_start": 0, "char_end": 10,
                     "level": 1, "is_implicit": False, "content": "PRIVATE_BODY"}] * 200
        prompt = plan_prompt(preview=source_preview("A sentence.", 120000), language="en",
                             intent=EnhancementIntent.RESTORE_SOURCE, toc_hint=TocHint.AUTO,
                             common_structure=sections)
        context = json.loads(prompt.split("\nCONTEXT:\n", 1)[1].split("\nFINAL INSTRUCTIONS:\n")[0])
        self.assertEqual(len(context["common_structure"]), 128)
        self.assertEqual(len(context["common_structure"][0]["title"]), 200)
        self.assertNotIn("PRIVATE_BODY", prompt)
        self.assertEqual(len(sections[0]["title"]), 1000)

    def test_frozen_compaction_caps_at_unchanged_coverage(self):
        text = ("A sentence about evidence. Another sentence about context.\n" * 3000)[:147401]
        metrics = prompt_metrics(fixture_prompt(text))
        self.assertEqual(metrics["visible_source_chars"], 120000)
        self.assertEqual(metrics["reference_count"], 3840)
        self.assertLessEqual(metrics["prompt_chars"], 290000)
        self.assertLessEqual(metrics["component_chars"]["references"], 63064)
        self.assertEqual(metrics["reference_preview_chars"], 0)
        for text in ("A short article. Another sentence.", "中\n文\n測\n試\n" * 100):
            self.assertLessEqual(len(fixture_prompt(text)), 14500)

    def test_wire_projection_preserves_canonical_mapping_without_mutation(self):
        from copy import deepcopy
        from test_native_fragmentation_baseline import FRAGMENTED
        from document_structure.common_parse_policy import common_analysis_view
        from document_structure.structure_plan_contract import compact_prompt_source
        for text in (FRAGMENTED * 100, "First topic. Second topic.", "\"quoted\" \\ text. Next."):
            preview = source_preview(text, 120000)
            original = deepcopy(preview)
            wire = compact_prompt_source(preview)
            self.assertEqual(preview, original)
            self.assertEqual(wire["omitted_ranges"], preview["omitted_ranges"])
            for canonical, displayed in zip(preview["segments"], wire["segments"]):
                view = common_analysis_view(canonical["text"])
                self.assertEqual("".join(p["text"] for p in displayed["passages"]), view.text)
                self.assertNotIn("analysis_text", displayed)
                positions = {}
                offset = 0
                for passage in displayed["passages"]:
                    if passage["boundary_id"] is not None:
                        self.assertNotIn(passage["boundary_id"], positions)
                        positions[passage["boundary_id"]] = offset
                    offset += len(passage["text"])
                self.assertEqual(len(positions), len(canonical["boundary_references"]))
                for ref in canonical["boundary_references"]:
                    display_position = positions[ref["boundary_id"]]
                    self.assertEqual(view.source_offsets[display_position], ref["local_position"])

    def test_components_reconcile_and_coverage_is_explicit(self):
        for text in ["A short article. Another sentence.",
                     ("A sentence about evidence. Another sentence about context.\n" * 3000)[:147401],
                     "中\n文\n測\n試\n" * 100]:
            with self.subTest(size=len(text)):
                prompt = fixture_prompt(text)
                metrics = prompt_metrics(prompt)
                self.assertEqual(sum(metrics["component_chars"].values()), len(prompt))
                self.assertGreaterEqual(metrics["component_chars"]["serialization_metadata"], 0)
                self.assertEqual(metrics["visible_source_chars"], min(len(text), 120000))
                self.assertEqual(metrics["input_token_estimate"], len(prompt.encode("utf-8")))
                self.assertIsNone(metrics["input_token_count"])

    def test_tokenizer_and_failure_fallback(self):
        prompt = fixture_prompt("A short article.")
        self.assertEqual(prompt_metrics(prompt, token_counter=lambda _: 42)["input_token_count"], 42)
        broken = Mock(side_effect=ValueError("private source must not be logged"))
        self.assertIsNone(prompt_metrics(prompt, token_counter=broken)["input_token_count"])

    def test_logs_never_include_source(self):
        prompt = fixture_prompt("PRIVATE_SENTINEL source. Another sentence.")
        with self.assertLogs("document_structure.plan_prompt_metrics", level="INFO") as logs:
            log_prompt_metrics(prompt, None)
        self.assertNotIn("PRIVATE_SENTINEL", " ".join(logs.output))
        self.assertIn("component_chars", " ".join(logs.output))

    def test_runtime_logs_before_call_without_changing_prompt(self):
        provider = Mock()
        from llm.llm_model_capabilities import LLMModelCapabilities
        provider.get_model_capabilities.return_value = LLMModelCapabilities("fixture", "responses", 1000000, 4000)
        provider.effective_output_budget = 4000
        provider.complete_text.return_value = "invalid"
        splitter = LLMSectionSplitter(llm_provider=provider)
        splitter._compute_preview_char_budget = lambda **_: 120000
        with self.assertLogs(level="INFO") as logs:
            with self.assertRaises(ValueError):
                splitter.split_candidate_with_provenance(raw_text="A short article. Another sentence.",
                                                        language=LanguageCode.EN)
        self.assertEqual(provider.complete_text.call_count, 3)
        output = " ".join(logs.output)
        self.assertIn("enhancement_prompt_metrics", output)
        self.assertIn("response_chars=7", output)


if __name__ == "__main__":
    for name, text in {
        "short": "A short article. Another sentence.",
        "long_147401": ("A sentence about evidence. Another sentence about context.\n" * 3000)[:147401],
        "fragmented": "中\n文\n測\n試\n" * 100,
    }.items():
        print(json.dumps({"fixture": name, **prompt_metrics(fixture_prompt(text))}, ensure_ascii=True))
    unittest.main()
