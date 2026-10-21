"""Full-source planning and local resource rejection without external services."""
import json
import unittest
from unittest.mock import Mock, patch

from document_structure.enhancement_resources import EnhancementResourceLimits
from document_structure.llm_section_splitter import LLMSectionSplitter
from document_structure.plan_prompt_budget import EnhancementPromptBudgetError
from document_structure.structure_plan_contract import (
    EnhancementIntent, TocObservation, source_preview, validate_candidate,
    resolve_candidate_positions,
)
from document_structure.common_parse_policy import TocHint
from document_structure.structured_document_builder import StructuredDocumentBuilder
from document_structure.section_splitter_selector import SectionSplitterSelector
from language.language_code import LanguageCode
from llm.llm_model_capabilities import LLMModelCapabilities
from test_structure_plan_application import Provider
from test_prepare_structured_reuse_before_raw_load import _build_pipeline, _StructuredStore, _TextLoader, _fake_document


class FullSourceEnhancementTests(unittest.TestCase):
    def test_147401_characters_include_middle_and_tail_in_both_entrypoints(self):
        raw = ("A readable sentence. Another idea.\n" * 5000)[:147401]
        for method in ("build_candidate_plan", "split_candidate_with_provenance"):
            captured = []
            def capture(payload, context):
                captured.append(context["source"])
                return payload
            provider = Provider(capture)
            provider.normalize_input_text_for_prompt = Mock(side_effect=AssertionError("no excerpts"))
            splitter = LLMSectionSplitter(llm_provider=provider)
            result = getattr(splitter, method)(raw_text=raw, language=LanguageCode.EN)
            source = captured[0]
            self.assertEqual(source["text"], raw)
            self.assertTrue(source["full_source"])
            self.assertEqual(source["omitted_ranges"], [])
            self.assertEqual(provider.calls, 1)
            provider.normalize_input_text_for_prompt.assert_not_called()
            if method == "split_candidate_with_provenance":
                self.assertEqual("".join(s.content for s in result[0]), raw)

    def test_resource_rejection_precedes_mapping_and_common_parser(self):
        for limits, raw in ((EnhancementResourceLimits(max_source_chars=10), "x" * 11),
                            (EnhancementResourceLimits(max_source_bytes=10), "中文文字"),
                            (EnhancementResourceLimits(max_line_breaks=2), "a\nb\nc\nd")):
            for method in ("build_candidate_plan", "split_candidate_with_provenance"):
                provider, common = Provider(), Mock()
                splitter = LLMSectionSplitter(llm_provider=provider, common_splitter=common,
                                              resource_limits=limits)
                with patch("document_structure.llm_section_splitter.source_preview") as mapping:
                    with self.assertRaisesRegex(EnhancementPromptBudgetError, "local_resource_limit"):
                        getattr(splitter, method)(raw_text=raw, language=LanguageCode.EN)
                    mapping.assert_not_called()
                common.split.assert_not_called()
                self.assertEqual(provider.calls, 0)

    def test_prompt_resource_limit_and_model_limit_are_distinct(self):
        for resource_limit, context, reason in ((100, 1000000, "local_resource_limit"),
                                                (100000, 2000, "enhancement_prompt_budget_exceeded")):
            provider = Provider()
            provider.get_model_capabilities = lambda: LLMModelCapabilities("fixture", "responses", context, 1000)
            splitter = LLMSectionSplitter(llm_provider=provider,
                resource_limits=EnhancementResourceLimits(max_prompt_bytes=resource_limit))
            with self.assertRaisesRegex(EnhancementPromptBudgetError, reason) as caught:
                splitter.split_candidate_with_provenance(raw_text="Some text.", language=LanguageCode.EN)
            self.assertEqual(StructuredDocumentBuilder._map_value_error_code(caught.exception), reason)
            self.assertEqual(provider.calls, 0)

    def test_missing_capacity_never_calls_provider(self):
        provider = Provider()
        provider.get_model_capabilities = Mock(side_effect=ValueError("private"))
        with self.assertRaisesRegex(EnhancementPromptBudgetError, "enhancement_budget_unavailable"):
            LLMSectionSplitter(llm_provider=provider).build_candidate_plan(raw_text="Readable text.", language=LanguageCode.EN)
        self.assertEqual(provider.calls, 0)

    def test_read_only_assembly_can_be_reused_without_generation(self):
        provider = Provider()
        preview, prompt, budget = LLMSectionSplitter(llm_provider=provider).assemble_candidate_request(
            "First. Second.", LanguageCode.EN, EnhancementIntent.RESTORE_SOURCE,
            TocHint.AUTO, TocObservation.NOT_CHECKED)
        self.assertTrue(preview["full_source"])
        self.assertEqual(budget["required_input_tokens"], len(prompt.encode("utf-8")) + 256)
        self.assertEqual(provider.calls, 0)

    def test_partial_or_discontinuous_source_never_dispatches(self):
        provider = Provider()
        partial = source_preview("A readable sentence. Another idea.", 10)
        with patch("document_structure.llm_section_splitter.source_preview", return_value=partial):
            with self.assertRaisesRegex(EnhancementPromptBudgetError, "incomplete_enhancement_source"):
                LLMSectionSplitter(llm_provider=provider).build_candidate_plan(
                    raw_text="A readable sentence. Another idea.", language=LanguageCode.EN)
        self.assertEqual(provider.calls, 0)

    def test_anchor_fallback_survives_reference_cap(self):
        raw = " ".join(f"Idea {i:03d}." for i in range(150))
        preview = source_preview(raw, None)
        self.assertEqual(len(preview["segments"][0]["boundary_references"]), 64)
        position = raw.index("Idea 100.")
        payload = {"schema_version": "structure_plan_v2", "intent": "restore_source",
            "outcome": "structured", "boundaries": [{
                "segment_id": preview["segments"][0]["segment_id"], "anchor": "Idea 100.",
                "title": "Later idea", "origin": "generated", "level": 1, "role": "main_body"}]}
        candidate = validate_candidate(json.dumps(payload), preview=preview, intent=EnhancementIntent.RESTORE_SOURCE)
        self.assertEqual(resolve_candidate_positions(candidate, preview), [position])

    def test_resource_logs_do_not_expose_source(self):
        limits = EnhancementResourceLimits(max_source_chars=1)
        with self.assertLogs("document_structure.enhancement_resources", level="WARNING") as logs:
            with self.assertRaises(EnhancementPromptBudgetError):
                limits.check_source("PRIVATE_SENTINEL")
        self.assertNotIn("PRIVATE_SENTINEL", " ".join(logs.output))
        self.assertIn("local_resource_limit", " ".join(logs.output))

    def test_capacity_and_resource_failures_preserve_existing_storage(self):
        for resource_failure in (False, True):
            store = _StructuredStore(exists_value=True, document=_fake_document("fixture"))
            old = store.document
            pipeline = _build_pipeline(loader=_TextLoader(), store=store)
            provider = Provider()
            if not resource_failure:
                provider.get_model_capabilities = lambda: LLMModelCapabilities("fixture", "responses", 2000, 1000)
            splitter = LLMSectionSplitter(llm_provider=provider, resource_limits=EnhancementResourceLimits(
                max_source_chars=1 if resource_failure else 2000000))
            pipeline.structured_document_builder = StructuredDocumentBuilder(
                section_splitter_selector=SectionSplitterSelector(llm_splitter=splitter))
            result = pipeline.prepare("fixture", mode="free_qa", force_rebuild=True,
                                      structured_parser_mode="llm_enhanced")
            self.assertFalse(result.structured_document_ready)
            self.assertIs(store.document, old)
            self.assertEqual(store.save_calls, 0)
            self.assertEqual(pipeline.profile_store.save_calls, 0)
            self.assertEqual(provider.calls, 0)


if __name__ == "__main__":
    unittest.main()
