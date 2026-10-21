"""NATIVE-05 production v2 selection and failure preservation without live services."""
import json
import unittest
from dataclasses import replace
from unittest.mock import Mock, patch

from document_structure.common_parse_policy import COMMON_POLICY_VERSION
from document_structure.heading_acceptance import heading_decision, validate_projection
from document_structure.llm_section_splitter import LLMSectionSplitter
from document_structure.section_splitter import CommonSectionSplitter
from document_structure.section_splitter_selector import SectionSplitterSelector
from document_structure.structure_plan_contract import source_preview, validate_candidate, EnhancementIntent, resolve_candidate_positions
from document_structure.structure_plan_application import apply_candidate
from document_structure.structured_document import StructuredDocument, StructuredSection
from document_structure.structured_document_builder import StructuredDocumentBuilder
from language.language_code import LanguageCode
from test_native_fragmentation_baseline import PROSE, FRAGMENTED
from test_prepare_structured_reuse_before_raw_load import _build_pipeline, _StructuredStore, _TextLoader, _fake_document


class Provider:
    def get_model_capabilities(self):
        from llm.llm_model_capabilities import LLMModelCapabilities
        return LLMModelCapabilities("fixture", "responses", 1000000, 4000)

    def __init__(self, transform=None):
        self.transform = transform
        self.calls = 0

    def normalize_input_text_for_prompt(self, **kwargs):
        return kwargs["text"][:120000]

    def complete_text(self, prompt):
        self.calls += 1
        from document_structure.semantic_structure_plan import read_semantic_prompt_context
        if "\nCONTEXT:\n" in prompt:
            self.context = read_semantic_prompt_context(prompt)
        context = self.context
        payload = {"schema_version": "semantic_structure_plan_v1",
            "boundaries": [{"start_quote": context["source"]["text"][:24], "title": None,
                "origin": "implicit", "level": 1, "role": "main_body"}]}
        return json.dumps(self.transform(payload, context) if self.transform else payload)


class StructurePlanApplicationTests(unittest.TestCase):
    def test_long_multilingual_compact_plan_exposes_full_source(self):
        for language, sentence in ((LanguageCode.EN, "A first idea. A second idea.\n"),
                                   (LanguageCode.ZH, "第一個觀點。第二個觀點。\n"),
                                   (LanguageCode.FR, "Une premiere idee. Une autre idee.\n")):
            with self.subTest(language=language):
                text = (sentence * 15000)[:130000]
                def plan(payload, context):
                    source = context["source"]
                    self.assertEqual(source["representation"], "readable_full_source_v1")
                    self.assertEqual(source["omitted_ranges"], [])
                    self.assertTrue(source["full_source"])
                    self.assertEqual(source["text"], text)
                    return payload
                provider = Provider(plan)
                sections, provenance = LLMSectionSplitter(llm_provider=provider).split_with_provenance(
                    raw_text=text, language=language)
                self.assertEqual("".join(section.content for section in sections), text)
                self.assertEqual(provenance["source_coverage"], "complete")
                self.assertEqual(provider.calls, 1)

    def test_missing_structured_prefix_is_preserved_without_model_guessing(self):
        text = "An introduction.\n\nA second topic."
        def missing_prefix(payload, context):
            payload["boundaries"] = [{"start_quote": "A second topic.",
                "title": "Second topic", "origin": "generated", "level": 1, "role": "main_body"}]
            return payload
        sections, _ = LLMSectionSplitter(llm_provider=Provider(missing_prefix)).split_with_provenance(
            raw_text=text, language=LanguageCode.EN)
        self.assertEqual(len(sections), 2)
        self.assertIsNone(sections[0].title)
        self.assertEqual("".join(section.content for section in sections), text)

    def test_quote_based_plan_applies_without_model_ids(self):
        text = "A first topic.\n\nA second topic."
        def references(payload, context):
            payload["boundaries"] = [{"start_quote": text[position:position + 15], "title": title, "origin": "generated",
                "level": 1, "role": "main_body"} for position, title in
                ((0, "First topic"), (text.index("A second"), "Second topic"))]
            return payload
        sections, provenance = LLMSectionSplitter(llm_provider=Provider(references)).split_with_provenance(
            raw_text=text, language=LanguageCode.EN)
        self.assertEqual(len(sections), 2)
        self.assertEqual("".join(section.content for section in sections), text)
        self.assertEqual(provenance["source_coverage"], "complete")

    def test_default_enhancement_can_split_semantically_without_common_ids(self):
        text = "A discussion of rituals.\n\nA discussion of community."
        def generate(payload, context):
            self.assertNotIn("common_structure", context)
            payload["boundaries"][0]["start_quote"] = "A discussion of rituals."
            payload["boundaries"][0].update(title="Rituals", origin="generated")
            payload["boundaries"].append({**payload["boundaries"][0],
                "start_quote": "A discussion of community.", "title": "Community"})
            return payload
        splitter = LLMSectionSplitter(llm_provider=Provider(generate))
        sections, provenance = splitter.split_with_provenance(raw_text=text, language=LanguageCode.EN)
        self.assertEqual(len(sections), 2)
        self.assertEqual("".join(s.content for s in sections), text)
        self.assertTrue(all(b["title_origin"] == "generated" for b in provenance["boundary_origins"]))

    def test_rejection_logs_safe_reason_not_private_response(self):
        provider = Provider(lambda p, c: {"private_text": "PRIVATE_SENTINEL"})
        with self.assertLogs("document_structure.llm_section_splitter", level="WARNING") as captured:
            document = self.build(provider)
        self.assertTrue(document.structure_error_code.startswith("enhancement_candidate_rejected:"))
        self.assertIn("reason=schema:", " ".join(captured.output))
        self.assertNotIn("PRIVATE_SENTINEL", " ".join(captured.output))

    def test_mislabelled_navigation_gets_generated_origin_not_source_authority(self):
        def mislabel(payload, context):
            payload["boundaries"][0].update(title="Semantic navigation", origin="source")
            return payload
        document = self.build(Provider(mislabel))
        self.assertIsNone(document.structure_error_code)
        self.assertEqual(document.parse_provenance["boundary_origins"][0]["title_origin"], "generated")

    def test_mislabelled_navigation_still_cannot_split_inside_a_word(self):
        def mislabel(payload, context):
            payload["boundaries"].append({**payload["boundaries"][0],
                "start_quote": PROSE[8:12], "title": "Semantic navigation", "origin": "source"})
            return payload
        document = self.build(Provider(mislabel), FRAGMENTED)
        self.assertTrue(document.structure_error_code.startswith("enhancement_candidate_rejected:"))

    def builder(self, provider):
        return StructuredDocumentBuilder(section_splitter_selector=SectionSplitterSelector(
            llm_splitter=LLMSectionSplitter(llm_provider=provider)))

    def build(self, provider, text=PROSE, **kwargs):
        return self.builder(provider).build("fixture", "fixture", text, LanguageCode.ZH,
            parser_mode="llm_enhanced", **kwargs)

    def test_single_body_is_applied_without_common_boundary_merge(self):
        text = "第一章\n" + PROSE + "\n第二章\n" + PROSE
        document = self.build(Provider(), text)
        self.assertIsNone(document.structure_error_code)
        self.assertEqual(len(document.chapters), 1)
        self.assertEqual(document.chapters[0].sections[0].content, text)
        self.assertEqual(document.parse_provenance["source_coverage"], "complete")
        self.assertFalse(document.parse_provenance["fallback_used"])

    def test_generated_title_origin_survives_serialization(self):
        def generate(payload, context):
            payload["boundaries"][0].update(title="閱讀線索", origin="generated")
            return payload
        provider = Provider(generate)
        document = self.build(provider, enhancement_intent="generate_reading_outline", toc_hint="present")
        self.assertIsNone(document.structure_error_code)
        restored = StructuredDocument.from_dict(document.to_dict())
        self.assertEqual(restored.chapters[0].metadata["title_origin"], "generated")
        self.assertEqual(restored.parse_provenance["boundary_origins"][0]["title_origin"], "generated")
        self.assertEqual(restored.parse_provenance["toc_hint"], "present")
        self.assertNotIn("toc_detection", restored.parse_provenance)
        self.assertEqual(provider.calls, 1)

    def test_fragmented_cross_line_anchor_resolves_to_original_source(self):
        preview = source_preview(FRAGMENTED, 1000)
        payload = {"schema_version": "structure_plan_v2", "intent": "restore_source",
            "outcome": "single_body", "boundaries": [{"segment_id": preview["segments"][0]["segment_id"],
            "anchor": PROSE[:8], "occurrence": 1, "title": None, "origin": "implicit",
            "role": "main_body", "level": 1}]}
        candidate = validate_candidate(json.dumps(payload), preview=preview, intent=EnhancementIntent.RESTORE_SOURCE)
        self.assertEqual(resolve_candidate_positions(candidate, preview), [0])
        sections, _ = apply_candidate(FRAGMENTED, candidate, preview)
        self.assertEqual(sections[0].content, FRAGMENTED)

    def test_invalid_occurrence_does_not_clamp(self):
        def bad(payload, context):
            payload["boundaries"][0]["occurrence"] = 999
            return payload
        document = self.build(Provider(bad))
        self.assertIsNotNone(document.structure_error_code)

    def test_generated_boundary_cannot_split_a_fragmented_word(self):
        def split_word(payload, context):
            payload["boundaries"].append({**payload["boundaries"][0],
                "start_quote": PROSE[8:12], "title": "Invented split", "origin": "generated"})
            return payload
        document = self.build(Provider(split_word), FRAGMENTED,
                              enhancement_intent="generate_reading_outline")
        self.assertIsNotNone(document.structure_error_code)

    def test_unsupported_source_heading_not_accepted(self):
        def invent(payload, context):
            payload["boundaries"][0].update(title=PROSE[:2], origin="source")
            return payload
        document = self.build(Provider(invent))
        self.assertIsNotNone(document.structure_error_code)

    def test_real_heading_and_implicit_prefix_preserve_all_text(self):
        text = PROSE + "\n\n第一章\n" + PROSE
        def headings(payload, context):
            payload["boundaries"] = [{**payload["boundaries"][0],
                "start_quote": "第一章", "title": "第一章", "origin": "source"}]
            return payload
        document = self.build(Provider(headings), text)
        self.assertIsNone(document.structure_error_code)
        self.assertEqual("".join(s.content for c in document.chapters for s in c.sections), text)
        self.assertTrue(any(c.title == "第一章" for c in document.chapters))

    def test_heading_negative_and_positive_controls(self):
        for text, title in (("我們討論第一章的內容。", "第一章"), ("序\n甲\n乙", "序"),
                            ('"Chapter 1"\nA sentence.', '"Chapter 1"'), ("短語\n正文", "短語")):
            self.assertNotEqual(heading_decision(text, 0, title), "accepted")
        for text, title in (("第一章\n" + PROSE, "第一章"), ("序\n\n" + PROSE, "序"),
                            ("生活背景\n\n" + PROSE, "生活背景")):
            self.assertEqual(heading_decision(text, 0, title), "accepted")

    def test_contradictory_bookmark_does_not_fabricate_title(self):
        sections = [StructuredSection("s", 0, "Invented", 1, PROSE, 0, len(PROSE))]
        self.assertEqual(validate_projection(PROSE, sections), [])

    def test_failed_enhancement_never_saves_or_invalidates_existing_assets(self):
        store = _StructuredStore(exists_value=True, document=_fake_document("fixture"))
        old = store.document
        pipeline = _build_pipeline(loader=_TextLoader(), store=store)
        provider = Provider(lambda payload, context: {"private_content": "SENSITIVE_SENTINEL"})
        pipeline.structured_document_builder = self.builder(provider)
        result = pipeline.prepare("fixture", mode="free_qa", force_rebuild=True,
            structured_parser_mode="llm_enhanced", enhancement_intent="generate_reading_outline")
        self.assertFalse(result.structured_document_ready)
        self.assertIs(store.document, old)
        self.assertEqual(store.save_calls, 0)
        self.assertEqual(pipeline.profile_builder.build_calls, 0)
        self.assertEqual(pipeline.profile_store.save_calls, 0)
        self.assertFalse(result.faiss_ready)
        self.assertNotIn("SENSITIVE_SENTINEL", str(result.errors))

    def test_successful_explicit_enhancement_saves_once_and_common_reuses_only_common_policy(self):
        store = _StructuredStore(exists_value=True, document=_fake_document("fixture"))
        pipeline = _build_pipeline(loader=_TextLoader(), store=store)
        provider = Provider()
        pipeline.structured_document_builder = self.builder(provider)
        result = pipeline.prepare("fixture", mode="base", force_rebuild=True,
            structured_parser_mode="llm_enhanced", toc_hint="absent")
        self.assertTrue(result.structured_document_ready, result.errors)
        self.assertEqual(store.save_calls, 1)
        self.assertEqual(provider.calls, 1)
        self.assertEqual(store.document.parse_provenance["common_parse_policy"]["parser_mode"], "llm_enhanced")
        result = pipeline.prepare("fixture", mode="base", toc_hint="absent")
        self.assertTrue(result.structured_document_ready, result.errors)
        self.assertEqual(store.save_calls, 2)
        self.assertEqual(provider.calls, 1)

    def test_api_prepare_executes_and_unconfirmed_reparse_is_rejected(self):
        import main
        from fastapi.testclient import TestClient
        store = _StructuredStore(exists_value=True, document=_fake_document("fixture"))
        pipeline = _build_pipeline(loader=_TextLoader(), store=store)
        provider = Provider()
        pipeline.structured_document_builder = self.builder(provider)
        client = TestClient(main.app)
        with patch.object(main.qa_coordinator, "document_preparation_pipeline", pipeline), \
                patch.object(main.section_task_coordinator, "document_preparation_pipeline", pipeline):
            for endpoint, mode_field in (("/documents/prepare", "structured_parser_mode"),
                                         ("/documents/reparse-structure", "parser_mode")):
                response = client.post(endpoint, json={"doc_name": "fixture", mode_field: "llm_enhanced",
                    "enhancement_intent": "generate_reading_outline", "toc_hint": "absent"})
                if endpoint == "/documents/prepare":
                    self.assertEqual(response.status_code, 200, response.text)
                    self.assertTrue(response.json()["success"])
                else:
                    self.assertEqual(response.status_code, 409, response.text)
                    self.assertEqual(response.json()["detail"], "confirmation_required")
            self.assertEqual(provider.calls, 1)
            self.assertEqual(store.save_calls, 1)


if __name__ == "__main__":
    unittest.main()
