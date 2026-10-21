"""NATIVE-03 source preservation, advisory intent and cache isolation contracts."""
import unittest
from unittest.mock import Mock

from api_schemas import PrepareDocumentRequest, PrepareTaskLayoutRequest, ReparseDocumentStructureRequest
from document_structure.common_parse_policy import COMMON_POLICY_VERSION, common_analysis_view
from document_structure.section_splitter import CommonSectionSplitter
from document_structure.structured_document_builder import StructuredDocumentBuilder
from doc_loaders.pdf_outline import PdfOutlineEntry, PdfOutlineResult
from doc_loaders.pdf_page_evidence import PdfPageTextBoundary
from language.language_code import LanguageCode
from test_native_fragmentation_baseline import PROSE, FRAGMENTED
from test_prepare_structured_reuse_before_raw_load import (
    _build_pipeline, _StructuredStore, _TextLoader, _fake_document,
)


class CommonParsePolicyTests(unittest.TestCase):
    def build(self, text=PROSE, **kwargs):
        return StructuredDocumentBuilder().build(
            document_id="fixture", title="fixture", raw_text=text,
            language=LanguageCode.ZH, **kwargs)

    def test_api_enum_default_and_validation(self):
        for schema in (PrepareDocumentRequest, PrepareTaskLayoutRequest, ReparseDocumentStructureRequest):
            options = {"doc_name": "fixture"}
            if schema is ReparseDocumentStructureRequest:
                options["parser_mode"] = "common"
            self.assertEqual(schema(**options).toc_hint, "auto")
            for hint in ("auto", "present", "absent"):
                self.assertEqual(schema(**options, toc_hint=hint).toc_hint, hint)
            for invalid in (None, "yes", "", 1):
                with self.assertRaises(ValueError):
                    schema(**options, toc_hint=invalid)

    def test_single_body_is_success_not_fabricated_toc(self):
        for hint in ("auto", "present", "absent"):
            document = self.build(toc_hint=hint)
            self.assertIsNone(document.structure_error_code)
            self.assertEqual(len(document.chapters), 1)
            self.assertEqual(document.chapters[0].sections[0].content, PROSE)
            policy = document.parse_provenance["common_parse_policy"]
            self.assertTrue(policy["implicit_single_body"])
            self.assertEqual(policy["toc_hint"], hint)
            self.assertEqual(policy["declaration_unconfirmed"], hint == "present")

    def test_absent_skips_layout_and_toc_detector(self):
        builder = StructuredDocumentBuilder()
        builder.toc_detector = Mock(side_effect=AssertionError("TOC search"))
        provider = Mock(side_effect=AssertionError("layout"))
        document = builder.build(document_id="fixture", title="fixture",
            raw_text="第一章\n" + PROSE + "\n第二章\n" + PROSE,
            language=LanguageCode.ZH, toc_hint="absent", page_evidence_provider=provider)
        self.assertIsNone(document.structure_error_code)
        self.assertEqual(len(document.chapters), 2)
        provider.assert_not_called()
        self.assertFalse(builder.toc_detector.mock_calls)
        text = "第一章\n" + PROSE
        outline = PdfOutlineResult(True, True, [PdfOutlineEntry("第一章", 1, 0)])
        document = builder.build(document_id="fixture", title="fixture", raw_text=text,
            language=LanguageCode.ZH, toc_hint="absent", outline=outline,
            page_boundaries=[PdfPageTextBoundary(0, 0, len(text), text)],
            page_evidence_provider=provider)
        self.assertIsNone(document.structure_error_code)
        self.assertEqual(document.parse_provenance["common_parse_policy"]["toc_observation"],
                         "validated_native_outline")
        self.assertEqual(document.chapters[0].title, "第一章")
        provider.assert_not_called()
        self.assertFalse(builder.toc_detector.mock_calls)

    def test_fragmented_prose_preserves_exact_source_and_real_preface(self):
        for text in (FRAGMENTED, "序\n\n" + FRAGMENTED,
                     PROSE + "\n第一章\n" + FRAGMENTED + "\n第二章\n" + PROSE):
            sections = CommonSectionSplitter().split(text, LanguageCode.ZH)
            self.assertEqual("".join(item.content for item in sections), text)
            self.assertEqual(sections[0].char_start, 0)
            self.assertEqual(sections[-1].char_end, len(text))
            for item in sections:
                self.assertEqual(item.content, text[item.char_start:item.char_end])
        sections = CommonSectionSplitter().split("序\n\n" + FRAGMENTED, LanguageCode.ZH)
        self.assertTrue(any(item.title == "序" for item in sections))

    def test_normal_short_verse_lists_and_page_breaks_are_not_joined(self):
        for text in ("春眠不覺曉\n處處聞啼鳥\n夜來風雨聲\n花落知多少",
                     "甲\n乙\n丙\n丁\n戊\n己\n庚\n辛",
                     PROSE + "\n\n" + PROSE, PROSE + "\f" + PROSE):
            self.assertEqual(common_analysis_view(text).text, text)

    def pipeline(self):
        store = _StructuredStore(exists_value=False, document=_fake_document("fixture"))
        pipeline = _build_pipeline(loader=_TextLoader(), store=store)
        pipeline.structured_document_builder = StructuredDocumentBuilder()
        return pipeline, store

    def test_prepare_hint_cache_change_and_warm_reuse(self):
        pipeline, store = self.pipeline()
        for hint, saves in (("auto", 1), ("auto", 1), ("absent", 2), ("absent", 2), ("present", 3)):
            result = pipeline.prepare("fixture", mode="base", toc_hint=hint)
            self.assertTrue(result.structured_document_ready, result.errors)
            self.assertEqual(store.save_calls, saves)
            self.assertEqual(store.document.parse_provenance["common_parse_policy"]["toc_hint"], hint)
        self.assertEqual(store.document.parse_provenance["common_parse_policy"]["version"], COMMON_POLICY_VERSION)

    def test_explicit_auto_invalidates_legacy_policy(self):
        pipeline, store = self.pipeline()
        store.exists_value = True
        result = pipeline.prepare("fixture", mode="base", toc_hint="auto")
        self.assertTrue(result.structured_document_ready, result.errors)
        self.assertEqual(store.save_calls, 1)

    def test_invalid_hint_fails_before_source_read(self):
        pipeline, store = self.pipeline()
        with self.assertRaises(ValueError):
            pipeline.prepare("fixture", mode="base", toc_hint="yes")
        self.assertEqual(pipeline.loader_factory.get_calls, 0)
        self.assertEqual(store.save_calls, 0)

    def test_prepare_and_load_retains_hint(self):
        pipeline, store = self.pipeline()
        result = pipeline.prepare_and_load("fixture", mode="base", toc_hint="absent")
        self.assertTrue(result.assets.structured_document_ready, result.assets.errors)
        self.assertEqual(result.structured_document.parse_provenance["common_parse_policy"]["toc_hint"], "absent")


if __name__ == "__main__":
    unittest.main()
