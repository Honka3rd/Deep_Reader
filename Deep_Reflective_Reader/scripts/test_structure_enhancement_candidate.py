"""Real planner/application with deterministic provider, no raw IO or writes."""
from dataclasses import replace
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from document_preparation.document_preparation_pipeline import DocumentPreparationPipeline
from document_preparation.structure_enhancement_candidate import EnhancementCandidateRejected
from document_structure.structured_document_builder import StructuredDocumentBuilder
from document_structure.section_splitter_selector import SectionSplitterSelector
from document_structure.llm_section_splitter import LLMSectionSplitter
import test_accepted_source_snapshot as fixtures
from test_structure_plan_application import Provider


def candidate_pipeline(provider):
    pipeline = object.__new__(DocumentPreparationPipeline)
    pipeline.structured_document_builder = StructuredDocumentBuilder(
        section_splitter_selector=SectionSplitterSelector(llm_splitter=LLMSectionSplitter(llm_provider=provider)))
    return pipeline


class CandidateTests(unittest.TestCase):
    def setUp(self):
        fixture = fixtures.AcceptedSourceTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.snapshot = fixture.resolve()
        self.provider = Provider()
        self.pipeline = candidate_pipeline(self.provider)

    def build(self, **kwargs):
        return self.pipeline.build_enhancement_candidate(snapshot=self.snapshot, **kwargs)

    def test_success_uses_only_builder_and_preserves_text_identity(self):
        before = self.snapshot
        # Pipeline intentionally has no loaders, stores, language/profile or
        # embedding providers. Any accidental dependency access would fail.
        with patch("subprocess.run", side_effect=AssertionError("engine")), \
             patch.object(DocumentPreparationPipeline, "prepare_and_load", side_effect=AssertionError("prepare")):
            result = self.build(toc_hint="absent")
        self.assertEqual(result.document.raw_text, before.canonical_text)
        self.assertEqual(self.snapshot, before)
        self.assertEqual(result.document.parse_provenance["extraction"],
                         before.extraction_identity.record(before.canonical_text))
        self.assertEqual(self.provider.calls, 1)
        self.assertEqual(result.document.sections, [])
        self.assertEqual(result.document.structure_nodes, [])
        self.assertIsNone(result.document.document_task_artifacts)

    def test_corrupt_snapshot_fails_before_provider(self):
        self.snapshot = replace(self.snapshot, text_length=1)
        with self.assertRaisesRegex(EnhancementCandidateRejected, "invalid_enhancement_snapshot"):
            self.build()
        self.assertEqual(self.provider.calls, 0)

    def test_invalid_options_fail_before_provider(self):
        for options in [dict(toc_hint="invent"), dict(enhancement_intent="invent")]:
            with self.assertRaises(ValueError):
                self.build(**options)
        self.assertEqual(self.provider.calls, 0)

    def test_provider_failure_has_no_common_success_fallback(self):
        self.provider.complete_text = Mock(side_effect=RuntimeError("private provider failure"))
        with self.assertRaises(EnhancementCandidateRejected):
            self.build()

    def test_invalid_plan_is_rejected(self):
        self.pipeline = candidate_pipeline(Provider(lambda payload, context: {"invalid": True}))
        with self.assertRaisesRegex(EnhancementCandidateRejected, "enhancement_candidate_rejected"):
            self.build()

    def test_defensive_validation_rejects_changed_text_and_missing_hierarchy(self):
        good = self.build().document
        for bad in [replace(good, raw_text="changed"), replace(good, chapters=[]),
                    replace(good, parse_provenance={})]:
            self.pipeline.structured_document_builder = Mock(build=Mock(return_value=bad))
            with self.assertRaises(EnhancementCandidateRejected):
                self.build()

    def test_common_context_and_legacy_intents_use_same_source(self):
        def check(payload, context):
            self.assertTrue(context["source"]["full_source"])
            return payload
        provider = Provider(check)
        self.pipeline = candidate_pipeline(provider)
        for intent in ["restore_source", "generate_reading_outline"]:
            result = self.build(enhancement_intent=intent)
            self.assertEqual(result.document.raw_text, self.snapshot.canonical_text)
        self.assertEqual(provider.calls, 2)


if __name__ == "__main__":
    unittest.main()
