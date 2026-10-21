#!/usr/bin/env python3
"""Read-only guards and explicit layout readiness, with isolated JSON persistence."""

from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient
import main
from app.task_layout_readiness import TaskLayoutNotReadyError
from test_task_layout_persistence_cache import (
    SectionTaskCoordinator, StructuredDocumentArtifactRepository, StructuredDocumentStore,
    _build_base_document, _FakeTaskUnitResolver, _FakeProfileStore,
    _FakeSectionTaskService, _FakeEnhancedParseEvaluator,
)


class ReadinessTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "cache-doc.structured.json"
        self.repository = StructuredDocumentArtifactRepository(
            store=StructuredDocumentStore(), base_dir=directory.name,
        )
        self.resolver = _FakeTaskUnitResolver()
        self.pipeline = Mock()
        self.pipeline.prepare_and_load.side_effect = AssertionError("read prepared document")
        self.pipeline.load_manual_structure_source_evidence.side_effect = AssertionError("read loaded raw/OCR/LLM evidence")
        self.pipeline.load_task_layout_page_boundaries.side_effect = AssertionError("ordinary read requested pages")
        self.coordinator = SectionTaskCoordinator(
            document_preparation_pipeline=self.pipeline,
            document_artifact_repository=self.repository,
            document_profile_store=_FakeProfileStore(),
            chapter_summary_service=_FakeSectionTaskService(),
            chapter_quiz_service=_FakeSectionTaskService(),
            task_unit_resolver=self.resolver,
            enhanced_parse_trigger_evaluator=_FakeEnhancedParseEvaluator(),
        )
        self.options = {"doc_name": "cache-doc", "task_unit_split_mode": "progressive"}

    def prepare(self):
        self.repository.save_document(_build_base_document(), doc_name="cache-doc")
        self.coordinator.ensure_document_task_layout(**self.options)

    def read_without_side_effects(self, *, reason=None, **overrides):
        before = self.path.read_bytes() if self.path.exists() else None
        with ExitStack() as stack:
            for method in ("save_document", "update_task_layout"):
                stack.enter_context(patch.object(self.repository, method, side_effect=AssertionError("read wrote")))
            stack.enter_context(patch.object(self.resolver, "resolve_with_options", side_effect=AssertionError("read resolved/embedded")))
            stack.enter_context(patch.object(self.coordinator.task_unit_id_normalizer, "normalize_document_task_unit_ids", side_effect=AssertionError("read repaired")))
            if reason:
                with self.assertRaises(TaskLayoutNotReadyError) as error:
                    self.coordinator.get_document_task_layout(**{**self.options, **overrides})
                self.assertEqual(error.exception.reason, reason)
            else:
                layout = self.coordinator.get_document_task_layout(**{**self.options, **overrides})
                self.assertTrue(layout.chapters[0].sections[0].task_units)
                self.assertNotIn("raw_text", layout.to_dict())
        self.assertEqual(before, self.path.read_bytes() if self.path.exists() else None)
        self.pipeline.prepare_and_load.assert_not_called()
        self.pipeline.load_manual_structure_source_evidence.assert_not_called()

    def test_cold_and_structure_only_reads_fail_without_preparation(self):
        self.read_without_side_effects(reason="missing_structure")
        self.repository.save_document(_build_base_document(), doc_name="cache-doc")
        self.read_without_side_effects(reason="missing_task_units")

    def test_ready_reads_and_refresh_guard(self):
        self.prepare()
        self.read_without_side_effects()
        self.read_without_side_effects(reason="explicit_preparation_required", refresh_task_units=True)
        self.read_without_side_effects(reason="task_unit_split_mode_mismatch", task_unit_split_mode="semantic_safe")

    def test_explicit_unavailable_pages_keep_character_anchors(self):
        self.prepare()
        self.pipeline.load_task_layout_page_boundaries.side_effect = None
        self.pipeline.load_task_layout_page_boundaries.return_value = []
        layout = self.coordinator.get_document_task_layout(**self.options, include_anchor_page_evidence=True)
        self.assertEqual(layout.chapters[0].anchor_evidence.anchor_type, "char_range")
        self.pipeline.load_task_layout_page_boundaries.assert_called_once()
        snapshot = self.pipeline.load_task_layout_page_boundaries.call_args.kwargs["document"]
        self.assertEqual(snapshot.raw_text, _build_base_document().raw_text)
        self.pipeline.load_manual_structure_source_evidence.assert_not_called()

    def test_stale_metadata_never_rebuilds(self):
        for key in ("source_hash", "resolver_version", "semantic_top_k_candidates"):
            with self.subTest(key=key):
                self.prepare()
                document = self.repository.load_document("cache-doc")
                document.document_task_artifacts.metadata["task_layout"][key] = "stale"
                self.repository.save_document(document, doc_name="cache-doc")
                self.read_without_side_effects(reason=key + "_mismatch")

    def test_invalid_json_and_hierarchy_never_repair(self):
        self.path.write_text("{", encoding="utf-8")
        self.read_without_side_effects(reason="invalid_structure")
        self.path.unlink()  # Discard only this corrupt temporary fixture before reseeding.
        self.prepare()
        document = self.repository.load_document("cache-doc")
        document.chapters[0].sections[0] = replace(
            document.chapters[0].sections[0], parent_chapter_id="wrong-parent",
        )
        # Bypass writer validation only to model corrupt persisted data.
        self.path.write_text(document.to_json(), encoding="utf-8")
        self.read_without_side_effects(reason="invalid_hierarchy")

    def test_duplicate_ids_require_explicit_repair(self):
        self.prepare()
        document = self.repository.load_document("cache-doc")
        first = document.chapters[0].sections[0].task_units[0]
        second_section = document.chapters[1].sections[0]
        second_section.task_units[0] = replace(second_section.task_units[0], unit_id=first.unit_id)
        self.path.write_text(document.to_json(), encoding="utf-8")
        self.read_without_side_effects(reason="duplicate_task_unit_ids")
        calls = self.resolver.calls
        self.coordinator.ensure_document_task_layout(**self.options)
        self.assertEqual(self.resolver.calls, calls)
        self.read_without_side_effects()

    def test_explicit_prepare_must_read_back_persistence(self):
        self.repository.save_document(_build_base_document(), doc_name="cache-doc")
        with patch.object(self.repository, "update_task_layout", return_value=_build_base_document()):
            with self.assertRaises(TaskLayoutNotReadyError) as error:
                self.coordinator.ensure_document_task_layout(**self.options)
        self.assertEqual(error.exception.reason, "missing_task_units")

    def test_repository_outage_is_not_disguised_as_readiness(self):
        with patch.object(self.repository, "load_document", side_effect=RuntimeError("storage unavailable")):
            with self.assertRaises(RuntimeError):
                self.coordinator.get_document_task_layout(**self.options)

    def test_http_read_prepare_and_readback_contract(self):
        def base_prepare(**kwargs):
            self.assertEqual(kwargs["mode"], "base")
            self.repository.save_document(_build_base_document(), doc_name="cache-doc")
            return SimpleNamespace(structured_document_ready=True, errors=[],
                                   effective_pdf_extraction_mode=None, pdf_extraction_provenance="unknown")

        pipeline = Mock()
        pipeline.prepare.side_effect = base_prepare
        with patch.object(main, "section_task_coordinator", self.coordinator), \
                patch.object(main.qa_coordinator, "document_preparation_pipeline", pipeline):
            client = TestClient(main.app)
            missing = client.post("/documents/task-layout", json=self.options)
            self.assertEqual(missing.status_code, 409, missing.text)
            self.assertIn("/documents/prepare-task-layout", missing.json()["detail"])
            pipeline.prepare.assert_not_called()
            prepared = client.post("/documents/prepare-task-layout", json={**self.options, "refresh_task_units": True})
            self.assertEqual(prepared.status_code, 200, prepared.text)
            pipeline.prepare.assert_called_once()
            calls = self.resolver.calls
            for _ in range(2):
                read = client.post("/documents/task-layout", json=self.options)
                self.assertEqual(read.status_code, 200, read.text)
                self.assertEqual(read.json()["chapters"], prepared.json()["chapters"])
            self.assertEqual(self.resolver.calls, calls)
            self.assertEqual(pipeline.prepare.call_count, 1)
            refresh = client.post("/documents/task-layout", json={**self.options, "refresh_task_units": True})
            self.assertEqual(refresh.status_code, 409)

    def test_failed_base_preparation_does_not_resolve(self):
        pipeline = Mock()
        pipeline.prepare.return_value = SimpleNamespace(structured_document_ready=False, errors=["failed"])
        with patch.object(main, "section_task_coordinator", self.coordinator), \
                patch.object(main.qa_coordinator, "document_preparation_pipeline", pipeline):
            response = TestClient(main.app).post("/documents/prepare-task-layout", json=self.options)
        self.assertEqual(response.status_code, 400)
        self.assertEqual(self.resolver.calls, 0)
        self.assertFalse(self.path.exists())

    def test_layout_only_preparation_preserves_structure_and_skips_pipeline(self):
        self.repository.save_document(_build_base_document(), doc_name="cache-doc")
        before = self.repository.load_document("cache-doc")
        pipeline = Mock()
        pipeline.prepare.side_effect = AssertionError("layout-only parsed source")
        with patch.object(main, "section_task_coordinator", self.coordinator), \
                patch.object(main.qa_coordinator, "document_preparation_pipeline", pipeline):
            client = TestClient(main.app)
            response = client.post("/documents/prepare-task-layout", json={**self.options, "layout_only": True})
            self.assertEqual(response.status_code, 200, response.text)
            self.assertIsNone(response.json()["pdf_extraction"]["effective_mode"])
            self.assertEqual(response.json()["pdf_extraction"]["provenance"], "unknown")
            conflict = client.post("/documents/prepare-task-layout", json={**self.options, "layout_only": True, "force_rebuild": True})
            self.assertEqual(conflict.status_code, 400)
            evidence = client.post("/documents/prepare-task-layout", json={**self.options, "layout_only": True, "include_anchor_page_evidence": True})
            self.assertEqual(evidence.status_code, 400)
            missing = client.post("/documents/prepare-task-layout", json={**self.options, "doc_name": "absent", "layout_only": True})
            self.assertEqual(missing.status_code, 409)
        pipeline.prepare.assert_not_called()
        after = self.repository.load_document("cache-doc")
        self.assertEqual(after.raw_text, before.raw_text)
        self.assertEqual([c.chapter_id for c in after.chapters], [c.chapter_id for c in before.chapters])
        self.read_without_side_effects()


if __name__ == "__main__":
    unittest.main()
