"""Constant-I/O content reads and lifecycle-only profile generation regressions."""
import unittest
import hashlib
import tempfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock, patch
from dataclasses import replace

from scripts.test_task_unit_content_endpoint import _build_coordinator, _build_cache_valid_document
from scripts.test_profile_before_structured_pipeline import (
    _build_real_pipeline_for_profile_tests, _FakeProfileBuilder, _FakeProfileStore)
from document_preparation.prepared_document_assets import PreparedDocumentAssets
from section_tasks.analysis_interaction_orchestrator import AnalysisInteractionOrchestrator
from section_tasks.quiz_interaction_orchestrator import QuizInteractionOrchestrator
from section_tasks.reading_interaction_artifact_store import DocumentReadingInteractionArtifactStore
from bundle_factory import BundleFactory
from section_tasks.interaction_policy_identity import policy_metadata
from section_tasks.reading_interaction_service_contracts import ReadingInteractionArtifact


class ContentReadRecovery(unittest.TestCase):
    def test_snapshot_preserves_current_and_stale_status_with_structure_version(self):
        document = _build_cache_valid_document()
        document.parse_provenance['storage_structure_version'] = 3
        c, _ = _build_coordinator(document)
        holder = {'document': document}
        repository = Mock()
        repository.load_document.side_effect = lambda name: holder['document']
        def mutate(name, mutation):
            artifacts, result = mutation(holder['document'])
            holder['document'] = replace(holder['document'], document_task_artifacts=artifacts)
            return result
        repository.mutate_document_artifacts.side_effect = mutate
        c.document_artifact_repository = repository
        store = DocumentReadingInteractionArtifactStore(repository)
        service = Mock()
        service.preflight.return_value = None
        c.analysis_interaction_orchestrator = AnalysisInteractionOrchestrator(service=service, artifact_store=store)
        c.quiz_interaction_orchestrator = QuizInteractionOrchestrator(service=service, artifact_store=store)
        c.critical_thinking_session_store = store
        chapter = document.chapters[0]
        section = chapter.sections[0]
        unit = section.task_units[0]
        for kind in ('analysis', 'quiz', 'critical_thinking_session'):
            request = c._build_content_block_availability_request(document=document, doc_name='Doc Content',
                task_unit=unit, section=section, chapter=chapter, content_block=unit.to_content_blocks()[0],
                interaction_type=kind, document_language=document.language)
            self.assertEqual(request.target.source_structure_version, 3)
            artifact = ReadingInteractionArtifact.from_target(target=request.target, interaction_type=kind,
                status='completed', payload={'text': 'saved evidence'},
                metadata={**policy_metadata(request), 'context': {'doc_name': 'Doc Content'}})
            if kind == 'analysis': store.save_analysis_artifact(artifact)
            elif kind == 'quiz': store.save_quiz_artifact(artifact)
            else: store.save_critical_thinking_session(artifact)
        repository.reset_mock()
        result = c.get_task_unit_contents(doc_name='Doc Content', task_unit_ids=[unit.unit_id])
        self.assertTrue(all(slot['status'] == 'completed' for slot in result[0].content_blocks[0].interaction_availability.values()))
        self.assertEqual(repository.load_document.call_count, 1)
        repository.mutate_document_artifacts.assert_not_called()
        holder['document'] = replace(holder['document'], language='zh')
        result = c.get_task_unit_contents(doc_name='Doc Content', task_unit_ids=[unit.unit_id])
        self.assertTrue(all(slot['status'] == 'stale_target' for slot in result[0].content_blocks[0].interaction_availability.values()))
        service.generate.assert_not_called()

    def test_one_snapshot_many_blocks_no_runtime_io_or_generation(self):
        for count in (1, 100):
            d = _build_cache_valid_document()
            section = d.chapters[0].sections[0]
            original = section.task_units[0]
            section.task_units[:] = [replace(original, unit_id=f"unit-{i}") for i in range(count)]
            c, repo = _build_coordinator(d)
            poison = Mock()
            poison.load_document.side_effect = AssertionError("per-block storage read")
            poison.mutate_document_artifacts.side_effect = AssertionError("write on read")
            store = DocumentReadingInteractionArtifactStore(poison)
            service = Mock()
            service.generate.side_effect = AssertionError("LLM on read")
            c.analysis_interaction_orchestrator = AnalysisInteractionOrchestrator(service=service, artifact_store=store)
            c.quiz_interaction_orchestrator = QuizInteractionOrchestrator(service=service, artifact_store=store)
            c.critical_thinking_session_store = store
            with patch.object(c, '_load_existing_document_profile', return_value=None) as profile:
                result = c.get_task_unit_contents(doc_name="Doc Content", task_unit_ids=[u.unit_id for u in section.task_units])
            self.assertEqual(len(result), count)
            self.assertEqual(repo.read_calls, 1)
            self.assertEqual(profile.call_count, 1)
            self.assertEqual(c.document_preparation_pipeline.prepare_calls, 0)
            self.assertEqual(repo.write_calls, 0)
            service.generate.assert_not_called()

    def test_missing_document_does_not_prepare(self):
        c, repo = _build_coordinator(_build_cache_valid_document())
        repo.load_document = Mock(side_effect=FileNotFoundError("missing"))
        with self.assertRaises(FileNotFoundError):
            c.get_task_unit_contents(doc_name="missing", task_unit_ids=["one"])
        self.assertEqual(c.document_preparation_pipeline.prepare_calls, 0)


class ProfileLifecycle(unittest.TestCase):
    def test_concurrent_initial_creation_builds_once(self):
        from profile.document_profile_store import DocumentProfileStore
        with tempfile.TemporaryDirectory() as directory:
            config = Mock()
            config.get_raw_profile_path.return_value = str(Path(directory) / 'profile.json')
            builder = _FakeProfileBuilder()
            pipeline = _build_real_pipeline_for_profile_tests(profile_builder=builder, profile_store=DocumentProfileStore())
            with patch('document_preparation.document_preparation_pipeline.FaissStorageConfig', return_value=config):
                with ThreadPoolExecutor(max_workers=2) as pool:
                    results = list(pool.map(lambda _: pipeline._prepare_profile('fixture', 'source', 'zh',
                        self.assets(profile_creation_allowed=True), False), range(2)))
            self.assertTrue(all(result[0] for result in results))
            self.assertEqual(builder.build_calls, 1)

    def test_postgres_pdf_cache_reuses_verified_sections_without_raw_text(self):
        from document_structure.structured_document import StructuredDocument, StructuredChapter, StructuredSection
        from document_structure.section_splitter_selector import SectionSplitterMode
        from document_preparation.pdf_extraction_identity import PdfExtractionIdentity
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'fixture.pdf'
            source.write_bytes(b'not a PDF: reuse must not parse this')
            text = 'Accepted canonical source.'
            identity = PdfExtractionIdentity.create(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                mode='fast', extraction_version='fixture', settings={})
            section = StructuredSection('s', 0, None, 2, text, 0, len(text))
            document = StructuredDocument('fixture', 'fixture', str(source), 'en', '',
                chapters=[StructuredChapter('c', None, 1, None, [section])],
                parse_provenance={'storage_structure_version': 1, 'pdf_extraction': identity.record(text)})
            pipeline = _build_real_pipeline_for_profile_tests(profile_builder=Mock(), profile_store=Mock())
            pipeline.structured_document_store = Mock()
            pipeline.structured_document_store.exists.return_value = True
            pipeline.structured_document_store.load.return_value = document
            pipeline.loader_factory = Mock(base_dir=directory)
            assets = self.assets(pdf_extraction_identity=identity, requested_pdf_extraction_mode='fast')
            from doc_loaders.source_session import SourceSession
            from doc_loaders.extraction_identity import normalize_pdf_identity
            assets.source_session = SourceSession(identity=normalize_pdf_identity(identity),
                source_path=source, extract=Mock(side_effect=AssertionError("cache extracted")))
            self.assertTrue(pipeline._try_reuse_existing_structured_document(doc_name='fixture', assets=assets,
                force_rebuild=False, parser_mode=SectionSplitterMode.COMMON))
            document.chapters[0].sections[0] = replace(section, content='corrupted')
            self.assertFalse(pipeline._try_reuse_existing_structured_document(doc_name='fixture', assets=assets,
                force_rebuild=False, parser_mode=SectionSplitterMode.COMMON))
            pipeline.loader_factory.get.assert_not_called()
            pipeline.profile_builder.build.assert_not_called()

    def assets(self, **kwargs):
        return PreparedDocumentAssets(doc_name="fixture", raw_text="source", language="zh",
            structured_document_ready=False, faiss_ready=False, profile_ready=False,
            bundle_ready=False, structured_document_path=None, faiss_namespace=None, **kwargs)

    def run_profile(self, store, *, force=False, allowed=False, builder=None, identity=None):
        builder = builder or _FakeProfileBuilder()
        pipeline = _build_real_pipeline_for_profile_tests(profile_builder=builder, profile_store=store)
        assets = self.assets(profile_creation_allowed=allowed, pdf_extraction_identity=identity)
        result = pipeline._prepare_profile("fixture", "source", "zh", assets, force)
        return result, assets, builder

    def test_pdf_identity_does_not_rebuild(self):
        store = _FakeProfileStore(exists_value=True)
        result, assets, builder = self.run_profile(store, identity=object())
        self.assertTrue(result[0])
        self.assertEqual(builder.build_calls, 0)
        self.assertEqual(store.save_calls + store.clear_calls, 0)
        self.assertFalse(assets.profile_generated)

    def test_corrupt_profile_is_not_cleared_or_regenerated(self):
        store = _FakeProfileStore(exists_value=True, load_should_raise=True)
        result, _, builder = self.run_profile(store, allowed=True)
        self.assertFalse(result[0])
        self.assertEqual(builder.build_calls + store.save_calls + store.clear_calls, 0)

    def test_missing_profile_requires_lifecycle_authority(self):
        store = _FakeProfileStore()
        result, _, builder = self.run_profile(store)
        self.assertFalse(result[0])
        self.assertEqual(builder.build_calls, 0)
        result, assets, builder = self.run_profile(store, allowed=True)
        self.assertTrue(result[0])
        self.assertTrue(assets.profile_generated)
        self.assertEqual(builder.build_calls, 1)

    def test_failed_explicit_rebuild_preserves_existing_profile(self):
        store = _FakeProfileStore(exists_value=True)
        result, _, _ = self.run_profile(store, force=True, builder=_FakeProfileBuilder(should_raise=True))
        self.assertFalse(result[0])
        self.assertTrue(store.exists_value)
        self.assertEqual(store.save_calls + store.clear_calls, 0)

    def test_bundle_read_and_index_cleanup_cannot_generate_or_clear_profile(self):
        factory = object.__new__(BundleFactory)
        factory.profile_store = _FakeProfileStore(exists_value=True)
        factory.profile_builder = Mock()
        factory.store = Mock()
        factory.fingerprint_handler = Mock()
        factory.bundle_cache = {}
        config = Mock()
        factory.clear_artifacts(config, "fixture")
        self.assertEqual(factory.profile_store.clear_calls, 0)
        factory.ensure_profile_ready(config, "source", "zh")
        factory.profile_store.load_should_raise = True
        with self.assertRaisesRegex(RuntimeError, "profile_unreadable"):
            factory.ensure_profile_ready(config, "source", "zh")
        factory.profile_store.exists_value = False
        with self.assertRaisesRegex(RuntimeError, "profile_not_ready"):
            factory.ensure_profile_ready(config, "source", "zh")
        factory.profile_builder.build.assert_not_called()


if __name__ == "__main__":
    unittest.main()
