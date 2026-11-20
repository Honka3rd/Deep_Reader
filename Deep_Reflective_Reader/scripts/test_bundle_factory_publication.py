"""Guarded factory/facade publication with real file/SQL source owners and FAISS."""
import os
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

from llama_index.core.node_parser import SentenceSplitter

import test_preparation_runtime_safety as runtime
from bundle_factory import BundleFactory
from bundle_provider import BundleProvider
from config.faiss_storage_config import FaissStorageConfig
from document_preparation.accepted_source_guard import AcceptedSourceGuard
from document_structure.structured_document_builder import StructuredDocumentBuilder
from language.language_code import LanguageCode
from retrieval.index_publication import FILES, JOURNAL
from retrieval.node_provider import NodeProvider

A = 'Alpha preserves the original accepted source.'
B = 'Beta contains a replacement accepted source.'


class FactoryPublicationTests(unittest.TestCase):
    fixture_type = runtime.RuntimeSafetyTests

    def setUp(self):
        self.fixture = self.fixture_type()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.path.write_text(A)
        self.fixture.seed(index=False)
        self.pipeline = self.fixture.pipeline
        self.repo = self.fixture.repository()
        self.config = FaissStorageConfig('book')
        self.nodes = NodeProvider(SentenceSplitter(chunk_size=512, chunk_overlap=0),
                                  Mock(detect=Mock(return_value='en')))
        self.factory = BundleFactory(
            self.pipeline.faiss_index_builder, self.pipeline.fingerprint_handler,
            self.pipeline.faiss_index_store, self.nodes,
            Mock(build=Mock(side_effect=AssertionError('profile generation forbidden'))),
            self.pipeline.profile_store)
        self.facade = BundleProvider(lambda name: self.config,
            lambda: self.pipeline.fingerprint_handler, lambda **kw: self.factory,
            Mock(get=Mock(side_effect=AssertionError('guarded facade extracted'))))

    def guard(self):
        return AcceptedSourceGuard(document=self.repo.load_document('book'),
            doc_name='book', source_path=lambda: self.fixture.loader.source_path('book'),
            current_document=lambda: self.repo.load_document('book'))

    def accept(self, text):
        self.fixture.path.write_text(text)
        result = self.fixture.loader.open_source('book').extract().result
        document = StructuredDocumentBuilder().build(document_id='book', title='book',
            raw_text=text, language=LanguageCode.EN)
        document = replace(document, source_path=str(self.fixture.path),
            parse_provenance={**document.parse_provenance, 'extraction': result.record()})
        self.repo.save_reparsed_document(document, 'book')
        return self.guard()

    def build(self, text, guard=None):
        return self.facade.get_bundle_from_raw_text('book', text,
            source_guard=guard if guard is not None else self.guard())

    def files(self):
        return {name: (self.config.base_dir/name).read_bytes()
                for name in FILES if (self.config.base_dir/name).exists()}

    def profile(self):
        return Path(self.config.get_raw_profile_path()).read_bytes()

    def preserved(self, before, profile):
        self.assertEqual(set(self.files()), set(FILES))
        self.assertEqual(self.files(), before)
        self.assertEqual(self.profile(), profile)
        loaded = self.factory.load_prepared(self.config, A)
        self.assertEqual([r.text() for r in loaded.id_to_record.values()], [A])
        self.factory.profile_builder.build.assert_not_called()

    def test_explicit_facade_failures_preserve_trio_profile_and_validated_reload(self):
        self.build(A)
        before, profile = self.files(), self.profile()
        guard = self.accept(B)
        for stage, owner, method in (
            ('parse', self.nodes, 'parse'),
            ('embedding', self.factory.builder, 'build_from_parsed_document'),
            ('candidate_save', self.factory.store, 'save'),
            ('metadata_save', self.factory.fingerprint_handler, 'save'),
        ):
            with self.subTest(stage=stage):
                with patch.object(owner, method, side_effect=OSError('controlled_failure')) as fail:
                    with self.assertRaisesRegex(OSError, 'controlled_failure'):
                        self.build(B, guard)
                    self.assertEqual(fail.call_count, 1)
                self.preserved(before, profile)
                self.assertFalse((self.config.base_dir/JOURNAL).exists())

    def test_first_build_failure_is_not_ready(self):
        for owner, method in ((self.nodes, 'parse'),
                              (self.factory.builder, 'build_from_parsed_document'),
                              (self.factory.store, 'save'),
                              (self.factory.fingerprint_handler, 'save')):
            with self.subTest(stage=method):
                before = self.profile()
                with patch.object(owner, method, side_effect=OSError('controlled_failure')):
                    with self.assertRaises(OSError):
                        self.build(A)
                self.assertEqual(self.files(), {})
                self.assertEqual(before, self.profile())
                self.assertFalse(self.config.exists())
                self.assertEqual(len(self.factory.bundle_cache), 0)

    def test_profile_readiness_failure_precedes_generation_or_publication(self):
        self.build(A)
        before, profile = self.files(), self.profile()
        guard = self.accept(B)
        for missing in (False, True):
            with self.subTest(missing=missing):
                path = Path(self.config.get_raw_profile_path())
                path.unlink() if missing else path.write_text('{}')
                with patch.object(self.nodes, 'parse', side_effect=AssertionError('parse')):
                    with self.assertRaisesRegex(RuntimeError, 'profile_(not_ready|unreadable)'):
                        self.build(B, guard)
                self.assertEqual(self.files(), before)
                self.assertFalse(path.exists()) if missing else self.assertEqual(path.read_text(), '{}')
                path.write_bytes(profile)
                self.preserved(before, profile)

    def test_profile_failure_after_embedding_does_not_replace_trio(self):
        self.build(A)
        before, profile = self.files(), self.profile()
        guard = self.accept(B)
        original = self.factory.builder.build_from_parsed_document
        def build_then_fail_profile(parsed):
            bundle = original(parsed)
            self.factory.profile_store.load = Mock(side_effect=ValueError('invalid profile'))
            return bundle
        with patch.object(self.factory.profile_store, 'load', wraps=self.factory.profile_store.load):
            with patch.object(self.factory.builder, 'build_from_parsed_document', side_effect=build_then_fail_profile):
                with self.assertRaisesRegex(RuntimeError, 'profile_unreadable'):
                    self.build(B, guard)
        self.preserved(before, profile)

    def test_normal_replacement_and_guarded_get_bundle_without_extraction(self):
        self.build(A)
        profile = self.profile()
        guard = self.accept(B)
        bundle = self.facade.get_bundle('book', source_guard=guard)
        self.assertEqual([r.text() for r in bundle.id_to_record.values()], [B])
        self.assertEqual(self.profile(), profile)
        self.assertEqual([r.text() for r in self.factory.load_prepared(self.config, B).id_to_record.values()], [B])
        self.facade.loader_factory.get.assert_not_called()
        self.factory.profile_builder.build.assert_not_called()

    def test_unguarded_build_is_explicit_readiness_without_mutation(self):
        with patch.object(self.nodes, 'parse', side_effect=AssertionError('unbound parse')):
            with self.assertRaisesRegex(RuntimeError, 'source_guard_required'):
                self.facade.get_bundle_from_raw_text('book', A)
        self.assertEqual(self.files(), {})
        self.build(A)
        before, profile = self.files(), self.profile()
        with self.assertRaisesRegex(RuntimeError, 'source_guard_required'):
            self.facade.get_bundle_from_raw_text('book', B, force_rebuild=True)
        self.preserved(before, profile)
        with patch.object(self.nodes, 'parse', side_effect=AssertionError('warm parse')):
            self.facade.get_bundle_from_raw_text('book', A)

    def test_guard_namespace_and_exact_text_are_bound_before_work(self):
        guard = self.guard()
        for config, text in ((FaissStorageConfig('other'), A), (self.config, A+'\n')):
            with self.subTest(text_matches=text == A):
                with patch.object(self.nodes, 'parse', side_effect=AssertionError('parse')):
                    with self.assertRaisesRegex(RuntimeError, 'source_guard_mismatch'):
                        self.factory.ensure_index_ready(config, text, source_guard=guard)
        self.assertEqual(self.files(), {})

    def test_source_replacement_or_reparse_during_embedding_rejects_old_writer(self):
        self.build(A)
        before, profile = self.files(), self.profile()
        for change in ('raw', 'same_bytes', 'incarnation', 'delete'):
            with self.subTest(change=change):
                guard = self.accept(B)
                original = self.factory.builder.build_from_parsed_document
                def build_then_change(parsed):
                    bundle = original(parsed)
                    if change == 'raw':
                        self.fixture.path.write_text('Unaccepted source')
                    elif change == 'same_bytes':
                        replacement = self.fixture.path.with_suffix('.tmp')
                        replacement.write_bytes(self.fixture.path.read_bytes())
                        os.replace(replacement, self.fixture.path)
                    elif change == 'incarnation':
                        self.repo.save_reparsed_document(self.repo.load_document('book'), 'book')
                    else:
                        self.fixture.path.unlink()
                    return bundle
                with patch.object(self.factory.builder, 'build_from_parsed_document', side_effect=build_then_change):
                    with self.assertRaises((ValueError, FileNotFoundError)):
                        self.build(B, guard)
                self.preserved(before, profile)

    def test_post_switch_source_change_rolls_back(self):
        import retrieval.index_publication as publication
        self.build(A)
        before, profile = self.files(), self.profile()
        guard = self.accept(B)
        original = publication.os.replace
        def move_then_change(source, destination):
            original(source, destination)
            if Path(source).parent.name == 'candidate' and Path(source).name == 'meta.json':
                self.fixture.path.write_text('Concurrent source')
        with patch.object(publication.os, 'replace', side_effect=move_then_change):
            with self.assertRaises(ValueError):
                self.build(B, guard)
        self.preserved(before, profile)
        self.assertFalse((self.config.base_dir/JOURNAL).exists())

    def test_interrupted_switch_retains_previous_snapshot_and_explicit_recovery(self):
        import retrieval.index_publication as publication
        class Interrupted(BaseException):
            pass
        self.build(A)
        before, profile = self.files(), self.profile()
        guard = self.accept(B)
        original = publication.os.replace
        def interrupt(source, destination):
            original(source, destination)
            if Path(source).parent.name == 'candidate':
                raise Interrupted()
        with patch.object(publication.os, 'replace', side_effect=interrupt):
            with self.assertRaises(Interrupted):
                self.build(B, guard)
        self.assertTrue((self.config.base_dir/JOURNAL).exists())
        snapshot = {str(p): p.read_bytes() for p in self.config.base_dir.rglob('*') if p.is_file()}
        loaded = self.factory.load_prepared(self.config, A)
        self.assertEqual([r.text() for r in loaded.id_to_record.values()], [A])
        self.assertEqual(snapshot, {str(p): p.read_bytes() for p in self.config.base_dir.rglob('*') if p.is_file()})
        self.factory.store.recover(self.config)
        self.preserved(before, profile)
        self.build(B, guard)
        self.assertFalse((self.config.base_dir/JOURNAL).exists())


@unittest.skipUnless(os.environ.get('DEEP_READER_TEST_POSTGRES_DSN'), 'requires disposable PostgreSQL DSN')
class PostgresFactoryPublicationTests(FactoryPublicationTests):
    fixture_type = runtime.PostgresRuntimeSafetyTests

    @classmethod
    def setUpClass(cls):
        cls.fixture_type.setUpClass()
        cls.addClassCleanup(cls.fixture_type.doClassCleanups)
