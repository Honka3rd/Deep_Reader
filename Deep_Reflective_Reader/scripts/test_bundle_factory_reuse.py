"""Same-factory reuse must admit current physical publication and profile."""
import json
import shutil
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import test_bundle_factory_publication as publication_tests
from bundle_factory import BundleFactory
from config.faiss_storage_config import FaissStorageConfig
from fingerprint_handler import FingerprintHandler
from language.language_code import LanguageCode
from profile.document_profile import DocumentProfile
from retrieval.index_publication import JOURNAL

A, B = publication_tests.A, publication_tests.B


class FactoryReuseTests(unittest.TestCase):
    def setUp(self):
        self.case = publication_tests.FactoryPublicationTests()
        self.case.setUp()
        self.addCleanup(self.case.doCleanups)
        self.factory = self.case.factory
        self.config = self.case.config
        self.old = self.case.build(A)

    def snapshot(self, config=None):
        root = (config or self.config).base_dir
        return {str(p.relative_to(root)): p.read_bytes()
                for p in root.rglob('*') if p.is_file()}

    def no_work(self):
        stack = ExitStack()
        for owner, name in ((self.factory.node_provider, 'parse'),
                            (self.factory.builder, 'build_from_parsed_document'),
                            (self.factory.profile_builder, 'build'),
                            (self.factory.profile_store, 'save'),
                            (self.factory.store, 'save'),
                            (self.factory.store, 'publish'),
                            (self.factory.store, 'load')):
            stack.enter_context(patch.object(owner, name, side_effect=AssertionError('read generated or wrote')))
        return stack

    def publish(self, text, config=None, topic='new-profile'):
        guard = self.case.accept(text)
        config = config or self.config
        candidate = self.factory.builder.build_from_parsed_document(
            self.factory.node_provider.parse(text, config))
        self.factory.store.publish(candidate, config, text=text,
            fingerprint=self.factory.fingerprint_handler, verify_source=guard.verify_source)
        self.factory.profile_store.save(DocumentProfile(topic, 'Existing source summary', LanguageCode.EN), config)

    def assert_text(self, bundle, text):
        self.assertEqual([r.text() for r in bundle.id_to_record.values()], [text.strip()])

    def test_other_publisher_source_and_profile_replace_cached_snapshot(self):
        old_topic = self.old.profile.topic
        self.publish(B)
        before = self.snapshot()
        with self.no_work():
            result = self.factory.ensure_index_ready(self.config, B)
            self.assert_text(result, B)
            self.assertEqual(result.profile.topic, 'new-profile')
            self.assertIsNot(result, self.old)
            self.assert_text(self.factory.load_prepared(self.config, B), B)
        self.assertEqual(before, self.snapshot())
        self.assertEqual(self.old.profile.topic, old_topic)
        self.assert_text(self.old, A)

    def test_same_logical_name_different_roots_never_share_returned_source(self):
        other = FaissStorageConfig('book', base_root=Path(self.case.fixture.temp.name)/'other')
        self.publish(B, other, 'other-profile')
        before = (self.snapshot(), self.snapshot(other))
        with self.no_work():
            for config, text in ((other, B), (self.config, A), (other, B)):
                result = self.factory.ensure_index_ready(config, text)
                self.assert_text(result, text)
                if config is other:
                    self.assertEqual(result.profile.topic, 'other-profile')
        self.assertEqual(before, (self.snapshot(), self.snapshot(other)))

    def test_current_profile_reloaded_even_when_index_is_unchanged(self):
        self.factory.profile_store.save(DocumentProfile('current', 'Current background', LanguageCode.EN), self.config)
        before = self.snapshot()
        with self.no_work():
            self.assertEqual(self.factory.get_or_load(self.config, A).profile.topic, 'current')
        self.assertEqual(before, self.snapshot())

    def test_missing_or_invalid_profile_never_uses_cached_profile(self):
        path = Path(self.config.get_raw_profile_path())
        good = path.read_bytes()
        for content in (None, '{}', '{bad'):
            with self.subTest(content=content):
                path.unlink(missing_ok=True) if content is None else path.write_text(content)
                self.factory._put_cache('book', self.old)
                before = self.snapshot()
                with self.no_work():
                    with self.assertRaisesRegex(RuntimeError, 'profile_(not_ready|unreadable)'):
                        self.factory.ensure_index_ready(self.config, A)
                self.assertEqual(before, self.snapshot())
                self.assertNotIn('book', self.factory.bundle_cache)
                path.write_bytes(good)

    def test_exact_text_and_configuration_required_for_cold_and_warm(self):
        original_fp = self.factory.fingerprint_handler
        for warm in (False, True):
            for text, fp in ((A+'\n', original_fp), ('\n'+A, original_fp),
                             (B, original_fp), (A, FingerprintHandler('changed', 512, 0)),
                             (A, FingerprintHandler('deterministic-audit', 256, 0))):
                with self.subTest(warm=warm, text_matches=text == A, model=fp.embedding_model):
                    self.factory.clear()
                    if warm:
                        self.factory._put_cache('book', self.old)
                    before = self.snapshot()
                    with patch.object(self.factory, 'fingerprint_handler', fp), self.no_work():
                        with self.assertRaisesRegex(RuntimeError, 'index_not_ready:explicit_prepare_required'):
                            self.factory.get_or_load(self.config, text)
                    self.assertEqual(before, self.snapshot())
                    self.assertNotIn('book', self.factory.bundle_cache)

    def test_missing_canonical_evidence_rejects_cold_and_warm_without_io(self):
        for warm in (False, True):
            self.factory.clear()
            if warm:
                self.factory._put_cache('book', self.old)
            before = self.snapshot()
            with patch.object(self.factory.store, 'load_validated', side_effect=AssertionError('no evidence')):
                with self.assertRaisesRegex(RuntimeError, 'canonical_source_required'):
                    self.factory.get_or_load(self.config)
            self.assertEqual(before, self.snapshot())

    def test_invalid_ids_missing_files_and_legacy_fingerprint_reject_warm(self):
        before = self.snapshot()
        for damage in ('ids', 'records', 'index', 'meta', 'legacy'):
            with self.subTest(damage=damage):
                for name, contents in before.items():
                    (self.config.base_dir/name).write_bytes(contents)
                self.factory._put_cache('book', self.old)
                if damage == 'ids':
                    path = Path(self.config.get_raw_records_path())
                    payload = json.loads(path.read_text())
                    record = next(iter(payload['records'].values()))
                    record['node_id'] += 100
                    path.write_text(json.dumps(payload))
                elif damage == 'legacy':
                    path = Path(self.config.get_raw_meta_path())
                    payload = json.loads(path.read_text());payload.pop('fingerprint_version')
                    path.write_text(json.dumps(payload))
                else:
                    getattr(self.config, damage+'_path').unlink()
                damaged = self.snapshot()
                with self.no_work():
                    with self.assertRaisesRegex(RuntimeError, 'index_not_ready:explicit_prepare_required'):
                        self.factory.get_or_load(self.config, A)
                self.assertEqual(damaged, self.snapshot())
        for name, contents in before.items():
            (self.config.base_dir/name).write_bytes(contents)

    def test_missing_directory_read_does_not_create_storage(self):
        config = FaissStorageConfig('book', base_root=Path(self.case.fixture.temp.name)/'absent')
        self.assertFalse(config.base_dir.exists())
        with self.no_work():
            with self.assertRaisesRegex(RuntimeError, 'index_not_ready:explicit_prepare_required'):
                self.factory.get_or_load(config, A)
        self.assertFalse(config.base_dir.parent.exists())

    def test_warm_revalidates_and_does_not_trust_mutated_memory(self):
        self.old.id_to_record.clear()
        before = self.snapshot()
        with self.no_work(), patch.object(self.factory.store, 'load_validated', wraps=self.factory.store.load_validated) as load:
            first = self.factory.get_or_load(self.config, A)
            second = self.factory.get_or_load(self.config, A)
            self.assertEqual(load.call_count, 2)
            self.assert_text(first, A);self.assert_text(second, A)
            self.assertIsNot(first, second)
        self.assertEqual(before, self.snapshot())

    def test_capacity_lru_invalidate_and_clear_are_memory_only(self):
        configs = [FaissStorageConfig(name) for name in ('one', 'two', 'three')]
        for config in configs:
            shutil.copytree(self.config.base_dir, config.base_dir)
        before = [self.snapshot(c) for c in configs]
        for capacity in (0, 1, 2):
            with self.subTest(capacity=capacity), self.no_work():
                self.factory.cache_capacity = capacity
                self.factory.clear()
                for config in configs[:2]:
                    self.assert_text(self.factory.get_or_load(config, A), A)
                self.assertEqual(len(self.factory.bundle_cache), min(capacity, 2))
                if capacity == 2:
                    self.factory.get_or_load(configs[0], A)
                    self.assertEqual(list(self.factory.bundle_cache), ['two', 'one'])
                    self.factory.get_or_load(configs[2], A)
                    self.assertEqual(list(self.factory.bundle_cache), ['one', 'three'])
                elif capacity == 1:
                    self.assertEqual(list(self.factory.bundle_cache), ['two'])
                self.factory.invalidate('two')
                self.assertNotIn('two', self.factory.bundle_cache)
                self.factory.clear()
                self.assertEqual(len(self.factory.bundle_cache), 0)
        self.assertEqual(before, [self.snapshot(c) for c in configs])

    def test_previous_journal_snapshot_is_validated_without_recovery_on_read(self):
        import retrieval.index_publication as publication
        class Interrupted(BaseException):
            pass
        guard = self.case.accept(B)
        candidate = self.factory.builder.build_from_parsed_document(self.factory.node_provider.parse(B, self.config))
        original = publication.os.replace
        def interrupt(source, destination):
            original(source, destination)
            if Path(source).parent.name == 'candidate':
                raise Interrupted()
        with patch.object(publication.os, 'replace', side_effect=interrupt):
            with self.assertRaises(Interrupted):
                self.factory.store.publish(candidate, self.config, text=B,
                    fingerprint=self.factory.fingerprint_handler, verify_source=guard.verify_source)
        self.assertTrue((self.config.base_dir/JOURNAL).exists())
        before = self.snapshot()
        with self.no_work():
            self.assert_text(self.factory.ensure_index_ready(self.config, A), A)
            with self.assertRaisesRegex(RuntimeError, 'index_not_ready'):
                self.factory.get_or_load(self.config, B)
        self.assertEqual(before, self.snapshot())
