"""PREP-05: exact canonical fingerprints with real splitting, FAISS and stores."""
import hashlib
import json
import os
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import Mock, patch

from llama_index.core.node_parser import SentenceSplitter
from config.faiss_storage_config import FaissStorageConfig
from retrieval.node_provider import NodeProvider
from retrieval.index_publication import FILES, JOURNAL
import test_preparation_runtime_safety as safety


class IndexIdentityTests(unittest.TestCase):
    fixture_class = safety.RuntimeSafetyTests

    def setUp(self):
        self.f = self.fixture_class()
        self.f.setUp()
        self.addCleanup(self.f.doCleanups)
        self.detector = Mock()
        self.detector.detect.return_value = 'en'
        self.f.pipeline.node_provider = NodeProvider(
            SentenceSplitter(chunk_size=128, chunk_overlap=0,
                             tokenizer=lambda text: list(text)), self.detector)
        self.config = FaissStorageConfig('book')

    def assert_positions(self, bundle, text):
        self.assertIsNotNone(bundle)
        self.assertTrue(bundle.id_to_record)
        for record in bundle.id_to_record.values():
            self.assertGreaterEqual(record.char_start(), 0)
            self.assertLessEqual(record.char_end(), len(text))
            # Real SentenceSplitter offsets delimit its stripped chunk exactly.
            self.assertEqual(text[record.char_start():record.char_end()], record.text())

    def snapshot(self):
        return {str(p): p.read_bytes() for p in Path(self.f.temp.name).rglob('*') if p.is_file()}

    def runtime(self, text, *, ready):
        f = self.f
        before = self.snapshot()
        document = f.store.load(f.config())
        with ExitStack() as stack:
            guarded = []
            for obj, method in ((f.loader, 'extract'), (f.pipeline.profile_builder, 'build'),
                                (f.pipeline.language_detector, 'detect'),
                                (self.detector, 'detect'), (f.pipeline.node_provider, 'parse'),
                                (f.pipeline.faiss_index_builder, 'build_from_parsed_document'),
                                (f.pipeline.faiss_index_store.embedder, 'get_text_embedding_batch')):
                guarded.append(stack.enter_context(
                    patch.object(obj, method, side_effect=AssertionError('warm generation'))))
            result = f.pipeline.prepare_and_load('book')
            for dependency in guarded:
                dependency.assert_not_called()
        self.assertEqual(result.assets.bundle_ready, ready, result.assets.errors)
        if ready:
            self.assert_positions(result.bundle, text)
        else:
            self.assertIsNone(result.bundle)
            self.assertIn('explicit_prepare_required', str(result.assets.errors))
        self.assertEqual(before, self.snapshot())
        self.assertEqual(document, f.store.load(f.config()))

    def prepare(self):
        f = self.f
        f.pipeline.faiss_index_builder.build_from_parsed_document.reset_mock()
        result = f.pipeline.prepare('book', pdf_extraction_mode='fast')
        self.assertTrue(result.bundle_ready, result.errors)
        return f.pipeline.faiss_index_builder.build_from_parsed_document.call_count

    def test_whitespace_changes_rebuild_once_identical_text_reuses(self):
        f = self.f
        original = f.seed()
        core = original.strip()
        for text in ('\n\n' + original, '\n' + original, original,
                     '  ' + core + '\t\n', core + '  \n\n', core + '\n', core,
                     core, '\n\t' + core + '\n\t', '\n\t' + core + '\n\t'):
            with self.subTest(text=repr(text)):
                current = f.pipeline.load_manual_structure_source_evidence('book').raw_text
                before = f.files()
                f.path.write_text(text)
                self.assertEqual(self.prepare(), int(current != text))
                if current == text:
                    self.assertEqual(before, f.files())
                else:
                    self.assertNotEqual(before['meta.json'], f.files()['meta.json'])
                canonical = f.pipeline.load_manual_structure_source_evidence('book').raw_text
                self.assertEqual(canonical, text)
                meta = json.loads(f.files()['meta.json'])
                self.assertEqual(meta['fingerprint_version'], 2)
                self.assertEqual(meta['content_length'], len(text))
                self.assertEqual(meta['content_hash'], hashlib.sha256(text.encode('utf-8')).hexdigest())
                self.runtime(text, ready=True)

    def legacy_meta(self, text):
        current = self.f.pipeline.fingerprint_handler.build_fingerprint(text)
        meta = {k: v for k, v in current.items() if k not in ('fingerprint_version', 'content_length')}
        meta['content_hash'] = hashlib.md5(text.strip().encode()).hexdigest()
        Path(self.config.get_raw_meta_path()).write_text(json.dumps(meta))

    def test_legacy_and_unknown_fingerprints_require_explicit_prepare(self):
        f = self.f
        text = f.seed()
        valid = f.files()['meta.json']
        for variant in ('legacy', 'unknown', 'wrong_length', 'wrong_hash'):
            with self.subTest(variant=variant):
                meta = json.loads(valid)
                if variant == 'legacy':
                    self.legacy_meta(text)
                else:
                    meta[{'unknown': 'fingerprint_version', 'wrong_length': 'content_length',
                          'wrong_hash': 'content_hash'}[variant]] = {
                              'unknown': 999, 'wrong_length': len(text) - 1, 'wrong_hash': '0' * 64}[variant]
                    Path(self.config.get_raw_meta_path()).write_text(json.dumps(meta))
                before = f.files()
                self.runtime(text, ready=False)
                f.pipeline.faiss_index_builder.build_from_parsed_document.reset_mock()
                # Explicit accepted-source recovery; no extraction/profile generation.
                with patch.object(f.loader, 'extract', side_effect=AssertionError('extract')), \
                     patch.object(f.pipeline.profile_builder, 'build', side_effect=AssertionError('profile')):
                    assets = f.pipeline.prepare('book')
                self.assertTrue(assets.bundle_ready, assets.errors)
                self.assertEqual(f.pipeline.faiss_index_builder.build_from_parsed_document.call_count, 1)
                self.assertNotEqual(before['meta.json'], f.files()['meta.json'])
                self.runtime(text, ready=True)

    def test_every_record_requires_valid_source_coordinates(self):
        f = self.f
        f.path.write_text(''.join(f'Sentence {i} has distinctive content.\n' for i in range(15)))
        text = f.seed()
        payload = json.loads(f.files()['records.json'])
        self.assertGreater(len(payload['records']), 1)
        for change in ('missing', 'negative', 'past_end', 'boolean', 'wrong_slice', 'wrong_chunk'):
            with self.subTest(change=change):
                candidate = json.loads(json.dumps(payload))
                record = list(candidate['records'].values())[1]
                if change == 'missing':
                    del record['char_end']
                elif change == 'negative':
                    record['char_start'] = -1
                elif change == 'past_end':
                    record['char_end'] = len(text) + 1
                elif change == 'boolean':
                    record['char_start'] = True
                elif change == 'wrong_slice':
                    record['char_start'] += 1
                else:
                    record['chunk_index'] = -1
                Path(self.config.get_raw_records_path()).write_text(json.dumps(candidate))
                self.runtime(text, ready=False)
                self.assertEqual(self.prepare(), 1)
                self.runtime(text, ready=True)

    def test_changed_text_build_failures_preserve_previous_trio(self):
        f = self.f
        text = f.seed()
        before = f.files()
        for method in ('parse', 'embedding', 'save', 'fingerprint'):
            with self.subTest(stage=method):
                f.path.write_text('\n\n' + text)
                obj, name = {'parse': (f.pipeline.node_provider, 'parse'),
                             'embedding': (f.pipeline.faiss_index_builder, 'build_from_parsed_document'),
                             'save': (f.pipeline.faiss_index_store, 'save'),
                             'fingerprint': (f.pipeline.fingerprint_handler, 'save')}[method]
                with patch.object(obj, name, side_effect=RuntimeError('injected')):
                    assets = f.pipeline.prepare('book', pdf_extraction_mode='fast')
                self.assertFalse(assets.faiss_ready)
                self.assertFalse(assets.bundle_ready)
                self.assertEqual(before, f.files())
                bundle = f.pipeline.faiss_index_store.load_validated(
                    self.config, text, f.pipeline.fingerprint_handler)
                self.assert_positions(bundle, text)
                self.runtime('\n\n' + text, ready=False)
        self.assertEqual(self.prepare(), 1)
        self.runtime('\n\n' + text, ready=True)

    def test_interrupted_publication_reads_only_matching_previous_snapshot(self):
        import retrieval.index_publication as publication
        f = self.f
        text = f.seed()
        class Interrupted(BaseException):
            pass
        for stage in FILES:
            for variant in ('changed', 'identical', 'legacy'):
                with self.subTest(stage=stage, variant=variant):
                    if variant == 'legacy':
                        self.legacy_meta(text)
                    before = f.files()
                    new = '\n\n' + text if variant == 'changed' else text
                    f.path.write_text(new)
                    original = publication.os.replace
                    def interrupt(source, target):
                        original(source, target)
                        if Path(source).parent.name == 'candidate' and Path(source).name == stage:
                            raise Interrupted()
                    with patch.object(publication.os, 'replace', side_effect=interrupt):
                        with self.assertRaises(Interrupted):
                            f.pipeline.prepare('book', pdf_extraction_mode='fast', force_rebuild=True)
                    self.assertTrue((self.config.base_dir / JOURNAL).exists())
                    disk_before = self.snapshot()
                    with publication.readable(self.config) as previous:
                        self.assertEqual(before, {name: (previous.base_dir / name).read_bytes() for name in FILES})
                    if variant != 'legacy':
                        self.assert_positions(f.pipeline.faiss_index_store.load_validated(
                            self.config, text, f.pipeline.fingerprint_handler), text)
                    self.runtime(new, ready=(variant == 'identical'))
                    self.assertEqual(disk_before, self.snapshot())
                    f.pipeline.faiss_index_store.recover(self.config)
                    self.assertEqual(before, f.files())
                    f.pipeline.faiss_index_builder.build_from_parsed_document.reset_mock()
                    assets = f.pipeline.prepare('book')
                    self.assertTrue(assets.bundle_ready, assets.errors)
                    self.assertEqual(f.pipeline.faiss_index_builder.build_from_parsed_document.call_count,
                                     int(variant != 'identical'))
                    self.runtime(new, ready=True)
                    text = new


@unittest.skipUnless(os.environ.get('DEEP_READER_TEST_POSTGRES_DSN'), 'requires disposable PostgreSQL DSN')
class PostgresIndexIdentityTests(IndexIdentityTests):
    fixture_class = safety.PostgresRuntimeSafetyTests

    @classmethod
    def setUpClass(cls):
        cls.fixture_class.setUpClass()
        cls.addClassCleanup(cls.fixture_class.doClassCleanups)


if __name__ == '__main__':
    unittest.main(verbosity=2)
