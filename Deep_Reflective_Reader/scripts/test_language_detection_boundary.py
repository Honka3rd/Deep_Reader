"""LANG-01/02: unknown language is not publication authority or log content."""
import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from config.faiss_storage_config import FaissStorageConfig
from language.document_language_detector import DocumentLanguageDetector
from language.language_code import LanguageCode
from language.language_profile_registry import LanguageProfileRegistry
from profile.document_profile_store import DocumentProfileStore
from retrieval.node_provider import NodeProvider
from test_document_profile_language_and_evidence_boundaries import _build_builder, _FakeLLMProvider
import test_preparation_runtime_safety as safety


INVALID = ('', 'unknown', 'cannot determine', '.', '...', ' . ', '. .',
           '!?', None, 42, True, [], {})


class DetectorBoundaryTests(unittest.TestCase):
    def test_invalid_normalization_is_total(self):
        for reply in INVALID:
            with self.subTest(reply=reply):
                self.assertEqual(LanguageProfileRegistry.normalize_detector_output(reply), LanguageCode.UNKNOWN)

    def test_invalid_detection_returns_no_evidence_and_calls_once(self):
        for reply in INVALID:
            with self.subTest(reply=reply):
                provider = Mock(); provider.complete_text.return_value = reply
                self.assertIsNone(DocumentLanguageDetector(provider).detect('這是中文原文。'))
                self.assertEqual(provider.complete_text.call_count, 1)

    def test_valid_codes_and_aliases(self):
        values = [(code, code) for code in LanguageProfileRegistry.get_supported_language_codes()]
        values += [('EN.', 'en'), ('Chinese.', 'zh'), ('zh-TW', 'zh'), ('English', 'en'), ('"de"', 'de')]
        for reply, expected in values:
            with self.subTest(reply=reply):
                provider = Mock(); provider.complete_text.return_value = reply
                self.assertEqual(DocumentLanguageDetector(provider).detect('Source'), expected)
                self.assertEqual(provider.complete_text.call_count, 1)

    def test_diagnostics_never_echo_source_or_output(self):
        marker = 'SYNTHETIC_PRIVATE_MARKER_01'
        for reply in (marker + ' 這是完整合成短文。', {'text': marker}, '...', ''):
            provider = Mock(); provider.complete_text.return_value = reply
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
                self.assertIsNone(DocumentLanguageDetector(provider).detect(marker))
            output = stream.getvalue()
            self.assertNotIn(marker, output)
            self.assertIn('outcome=unknown', output)
            self.assertIn('reason=', output)
            self.assertIn('output_chars=', output)
            self.assertLess(len(output), 300)

    def test_persisted_language_reuse_is_read_only(self):
        with tempfile.TemporaryDirectory() as directory:
            profile = Path(directory) / 'profile.json'; records = Path(directory) / 'records.json'
            config = SimpleNamespace(get_raw_profile_path=lambda: str(profile), get_raw_records_path=lambda: str(records))
            provider = Mock(); provider.complete_text.side_effect = AssertionError('unexpected model call')
            detector = DocumentLanguageDetector(provider)
            profile.write_text(json.dumps({'document_language': 'Chinese.'}))
            records.write_text(json.dumps({'document_language': 'en'}))
            before = (profile.read_bytes(), records.read_bytes())
            self.assertEqual(detector.detect('Source', config), 'zh')
            self.assertEqual(before, (profile.read_bytes(), records.read_bytes()))
            profile.unlink()
            self.assertEqual(detector.detect('Source', config), 'en')
            self.assertEqual(records.read_bytes(), before[1])
            provider.complete_text.assert_not_called()

    def test_retrieval_stops_before_parser_without_evidence(self):
        provider = Mock(); provider.complete_text.return_value = 'unknown'
        parser = Mock(); parser.get_nodes_from_documents.return_value = []
        with self.assertRaisesRegex(ValueError, '^document_language_not_ready$'):
            NodeProvider(parser, DocumentLanguageDetector(provider)).parse('Source', None)
        parser.get_nodes_from_documents.assert_not_called()
        self.assertEqual(provider.complete_text.call_count, 1)

    def test_retrieval_preserves_valid_language(self):
        provider = Mock(); provider.complete_text.return_value = 'Chinese.'
        parser = Mock(); parser.get_nodes_from_documents.return_value = []
        result = NodeProvider(parser, DocumentLanguageDetector(provider)).parse('原文', None)
        self.assertEqual(result.document_language, 'zh')
        parser.get_nodes_from_documents.assert_called_once()


class PreparationCases:
    def setUp(self):
        self.f = self.fixture(); self.f.setUp(); self.addCleanup(self.f.doCleanups)
        self.f.path.write_text('第一章\n這份中文文件描述閱讀、學習與科學研究，保存完整的原始內容。')
        payload = dict(topic='essay', summary='A valid source summary', text_form='essay',
            discourse_mode='expository', document_structure_shape='essay_sections',
            likely_heading_style='plain_title_headings', title_uniqueness_risk='low', confidence=.8, notes=[])
        self.classifier = _FakeLLMProvider([json.dumps(payload)] * 20)
        self.f.pipeline.profile_builder = _build_builder(self.classifier)
        self.provider = Mock()
        self.f.pipeline.language_detector = DocumentLanguageDetector(self.provider)
        self.profile_path = Path(FaissStorageConfig('book').get_raw_profile_path())

    def test_initial_unknown_cannot_publish(self):
        for reply in INVALID:
            with self.subTest(reply=reply):
                self.provider.complete_text.return_value = reply
                with patch.object(self.f.pipeline.structured_document_builder, 'build', wraps=self.f.pipeline.structured_document_builder.build) as build:
                    result = self.f.pipeline.prepare('book', mode='base')
                self.assertIsNone(result.language)
                self.assertFalse(result.profile_ready)
                self.assertFalse(result.structured_document_ready)
                self.assertIn('detect_language_empty_result', result.errors)
                self.assertIn('prepare_profile_skipped:missing_language', result.errors)
                self.assertIn('prepare_structured_document_skipped:missing_language', result.errors)
                build.assert_not_called()
                self.assertFalse(self.profile_path.exists())
                self.assertFalse(self.f.store.exists(self.f.config()))
        self.assertEqual(self.classifier.prompts, [])
        self.assertEqual(self.provider.complete_text.call_count, len(INVALID))

    def test_explicit_rebuild_unknown_preserves_accepted_assets(self):
        self.f.seed()
        notes = safety.NoteInteractionService(self.f.repository())
        state = notes.read('book', target_level='document')
        notes.save('book', target_level='document', text='Keep this note', expected_revision=0,
                   mutation_id='language-note', target_token=state['target_token'])
        before = self.f.store.load(self.f.config()).to_dict()
        old_profile = self.profile_path.read_bytes(); old_index = self.f.files()
        calls = len(self.classifier.prompts)
        for reply in ('unknown', '...', None):
            with self.subTest(reply=reply):
                self.provider.complete_text.return_value = reply
                result = self.f.pipeline.prepare('book', force_rebuild=True)
                self.assertFalse(result.profile_ready)
                self.assertFalse(result.structured_document_ready)
                self.assertFalse(result.bundle_ready)
                self.assertEqual(self.profile_path.read_bytes(), old_profile)
                self.assertEqual(self.f.store.load(self.f.config()).to_dict(), before)
                self.assertEqual(self.f.files(), old_index)
        self.assertEqual(len(self.classifier.prompts), calls)
        with patch.object(self.f.pipeline.language_detector, 'detect', side_effect=AssertionError('runtime detection')), \
             patch.object(self.f.pipeline.profile_builder, 'build', side_effect=AssertionError('runtime profile')):
            self.assertTrue(self.f.pipeline.prepare_and_load('book').assets.bundle_ready)
        self.assertEqual(self.profile_path.read_bytes(), old_profile)
        self.assertEqual(self.f.store.load(self.f.config()).to_dict(), before)

    def test_valid_preparation_keeps_detected_language(self):
        for reply, expected in [('Chinese.', 'zh'), ('en', 'en')]:
            self.provider.complete_text.return_value = reply
            result = self.f.pipeline.prepare('book', mode='base', force_rebuild=True)
            self.assertTrue(result.profile_ready, result.errors)
            self.assertTrue(result.structured_document_ready, result.errors)
            self.assertEqual(result.language, expected)
            self.assertEqual(DocumentProfileStore.load(FaissStorageConfig('book')).document_language_code, expected)
            self.assertEqual(self.f.store.load(self.f.config()).language, expected)


class FilePreparationTests(PreparationCases, unittest.TestCase):
    fixture = safety.RuntimeSafetyTests


@unittest.skipUnless(os.environ.get('DEEP_READER_TEST_POSTGRES_DSN'), 'requires disposable PostgreSQL DSN')
class PostgresPreparationTests(PreparationCases, unittest.TestCase):
    fixture = safety.PostgresRuntimeSafetyTests

    @classmethod
    def setUpClass(cls):
        cls.fixture.setUpClass()
        cls.addClassCleanup(cls.fixture.doClassCleanups)
