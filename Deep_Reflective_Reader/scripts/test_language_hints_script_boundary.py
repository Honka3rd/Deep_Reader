"""LANG-03 conservative script filtering and real file/PostgreSQL persistence.

No live models. SQL uses the existing disposable-schema test contract.
"""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from uuid import uuid4

from language.language_script_registry import LanguageScriptRegistry
from profile.document_profile import ScriptSystem
from document_structure.structured_document import StructuredDocument, StructuredChapter, StructuredSection
from document_structure.learning_repository import OperationWrite
from reading_preferences.models import BookSettings, LanguageLearningSettings, WorkspaceScope
from scripts import test_learning_file_repository as files
from scripts import test_learning_hints_service as hints
from scripts import test_learning_postgres_repository as postgres
from section_tasks.learning_generation_policy import GenerationBinding, GenerationCommand
from section_tasks.learning_hints_output import parse_hints, LearningHintsError
from section_tasks.learning_hints_service import LearningHintsService
from section_tasks.learning_sentence_segmenter import segment_learning_block
from shared.task_unit_model import TaskUnit

LATIN = 'Scientific research requires reliable evidence and very careful analysis.'
CHINESE = '這本書介紹科學研究的方法與證據。'
CASES = (
    ('latin', LATIN, 'research', 'zh', ['zh'], False, 0, 1),
    ('simplified', '这个问题需要通过研究与数据分析来解决。', '研究', 'zh', ['zh'], False, 1, 0),
    ('traditional', '這個問題需要透過研究與數據分析來解決。', '研究', 'zh', ['zh'], False, 1, 0),
    ('neutral', '中文文本中的人物情感。', '人物', 'zh', ['zh'], False, 1, 0),
    ('name', '這本書介紹 OpenAI 的研究方法與相關證據。', '研究', 'zh', ['zh'], False, 1, 0),
    ('short_name', 'OpenAI', 'OpenAI', 'zh', ['zh'], True, 0, 0),
    ('short_uncertain', 'Research', 'Research', 'zh', ['zh'], False, 1, 0),
    ('mixed', '這是研究方法，research requires evidence.', '研究', 'zh', ['zh', 'en'], False, 0, 0),
    ('uncertain', '這是研究方法，research requires evidence.', '研究', 'zh', ['zh', None], False, 0, 0),
    ('japanese', 'こんにちは世界。', '世界', 'ja', ['ja'], False, 1, 0),
    ('wrong_english', '這段完全由中文字符組成並且沒有任何英文詞語。', '中文', 'en', ['en'], False, 0, 1),
)


def proposal_data(prepared, form, *, languages=None, proper=False):
    data = hints.response_data(prepared)
    if languages is not None:
        for row in data['sentences']:
            row['language_codes'] = languages
    data['candidates'] = [hints.proposal(prepared.plan.groups[0][0], form,
        proper_name=proper, meaning='這是對原文詞語的簡明解釋。' if prepared.key.source_language == 'en'
        else 'A careful study of the subject.')]
    return data


class ScriptBoundaryTests(unittest.TestCase):
    def test_annotation_matrix(self):
        for name, text, form, language, annotations, proper, count, omitted in CASES:
            with self.subTest(name=name):
                _, p = hints.prepare(text=text, source=language, target='zh' if language == 'en' else 'en')
                data = proposal_data(p, form, languages=annotations, proper=proper)
                items, assessment = parse_hints(json.dumps(data), p.key, p.sentences, text, p.plan.groups[0])
                self.assertEqual(len(items), count)
                self.assertEqual(assessment.omitted_sentences, omitted)
                self.assertEqual(assessment.accepted_count, count)
                self.assertEqual(assessment.candidate_count, assessment.accepted_count + assessment.filtered_count + assessment.rejected_count)
                self.assertEqual('script_mismatch' in {i.code for i in assessment.issues}, bool(omitted))

    def test_conservative_threshold_and_han_extension(self):
        for text, rejected in (
            ('aaaaa bbbbb ccccc ddddd eeeee fffff ggggg hhhhh', True),
            ('aaaaa bbbbb ccccc ddddd eeeee fffff ggggg hhhh', False),
            ('aaaaaa bbbbbb cccccc dddddd eeeeee ffffff gggggg', False),
            (LATIN[:-1] + ' 㐀', False), (LATIN[:-1] + ' 中', False),
            (LATIN[:-1] + ' 𠀀', False), (LATIN[:-1] + ' 﨑', False), ('12345 !?', False),
        ):
            with self.subTest(text=text):
                _, p = hints.prepare(text=text, source='zh', target='en')
                data = hints.response_data(p)
                _, report = parse_hints(json.dumps(data), p.key, p.sentences, text, p.plan.groups[0])
                self.assertEqual(bool(report.omitted_sentences), rejected)

    def test_advisory_default_is_not_redefined_as_language_authority(self):
        registry = LanguageScriptRegistry()
        self.assertEqual(registry.get_entry('zh').default_script_system, ScriptSystem.MIXED)
        self.assertEqual(registry.detect_script_system(text='中文文本', language='zh'), ScriptSystem.MIXED)
        self.assertEqual(registry.detect_script_system(text='', language='zh'), ScriptSystem.MIXED)

    def test_policy_identity_changes_without_sentence_or_prompt_change(self):
        _, p = hints.prepare(text=CHINESE, source='zh', target='en')
        self.assertEqual(p.key.policy_version, 'learning-hints-6')
        old = replace(p.key, policy_version='learning-hints-5')
        self.assertNotEqual(old.cache_id, p.key.cache_id)
        self.assertEqual(old.resource, p.key.resource)
        self.assertEqual(p.key.prompt_version, 'learning-hints-prompt-4')


class PersistenceCases:
    def seed(self, text=LATIN):
        unit = TaskUnit('unit', 'Unit', None, text, ['section'], False, parent_section_id='section')
        section = StructuredSection('section', 0, 'Section', 2, text, 0, len(text), parent_chapter_id='chapter', task_units=[unit])
        document = StructuredDocument('book', 'Book', None, 'zh', text,
            chapters=[StructuredChapter('chapter', 'Chapter', 1, None, [section])])
        self.docs.save_reparsed_document(document, 'book')
        self.prefs.save_workspace('en', expected_revision=0, mutation_id='native')
        self.prefs.save_book('book', BookSettings('language_learning', 'zh',
            (LanguageLearningSettings('zh', 'tocfl_reading', '1', '1'),)), expected_revision=0,
            expected_workspace_revision=1, resolved_source_language='zh', mutation_id='settings')
        self.before = self.docs.load_document('book').to_dict()
        block = self.repo.read_sources('book').blocks[0]
        # Consume the actual reading projection, including its existing OCR normalization.
        self.sentences = segment_learning_block(source=block.source, effective_text=block.effective_text, source_language='zh')
        self.provider = hints.Provider()
        self.service = LearningHintsService(self.provider, provider_name='test')
        self.text = block.effective_text
        self.select()

    def select(self):
        self.prepared = self.service.prepare(sentences=self.sentences, effective_text=self.text,
            source_language='zh', explanation_language='en', proficiency=hints.proficiency('zh', '1'))
        self.binding = GenerationBinding(self.prepared.key, 1, 1)
        self.provider.response = json.dumps(proposal_data(self.prepared, 'research' if self.text == LATIN else '研究'))

    def write(self, entry, action, now, **fields):
        return self.repo.write_operation('book', OperationWrite(entry.operation.operation_id,
            entry.revision, entry.last_fence, now, action, **fields))

    def complete(self, command='generate', now=10):
        p = self.prepared
        entry = self.repo.claim('book', GenerationCommand(command, self.binding), sentences=self.sentences,
            plan=p.plan, now_ms=now, lease_until_ms=now+100).next_entry
        entry = self.write(entry, 'dispatch', now+1)
        batch = self.service.generate_batch(p, entry.operation, now_ms=now+1)
        entry = self.write(entry, 'checkpoint', now+2, batch_index=0, batch=batch)
        saved = self.service.assemble(p, entry.operation, self.repo.read_checkpoints('book', entry.operation.operation_id))
        return self.write(entry, 'commit', now+3, result=saved)

    def test_rejected_sentence_persists_assessment_reloads_and_reuses(self):
        self.seed()
        entry = self.complete()
        saved = self.restart().read_batch('book', (self.prepared.key,))[0].view.saved
        self.assertEqual(saved, entry.saved)
        self.assertEqual(saved.items, ())
        self.assertEqual(saved.assessment.omitted_sentences, 1)
        self.assertEqual(saved.assessment.rejected_count, 1)
        self.assertIn('script_mismatch', {i.code for i in saved.assessment.issues})
        reuse = self.repo.claim('book', GenerationCommand('reuse', self.binding), sentences=self.sentences,
            plan=self.prepared.plan, now_ms=30, lease_until_ms=100)
        self.assertEqual(reuse.outcome, 'reuse')
        self.assertEqual(len(self.provider.calls), 1)
        self.assertEqual(self.docs.load_document('book').to_dict(), self.before)

    def test_old_policy_saved_is_preserved_and_new_read_does_not_generate(self):
        self.seed(CHINESE)
        with patch('section_tasks.learning_hints_service.POLICY_VERSION', 'learning-hints-5'):
            self.select()
            old_key = self.prepared.key
            old = self.complete().saved
        self.select()
        self.assertNotEqual(old_key.cache_id, self.prepared.key.cache_id)
        self.assertEqual(old.items[0].source_form, '研究')
        for _ in range(2):
            rows = self.restart().read_batch('book', (old_key, self.prepared.key))
            self.assertEqual(rows[0].view.saved, old)
            self.assertIsNone(rows[1].view.saved)
        self.assertEqual(len(self.provider.calls), 1)
        new = self.complete('new-policy', 50).saved
        self.assertEqual(new.items, old.items)
        self.assertEqual(self.restart().read_batch('book', (old_key,))[0].view.saved, old)
        self.assertEqual(len(self.provider.calls), 2)
        self.assertEqual(self.docs.load_document('book').to_dict(), self.before)

    def test_failed_refresh_keeps_valid_saved_and_checkpoint(self):
        self.seed(CHINESE)
        old = self.complete()
        old_batches = self.repo.read_checkpoints('book', old.operation.operation_id)
        entry = self.repo.claim('book', GenerationCommand('refresh', self.binding, 'regenerate'),
            sentences=self.sentences, plan=self.prepared.plan, now_ms=100, lease_until_ms=200).next_entry
        entry = self.write(entry, 'dispatch', 110)
        self.provider.response = '{}'
        with self.assertRaises(LearningHintsError):
            self.service.generate_batch(self.prepared, entry.operation, now_ms=110)
        entry = self.write(entry, 'fail', 120, reason='invalid_output', outcome_known=True)
        self.assertEqual(entry.saved, old.saved)
        self.assertEqual(self.restart().read_batch('book', (self.prepared.key,))[0].view.saved, old.saved)
        self.assertEqual(self.repo.read_checkpoints('book', old.operation.operation_id), old_batches)
        self.assertEqual(len(self.provider.calls), 2)
        self.assertEqual(self.docs.load_document('book').to_dict(), self.before)


class FileScriptPersistenceTests(PersistenceCases, unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.docs, self.prefs, self.repo = files.repositories(self.root)

    def restart(self):
        return files.repositories(self.root)[2]


class PostgresScriptPersistenceTests(PersistenceCases, unittest.TestCase):
    setUpClass = classmethod(postgres.LearningPostgresTests.setUpClass.__func__)
    drop_schema = classmethod(postgres.LearningPostgresTests.drop_schema.__func__)

    def setUp(self):
        self.namespace = uuid4().hex
        self.docs = postgres.PostgresStructuredDocumentArtifactRepository(
            postgres.PostgresStructuredDocumentStore(self.scoped, self.namespace))
        self.prefs = postgres.PostgresReadingPreferencesRepository(self.scoped, WorkspaceScope(self.namespace))
        self.repo = postgres.repository(self.scoped, self.namespace)

    def restart(self):
        return postgres.repository(self.scoped, self.namespace)


if __name__ == '__main__':
    unittest.main()
