"""TASK-05: real legacy section/chapter HTTP publication and corrupt-cache reads.

Providers are deterministic; repository mutations, reloads and SQL transactions are real.
"""
from contextlib import contextmanager
from dataclasses import replace
import json
import tempfile
import unittest
from unittest.mock import patch
from uuid import uuid4

from fastapi.testclient import TestClient
import main
from app.section_task_coordinator import SectionTaskCoordinator
from document_structure.structured_document_artifact_repository import StructuredDocumentArtifactRepository
from db.postgres_structured_document_artifact_repository import PostgresStructuredDocumentArtifactRepository
from db.postgres_structured_document_store import PostgresStructuredDocumentStore
from scripts.test_reading_interaction_artifact_store import _document
from scripts.test_summary_note_postgres import PostgresNotes
from scripts.test_summary_persistence_cache import (
    _FakePreparationPipeline, _FakeProfileStore, _FakeTaskUnitResolver, _FakeEnhancedParseEvaluator,
)
from section_tasks.chapter_summary_service import ChapterSummaryService
from section_tasks.chapter_quiz_service import ChapterQuizService
from section_tasks.section_task_context_builder import SectionTaskContextBuilder
from section_tasks.section_task_prompt_builder_factory import SectionTaskPromptBuilderFactory
from section_tasks.section_task_prompt_common import SectionTaskPromptCommon
from section_tasks.topic_guidance_registry import TopicGuidanceRegistry
from section_tasks.summary_task_prompt_builder import SummaryTaskPromptBuilder
from section_tasks.section_quiz_task_prompt_builder import SectionQuizTaskPromptBuilder
from section_tasks.chapter_quiz_task_prompt_builder import ChapterQuizTaskPromptBuilder
from shared.task_artifacts import SummaryArtifact, QuizArtifact, TaskArtifacts

BODY = 'This passage explains why clear assumptions support sound conclusions. ' * 12
BAD = (None, True, 123, [], {}, '', ' \n\t')


def questions():
    return [dict(question_id=f'q{i}', question_text=f'Question {i}', answer_text=f'Answer {i}') for i in range(4)]


class Provider:
    def __init__(self):
        self.response = 'A valid saved summary.'
        self.calls = 0
        self.during = None

    def complete_text(self, prompt):
        self.calls += 1
        if self.during:
            self.during()
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


class OutputCases:
    def seed(self, body=BODY):
        doc = _document(); chapter = doc.chapters[0]; section = chapter.sections[0]
        section = replace(section, content=body, char_start=0, char_end=len(body),
                          task_units=[replace(section.task_units[0], content=body)])
        self.repo.save_reparsed_document(replace(doc, document_id='book', raw_text=body,
            chapters=[replace(chapter, sections=[section])]), 'book')
        self.provider = Provider()
        common = SectionTaskPromptCommon(TopicGuidanceRegistry())
        factory = SectionTaskPromptBuilderFactory(SummaryTaskPromptBuilder(common),
            SectionQuizTaskPromptBuilder(common), ChapterQuizTaskPromptBuilder(common))
        resolver = _FakeTaskUnitResolver(); context = SectionTaskContextBuilder()
        self.co = SectionTaskCoordinator(_FakePreparationPipeline(self.repo), self.repo,
            _FakeProfileStore(), ChapterSummaryService(self.provider, context, factory, resolver),
            ChapterQuizService(self.provider, context, factory, resolver), resolver, _FakeEnhancedParseEvaluator())
        self.co.ensure_document_task_layout(doc_name='book')
        doc = self.repo.load_document('book')
        self.chapter_id = doc.chapters[0].chapter_id
        self.section_id = doc.chapters[0].sections[0].section_id

    @contextmanager
    def client(self):
        with patch.object(main, 'section_task_coordinator', self.co), TestClient(main.app) as client:
            yield client

    def request(self, client, level, kind, refresh=False):
        route = 'summarize-chapter' if (level, kind) == ('chapter', 'summary') else level + '-' + kind
        return client.post('/documents/' + route, json={'doc_name': 'book',
            level + '_id': self.section_id if level == 'section' else self.chapter_id,
            'refresh_' + kind: refresh})

    def success(self, response):
        self.assertEqual(response.status_code, 200, response.text)
        self.assertTrue(response.json()['success'], response.text)
        return response.json()

    def failure(self, response):
        self.assertNotEqual(response.status_code, 200, response.text)
        self.assertFalse(response.json()['success'], response.text)

    def snapshot(self):
        return self.repo.load_document('book').to_dict()

    def saved(self, level, kind):
        doc = self.repo.load_document('book')
        artifacts = (doc.chapters[0].sections[0].task_artifacts if level == 'section'
                     else next(iter(doc.document_task_artifacts.chapter_artifacts.values())))
        return getattr(artifacts, kind) if artifacts else None

    def corrupt(self, level, kind, value):
        """Seed old stored corruption through real storage, never a mocked successful save."""
        def mutation(doc):
            def change(artifacts):
                old = getattr(artifacts, kind)
                return replace(artifacts, **{kind: replace(old, **{('content' if kind == 'summary' else 'items'): value})})
            if level == 'section':
                c = doc.chapters[0]; s = c.sections[0]
                return replace(doc, chapters=[replace(c, sections=[replace(s, task_artifacts=change(s.task_artifacts))])])
            artifacts = doc.document_task_artifacts
            return replace(doc, document_task_artifacts=replace(artifacts,
                chapter_artifacts={key: change(item) for key, item in artifacts.chapter_artifacts.items()}))
        self.repo._mutate_hierarchy('book', mutation)

    def test_invalid_quiz_fields_reject_without_initial_save(self):
        for level in ('section', 'chapter'):
            self.seed()
            with self.client() as client:
                for field in ('question_id', 'question_text', 'answer_text'):
                    for bad in BAD:
                        with self.subTest(level=level, field=field, bad=bad):
                            rows = questions(); rows[2][field] = bad
                            self.provider.response = json.dumps(rows)
                            before = self.snapshot(); calls = self.provider.calls
                            self.failure(self.request(client, level, 'quiz'))
                            self.assertEqual(self.provider.calls, calls + 1)
                            self.assertEqual(self.snapshot(), before)

    def test_duplicate_and_malformed_quiz_refresh_preserves_valid_cache(self):
        duplicate = questions(); duplicate[3]['question_id'] = ' q0 '
        malformed = [None, [], {}, True, 'not json', json.dumps(questions()[:3]),
                     json.dumps(questions() + [None]), json.dumps(duplicate), TimeoutError('offline')]
        for level in ('section', 'chapter'):
            self.seed(); self.provider.response = json.dumps(questions())
            with self.client() as client:
                valid = self.success(self.request(client, level, 'quiz'))['questions']
                for bad in malformed:
                    with self.subTest(level=level, bad=repr(bad)):
                        before = self.snapshot(); calls = self.provider.calls
                        self.provider.response = bad
                        self.failure(self.request(client, level, 'quiz', True))
                        self.assertEqual(self.snapshot(), before)
                        cached = self.success(self.request(client, level, 'quiz'))
                        self.assertTrue(cached['cache_hit'])
                        self.assertEqual(cached['questions'], valid)
                        self.assertEqual(self.provider.calls, calls + 1)

    def test_invalid_summary_initial_and_refresh_preserves_saved_result(self):
        for level in ('section', 'chapter'):
            self.seed()
            with self.client() as client:
                before = self.snapshot()
                self.provider.response = ' \n\t'
                self.failure(self.request(client, level, 'summary'))
                self.assertEqual(self.snapshot(), before)
                self.provider.response = ' A valid saved summary. '
                valid = self.success(self.request(client, level, 'summary'))['result']
                for bad in BAD + (TimeoutError('offline'), ConnectionError('offline')):
                    with self.subTest(level=level, bad=repr(bad)):
                        before = self.snapshot(); calls = self.provider.calls
                        self.provider.response = bad
                        self.failure(self.request(client, level, 'summary', True))
                        self.assertEqual(self.snapshot(), before)
                        cached = self.success(self.request(client, level, 'summary'))
                        self.assertTrue(cached['cache_hit']); self.assertEqual(cached['result'], valid)
                        self.assertEqual(self.provider.calls, calls + 1)

    def test_valid_generation_refresh_and_reloaded_reuse(self):
        for level in ('section', 'chapter'):
            for kind in ('summary', 'quiz'):
                self.seed(); self.provider.response = 'First summary' if kind == 'summary' else json.dumps(questions())
                with self.client() as client:
                    self.assertFalse(self.success(self.request(client, level, kind))['cache_hit'])
                    self.assertIsNotNone(self.saved(level, kind))
                    rows = questions(); rows[0]['answer_text'] = 'Changed answer'
                    self.provider.response = 'Next summary' if kind == 'summary' else json.dumps(rows)
                    new = self.success(self.request(client, level, kind, True))
                    self.assertFalse(new['cache_hit'])
                    self.assertEqual(self.provider.calls, 2)
                    before = self.snapshot()
                    with patch.object(self.repo, '_mutate_hierarchy', side_effect=AssertionError('cache write')), \
                         patch.object(self.provider, 'complete_text', side_effect=AssertionError('cache generate')):
                        cached = self.success(self.request(client, level, kind))
                    self.assertTrue(cached['cache_hit'])
                    self.assertEqual(cached['result' if kind == 'summary' else 'questions'], new['result' if kind == 'summary' else 'questions'])
                    self.assertEqual(self.snapshot(), before)

    def test_corrupt_quiz_cache_rejected_as_whole_and_only_explicit_generation_repairs(self):
        duplicate = questions(); duplicate[3]['question_id'] = 'q0'
        wrong = questions(); wrong[0]['question_text'] = None
        for level in ('section', 'chapter'):
            for bad in (duplicate, wrong, questions() + [None]):
                with self.subTest(level=level, bad=repr(bad)):
                    self.seed(); self.provider.response = json.dumps(questions())
                    with self.client() as client:
                        self.success(self.request(client, level, 'quiz'))
                        self.corrupt(level, 'quiz', bad)
                        before = self.snapshot(); calls = self.provider.calls
                        with patch.object(self.repo, '_mutate_hierarchy', side_effect=AssertionError('read write')):
                            artifact = self.saved(level, 'quiz')
                            self.assertIsNone(self.co._quiz_questions_from_artifact(artifact))
                        self.assertEqual(self.snapshot(), before); self.assertEqual(self.provider.calls, calls)
                        self.assertFalse(self.success(self.request(client, level, 'quiz'))['cache_hit'])
                        self.assertEqual(self.provider.calls, calls + 1)
                        self.assertEqual(len(self.saved(level, 'quiz').items), 4)

    def test_corrupt_summary_cache_regenerates_without_stringification(self):
        for level in ('section', 'chapter'):
            for bad in BAD:
                with self.subTest(level=level, bad=bad):
                    self.seed()
                    with self.client() as client:
                        self.success(self.request(client, level, 'summary'))
                        self.corrupt(level, 'summary', bad)
                        self.assertFalse(self.success(self.request(client, level, 'summary'))['cache_hit'])
                        self.assertEqual(self.provider.calls, 2)
                        self.assertEqual(self.saved(level, 'summary').content, 'A valid saved summary.')

    def test_legacy_single_question_and_ordinary_strings_remain_reusable(self):
        for level in ('section', 'chapter'):
            self.seed()
            with self.client() as client:
                self.success(self.request(client, level, 'summary'))
                self.corrupt(level, 'summary', 'None')
                self.provider.response = json.dumps(questions())
                self.success(self.request(client, level, 'quiz'))
                self.corrupt(level, 'quiz', [dict(question_id='7', question_text='None', answer_text='123')])
                before = self.snapshot(); calls = self.provider.calls
                summary = self.success(self.request(client, level, 'summary'))
                quiz = self.success(self.request(client, level, 'quiz'))
                self.assertTrue(summary['cache_hit']); self.assertEqual(summary['result'], 'None')
                self.assertTrue(quiz['cache_hit']); self.assertEqual(len(quiz['questions']), 1)
                self.assertEqual(self.snapshot(), before); self.assertEqual(self.provider.calls, calls)

    def test_corrupt_availability_http_is_read_only(self):
        for level in ('section', 'chapter'):
            for kind in ('summary', 'quiz'):
                self.seed(); self.provider.response = 'Valid summary' if kind == 'summary' else json.dumps(questions())
                with self.client() as client:
                    self.success(self.request(client, level, kind))
                    duplicate = questions(); duplicate[1]['question_id'] = 'q0'
                    self.corrupt(level, kind, None if kind == 'summary' else duplicate)
                    before = self.snapshot(); calls = self.provider.calls
                    with patch.object(self.repo, '_mutate_hierarchy', side_effect=AssertionError('availability write')), \
                         patch.object(self.provider, 'complete_text', side_effect=AssertionError('availability generate')):
                        result = client.post('/documents/task-layout', json={'doc_name': 'book'})
                    self.assertEqual(result.status_code, 200, result.text)
                    flags = []
                    def inspect(value):
                        if isinstance(value, dict):
                            if kind + '_cache_valid' in value:
                                flags.append(value[kind + '_cache_valid'])
                            for child in value.values():
                                inspect(child)
                        elif isinstance(value, list):
                            for child in value:
                                inspect(child)
                    inspect(result.json())
                    self.assertIn(False, flags); self.assertNotIn(True, flags)
                    self.assertEqual(self.snapshot(), before); self.assertEqual(self.provider.calls, calls)

    def test_summary_service_rejects_invalid_payload_before_orchestration(self):
        self.seed()
        unit = self.repo.load_document('book').chapters[0].sections[0].task_units[0]
        for bad in BAD:
            with self.subTest(bad=bad):
                self.provider.response = bad
                result = self.co.chapter_summary_service.summarize_task_unit(
                    task_unit=unit, document_title='Book', document_profile=None, task_unit_index=0)
                self.assertFalse(result.success); self.assertIsNone(result.payload)

    def test_valid_late_generation_cannot_publish_after_reparse(self):
        for level in ('section', 'chapter'):
            for kind in ('summary', 'quiz'):
                self.seed(); self.provider.response = 'Valid but late' if kind == 'summary' else json.dumps(questions())
                state = []
                def reparse():
                    self.repo.save_reparsed_document(self.repo.load_document('book'), 'book')
                    state.append(self.snapshot())
                self.provider.during = reparse
                with self.client() as client:
                    self.failure(self.request(client, level, kind))
                self.assertEqual(self.snapshot(), state[0])
                self.assertEqual(self.provider.calls, 1)


class FileOutputs(OutputCases, unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        self.repo = StructuredDocumentArtifactRepository(base_dir=temp.name)


class PostgresOutputs(OutputCases, unittest.TestCase):
    setUpClass = classmethod(PostgresNotes.setUpClass.__func__)
    drop_schema = classmethod(PostgresNotes.drop_schema.__func__)

    def setUp(self):
        self.repo = PostgresStructuredDocumentArtifactRepository(
            PostgresStructuredDocumentStore(self.scoped, namespace=uuid4().hex))


class PayloadContracts(unittest.TestCase):
    def test_deserialization_does_not_mask_invalid_payloads(self):
        for bad in (None, True, 3, [], {}):
            with self.subTest(bad=bad):
                self.assertFalse(SummaryArtifact.from_dict({'content': bad}).content)
        for bad in (None, {}, [questions()[0], None]):
            with self.subTest(bad=bad):
                try:
                    parsed = QuizArtifact.from_dict({'items': bad})
                except (TypeError, ValueError) as error:
                    self.fail(f'corrupt optional artifact must remain loadable: {type(error).__name__}')
                self.assertFalse(parsed.items)

    def test_valid_legacy_roundtrip(self):
        artifact = TaskArtifacts(summary=SummaryArtifact('None'), quiz=QuizArtifact(questions()[:1]))
        self.assertEqual(TaskArtifacts.from_dict(artifact.to_dict()), artifact)


if __name__ == '__main__':
    unittest.main()
