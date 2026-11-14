"""APP-05: real HTTP/source/session publication at every supported target level.

Run with the backend on PYTHONPATH and DEEP_READER_TEST_POSTGRES_DSN set to a
disposable-test database. Providers are deterministic; SQL uses isolated schemas.
"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import os
import unittest
from unittest.mock import patch
from uuid import uuid4

from fastapi.testclient import TestClient
import main
from scripts.test_critical_thinking_publication import (
    ANSWER, EVALUATION, CriticalThinkingPublicationTests,
    PostgresCriticalThinkingPublicationTests,
)
from scripts.test_reading_interaction_artifact_store import _document
from shared.task_unit_model import TaskUnitContentBlock


BASE = '/documents/reading-interactions/critical-thinking/'
LEVELS = ('document', 'chapter', 'section', 'task_unit', 'content_block')


class CriticalThinkingTargetPublicationTests(unittest.TestCase):
    # Reuse storage/service setup only, without counting inherited APP-03 tests.
    setUp = CriticalThinkingPublicationTests.setUp
    repository = CriticalThinkingPublicationTests.repository
    coordinator = CriticalThinkingPublicationTests.coordinator

    def target(self, client, level):
        document = self.repo.load_document('book')
        chapter = document.chapters[0]
        section = chapter.sections[0]
        unit = section.task_units[0]
        target = {'doc_name': 'book', 'target_type': level}
        if level != 'document':
            target['chapter_id'] = chapter.chapter_id
        if level in ('section', 'task_unit', 'content_block'):
            target['section_id'] = section.section_id
        if level in ('task_unit', 'content_block'):
            target['task_unit_id'] = unit.unit_id
        if level == 'content_block':
            response = client.post('/documents/book/task-units/content', json={
                'task_unit_ids': [unit.unit_id], 'include_raw_content': True})
            self.assertEqual(response.status_code, 200, response.text)
            block = response.json()['contents'][0]['content_blocks'][0]
            self.assertIsNone((block.get('metadata') or {}).get('source_structure_version'))
            target['content_block_id'] = block['block_id']
        return target

    def post(self, client, action, target, **fields):
        response = client.post(BASE + action, json={'target': target, **fields})
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()['envelope']

    def assert_read(self, client, target, session_id, status):
        before = self.repo.load_document('book')
        with patch.object(self.service, '_question_generator') as question, \
                patch.object(self.service, '_evaluation_generator') as evaluation, \
                patch.object(self.repo, 'mutate_document_artifacts') as write:
            response = client.post(BASE + 'read', json={'target': target, 'session_id': session_id})
            self.assertEqual(response.status_code, 200, response.text)
            result = response.json()
            question.assert_not_called()
            evaluation.assert_not_called()
            write.assert_not_called()
        self.assertEqual(result['envelope']['status'], status)
        self.assertEqual(result['payload']['submitted_answer'], ANSWER)
        self.assertEqual(self.repo.load_document('book'), before)

    def test_five_levels_generate_submit_read_and_timeout_retry(self):
        with patch.object(main, 'section_task_coordinator', self.co), TestClient(main.app) as client:
            for level in LEVELS:
                for timeout in (False, True):
                    with self.subTest(level=level, timeout=timeout):
                        target = self.target(client, level)
                        target['source_structure_version'] = None
                        generated = self.post(client, 'generate-question', target)
                        self.assertEqual(generated['status'], 'question_generated')
                        session_id = generated['session_id']

                        def evaluate(session, instruction):
                            # Provider-time read proves the submitted answer is durable.
                            document = self.repo.load_document('book')
                            saved = self.store._session_from_document(document, session_id)
                            self.assertEqual(saved.status, 'answer_submitted')
                            self.assertEqual(saved.payload['answer'], ANSWER)
                            # A second thread/SQL connection can acquire the writer boundary.
                            pool = ThreadPoolExecutor(max_workers=1)
                            try:
                                future = pool.submit(self.repo.mutate_document_artifacts, 'book',
                                                     lambda current: (current.document_task_artifacts, 'free'))
                                self.assertEqual(future.result(timeout=4), 'free')
                            finally:
                                pool.shutdown(wait=False)
                            if timeout:
                                raise TimeoutError('PRIVATE PROVIDER DETAILS')
                            return dict(EVALUATION)

                        with patch.object(self.service, '_evaluation_generator', side_effect=evaluate) as provider:
                            result = self.post(client, 'submit-answer', target,
                                               session_id=session_id, answer=ANSWER)
                        self.assertEqual(provider.call_count, 1)
                        self.assertNotIn('PRIVATE', repr(result))
                        status = 'evaluation_failed' if timeout else 'completed'
                        self.assertEqual(result['status'], status)
                        self.assert_read(client, target, session_id, status)
                        if timeout:
                            # A fresh coordinator/store reads the persisted failed answer.
                            self.co = self.coordinator()
                            with patch.object(main, 'section_task_coordinator', self.co):
                                retried = self.post(client, 'retry-evaluation', target, session_id=session_id)
                            self.assertEqual(retried['status'], 'completed')
                            self.assert_read(client, target, session_id, 'completed')

    def test_segmented_block_absent_null_and_unavailable_evidence(self):
        with patch.object(main, 'section_task_coordinator', self.co), TestClient(main.app) as client:
            target = self.target(client, 'content_block')
            for evidence in ({}, {'source_hash': None, 'source_structure_version': None}):
                with self.subTest(evidence=evidence):
                    result = self.post(client, 'generate-question', {**target, **evidence})
                    self.assertEqual(result['status'], 'question_generated')
            # Supplying a version when selected metadata has none must still reject.
            for evidence in ({'source_structure_version': 1}, {'source_hash': 'stale-hash'}):
                with self.subTest(evidence=evidence):
                    before = self.repo.load_document('book')
                    with patch.object(self.service, '_question_generator') as provider:
                        response = client.post(BASE + 'generate-question', json={'target': {**target, **evidence}})
                        self.assertEqual(response.status_code, 409, response.text)
                        self.assertIn('stale_content_block_target', response.text)
                        provider.assert_not_called()
                    self.assertEqual(self.repo.load_document('book'), before)

    def test_explicit_block_evidence_is_not_document_version(self):
        # Custom blocks legitimately carry their own evidence; never add a version
        # to ordinary segmented blocks merely to satisfy publication validation.
        document = _document()
        unit = document.chapters[0].sections[0].task_units[0]
        unit.content_blocks.append(TaskUnitContentBlock(
            block_id='custom-evidence-block', content=unit.content,
            metadata={'source_hash': 'accepted-block-hash', 'source_structure_version': 37}))
        self.repo.save_reparsed_document(document, 'book')
        with patch.object(main, 'section_task_coordinator', self.co), TestClient(main.app) as client:
            target = self.target(client, 'content_block')
            target['content_block_id'] = 'custom-evidence-block'
            for evidence in ({}, {'source_hash': 'accepted-block-hash', 'source_structure_version': 37}):
                with self.subTest(evidence=evidence):
                    current = {**target, **evidence}
                    generated = self.post(client, 'generate-question', current)
                    completed = self.post(client, 'submit-answer', current,
                                          session_id=generated['session_id'], answer=ANSWER)
                    self.assertEqual(completed['status'], 'completed')
                    stored = self.repo.load_document('book')
                    saved = self.store._session_from_document(stored, generated['session_id'])
                    self.assertEqual(saved.metadata['interaction_policy'].get('source_structure_version'),
                                     stored.parse_provenance.get('storage_structure_version'))
                    self.assertNotEqual(saved.metadata['interaction_policy'].get('source_structure_version'), 37)
            for evidence in ({'source_hash': 'outdated'}, {'source_structure_version': 36}):
                before = self.repo.load_document('book')
                with patch.object(self.service, '_question_generator') as provider:
                    response = client.post(BASE + 'generate-question', json={'target': {**target, **evidence}})
                    self.assertEqual(response.status_code, 409, response.text)
                    provider.assert_not_called()
                self.assertEqual(self.repo.load_document('book'), before)

    def test_late_question_and_evaluation_reject_new_incarnation_at_five_levels(self):
        with patch.object(main, 'section_task_coordinator', self.co), TestClient(main.app) as client:
            for level in LEVELS:
                for stage in ('question', 'evaluation'):
                    with self.subTest(level=level, stage=stage):
                        self.repo.save_reparsed_document(_document(), 'book')
                        target = self.target(client, level)
                        fields = {}
                        if stage == 'evaluation':
                            fields = {'session_id': self.post(client, 'generate-question', target)['session_id'],
                                      'answer': ANSWER}
                        after = []

                        def delayed(*args):
                            # Same IDs and text, different incarnation: hash alone is insufficient.
                            self.repo.save_reparsed_document(_document(), 'book')
                            after.append(self.repo.load_document('book'))
                            return {'question': 'Which assumption matters most?'} if stage == 'question' else dict(EVALUATION)

                        attribute = '_question_generator' if stage == 'question' else '_evaluation_generator'
                        action = 'generate-question' if stage == 'question' else 'submit-answer'
                        with patch.object(self.service, attribute, side_effect=delayed):
                            response = client.post(BASE + action, json={'target': target, **fields})
                        self.assertEqual(response.status_code, 409, response.text)
                        self.assertEqual(response.json()['detail'], 'stale_critical_thinking_source')
                        self.assertEqual(self.repo.load_document('book'), after[0])

    def test_competing_retry_result_cannot_overwrite_new_session_revision(self):
        with patch.object(main, 'section_task_coordinator', self.co), TestClient(main.app) as client:
            for level in LEVELS:
                with self.subTest(level=level):
                    target = self.target(client, level)
                    session_id = self.post(client, 'generate-question', target)['session_id']
                    with patch.object(self.service, '_evaluation_generator', side_effect=TimeoutError):
                        self.post(client, 'submit-answer', target, session_id=session_id, answer=ANSWER)
                    after = []

                    def delayed(session, instruction):
                        with patch.object(self.service, '_evaluation_generator', return_value={
                                **EVALUATION, 'feedback': 'Newer result must survive.'}):
                            newer = self.service.evaluate_answer(session)
                        self.store.save_critical_thinking_session(newer, expected_session=session)
                        after.append(self.repo.load_document('book'))
                        return dict(EVALUATION)

                    with patch.object(self.service, '_evaluation_generator', side_effect=delayed):
                        response = client.post(BASE + 'retry-evaluation', json={
                            'target': target, 'session_id': session_id})
                    self.assertEqual(response.status_code, 409, response.text)
                    self.assertEqual(response.json()['detail'], 'stale_critical_thinking_session')
                    self.assertEqual(self.repo.load_document('book'), after[0])
                    self.assert_read(client, target, session_id, 'completed')

    def test_block_lifecycle_changes_reject_late_evaluation_without_writes(self):
        from document_structure.reading_content import ReadingContentService
        with patch.object(main, 'section_task_coordinator', self.co), TestClient(main.app) as client:
            for change in ('reimport', 'rename', 'edit', 'undo'):
                with self.subTest(change=change):
                    self.repo.save_reparsed_document(_document(), 'book')
                    target = self.target(client, 'content_block')
                    session_id = self.post(client, 'generate-question', target)['session_id']
                    after = []

                    def delayed(*args):
                        if change == 'reimport':
                            self.repo.delete_document('book')
                            self.repo.save_reparsed_document(_document(), 'book')
                        elif change == 'rename':
                            for old, new in (('book', 'away'), ('away', 'book')):
                                self.repo.rename_document(old, new,
                                    lambda document, name: replace(document, document_id=name))
                        else:
                            content = ReadingContentService(self.repo)
                            layout = content.read('book')
                            content.mutate('book', action='edit', block_id=layout['blocks'][0]['block_id'],
                                text='Changed effective reading source.', expected_revision=layout['revision'],
                                source_token=layout['source_token'], mutation_id=uuid4().hex,
                                discard_interactions=False)
                            if change == 'undo':
                                layout = content.read('book')
                                content.mutate('book', action='undo', expected_revision=layout['revision'],
                                    source_token=layout['source_token'], mutation_id=uuid4().hex)
                        after.append(self.repo.load_document('book'))
                        return dict(EVALUATION)

                    with patch.object(self.service, '_evaluation_generator', side_effect=delayed):
                        result = client.post(BASE + 'submit-answer', json={
                            'target': target, 'session_id': session_id, 'answer': ANSWER})
                    self.assertEqual(result.status_code, 409, result.text)
                    self.assertEqual(result.json()['detail'], 'stale_critical_thinking_source')
                    self.assertEqual(self.repo.load_document('book'), after[0])

    def test_block_interruption_preserves_answer_for_reloaded_http_retry(self):
        with patch.object(main, 'section_task_coordinator', self.co), TestClient(main.app) as client:
            target = self.target(client, 'content_block')
            session_id = self.post(client, 'generate-question', target)['session_id']
            args = {key: value for key, value in target.items() if key != 'target_type'}
            with patch.object(self.service, '_evaluation_generator', side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    self.co.submit_critical_thinking_answer(**args, target_level='content_block',
                                                            session_id=session_id, answer=ANSWER)
            self.co = self.coordinator()
            with patch.object(main, 'section_task_coordinator', self.co):
                self.assert_read(client, target, session_id, 'answer_submitted')
                result = self.post(client, 'retry-evaluation', target, session_id=session_id)
                self.assertEqual(result['status'], 'completed')
                self.assert_read(client, target, session_id, 'completed')


@unittest.skipUnless(os.environ.get('DEEP_READER_TEST_POSTGRES_DSN'), 'requires disposable PostgreSQL DSN')
class PostgresCriticalThinkingTargetPublicationTests(CriticalThinkingTargetPublicationTests):
    setUpClass = classmethod(PostgresCriticalThinkingPublicationTests.setUpClass.__func__)
    drop_schema = classmethod(PostgresCriticalThinkingPublicationTests.drop_schema.__func__)
    repository = PostgresCriticalThinkingPublicationTests.repository

    def test_non_block_client_document_version_guard_is_preserved(self):
        with patch.object(main, 'section_task_coordinator', self.co), TestClient(main.app) as client:
            for level in LEVELS[:-1]:
                with self.subTest(level=level):
                    target = self.target(client, level)
                    version = self.repo.load_document('book').parse_provenance['storage_structure_version']
                    generated = self.post(client, 'generate-question', {
                        **target, 'source_structure_version': version})
                    self.assertEqual(generated['status'], 'question_generated')
                    before = self.repo.load_document('book')
                    with patch.object(self.service, '_question_generator') as provider:
                        response = client.post(BASE + 'generate-question', json={
                            'target': {**target, 'source_structure_version': version + 1}})
                        self.assertEqual(response.status_code, 409, response.text)
                        self.assertEqual(response.json()['detail'], 'stale_structure_version')
                        provider.assert_not_called()
                    self.assertEqual(self.repo.load_document('book'), before)


if __name__ == '__main__':
    unittest.main(verbosity=2)
