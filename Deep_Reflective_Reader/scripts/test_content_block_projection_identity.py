"""APP-06: exact default/segmented HTTP source identity, real file and SQL stores.

Providers are deterministic. SQL schemas are disposable; no private inputs required.
"""
from dataclasses import replace
import hashlib
import os
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
import main
from scripts.test_critical_thinking_publication import (
    ANSWER, EVALUATION, CriticalThinkingPublicationTests,
    PostgresCriticalThinkingPublicationTests,
)
from scripts.test_reading_interaction_artifact_store import _document
from shared.task_unit_model import TaskUnitContentBlock
from document_structure.text_normalization import normalize_ocr_text

BASE = '/documents/reading-interactions/critical-thinking/'
FIRST = 'This paragraph explains why the observations alone do not establish causation.'
SECOND = 'This different paragraph describes an alternative explanation and the evidence required to distinguish it.'
BODY = FIRST + '\n\n' + SECOND


class ContentBlockProjectionIdentityTests(unittest.TestCase):
    setUp = CriticalThinkingPublicationTests.setUp
    repository = CriticalThinkingPublicationTests.repository
    coordinator = CriticalThinkingPublicationTests.coordinator

    def seed(self, content=BODY, blocks=None):
        document = _document()
        chapter = document.chapters[0]
        section = chapter.sections[0]
        unit = replace(section.task_units[0], content=content, content_blocks=blocks or [])
        section = replace(section, content=content, char_end=len(content), task_units=[unit])
        document = replace(document, raw_text=content, chapters=[replace(chapter, sections=[section])])
        self.repo.save_reparsed_document(document, 'book')
        current = self.repo.load_document('book')
        self.chapter = current.chapters[0]
        self.section = self.chapter.sections[0]
        self.unit = self.section.task_units[0]

    def set_blocks(self, blocks):
        # Preserve the actual SQL-generated owning unit ID while installing explicit blocks.
        current = self.repo.load_document('book')
        chapter = current.chapters[0]
        section = chapter.sections[0]
        unit = replace(section.task_units[0], content_blocks=blocks)
        updated = replace(current, chapters=[replace(chapter, sections=[replace(section, task_units=[unit])])])
        self.repo.save_document(updated, 'book')
        current = self.repo.load_document('book')
        self.chapter = current.chapters[0]
        self.section = self.chapter.sections[0]
        self.unit = self.section.task_units[0]

    def blocks(self, client, mode=None, route='batch'):
        args = {} if mode is None else {'segmented': mode}
        if route == 'single':
            response = client.get('/documents/book/task-units/' + self.unit.unit_id + '/content', params=args)
        else:
            response = client.post('/documents/book/task-units/content', json={
                'task_unit_ids': [self.unit.unit_id], **args})
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        return (payload if route == 'single' else payload['contents'][0])['content_blocks']

    def target(self, block):
        return dict(doc_name='book', target_type='content_block', chapter_id=self.chapter.chapter_id,
                    section_id=self.section.section_id, task_unit_id=self.unit.unit_id,
                    content_block_id=block['block_id'],
                    source_hash=(block.get('metadata') or {}).get('source_hash'),
                    source_structure_version=(block.get('metadata') or {}).get('source_structure_version'))

    def post(self, client, action, target, **fields):
        response = client.post(BASE + action, json={'target': target, **fields})
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def resolved(self, block):
        args = self.target(block)
        args['target_level'] = args.pop('target_type')
        return self.co.resolve_reading_target(**args)

    def generate(self, client, block):
        with patch.object(self.service, '_question_generator', return_value={
                'question': 'Which assumption does the passage rely upon?'}) as provider:
            result = self.post(client, 'generate-question', self.target(block))
        self.assertEqual(result['envelope']['status'], 'question_generated', result)
        provider.assert_called_once()
        source = provider.call_args.args[0].target.content
        self.assertEqual(source, block['content'])
        session_id = result['envelope']['session_id']
        saved = self.store._session_from_document(self.repo.load_document('book'), session_id)
        self.assertEqual(saved.metadata['interaction_policy']['source_sha256'],
                         hashlib.sha256(block['content'].encode()).hexdigest())
        self.assertEqual(saved.target_id, block['block_id'])
        return session_id

    def test_single_batch_omitted_false_true_exact_source_and_policy(self):
        for body in (BODY, 'Section One\n' + BODY, 'Short.\n\n' + SECOND):
            self.seed(body)
            with patch.object(main, 'section_task_coordinator', self.co), TestClient(main.app) as client:
                for route in ('single', 'batch'):
                    for mode in (None, False, True):
                        with self.subTest(body=body[:15], route=route, mode=mode):
                            before = self.repo.load_document('book')
                            with patch.object(self.repo, 'mutate_document_artifacts') as write, \
                                    patch.object(self.service, '_question_generator') as provider:
                                blocks = self.blocks(client, mode, route)
                            write.assert_not_called()
                            provider.assert_not_called()
                            self.assertEqual(self.repo.load_document('book'), before)
                            for block in blocks:
                                self.assertEqual(self.resolved(block).content, block['content'])
                                if block['content'] == 'Short.':
                                    with patch.object(self.service, '_question_generator') as provider:
                                        result = self.post(client, 'generate-question', self.target(block))
                                    self.assertEqual(result['envelope']['status'], 'insufficient_content')
                                    provider.assert_not_called()
                                else:
                                    session_id = self.generate(client, block)
                                    read = self.post(client, 'read', self.target(block), session_id=session_id)
                                    self.assertEqual(read['envelope']['status'], 'question_generated')
                            if mode is not True:
                                self.assertEqual(len(blocks), 1)
                                self.assertEqual(blocks[0]['content'], normalize_ocr_text(body))
                            else:
                                self.assertEqual(blocks[0]['block_id'], self.unit.unit_id + ':content:0')
                                self.assertNotIn('Section One', blocks[0]['content'])

    def test_availability_and_explicit_session_read_do_not_cross_projections(self):
        for generated_mode in (False, True):
            self.seed()
            with patch.object(main, 'section_task_coordinator', self.co), TestClient(main.app) as client:
                selected = self.blocks(client, generated_mode)[0]
                other = self.blocks(client, not generated_mode)[0]
                self.assertNotEqual(selected['block_id'], other['block_id'])
                session_id = self.generate(client, selected)
                for route in ('single', 'batch'):
                    current = self.blocks(client, generated_mode, route)[0]
                    alternative = self.blocks(client, not generated_mode, route)[0]
                    self.assertEqual(current['interaction_availability']['critical_thinking']['session_id'], session_id)
                    slot = alternative['interaction_availability']['critical_thinking']
                    self.assertEqual(slot['status'], 'not_generated')
                    self.assertIsNone(slot['session_id'])
                    for fields in ({}, {'session_id': session_id}):
                        response = self.post(client, 'read', self.target(alternative), **fields)
                        self.assertEqual(response['envelope']['status'], 'not_generated')

    def test_custom_blocks_normalize_and_disambiguate_without_persistence_migration(self):
        self.seed()
        colliding_id = self.unit.unit_id + ':content:0'
        for block_id in (colliding_id, 'custom-public-block'):
            with self.subTest(block_id=block_id):
                custom = TaskUnitContentBlock(block_id=block_id, content='Custom source.\r\n' + SECOND,
                    artifact_ids=['unverified-old-id'],
                    interaction_availability={'critical_thinking': {'status': 'question_generated', 'session_id': 'unverified-old-id'}},
                    metadata={'content_block_id': block_id, 'source_hash': 'custom-evidence', 'source_structure_version': 37})
                self.set_blocks([custom])
                before = self.repo.load_document('book')
                with patch.object(main, 'section_task_coordinator', self.co), TestClient(main.app) as client:
                    for route in ('single', 'batch'):
                        for mode in (None, False):
                            block = self.blocks(client, mode, route)[0]
                            self.assertEqual(self.resolved(block).content, block['content'])
                            self.assertEqual(block['metadata']['content_block_id'], block['block_id'])
                            if block_id == colliding_id:
                                self.assertNotEqual(block['block_id'], colliding_id)
                                self.assertIsNone(block['artifact_ids'])
                                self.assertIsNone(block['artifact_target_refs'])
                            else:
                                self.assertEqual(block['block_id'], block_id)
                    self.assertEqual(self.repo.load_document('book'), before)
                    self.generate(client, block)
                    for segmented_block in self.blocks(client, True):
                        self.generate(client, segmented_block)
                    for evidence in ({'source_hash': 'stale'}, {'source_structure_version': 36}):
                        with patch.object(self.service, '_question_generator') as provider:
                            response = client.post(BASE + 'generate-question', json={
                                'target': {**self.target(block), **evidence}})
                        self.assertEqual(response.status_code, 409, response.text)
                        provider.assert_not_called()

    def test_same_text_custom_evidence_has_distinct_identity(self):
        self.seed(FIRST)
        custom = TaskUnitContentBlock(block_id=self.unit.unit_id + ':content:0', content=FIRST,
                                      metadata={'source_structure_version': 37, 'source_hash': 'custom'})
        self.set_blocks([custom])
        with patch.object(main, 'section_task_coordinator', self.co), TestClient(main.app) as client:
            default = self.blocks(client, False)[0]
            segmented = self.blocks(client, True)[0]
            self.assertNotEqual(default['block_id'], segmented['block_id'])
            self.assertEqual(default['content'], segmented['content'])
            self.generate(client, default)
            self.generate(client, segmented)

    def test_legacy_single_paragraph_and_segmented_sessions_remain_addressable(self):
        self.seed(FIRST)
        with patch.object(main, 'section_task_coordinator', self.co), TestClient(main.app) as client:
            default = self.blocks(client)[0]
            self.assertEqual(default['block_id'], self.unit.unit_id + ':content:0')
            self.assertIsNone(default['metadata'])
            session_id = self.generate(client, default)
            segmented = self.blocks(client, True)[0]
            self.assertEqual(segmented['block_id'], default['block_id'])
            self.assertEqual(segmented['interaction_availability']['critical_thinking']['session_id'], session_id)
            self.assertEqual(self.post(client, 'read', self.target(segmented))['envelope']['session_id'], session_id)

    def test_primary_reading_blocks_do_not_duplicate_compatibility_text(self):
        from document_structure.reading_content import ReadingContentService
        self.seed()
        reading = ReadingContentService(self.repo).read('book')
        self.assertEqual([block['content'] for block in reading['blocks']], [FIRST, SECOND])
        self.assertTrue(all(':compat:' not in block['block_id'] for block in reading['blocks']))

    def test_projection_mismatch_cannot_borrow_availability(self):
        self.seed()
        block = TaskUnitContentBlock(block_id=self.unit.unit_id + ':content:0', content=BODY)
        with self.assertRaisesRegex(ValueError, 'projection_source_mismatch'):
            self.co._build_content_block_availability_request(document=self.repo.load_document('book'),
                doc_name='book', task_unit=self.unit, section=self.section, chapter=self.chapter,
                content_block=block, interaction_type='critical_thinking_session')

    def test_compatibility_timeout_retry_and_competing_result_cas(self):
        self.seed()
        with patch.object(main, 'section_task_coordinator', self.co), TestClient(main.app) as client:
            block = self.blocks(client)[0]
            target = self.target(block)
            session_id = self.generate(client, block)
            with patch.object(self.service, '_evaluation_generator', side_effect=TimeoutError):
                failed = self.post(client, 'submit-answer', target, session_id=session_id, answer=ANSWER)
            self.assertEqual(failed['envelope']['status'], 'evaluation_failed')
            saved = self.store._session_from_document(self.repo.load_document('book'), session_id)
            self.assertEqual(saved.payload['answer'], ANSWER)
            snapshots = []
            def delayed(session, instruction):
                with patch.object(self.service, '_evaluation_generator', return_value=EVALUATION):
                    newer = self.service.evaluate_answer(session)
                self.store.save_critical_thinking_session(newer, expected_session=session)
                snapshots.append(self.repo.load_document('book'))
                return dict(EVALUATION)
            with patch.object(self.service, '_evaluation_generator', side_effect=delayed):
                response = client.post(BASE + 'retry-evaluation', json={'target': target, 'session_id': session_id})
            self.assertEqual(response.status_code, 409, response.text)
            self.assertEqual(self.repo.load_document('book'), snapshots[0])
            self.assertEqual(self.post(client, 'read', target, session_id=session_id)['envelope']['status'], 'completed')

    def test_compatibility_late_source_and_epoch_changes_reject_without_writes(self):
        for changed in (BODY, BODY + '\nNew source.'):
            self.seed()
            with patch.object(main, 'section_task_coordinator', self.co), TestClient(main.app) as client:
                target = self.target(self.blocks(client)[0])
                snapshots = []
                def delayed(request, instruction):
                    self.seed(changed)
                    snapshots.append(self.repo.load_document('book'))
                    return {'question': 'Which assumption matters most?'}
                with patch.object(self.service, '_question_generator', side_effect=delayed):
                    response = client.post(BASE + 'generate-question', json={'target': target})
                self.assertEqual(response.status_code, 409, response.text)
                self.assertEqual(self.repo.load_document('book'), snapshots[0])

    def test_duplicate_and_occupied_alias_cannot_be_published(self):
        self.seed()
        original_id = self.unit.unit_id + ':content:0'
        alias = self.unit.unit_id + ':compat:' + hashlib.sha256(original_id.encode()).hexdigest()
        for second_id in (original_id, alias):
            blocks = [TaskUnitContentBlock(block_id=original_id, content=BODY),
                      TaskUnitContentBlock(block_id=second_id, content=SECOND)]
            before = self.repo.load_document('book')
            with self.assertRaisesRegex(ValueError, 'duplicate_content_block_id'):
                self.set_blocks(blocks)
            self.assertEqual(self.repo.load_document('book'), before)
            # A malformed legacy snapshot also fails before provider dispatch.
            malformed = replace(before, chapters=[replace(self.chapter, sections=[replace(
                self.section, task_units=[replace(self.unit, content_blocks=blocks)])])])
            with patch.object(self.repo, 'load_document', return_value=malformed), \
                    patch.object(self.service, '_question_generator') as provider, \
                    patch.object(main, 'section_task_coordinator', self.co), TestClient(main.app) as client:
                response = client.post(BASE + 'generate-question', json={'target': {
                    'doc_name': 'book', 'target_type': 'content_block',
                    'task_unit_id': self.unit.unit_id, 'content_block_id': original_id}})
                self.assertIn(response.status_code, (400, 409, 422), response.text)
                provider.assert_not_called()
            self.assertEqual(self.repo.load_document('book'), before)


@unittest.skipUnless(os.environ.get('DEEP_READER_TEST_POSTGRES_DSN'), 'requires disposable PostgreSQL DSN')
class PostgresContentBlockProjectionIdentityTests(ContentBlockProjectionIdentityTests):
    setUpClass = classmethod(PostgresCriticalThinkingPublicationTests.setUpClass.__func__)
    drop_schema = classmethod(PostgresCriticalThinkingPublicationTests.drop_schema.__func__)
    repository = PostgresCriticalThinkingPublicationTests.repository
