"""TASK-02 durable current selection: real file/SQL/HTTP, deterministic providers."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import copy
import os
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
import main
import section_tasks.reading_interaction_artifact_store as adapter
from scripts.test_critical_thinking_publication import (
    ANSWER, CriticalThinkingPublicationTests, PostgresCriticalThinkingPublicationTests,
)
from scripts.test_content_block_projection_identity import ContentBlockProjectionIdentityTests

BASE = '/documents/reading-interactions/critical-thinking/'
CURRENT = '_critical_thinking_current'


class CurrentSessionTests(unittest.TestCase):
    repository = CriticalThinkingPublicationTests.repository
    coordinator = CriticalThinkingPublicationTests.coordinator
    seed = ContentBlockProjectionIdentityTests.seed
    blocks = ContentBlockProjectionIdentityTests.blocks
    target = ContentBlockProjectionIdentityTests.target
    post = ContentBlockProjectionIdentityTests.post

    def setUp(self):
        CriticalThinkingPublicationTests.setUp(self)
        self.seed()

    def generate(self, client, target, digit):
        with patch.object(adapter, 'uuid4', return_value=SimpleNamespace(hex=digit * 32)):
            result = self.post(client, 'generate-question', target)
        self.assertEqual(result['envelope']['status'], 'question_generated')
        return result['envelope']['session_id']

    def reload(self):
        # Recreate file/SQL owner, adapter and coordinator against persisted storage.
        self.repo = type(self.repo)(self.repo.store) if hasattr(self.repo.store, 'dsn') else type(self.repo)(base_dir=self.temp.name)
        self.store = adapter.DocumentReadingInteractionArtifactStore(self.repo)
        self.co = self.coordinator()

    def edit_metadata(self, edit):
        def mutation(document):
            artifacts = document.document_task_artifacts
            metadata = copy.deepcopy(artifacts.metadata)
            edit(metadata['reading_interaction_artifacts'])
            return replace(artifacts, metadata=metadata), None
        self.repo.mutate_document_artifacts('book', mutation)

    def current(self, client, target):
        return self.post(client, 'read', target)['envelope']

    def assert_current_and_availability(self, client, target, expected):
        before = self.repo.load_document('book')
        with patch.object(self.repo, 'mutate_document_artifacts', side_effect=AssertionError('read wrote')), \
                patch.object(self.service, '_question_generator', side_effect=AssertionError('read generated')), \
                patch.object(self.service, '_evaluation_generator', side_effect=AssertionError('read evaluated')):
            envelope = self.current(client, target)
            self.assertEqual(envelope['session_id'], expected)
            if expected is None:
                self.assertEqual(envelope['status'], 'not_generated')
            for route in ('single', 'batch'):
                blocks = self.blocks(client, True, route)
                block = next(b for b in blocks if b['block_id'] == target['content_block_id'])
                slot = block['interaction_availability']['critical_thinking']
                self.assertEqual(slot['session_id'], expected)
                self.assertEqual(slot['status'], envelope['status'])
        self.assertEqual(self.repo.load_document('book'), before)

    def test_http_reload_old_answer_timeout_retry_does_not_steal_current(self):
        with patch.object(main, 'section_task_coordinator', self.co), TestClient(main.app) as client:
            target = self.target(self.blocks(client, True)[0])
            first = self.generate(client, target, 'f')
            second = self.generate(client, target, 'a')
        self.reload()
        with patch.object(main, 'section_task_coordinator', self.co), TestClient(main.app) as client:
            self.assert_current_and_availability(client, target, second)
            with patch.object(self.service, '_evaluation_generator', side_effect=TimeoutError('private')):
                failed = self.post(client, 'submit-answer', target, session_id=first, answer=ANSWER)
            self.assertEqual(failed['envelope']['status'], 'evaluation_failed')
            history = self.post(client, 'read', target, session_id=first)
            self.assertEqual(history['payload']['submitted_answer'], ANSWER)
            self.assert_current_and_availability(client, target, second)
            self.assertEqual(self.post(client, 'retry-evaluation', target, session_id=first)['envelope']['status'], 'completed')
            self.assert_current_and_availability(client, target, second)
            self.assertEqual(self.post(client, 'read', target, session_id=first)['envelope']['status'], 'completed')

    def test_all_five_http_target_levels_select_latest_after_reload(self):
        from scripts.test_critical_thinking_target_publication import CriticalThinkingTargetPublicationTests, LEVELS
        expected = []
        with patch.object(main, 'section_task_coordinator', self.co), TestClient(main.app) as client:
            for level in LEVELS:
                target = CriticalThinkingTargetPublicationTests.target(self, client, level, segmented=True)
                old = self.generate(client, target, 'f')
                current = self.generate(client, target, 'a')
                expected.append((target, old, current))
        self.reload()
        with patch.object(main, 'section_task_coordinator', self.co), TestClient(main.app) as client:
            for target, old, current in expected:
                with self.subTest(level=target['target_type']):
                    self.assertEqual(self.current(client, target)['session_id'], current)
                    self.assertEqual(self.post(client, 'read', target, session_id=old)['envelope']['session_id'], old)

    def test_current_is_target_scoped(self):
        with patch.object(main, 'section_task_coordinator', self.co), TestClient(main.app) as client:
            targets = [self.target(b) for b in self.blocks(client, True)]
            self.assertGreaterEqual(len(targets), 2)
            self.generate(client, targets[0], 'f')
            latest = self.generate(client, targets[0], 'a')
            other = self.generate(client, targets[1], 'b')
            self.assert_current_and_availability(client, targets[0], latest)
            self.assert_current_and_availability(client, targets[1], other)
            self.assertEqual(self.post(client, 'read', targets[0], session_id=other)['envelope']['status'], 'not_generated')

    def test_legacy_ambiguous_history_is_read_only_until_explicit_new_question(self):
        with patch.object(main, 'section_task_coordinator', self.co), TestClient(main.app) as client:
            target = self.target(self.blocks(client, True)[0])
            first = self.generate(client, target, 'f')
            second = self.generate(client, target, 'a')
            self.edit_metadata(lambda metadata: metadata.pop(CURRENT, None))
            self.assert_current_and_availability(client, target, None)
            for session_id in (first, second):
                self.assertEqual(self.post(client, 'read', target, session_id=session_id)['envelope']['status'], 'question_generated')
            self.post(client, 'submit-answer', target, session_id=first, answer=ANSWER)
            self.assert_current_and_availability(client, target, None)
            self.assertNotIn(CURRENT, self.repo.load_document('book').document_task_artifacts.metadata['reading_interaction_artifacts'])
            newest = self.generate(client, target, '0')
            self.assert_current_and_availability(client, target, newest)

    def test_single_legacy_session_is_readable_without_backfill(self):
        with patch.object(main, 'section_task_coordinator', self.co), TestClient(main.app) as client:
            target = self.target(self.blocks(client, True)[0])
            session_id = self.generate(client, target, 'f')
            self.edit_metadata(lambda metadata: metadata.pop(CURRENT, None))
            self.assert_current_and_availability(client, target, session_id)

    def test_invalid_pointer_does_not_guess_or_borrow_another_target(self):
        with patch.object(main, 'section_task_coordinator', self.co), TestClient(main.app) as client:
            target, other = [self.target(b) for b in self.blocks(client, True)][:2]
            first = self.generate(client, target, 'f')
            other_id = self.generate(client, other, 'a')
            # Preserve an explicit bad reference; never fall back to the lone matching history.
            for invalid in ({'session_id': 'absent'}, {'session_id': other_id}, {}, {'session_id': 4}, None, []):
                with self.subTest(invalid=invalid):
                    def change(metadata):
                        self.assertIn(CURRENT, metadata)
                        pointers = metadata[CURRENT]
                        key = next(k for k,v in pointers.items() if v.get('session_id') == first)
                        pointers[key] = invalid
                    original = copy.deepcopy(self.repo.load_document('book').document_task_artifacts.metadata['reading_interaction_artifacts'])
                    self.edit_metadata(change)
                    self.assert_current_and_availability(client, target, None)
                    self.edit_metadata(lambda metadata: (metadata.clear(), metadata.update(original)))

    def test_concurrent_questions_follow_atomic_publication_order(self):
        barrier = threading.Barrier(2)
        published = []
        owner = self.repo.mutate_document_artifacts
        def mutate(name, mutation):
            def observed(document):
                updated, result = mutation(document)
                metadata = document.document_task_artifacts.metadata if document.document_task_artifacts else {}
                old = metadata.get('reading_interaction_artifacts', {}).get('critical_thinking_session', {})
                sessions = updated.metadata['reading_interaction_artifacts']['critical_thinking_session']
                new_ids = set(sessions) - set(old)
                self.assertEqual(len(new_ids), 1)
                published.append(new_ids.pop())
                return updated, result
            return owner(name, observed)
        def provider(request, instruction):
            barrier.wait(timeout=8)
            return {'question': 'Which assumption supports this claim?'}
        with patch.object(self.service, '_question_generator', side_effect=provider), \
                patch.object(self.repo, 'mutate_document_artifacts', side_effect=mutate), ThreadPoolExecutor(max_workers=2) as pool:
            results = [future.result(timeout=20) for future in [
                pool.submit(self.co.generate_critical_thinking_question, **self.args) for _ in range(2)]]
        self.reload()
        self.assertEqual(self.co.read_critical_thinking_session(**self.args).session_id, published[-1])
        self.assertEqual({r.session_id for r in results}, set(published))
        for session_id in published:
            self.assertEqual(self.co.read_critical_thinking_session(**self.args, session_id=session_id).status, 'question_generated')

    def test_failed_new_publication_rolls_back_session_and_current(self):
        first = self.co.generate_critical_thinking_question(**self.args).session_id
        before = self.repo.load_document('book')
        if hasattr(self.repo.store, 'dsn'):
            import psycopg
            execute = psycopg.Connection.execute
            def fail(connection, query, *args, **kwargs):
                result = execute(connection, query, *args, **kwargs)
                if 'UPDATE documents SET metadata_payload=jsonb_set' in str(query):
                    raise OSError('after SQL update')
                return result
            fault = patch.object(psycopg.Connection, 'execute', fail)
        else:
            fault = patch('document_structure.structured_document_artifact_repository.os.replace', side_effect=OSError('atomic replace'))
        with fault, self.assertRaisesRegex(RuntimeError, 'save_unconfirmed'):
            self.co.generate_critical_thinking_question(**self.args)
        self.assertEqual(self.repo.load_document('book'), before)
        self.assertEqual(self.co.read_critical_thinking_session(**self.args).session_id, first)

    def test_lost_acknowledgement_preserves_current_and_single_generation(self):
        owner = self.repo.mutate_document_artifacts
        def mutate(name, mutation):
            owner(name, mutation)
            raise OSError('lost acknowledgement')
        with patch.object(self.repo, 'mutate_document_artifacts', side_effect=mutate), \
                patch.object(self.service, '_question_generator', return_value={'question': 'Which premise matters?'}) as provider:
            generated = self.co.generate_critical_thinking_question(**self.args)
        provider.assert_called_once()
        self.reload()
        self.assertEqual(self.co.read_critical_thinking_session(**self.args).session_id, generated.session_id)


@unittest.skipUnless(os.environ.get('DEEP_READER_TEST_POSTGRES_DSN'), 'requires disposable PostgreSQL DSN')
class PostgresCurrentSessionTests(CurrentSessionTests):
    setUpClass = classmethod(PostgresCriticalThinkingPublicationTests.setUpClass.__func__)
    drop_schema = classmethod(PostgresCriticalThinkingPublicationTests.drop_schema.__func__)
    repository = PostgresCriticalThinkingPublicationTests.repository


if __name__ == '__main__':
    unittest.main(verbosity=2)
