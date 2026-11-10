"""Ordinary/learning reuse on real file and disposable SQL storage; no paid calls."""
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from fastapi import FastAPI
from fastapi.testclient import TestClient
from app.translation_action import TranslationAction
from section_tasks.translation_contracts import TranslationError
from section_tasks.translation_records import translation_snapshot
from scripts import test_learning_orchestration as orchestration
from scripts import test_learning_file_repository as files
from scripts import test_learning_postgres_repository as sql
from translation_routes import router

class SharedCases:
    setup_service = orchestration.OrchestrationCases.setup_service
    make_service = orchestration.OrchestrationCases.make_service
    settings = orchestration.OrchestrationCases.settings
    request = orchestration.OrchestrationCases.request
    block_provider = orchestration.OrchestrationCases.block_provider
    change = orchestration.OrchestrationCases.change

    def setup_shared(self):
        self.setup_service()
        self.action = TranslationAction(self.docs, SimpleNamespace(load=lambda config:
            SimpleNamespace(document_language='en')), shared_translation=self.service)
        self.action._shared.wait_seconds = 5
        self.block_id = self.block.source.content_block_id

    def normal(self, command='ordinary'):
        return self.action.generate(doc_name='book', content_block_id=self.block_id,
            source_language='en', target_language='zh', mutation_id=command)

    def records(self):
        return self.action.read(doc_name='book', content_block_id=self.block_id)['translations']

    def test_executor_rejection_records_failure_and_releases_shared_capacity(self):
        self.settings(normal=True)
        with patch.object(self.service.worker, 'start', side_effect=RuntimeError('executor closed')):
            for index in range(5):
                command = f'rejected-{index}'
                with self.assertRaisesRegex(RuntimeError, 'executor closed'):
                    self.normal(command)
                receipt = self.repo.lookup_command('book', 'translation:' + command)
                status = self.service.status('book', receipt.operation_id)
                self.assertEqual(status.state, 'failed')
                self.assertEqual(status.snapshot.operation.failure_code, 'worker_stopped')
                self.assertFalse(status.snapshot.operation.outcome_uncertain)
        self.assertFalse(self.trans_provider.calls)
        self.assertEqual(self.records(), [])
        self.assertEqual(self.normal('accepted').translation.status, 'completed')
        self.assertEqual(len(self.trans_provider.calls), 1)

    def test_normal_then_learning_reuses_exact_result_without_provider_preflight(self):
        self.settings(normal=True)
        result = self.normal()
        self.assertEqual(result.record['translated_text'], '這是一句完整譯文。 這是一句完整譯文。')
        self.assertEqual(len(self.trans_provider.calls), 1)
        self.assertEqual(translation_snapshot(self.docs.load_document('book'), self.block_id).records(), [])
        files.configure(self.prefs)
        with patch.object(self.trans_provider, 'get_model_capabilities', side_effect=AssertionError('must reuse')):
            value = self.request(command_id='learning')
        self.assertEqual(value.outcome, 'reuse')
        page = self.service.read('book', (self.block_id,))
        self.assertEqual(len(page.blocks[0].components[0].view.saved.rows), 2)
        self.assertIsNone(page.blocks[0].components[1].view.saved)
        self.assertEqual(len(self.records()), 1)

    def test_learning_then_normal_read_and_generate_reuse(self):
        self.request(); self.service.shutdown()
        self.settings(normal=True)
        with patch.object(self.trans_provider, 'get_model_capabilities', side_effect=AssertionError('must reuse')):
            first = self.records(); result = self.normal()
        self.assertEqual(first, [result.record])
        self.assertEqual(len(self.trans_provider.calls), 1)

    def test_no_native_language_required(self):
        self.settings(normal=True)
        workspace = self.prefs.read_workspace()
        self.prefs.save_workspace(None, expected_revision=workspace.revision, mutation_id='clear-native')
        result = self.normal()
        self.assertEqual(result.translation.status, 'completed')
        self.assertIsNone(self.prefs.read_workspace().native_language)

    def test_concurrent_normal_requests_one_provider_call(self):
        self.settings(normal=True); entered, release = self.block_provider()
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(self.normal, 'first'); self.assertTrue(entered.wait(5))
            second = pool.submit(self.normal, 'second'); release.set()
            self.assertEqual(first.result(10).record, second.result(10).record)
        self.assertEqual(len(self.trans_provider.calls), 1)

    def test_normal_inflight_survives_mode_switch_and_learning_reuses(self):
        self.settings(normal=True); entered, release = self.block_provider()
        with ThreadPoolExecutor(max_workers=1) as pool:
            first = pool.submit(self.normal); self.assertTrue(entered.wait(5)); files.configure(self.prefs)
            other = self.request(command_id='learning'); self.assertIn(other.outcome, ('reconcile', 'coalesced'))
            release.set(); first.result(10)
        self.assertEqual(self.request(command_id='learning-reuse').outcome, 'reuse')
        self.assertEqual(len(self.trans_provider.calls), 1)

    def test_source_edit_rejects_late_translation(self):
        self.settings(normal=True); entered, release = self.block_provider()
        with ThreadPoolExecutor(max_workers=1) as pool:
            first = pool.submit(self.normal); self.assertTrue(entered.wait(5)); self.change(); release.set()
            with self.assertRaisesRegex(TranslationError, 'source_changed'): first.result(10)
        self.assertEqual(self.records(), [])

    def test_unknown_provider_outcome_does_not_retry(self):
        self.settings(normal=True); self.trans_provider.failure = TimeoutError('private')
        for command in ('ordinary', 'ordinary', 'another-command'):
            with self.assertRaisesRegex(TranslationError, 'outcome_unconfirmed'): self.normal(command)
        self.assertEqual(len(self.trans_provider.calls), 1)

    def test_known_invalid_output_can_be_explicitly_retried(self):
        self.settings(normal=True); self.trans_provider.response = '{}'
        with self.assertRaisesRegex(TranslationError, 'invalid_output'): self.normal()
        self.trans_provider.response = None
        self.assertEqual(self.normal('retry').translation.status, 'completed')
        self.assertEqual(len(self.trans_provider.calls), 2)

    def test_timeout_later_read_reuses_without_new_generation(self):
        self.settings(normal=True); self.action._shared.wait_seconds = .01
        entered, release = self.block_provider()
        with self.assertRaisesRegex(TranslationError, 'outcome_unconfirmed'): self.normal()
        self.assertTrue(entered.is_set()); release.set(); self.service.shutdown()
        self.assertEqual(len(self.records()), 1); self.normal()
        self.assertEqual(len(self.trans_provider.calls), 1)

    def test_legacy_paragraph_preserved_and_aligned_result_takes_precedence(self):
        from section_tasks.translation_contracts import TranslationResult
        legacy = TranslationAction(self.docs, self.action._profiles, SimpleNamespace(translate=lambda **kw:
            TranslationResult('en', 'zh', 'completed', '舊段落譯文。')))
        legacy.generate(doc_name='book', content_block_id=self.block_id, source_language='en', target_language='zh')
        self.assertEqual(self.records()[0]['translated_text'], '舊段落譯文。')
        self.normal(); self.assertFalse(self.trans_provider.calls)
        self.request(); self.service.shutdown()
        self.assertEqual(len(self.trans_provider.calls), 1)
        self.assertNotEqual(self.records()[0]['translated_text'], '舊段落譯文。')
        self.assertEqual(translation_snapshot(self.docs.load_document('book'), self.block_id).records()[0]['translated_text'], '舊段落譯文。')

    def test_lost_claim_ack_resumes_only_on_explicit_retry(self):
        from document_structure.learning_repository import LearningCommitUnknown
        self.settings(normal=True)
        claim = self.repo.claim
        def lost(*args, **kw):
            claim(*args, **kw)
            raise LearningCommitUnknown('lost claim acknowledgement')
        with patch.object(self.repo, 'claim', side_effect=lost):
            with self.assertRaisesRegex(TranslationError, 'outcome_unconfirmed'): self.normal()
        self.assertFalse(self.trans_provider.calls)
        self.assertEqual(self.records(), [])
        self.normal(); self.assertEqual(len(self.trans_provider.calls), 1)

    def test_read_only_catalog_and_bounded_pagination(self):
        self.settings(normal=True); self.normal()
        before = self.records()
        with patch.object(self.repo, 'claim', side_effect=AssertionError('read cannot claim')), \
             patch.object(self.repo, 'write_operation', side_effect=AssertionError('read cannot write')):
            page = self.repo.read_translations('book', self.block_id, limit=1)
            self.assertEqual(len(page.records), 1); self.assertIsNotNone(page.next_cursor)
            self.assertEqual(self.repo.read_translations('book', self.block_id, after=page.next_cursor, limit=1).records, ())
            self.assertEqual(self.records(), before)
        self.assertEqual(len(self.trans_provider.calls), 1)

    def test_fresh_book_needs_no_learning_preferences(self):
        self.docs.save_reparsed_document(files.fixture('fresh'), 'fresh')
        fresh_id = self.repo.read_sources('fresh').blocks[0].source.content_block_id
        result = self.action.generate(doc_name='fresh', content_block_id=fresh_id,
            source_language='en', target_language='zh', mutation_id='fresh')
        self.assertEqual(result.translation.status, 'completed')
        self.assertEqual(self.prefs.read_book('fresh').revision, 0)

    def test_explicit_source_language_pair_does_not_reuse_other_source(self):
        self.settings(normal=True); self.normal()
        other = self.action.generate(doc_name='book', content_block_id=self.block_id,
            source_language='fr', target_language='zh', mutation_id='other-source')
        self.assertEqual(other.translation.source_language, 'fr')
        self.assertEqual(len(self.trans_provider.calls), 2)
        # Legacy presentation has a single target record, selected deterministically.
        self.assertEqual(len(self.records()), 1)
        self.assertEqual(self.records()[0]['source_language'], 'en')
        self.assertEqual(len(self.repo.read_translations('book', self.block_id).records), 2)

    def test_failed_learning_refresh_preserves_ordinary_projection(self):
        self.settings(normal=True); original = self.normal().record
        files.configure(self.prefs); self.trans_provider.response = '{}'
        self.request(command_id='refresh', intent='regenerate'); self.service.shutdown()
        self.assertEqual(self.records(), [original])
        self.assertEqual(len(self.trans_provider.calls), 2)

    def test_same_command_cannot_change_source_language(self):
        self.settings(normal=True); self.normal()
        with self.assertRaisesRegex(TranslationError, 'mutation_conflict'):
            self.action.generate(doc_name='book', content_block_id=self.block_id,
                source_language='fr', target_language='zh', mutation_id='ordinary')
        self.assertEqual(len(self.trans_provider.calls), 1)

    def test_transport_shape_unchanged_and_shared_prompt_reported(self):
        self.settings(normal=True)
        app = FastAPI(); app.include_router(router); app.state.translation_action = self.action
        with TestClient(app) as client:
            request = dict(doc_name='book', content_block_id=self.block_id)
            response = client.post('/documents/reading-interactions/translate/generate', json={**request,
                'source_language':'en','target_language':'zh','mutation_id':'http'})
            self.assertEqual(response.status_code, 200, response.text)
            generated = response.json(); self.assertTrue(generated['envelope']['persisted'])
            self.assertEqual(generated['envelope']['prompt_instruction_version'], self.normal().prompt_version)
            read = client.post('/documents/reading-interactions/translate/read', json=request)
            self.assertEqual(read.status_code, 200, read.text)
            self.assertEqual(read.json()['translations'], [generated['record']])

class SharedFileTests(SharedCases, unittest.TestCase):
    def setUp(self):
        files.LearningFileRepositoryTests.setUp(self); self.setup_shared()

class SharedSqlTests(SharedCases, unittest.TestCase):
    setUpClass = classmethod(sql.LearningPostgresTests.setUpClass.__func__)
    drop_schema = classmethod(sql.LearningPostgresTests.drop_schema.__func__)
    def setUp(self):
        sql.LearningPostgresTests.setUp(self); self.setup_shared()

    def test_sql_commit_failure_cannot_publish_aggregate(self):
        from db import learning_rows
        self.settings(normal=True); persist = learning_rows.persist
        def fail(connection, row_id, scope, doc, state, previous):
            persist(connection, row_id, scope, doc, state, previous)
            if any(op.state == 'succeeded' for op in state['operations'].values()):
                raise RuntimeError('injected after SQL writes')
        with patch.object(learning_rows, 'persist', side_effect=fail):
            with self.assertRaisesRegex(TranslationError, 'provider_failed'):
                self.normal()
        self.assertEqual(self.records(), [])
        self.assertEqual(translation_snapshot(self.docs.load_document('book'), self.block_id).records(), [])
        self.assertEqual(self.normal('retry').translation.status, 'completed')
        self.assertEqual(len(self.trans_provider.calls), 2)

if __name__ == '__main__': unittest.main()
