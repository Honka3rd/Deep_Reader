"""LLRN-16 actual HTTP/DI with file and disposable PostgreSQL owners."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, Thread
from time import monotonic, sleep
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from uuid import uuid4

from dependency_injector import providers
from fastapi import FastAPI
from fastapi.testclient import TestClient
import psycopg
from psycopg import sql

import api_learning_projection as project
from api_learning_schemas import LearningErrorResponse, LearningReadResponse, learning_source_token
from app.task_layout_readiness import TaskLayoutNotReadyError
from config.app_DI_config import AppDIConfig, LearningRuntimeConfig
from config.container import ApplicationLookupContainer
from document_structure.learning_repository import LearningCommitUnknown, LearningStorageError
from document_structure.structured_document_artifact_repository import StructuredDocumentArtifactRepository
from document_structure.reading_content import ReadingContentService
from learning_routes import router, learning_lifespan, get_learning_runtime
from reading_preferences.models import PreferenceCommitUnknown, PreferenceStorageError
from scripts.test_learning_file_repository import fixture, configure
from scripts.test_learning_container_wiring import OnceProvider
from scripts import test_reading_preferences_postgres as postgres_tests
from section_tasks.learning_generation_policy import AdmissionDecision, BatchPlan

BASE = '/documents/reading-interactions/learning/'
BOOK = '/documents/reading-preferences/'
WORKSPACE = '/reading-preferences/workspace'


def file_container(root):
    container = ApplicationLookupContainer.build(AppDIConfig(structured_storage_backend='file',
        learning_runtime=LearningRuntimeConfig(str(root/'prefs'), str(root/'learning'))))
    container.file_structured_document_artifact_repository.override(providers.Object(
        StructuredDocumentArtifactRepository(base_dir=str(root/'documents'), namespace='http-tests')))
    return container


class HttpCases:
    def prepare(self):
        self.provider = OnceProvider()
        self.container.llm_provider.override(providers.Object(self.provider))
        self.container.document_profile_store.override(providers.Object(SimpleNamespace(
            load_source=lambda config: SimpleNamespace(document_language_code='en'))))
        self.docs = self.container.structured_document_artifact_repository()
        self.prefs = self.container.reading_preferences_repository()
        self.repo = self.container.learning_repository()
        self.docs.save_reparsed_document(fixture(), 'book')
        configure(self.prefs)
        self.block = self.repo.read_sources('book').blocks[0]
        self.app = FastAPI(lifespan=learning_lifespan)
        self.app.include_router(router)
        self.app.state.learning_container = self.container
        self.client = TestClient(self.app)
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)
        self.service = self.app.state.learning_runtime.service

    def post(self, path, body, status=200):
        response = self.client.post(path, json=body)
        self.assertEqual(response.status_code, status, response.text)
        self.assertEqual(response.headers.get('Cache-Control'), 'no-store')
        if status != 200:
            LearningErrorResponse.model_validate(response.json())
        return response.json()

    def payload(self, command='generate', **kw):
        return dict(doc_name='book', content_block_id=self.block.source.content_block_id,
            component='translation', command_id=command, expected_source_token=learning_source_token(self.block.source),
            expected_workspace_revision=1, expected_book_revision=1, **kw)

    def read(self):
        return self.post(BASE+'read', {'doc_name': 'book', 'content_block_ids': [self.block.source.content_block_id]})

    def finish(self, operation='generate'):
        end = monotonic()+8
        while monotonic() < end:
            state = self.service.status('book', operation).state
            if state in {'succeeded', 'failed', 'unknown', 'stale'}:
                return state
            sleep(.01)
        self.fail('worker did not settle')

    def apply(self, revision=1, **kw):
        body = {'doc_name': 'book', 'expected_revision': revision, 'expected_workspace_revision': 1,
            'mutation_id': 'settings-'+str(revision), 'selection': {'mode_id': 'normal', 'version': '1'}}
        body.update(kw)
        return self.post(BOOK+'apply', body)

    def test_wait_route_returns_durable_completion_without_repeated_http_reads(self):
        result = self.post(BASE+'generate', self.payload('wait-once'))
        status = self.post(BASE+'operations/wait', {'handle': result['operation']})
        self.assertEqual(status['state'], 'succeeded')
        self.assertEqual(status['result_revision'], 1)
        self.assertEqual(len(self.provider.calls), 1)
        self.assertEqual(self.post(BASE+'operations/wait', {'handle': {'doc_name':'book','operation_id':'absent'}})['state'], 'missing')

    def test_all_read_routes_are_pure_and_provider_independent(self):
        before = self.snapshot()
        unavailable = Mock(side_effect=AssertionError('provider on read'))
        self.container.llm_provider.override(providers.Callable(unavailable))
        for path in ('/reading-preferences/options', WORKSPACE):
            response = self.client.get(path)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.headers['Cache-Control'], 'no-store')
        self.post(BOOK+'read', {'doc_name': 'book'})
        self.assertEqual([c['state'] for c in self.read()['blocks'][0]['components']], ['missing', 'missing'])
        self.assertEqual(self.post(BASE+'operations/read', {'handle': {'doc_name': 'book', 'operation_id': 'absent'}})['state'], 'missing')
        self.assertFalse(self.post(BASE+'commands/read', {'doc_name': 'book', 'command_id': 'absent'})['found'])
        self.assertEqual(before, self.snapshot())
        unavailable.assert_not_called()

    def test_generate_status_receipt_saved_and_reuse(self):
        result = self.post(BASE+'generate', self.payload())
        self.assertEqual(result['outcome'], 'claimed')
        self.assertIsNone(result['result_revision'])
        self.assertEqual(self.finish(), 'succeeded')
        state = self.post(BASE+'operations/read', {'handle': result['operation']})
        self.assertEqual(state['durable_state'], 'succeeded')
        self.assertIsNotNone(state['result_revision'])
        page = self.read()
        LearningReadResponse.model_validate(page)
        self.assertEqual(page['blocks'][0]['components'][0]['state'], 'ready')
        receipt = self.post(BASE+'commands/read', {'doc_name': 'book', 'command_id': 'generate'})
        self.assertEqual(receipt['receipt']['outcome'], 'claimed')
        self.container.llm_provider.override(providers.Object(None))
        reused = self.post(BASE+'generate', self.payload('reuse'))
        self.assertEqual(reused['outcome'], 'reuse')
        self.assertEqual(reused['result_revision'], state['result_revision'])
        self.assertEqual(len(self.provider.calls), 1)

    def test_hints_are_independent_and_empty_is_saved(self):
        body = self.payload('hints'); body['component'] = 'hints'
        self.post(BASE+'generate', body)
        self.assertEqual(self.finish('hints'), 'succeeded')
        components = self.read()['blocks'][0]['components']
        self.assertEqual([c['state'] for c in components], ['missing', 'ready'])
        self.assertEqual(components[1]['saved']['items'], [])

    def test_same_language_settings_and_generation_are_rejected_without_calls(self):
        response = self.client.put(WORKSPACE, json={'native_language': 'en', 'expected_revision': 1, 'mutation_id': 'native-en'})
        self.assertEqual(response.status_code, 200)
        settings = self.post(BOOK+'read', {'doc_name': 'book'})
        self.assertIsNone(settings['effective_mode'])
        body = self.payload();body['expected_workspace_revision'] = 2
        error = self.post(BASE+'generate', body, 422)['detail']
        self.assertEqual(error['kind'], 'ineligible')
        self.assertIn('same_language', [reason['code'] for reason in error['reasons']])
        selection = {'mode_id': 'language_learning', 'version': '1', 'language_code': 'en',
                     'scheme_id': 'cefr_reading', 'scheme_version': '1', 'level_id': 'B1'}
        error = self.post(BOOK+'apply', {'doc_name': 'book', 'expected_revision': 1, 'expected_workspace_revision': 2,
            'mutation_id': 'same', 'selection': selection}, 422)['detail']
        self.assertEqual(error['code'], 'preference_settings_invalid')
        self.assertEqual(self.provider.calls, [])

    def test_normal_mode_generation_has_explicit_reason(self):
        self.apply()
        error = self.post(BASE+'generate', self.payload(), 422)['detail']
        self.assertEqual(error['reasons'][0]['code'], 'learning_mode_not_active')

    def test_apply_preserves_learning_settings_and_override_omission(self):
        result = self.apply(source_language_override='fr')
        self.assertEqual(result['active_mode'], 'normal')
        self.assertEqual(result['saved_settings'][0]['selection']['level_id'], 'B1')
        unchanged = self.apply(2)
        self.assertEqual(unchanged['source_language_override'], 'fr')
        self.assertEqual(unchanged['revision'], 2)
        self.assertIsNone(self.apply(2, mutation_id='clear-override', source_language_override=None)['source_language_override'])
        self.assertEqual(self.provider.calls, [])

    def test_source_and_settings_cas_fail_before_generation(self):
        for field, value, code in [('expected_source_token', '0'*64, 'learning_source_changed'),
            ('expected_book_revision', 0, 'learning_settings_changed'),
            ('expected_workspace_revision', 0, 'learning_settings_changed')]:
            body = self.payload();body[field] = value
            self.assertEqual(self.post(BASE+'generate', body, 409)['detail']['code'], code)
        self.assertIsNone(self.repo.lookup_command('book', 'generate'))
        self.assertEqual(self.provider.calls, [])

    def test_preference_revision_conflicts(self):
        self.apply()
        error = self.post(BOOK+'apply', {'doc_name': 'book', 'expected_revision': 1, 'expected_workspace_revision': 1,
            'mutation_id': 'stale', 'selection': {'mode_id': 'normal', 'version': '1'}}, 409)
        self.assertEqual(error['detail']['code'], 'preference_revision_conflict')
        response = self.client.put(WORKSPACE, json={'native_language': 'fr', 'expected_revision': 0, 'mutation_id': 'stale'})
        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.headers['Cache-Control'], 'no-store')

    def test_missing_document_block_and_storage_failure_are_distinct(self):
        self.assertEqual(self.post(BOOK+'read', {'doc_name': 'absent'}, 404)['detail']['code'], 'learning_document_missing')
        body = self.payload(); body['content_block_id'] = 'absent'
        self.assertEqual(self.post(BASE+'generate', body, 404)['detail']['code'], 'learning_block_missing')
        with patch.object(self.repo, 'resolve_blocks', side_effect=LearningStorageError('private path')):
            error = self.post(BASE+'read', {'doc_name': 'book', 'content_block_ids': ['block']}, 503)
        self.assertEqual(error['detail']['kind'], 'unavailable')
        self.assertNotIn('private path', json.dumps(error))

    def test_strict_validation_is_sanitized_on_both_families(self):
        for path, body in [(BASE+'generate', self.payload()|{'raw_text': 'private content'}),
            (BOOK+'apply', {'doc_name': '../private content'}),
            (BASE+'read', {'doc_name': 'book', 'content_block_ids': [self.block.source.content_block_id]*2})]:
            error = self.post(path, body, 422)
            self.assertEqual(error['detail']['kind'], 'invalid_request')
            self.assertNotIn('private content', json.dumps(error))
        response = self.client.post(BASE+'generate', content='{private-invalid-json', headers={'Content-Type': 'application/json'})
        self.assertEqual(response.status_code, 422)
        self.assertNotIn('private-invalid', response.text)
        self.assertEqual(response.headers['Cache-Control'], 'no-store')

    def test_foreign_operation_and_command_handles_cannot_read_another_book(self):
        result = self.post(BASE+'generate', self.payload());self.finish()
        handle = result['operation']|{'doc_name': 'other-book'}
        self.assertEqual(self.post(BASE+'operations/read', {'handle': handle})['state'], 'missing')
        self.assertFalse(self.post(BASE+'commands/read', {'doc_name': 'other-book', 'command_id': 'generate'})['found'])
        self.assertEqual(self.post(BASE+'operations/reconcile', {'handle': handle})['outcome'], 'missing')

    def test_failed_refresh_retains_saved_and_other_component(self):
        self.post(BASE+'generate', self.payload());self.finish()
        saved = self.read()['blocks'][0]['components'][0]['saved']
        self.provider.failure = TimeoutError('private provider response')
        self.post(BASE+'generate', self.payload('refresh', intent='regenerate'));self.finish('refresh')
        component = self.read()['blocks'][0]['components'][0]
        self.assertEqual(component['saved'], saved)
        self.assertEqual(component['operation_state'], 'unknown')

    def test_provider_unavailable_preflight_writes_no_claim(self):
        self.container.llm_provider.override(providers.Object(None))
        error = self.post(BASE+'generate', self.payload(), 503)
        self.assertEqual(error['detail']['code'], 'learning_provider_unavailable')
        self.assertIsNone(self.repo.lookup_command('book', 'generate'))

    def test_admission_denial_is_durable_outcome_without_provider_call(self):
        with patch.object(self.container.learning_admission(), 'evaluate', return_value=AdmissionDecision(False, 'restricted')):
            result = self.post(BASE+'generate', self.payload())
        self.assertEqual(result['outcome'], 'denied')
        self.assertEqual(result['reason'], 'restricted')
        self.assertIsNone(result['result_revision'])
        self.assertEqual(self.provider.calls, [])

    def test_unknown_requires_explicit_duplicate_cost_ack_for_retry(self):
        self.provider.failure = TimeoutError('unknown remote outcome')
        result = self.post(BASE+'generate', self.payload());self.assertEqual(self.finish(), 'unknown')
        self.provider.failure = None
        body = self.payload('retry-without-ack', intent='retry', retry_operation=result['operation'])
        refused = self.post(BASE+'generate', body)
        self.assertEqual(refused['outcome'], 'reconcile')
        self.assertEqual(len(self.provider.calls), 1)
        body = self.payload('retry-with-ack', intent='retry', retry_operation=result['operation'],
                            acknowledge_possible_duplicate_cost=True)
        self.assertEqual(self.post(BASE+'generate', body)['outcome'], 'claimed')
        self.assertEqual(self.finish('retry-with-ack'), 'succeeded')
        self.assertEqual(len(self.provider.calls), 2)

    def test_partial_failure_exposes_only_durable_batches(self):
        translation = self.service.context.services['translation']
        original_prepare = translation.prepare
        original_call = self.provider.complete_text_once_with_output_budget
        def prepare(**kw):
            value = original_prepare(**kw)
            return replace(value, plan=BatchPlan(tuple((s.sentence_id,) for s in value.sentences.sentences)))
        def call(*args):
            if self.provider.calls:self.provider.failure = TimeoutError('second batch unknown')
            return original_call(*args)
        with patch.object(translation, 'prepare', side_effect=prepare), patch.object(self.provider, 'complete_text_once_with_output_budget', side_effect=call):
            result = self.post(BASE+'generate', self.payload());self.assertEqual(self.finish(), 'unknown')
        value = self.post(BASE+'operations/partial', {'handle': result['operation']})
        self.assertEqual(value['completed_batches'], 1)
        self.assertEqual(value['total_batches'], 2)
        self.assertEqual(len(value['batches']), 1)
        self.assertIsNone(self.read()['blocks'][0]['components'][0]['saved'])

    def test_lost_claim_ack_lookup_and_explicit_reconcile(self):
        claim = self.repo.claim
        def lose(*args, **kw):
            claim(*args, **kw)
            raise LearningCommitUnknown('private commit detail')
        with patch.object(self.repo, 'claim', side_effect=lose):
            result = self.post(BASE+'generate', self.payload())
        self.assertEqual(result['outcome'], 'unknown')
        self.assertIsNone(result['operation'])
        self.assertEqual(self.provider.calls, [])
        receipt = self.post(BASE+'commands/read', {'doc_name': 'book', 'command_id': 'generate'})['receipt']
        resumed = self.post(BASE+'operations/reconcile', {'handle': receipt['operation']})
        self.assertEqual(resumed['outcome'], 'resuming')
        self.assertEqual(self.finish(), 'succeeded')
        self.assertEqual(len(self.provider.calls), 1)

    def test_lost_preference_commit_ack_requires_read(self):
        save = self.prefs.save_workspace
        def lose(*args, **kw):
            save(*args, **kw)
            raise PreferenceCommitUnknown('private database detail')
        with patch.object(self.prefs, 'save_workspace', side_effect=lose):
            response = self.client.put(WORKSPACE, json={'native_language': 'fr', 'expected_revision': 1, 'mutation_id': 'native-fr'})
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()['detail']['recovery'], 'read_preferences')
        self.assertEqual(self.client.get(WORKSPACE).json()['native_language'], 'fr')

    def test_unexpected_response_failure_after_acceptance_is_unknown(self):
        with patch.object(project, 'submission_response', side_effect=RuntimeError('private internal details')):
            error = self.post(BASE+'generate', self.payload(), 503)
        self.assertEqual(error['detail'], {'kind': 'unknown', 'code': 'learning_commit_unknown', 'recovery': 'lookup_command'})
        self.assertEqual(self.finish(), 'succeeded')
        self.assertEqual(len(self.provider.calls), 1)

    def test_partial_is_committed_batches_with_coherent_metadata(self):
        result = self.post(BASE+'generate', self.payload());self.finish()
        partial = self.post(BASE+'operations/partial', {'handle': result['operation'], 'offset': 0, 'limit': 1})
        self.assertEqual(len(partial['batches']), 1)
        self.assertEqual(partial['completed_batches'], 1)
        self.assertNotIn('result_revision', partial)

    def test_partial_settings_race_is_conflict_not_stale_content(self):
        result = self.post(BASE+'generate', self.payload());self.finish()
        read = self.repo.read_checkpoints
        def race(*args, **kw):
            batches = read(*args, **kw)
            self.prefs.save_workspace('fr', expected_revision=1, mutation_id='race')
            return batches
        with patch.object(self.repo, 'read_checkpoints', side_effect=race):
            error = self.post(BASE+'operations/partial', {'handle': result['operation']}, 409)
        self.assertEqual(error['detail']['code'], 'learning_operation_conflict')

    def test_source_edit_makes_old_result_and_operation_stale(self):
        result = self.post(BASE+'generate', self.payload());self.finish()
        content = ReadingContentService(self.docs); state = content.read('book')
        content.mutate('book', action='edit', expected_revision=state['revision'], source_token=state['source_token'],
            mutation_id='edit', block_id=self.block.source.content_block_id, text='Changed source.', discard_interactions=False)
        status = self.post(BASE+'operations/read', {'handle': result['operation']})
        self.assertEqual(status['state'], 'stale')
        self.assertFalse(status['source_current'])
        self.assertIsNone(self.read()['blocks'][0]['components'][0]['saved'])

    def test_two_clients_coalesce_and_shutdown_owns_accepted_work(self):
        entered, release = Event(), Event()
        original = self.provider.complete_text_once_with_output_budget
        def block(*args):
            entered.set()
            if not release.wait(10):raise TimeoutError('test release missing')
            return original(*args)
        self.provider.complete_text_once_with_output_budget = block
        self.addCleanup(release.set)
        with ThreadPoolExecutor(2) as pool:
            first = pool.submit(self.post, BASE+'generate', self.payload('first'))
            second = pool.submit(self.post, BASE+'generate', self.payload('second'))
            results = [first.result(timeout=5), second.result(timeout=5)]
        self.assertEqual(sorted(r['outcome'] for r in results), ['claimed', 'coalesced'])
        self.assertTrue(entered.wait(5))
        release.set()
        winner = next(r for r in results if r['outcome']=='claimed')['operation']['operation_id']
        self.assertEqual(self.finish(winner), 'succeeded')
        self.assertEqual(len(self.provider.calls), 1)


class FileHttpTests(HttpCases, unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.container = file_container(self.root)
        self.prepare()

    def snapshot(self):
        return {str(p.relative_to(self.root)): p.read_bytes() for p in self.root.rglob('*') if p.is_file()}


class PostgresHttpTests(HttpCases, unittest.TestCase):
    setUpClass = classmethod(postgres_tests.PostgresPreferencesTests.setUpClass.__func__)
    drop_schema = classmethod(postgres_tests.PostgresPreferencesTests.drop_schema.__func__)

    def setUp(self):
        self.container = ApplicationLookupContainer.build(AppDIConfig(structured_storage_backend='postgres',
            postgres_dsn=self.scoped, postgres_namespace=uuid4().hex))
        self.prepare()

    def snapshot(self):
        with psycopg.connect(self.scoped) as conn:
            tables = conn.execute('SELECT tablename FROM pg_tables WHERE schemaname=current_schema() ORDER BY tablename').fetchall()
            return {table: conn.execute(sql.SQL('SELECT to_jsonb(t) FROM {} t ORDER BY to_jsonb(t)::text').format(sql.Identifier(table))).fetchall()
                    for table, in tables}


class ApplicationLifecycleTests(unittest.TestCase):
    def test_missing_runtime_and_options_remain_explicit(self):
        app = FastAPI(lifespan=learning_lifespan);app.include_router(router)
        with TestClient(app) as client:
            self.assertEqual(client.get('/reading-preferences/options').status_code, 200)
            response = client.get(WORKSPACE)
            self.assertEqual(response.status_code, 503)
            self.assertEqual(response.json()['detail']['code'], 'preference_storage_unavailable')

    def test_actual_main_registers_all_contracts_and_lifespan(self):
        with TemporaryDirectory() as directory:
            container = file_container(Path(directory))
            container.section_task_coordinator.override(providers.Object(Mock()))
            container.smart_cleanup_service.override(providers.Object(Mock()))
            container.translation_action.override(providers.Object(Mock()))
            provider = Mock(side_effect=AssertionError('startup generation dependency'))
            container.llm_provider.override(providers.Callable(provider))
            spec = importlib.util.spec_from_file_location('llrn16_main_acceptance', Path(__file__).resolve().parents[1]/'main.py')
            module = importlib.util.module_from_spec(spec)
            with patch('app.qa_coordinator.QACoordinator', return_value=SimpleNamespace(container=container)):
                spec.loader.exec_module(module)
            with TestClient(module.app) as client:
                self.assertEqual(client.get(WORKSPACE).json()['revision'], 0)
                schema = client.get('/openapi.json').json()
                for route in router.routes:
                    for method in route.methods:
                        operation = schema['paths'][route.path][method.lower()]
                        self.assertIn('LearningErrorResponse', json.dumps(operation['responses']['422']))
                self.assertIn('/documents/task-layout', schema['paths'])
                self.assertIn('/documents/reading-interactions/translate/read', schema['paths'])
                self.assertIs(module.app.state.learning_runtime, container.learning_runtime())
                provider.assert_not_called()
                self.assertEqual(list(Path(directory).rglob('*')), [])
                container.structured_document_artifact_repository().save_reparsed_document(fixture(), 'book')
                module._RAW_DOCUMENT_BASE_DIR = Path(directory)/'raw'
                module.section_task_coordinator.get_document_task_layout.side_effect = TaskLayoutNotReadyError('not_prepared')
                def snapshot():
                    return {str(p): p.read_bytes() for p in Path(directory).rglob('*') if p.is_file()}
                before = snapshot()
                self.assertEqual(client.get('/documents').status_code, 200)
                self.assertEqual(client.get('/documents/book/reading-content').status_code, 200)
                self.assertEqual(client.post('/documents/task-layout', json={'doc_name': 'book'}).status_code, 409)
                self.assertEqual(snapshot(), before)
                provider.assert_not_called()
            self.assertFalse(hasattr(module.app.state, 'learning_runtime'))

    def test_lifespan_waits_for_accepted_provider_work(self):
        with TemporaryDirectory() as directory:
            container = file_container(Path(directory));provider = OnceProvider()
            entered, release, closed = Event(), Event(), Event()
            original = provider.complete_text_once_with_output_budget
            def block(*args):
                entered.set()
                if not release.wait(10):raise TimeoutError('test release missing')
                return original(*args)
            provider.complete_text_once_with_output_budget = block
            container.llm_provider.override(providers.Object(provider))
            container.document_profile_store.override(providers.Object(SimpleNamespace(
                load_source=lambda config: SimpleNamespace(document_language_code='en'))))
            container.structured_document_artifact_repository().save_reparsed_document(fixture(), 'book')
            configure(container.reading_preferences_repository())
            block_source = container.learning_repository().read_sources('book').blocks[0].source
            app = FastAPI(lifespan=learning_lifespan);app.include_router(router);app.state.learning_container=container
            client = TestClient(app);client.__enter__()
            service = app.state.learning_runtime.service
            response = client.post(BASE+'generate', json={'doc_name': 'book', 'content_block_id': block_source.content_block_id,
                'component': 'translation', 'command_id': 'drain', 'expected_source_token': learning_source_token(block_source),
                'expected_workspace_revision': 1, 'expected_book_revision': 1})
            self.assertEqual(response.status_code, 200)
            self.assertTrue(entered.wait(5))
            thread = Thread(target=lambda: (client.__exit__(None, None, None), closed.set()))
            thread.start()
            try:
                self.assertFalse(closed.wait(.1))
            finally:
                release.set();thread.join(10)
            self.assertTrue(closed.is_set())
            self.assertEqual(service.status('book', 'drain').state, 'succeeded')
            self.assertEqual(len(provider.calls), 1)

    def test_lifespan_drains_and_can_start_again(self):
        with TemporaryDirectory() as directory:
            container = file_container(Path(directory))
            container.document_profile_store.override(providers.Object(SimpleNamespace(load_source=lambda config: None)))
            app = FastAPI(lifespan=learning_lifespan);app.include_router(router);app.state.learning_container=container
            with TestClient(app):
                first = app.state.learning_runtime
                closed = Mock(wraps=first.service.shutdown)
                first.service.shutdown = closed
            closed.assert_called_once_with(wait=True)
            with TestClient(app):
                self.assertIsNot(app.state.learning_runtime, first)
                self.assertIs(app.state.learning_runtime.service.repository, first.service.repository)


if __name__ == '__main__':
    unittest.main()
