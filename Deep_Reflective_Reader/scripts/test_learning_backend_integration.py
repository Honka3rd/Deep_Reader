"""LLRN-20 continuous HTTP/runtime/storage journeys; no paid provider calls.

Run through accept_learning_backend for deduplicated regression and hash evidence.
SQL uses the existing explicit disposable-schema fixture, never default app storage.
"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from uuid import uuid4

from dependency_injector import providers
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api_learning_schemas import learning_source_token
from config.app_DI_config import AppDIConfig
from config.container import ApplicationLookupContainer
from document_structure.learning_repository import LearningCommitUnknown
from document_structure.reading_content import ReadingContentService
from learning_routes import router, learning_lifespan
from scripts import test_learning_routes as http
from scripts.test_learning_container_wiring import OnceProvider
from scripts.test_learning_file_repository import fixture, configure
from scripts.test_learning_lifecycle import renamed


EVIDENCE = []


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    default=str).encode()).hexdigest()


class IntegrationJourneys:
    post = http.HttpCases.post
    finish = http.HttpCases.finish

    def prepare(self):
        self.provider = OnceProvider()
        self.profile = SimpleNamespace(document_language_code='en')
        self.client = None
        self.start()
        self.addCleanup(self.close)
        self.docs.save_reparsed_document(fixture(), 'book')
        configure(self.prefs)
        self.block_id = self.repo.read_sources('book').blocks[0].source.content_block_id
        self.steps = []
        self.record('initial')

    def start(self):
        self.container = self.new_container()
        self.container.llm_provider.override(providers.Object(self.provider))
        self.container.document_profile_store.override(providers.Object(SimpleNamespace(
            load_source=lambda config: self.profile)))
        self.docs = self.container.structured_document_artifact_repository()
        self.prefs = self.container.reading_preferences_repository()
        self.repo = self.container.learning_repository()
        app = FastAPI(lifespan=learning_lifespan)
        app.include_router(router)
        app.state.learning_container = self.container
        self.client = TestClient(app)
        self.client.__enter__()
        self.service = app.state.learning_runtime.service

    def close(self):
        if self.client is not None:
            self.client.__exit__(None, None, None)
            self.client = None

    def restart(self):
        self.close()
        self.start()

    def record(self, phase, doc='book'):
        value = {
            'phase': phase, 'calls': len(self.provider.calls),
            'source': digest(asdict(self.repo.read_sources(doc))),
            'profile': digest(vars(self.profile)),
            'settings': digest((asdict(self.prefs.read_workspace()), asdict(self.prefs.read_book(doc)))),
        }
        self.steps.append(value)
        return value

    def tearDown(self):
        self.close()
        EVIDENCE.append({'test': self.id(), 'adapter': self.adapter,
                         'provider_calls': len(self.provider.calls), 'steps': self.steps})

    def page(self, doc='book'):
        return self.post(http.BASE+'read', {'doc_name': doc, 'content_block_ids': [self.block_id]})

    def components(self, doc='book'):
        return self.page(doc)['blocks'][0]['components']

    def payload(self, component='translation', doc='book', **extra):
        source = self.repo.resolve_blocks(doc, (self.block_id,))[0].source
        return dict(doc_name=doc, content_block_id=self.block_id, component=component,
                    command_id=uuid4().hex, expected_source_token=learning_source_token(source),
                    expected_workspace_revision=self.prefs.read_workspace().revision,
                    expected_book_revision=self.prefs.read_book(doc).revision, **extra)

    def generate(self, component='translation', doc='book', **extra):
        body = self.payload(component, doc, **extra)
        receipt = self.post(http.BASE+'generate', body)
        self.assertEqual(receipt['outcome'], 'claimed')
        if doc == 'book':
            self.assertEqual(self.finish(body['command_id']), 'succeeded')
        return receipt

    def select(self, level='B1', source='en', mode='language_learning'):
        selection = {'mode_id': mode, 'version': '1'}
        if mode == 'language_learning':
            selection.update(language_code=source, scheme_id='cefr_reading', scheme_version='1', level_id=level)
        return self.post(http.BOOK+'apply', {
            'doc_name': 'book', 'expected_revision': self.prefs.read_book('book').revision,
            'expected_workspace_revision': self.prefs.read_workspace().revision,
            'mutation_id': uuid4().hex, 'source_language_override': source, 'selection': selection})

    def native(self, language):
        response = self.client.put(http.WORKSPACE, json={
            'native_language': language, 'expected_revision': self.prefs.read_workspace().revision,
            'mutation_id': uuid4().hex})
        self.assertEqual(response.status_code, 200, response.text)

    def change(self, action='edit'):
        content = ReadingContentService(self.docs)
        current = content.read('book')
        args = dict(action=action, expected_revision=current['revision'],
                    source_token=current['source_token'], mutation_id=uuid4().hex)
        if action == 'edit':
            args.update(block_id=self.block_id, text='Changed source.', discard_interactions=False)
        return content.mutate('book', **args)

    def held_provider(self):
        entered, release = Event(), Event()
        original = self.provider.complete_text_once_with_output_budget
        def call(*args):
            entered.set()
            if not release.wait(10):
                raise TimeoutError('test worker release missing')
            return original(*args)
        self.provider.complete_text_once_with_output_budget = call
        self.addCleanup(release.set)
        return entered, release

    def test_cache_dimensions_survive_settings_roundtrip_and_new_runtime(self):
        self.generate(); self.generate('hints')
        original = [c['saved'] for c in self.components()]
        self.select('C2')
        self.assertEqual([c['state'] for c in self.components()], ['ready', 'missing'])
        self.assertEqual(self.post(http.BASE+'generate', self.payload())['outcome'], 'reuse')
        self.generate('hints')
        self.select('B1')
        self.assertEqual([c['saved'] for c in self.components()], original)
        self.record('level-roundtrip')
        self.native('fr')
        self.assertEqual([c['state'] for c in self.components()], ['missing', 'missing'])
        self.generate('hints')  # Empty hints are valid in every explanation language.
        self.native('zh')
        self.assertEqual([c['saved'] for c in self.components()], original)
        self.select(source='fr')
        self.assertEqual([c['state'] for c in self.components()], ['missing', 'missing'])
        self.generate(); self.generate('hints')
        self.select(mode='normal')
        self.assertEqual(self.components(), [])
        self.select()
        self.restart()
        before = self.snapshot()
        self.assertEqual([c['saved'] for c in self.components()], original)
        for component in ('translation', 'hints'):
            # Saved reads must not perform provider preflight, even after restart.
            with self.container.llm_provider.override(providers.Object(None)):
                self.assertEqual(self.post(http.BASE+'generate', self.payload(component))['outcome'], 'reuse')
        # Reuse records command receipts; reads themselves remain byte/row pure.
        after_commands = self.snapshot()
        self.page(); self.page()
        self.assertEqual(after_commands, self.snapshot())
        self.assertNotEqual(before, after_commands)
        end = self.record('restart-reuse')
        self.assertEqual(end['source'], self.steps[0]['source'])
        self.assertEqual(end['profile'], self.steps[0]['profile'])
        self.assertEqual(len(self.provider.calls), 6)

    def test_all_levels_invalidated_after_edit_undo_and_reparse(self):
        self.generate()
        for level in ('A1', 'B1', 'C2'):
            self.select(level); self.generate('hints')
        old = self.page()['blocks'][0]['source_token']
        self.change(); self.change('undo')
        self.assertNotEqual(self.page()['blocks'][0]['source_token'], old)
        for level in ('A1', 'B1', 'C2'):
            self.select(level)
            self.assertEqual([c['state'] for c in self.components()], ['missing', 'missing'])
        self.assertEqual(len(self.provider.calls), 4)
        self.record('undo-invalidates-every-level')
        self.generate(); self.generate('hints')
        self.docs.save_reparsed_document(self.docs.load_document('book'), 'book')
        old_block_id = self.block_id
        self.block_id = self.repo.read_sources('book').blocks[0].source.content_block_id
        if old_block_id != self.block_id:
            self.post(http.BASE+'read', {'doc_name': 'book', 'content_block_ids': [old_block_id]}, 404)
        self.assertEqual([c['state'] for c in self.components()], ['missing', 'missing'])
        self.record('reparse-invalidates')

    def test_inflight_source_edit_then_undo_cannot_commit_old_output(self):
        entered, release = self.held_provider()
        body = self.payload()
        receipt = self.post(http.BASE+'generate', body)
        self.assertTrue(entered.wait(5))
        self.change(); self.change('undo')
        release.set()
        self.close(); self.start()  # Drain the old worker before inspecting persisted state.
        status = self.post(http.BASE+'operations/read', {'handle': receipt['operation']})
        self.assertEqual(status['state'], 'stale')
        self.assertFalse(status['source_current'])
        self.assertEqual([c['state'] for c in self.components()], ['missing', 'missing'])
        self.assertEqual(len(self.provider.calls), 1)
        self.record('late-output-fenced')

    def test_inflight_settings_roundtrip_fences_revision_even_when_key_matches(self):
        entered, release = self.held_provider()
        receipt = self.post(http.BASE+'generate', self.payload())
        self.assertTrue(entered.wait(5))
        self.select('C2'); self.select('B1')
        release.set(); self.restart()
        status = self.post(http.BASE+'operations/read', {'handle': receipt['operation']})
        self.assertFalse(status['settings_current'])
        self.assertEqual(status['state'], 'stale')
        self.assertIsNone(self.components()[0]['saved'])
        self.assertEqual(len(self.provider.calls), 1)
        self.record('settings-roundtrip-fenced')

    def test_unknown_restart_read_replay_and_consent_never_blindly_resend(self):
        self.provider.failure = TimeoutError('unknown external outcome')
        body = self.payload()
        receipt = self.post(http.BASE+'generate', body)
        self.assertEqual(self.finish(body['command_id']), 'unknown')
        self.restart(); self.provider.failure = None
        before = self.snapshot()
        self.page()
        self.post(http.BASE+'operations/read', {'handle': receipt['operation']})
        self.post(http.BASE+'commands/read', {'doc_name': 'book', 'command_id': body['command_id']})
        self.assertEqual(before, self.snapshot())
        replay = self.post(http.BASE+'generate', body)
        self.assertTrue(replay['replay'])
        self.post(http.BASE+'operations/reconcile', {'handle': receipt['operation']})
        retry = self.payload(intent='retry', retry_operation=receipt['operation'])
        self.assertEqual(self.post(http.BASE+'generate', retry)['outcome'], 'reconcile')
        self.assertEqual(len(self.provider.calls), 1)
        self.record('unknown-no-resend')
        self.generate(intent='retry', retry_operation=receipt['operation'], acknowledge_possible_duplicate_cost=True)
        self.assertEqual(len(self.provider.calls), 2)
        self.record('explicit-consented-retry')

    def test_lost_claim_ack_restart_concurrent_reconcile_dispatches_once(self):
        claim = self.repo.claim
        def lose(*args, **kw):
            claim(*args, **kw)
            raise LearningCommitUnknown('simulated acknowledgement loss')
        body = self.payload()
        with patch.object(self.repo, 'claim', side_effect=lose):
            self.assertEqual(self.post(http.BASE+'generate', body)['outcome'], 'unknown')
        self.restart()
        receipt = self.post(http.BASE+'commands/read', {'doc_name': 'book', 'command_id': body['command_id']})['receipt']
        self.assertEqual(len(self.provider.calls), 0)
        entered, release = self.held_provider()
        with ThreadPoolExecutor(2) as pool:
            futures = [pool.submit(self.post, http.BASE+'operations/reconcile', {'handle': receipt['operation']}) for _ in range(2)]
            results = [f.result(timeout=5) for f in futures]
        self.assertIn('resuming', [r['outcome'] for r in results])
        self.assertTrue(entered.wait(5))
        release.set()
        self.assertEqual(self.finish(body['command_id']), 'succeeded')
        self.assertEqual(len(self.provider.calls), 1)
        self.record('lost-ack-recovered-once')

    def test_rename_preserves_cache_delete_and_recreation_cannot_revive_it(self):
        self.generate(); self.generate('hints')
        saved = [c['saved'] for c in self.components()]
        self.assertTrue(self.docs.rename_document('book', 'renamed', renamed))
        page = self.page('renamed')['blocks'][0]
        current = [c['saved'] for c in page['components']]
        self.assertEqual(current[1], saved[1])
        self.assertEqual(current[0]['result_revision'], saved[0]['result_revision'])
        self.assertEqual([r['translated_text'] for r in current[0]['rows']],
                         [r['translated_text'] for r in saved[0]['rows']])
        self.assertEqual([r['sentence_id'] for r in current[0]['rows']],
                         [s['sentence_id'] for s in page['sentences'] if s['sentence_id']])
        self.assertNotEqual(current[0]['rows'][0]['sentence_id'], saved[0]['rows'][0]['sentence_id'])
        for component in ('translation', 'hints'):
            self.assertEqual(self.post(http.BASE+'generate', self.payload(component, 'renamed'))['outcome'], 'reuse')
        self.record('renamed', 'renamed')
        self.docs.delete_document('renamed')
        self.post(http.BOOK+'read', {'doc_name': 'renamed'}, 404)
        self.docs.save_reparsed_document(fixture('renamed'), 'renamed')
        self.block_id = self.repo.read_sources('renamed').blocks[0].source.content_block_id
        self.assertEqual(self.prefs.read_book('renamed').settings.active_mode, 'normal')
        self.assertEqual(self.components('renamed'), [])
        self.assertEqual(len(self.provider.calls), 2)
        self.record('recreated', 'renamed')


class FileBackendIntegration(IntegrationJourneys, unittest.TestCase):
    adapter = 'file'
    snapshot = http.FileHttpTests.snapshot

    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.prepare()

    def new_container(self):
        return http.file_container(self.root)


class PostgresBackendIntegration(IntegrationJourneys, unittest.TestCase):
    adapter = 'postgres'
    setUpClass = classmethod(http.PostgresHttpTests.setUpClass.__func__)
    drop_schema = classmethod(http.PostgresHttpTests.drop_schema.__func__)
    snapshot = http.PostgresHttpTests.snapshot

    def setUp(self):
        self.namespace = uuid4().hex
        self.prepare()

    def new_container(self):
        return ApplicationLookupContainer.build(AppDIConfig(structured_storage_backend='postgres',
            postgres_dsn=self.scoped, postgres_namespace=self.namespace))


if __name__ == '__main__':
    unittest.main()
