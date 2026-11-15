"""TASK-04: explicit default translation sources, real file/SQL and HTTP.

No paid providers; aliases must remain absent from primary source pagination.
"""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace
import hashlib
import json
import unittest
from unittest.mock import Mock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
import main
from app.section_task_coordinator import SectionTaskCoordinator
from document_structure.reading_content import original_blocks, ReadingContentService
from scripts import test_shared_translation as shared
from scripts import test_learning_file_repository as files
from scripts import test_learning_postgres_repository as sql
from section_tasks.translation_contracts import TranslationError
from shared.task_unit_model import TaskUnitContentBlock
from translation_routes import router

FIRST = 'A complete first paragraph with enough text for translation.'
SECOND = 'A different paragraph supplies independent evidence and a new conclusion.'
BODY = FIRST + '\n\n' + SECOND
BASE = '/documents/reading-interactions/translate/'


class CompatibilityCases:
    setup_shared = shared.SharedCases.setup_shared
    setup_service = shared.SharedCases.setup_service
    make_service = shared.SharedCases.make_service
    settings = shared.SharedCases.settings
    request = shared.SharedCases.request
    block_provider = shared.SharedCases.block_provider

    def seed(self, text=BODY):
        doc = self.docs.load_document('book')
        chapter = doc.chapters[0]; section = chapter.sections[0]
        unit = replace(section.task_units[0], content=text, content_blocks=[])
        section = replace(section, content=text, task_units=[unit], char_start=0, char_end=len(text))
        self.docs.save_reparsed_document(replace(doc, raw_text=text,
            chapters=[replace(chapter, sections=[section])]), 'book')
        self.unit = self.docs.load_document('book').chapters[0].sections[0].task_units[0]

    @contextmanager
    def clients(self):
        profiles = Mock(); profiles.exists.return_value = False
        coordinator = SectionTaskCoordinator(Mock(), self.docs, profiles, Mock(), Mock(), Mock(), Mock())
        app = FastAPI(); app.include_router(router); app.state.translation_action = self.action
        with patch.object(main, 'section_task_coordinator', coordinator), \
                TestClient(main.app) as content, TestClient(app) as translations:
            yield content, translations

    def blocks(self, client, mode=None, route='batch'):
        args = {} if mode is None else {'segmented': mode}
        if route == 'single':
            response = client.get('/documents/book/task-units/' + self.unit.unit_id + '/content', params=args)
        else:
            response = client.post('/documents/book/task-units/content', json={
                'task_unit_ids': [self.unit.unit_id], **args})
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        return (result if route == 'single' else result['contents'][0])['content_blocks']

    def read(self, client, block):
        response = client.post(BASE + 'read', json={'doc_name': 'book', 'content_block_id': block['block_id']})
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()['translations']

    def generate(self, client, block, command):
        response = client.post(BASE + 'generate', json={'doc_name': 'book',
            'content_block_id': block['block_id'], 'source_language': 'en',
            'target_language': 'zh', 'mutation_id': command})
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()['record']

    def assert_saved_source(self, block):
        page = self.repo.read_translations('book', block['block_id'])
        self.assertEqual(len(page.records), 1)
        saved = page.records[0].value
        self.assertEqual(saved.key.resource.source.effective_content_hash,
                         hashlib.sha256(block['content'].encode()).hexdigest())
        resolved, = self.repo.resolve_blocks('book', (block['block_id'],))
        self.assertEqual(resolved.effective_text, block['content'])
        self.assertEqual(saved.key.resource.source, resolved.source)
        self.assertEqual(self.repo.read_sentences('book', (saved.key.resource,))[0].state, 'ready')
        self.assertIsNotNone(self.repo.read_batch('book', (saved.key,))[0].view.saved)
        op = page.records[0].operation
        self.assertTrue(self.repo.read_checkpoints('book', op.operation_id))
        return saved

    def test_single_batch_modes_exact_source_saved_reuse_and_no_duplicate_enumeration(self):
        self.seed()
        with self.clients() as (content, client):
            for route in ('single', 'batch'):
                for mode in (None, False, True):
                    with self.subTest(route=route, segmented=mode):
                        block = self.blocks(content, mode, route)[0]
                        before = len(self.trans_provider.calls)
                        existing = self.read(client, block)
                        record = self.generate(client, block, route + str(mode))
                        self.assertEqual(len(self.trans_provider.calls), before + (not existing))
                        if not existing:
                            payload = json.loads(self.trans_provider.calls[-1][0].split('INPUT_JSON:\n')[1])
                            self.assertEqual(payload['context'], block['content'])
                        self.assert_saved_source(block)
                        self.block, = self.repo.resolve_blocks('book', (block['block_id'],))
                        self.assertEqual(self.request(command_id='learning-' + route + str(mode)).outcome, 'reuse')
                        self.assertEqual(len(self.trans_provider.calls), before + (not existing))
                        doc_before = self.docs.load_document('book').to_dict()
                        with patch.object(self.repo, 'claim', side_effect=AssertionError('read/reuse claim')), \
                                patch.object(self.trans_provider, 'get_model_capabilities', side_effect=AssertionError('read/reuse provider')):
                            self.assertEqual(self.read(client, block), [record])
                            self.assertEqual(self.generate(client, block, 'reuse-' + route + str(mode)), record)
                        self.assertEqual(self.docs.load_document('book').to_dict(), doc_before)
            primary = self.repo.read_sources('book').blocks
            self.assertEqual([b.effective_text for b in primary], [FIRST, SECOND])
            self.assertTrue(all(':compat:' not in b.source.content_block_id for b in primary))
            self.assertEqual(len(original_blocks(self.docs.load_document('book'))), 2)
            self.assertEqual(len(self.trans_provider.calls), 2)

    def test_projections_do_not_borrow_saved_translations_in_either_direction(self):
        for first_mode in (False, True):
            self.seed()
            with self.clients() as (content, client):
                full = self.blocks(content, False)[0]; paragraph = self.blocks(content, True)[0]
                self.assertNotEqual(full['block_id'], paragraph['block_id'])
                first, other = (paragraph, full) if first_mode else (full, paragraph)
                self.assertEqual(self.read(client, first), [])
                self.generate(client, first, 'first-' + str(first_mode))
                self.assertEqual(self.read(client, other), [])
                self.generate(client, other, 'other-' + str(first_mode))
                a = self.assert_saved_source(first); b = self.assert_saved_source(other)
                self.assertNotEqual(a.key, b.key)
                self.assertNotEqual(a.key.resource.source.effective_content_hash, b.key.resource.source.effective_content_hash)

    def test_heading_short_first_and_custom_colliding_block_sources(self):
        for case, text in enumerate(('Section\n' + BODY, 'Short.\n\n' + SECOND, BODY)):
            self.seed(text)
            if case == 2:
                doc = self.docs.load_document('book'); c = doc.chapters[0]; s = c.sections[0]
                custom = TaskUnitContentBlock(block_id=self.unit.unit_id + ':content:0',
                    content='Custom full source.\n\nDifferent second paragraph.', block_type='paragraph')
                self.docs.save_document(replace(doc, chapters=[replace(c, sections=[replace(s,
                    task_units=[replace(self.unit, content_blocks=[custom])])])]), 'book')
            with self.clients() as (content, client):
                for mode in (False, True):
                    with self.subTest(case=case, mode=mode):
                        block = self.blocks(content, mode)[0]
                        self.generate(client, block, f'case-{case}-{mode}')
                        payload = json.loads(self.trans_provider.calls[-1][0].split('INPUT_JSON:\n')[1])
                        self.assertEqual(payload['context'], block['content'])
                        self.assert_saved_source(block)

    def test_change_and_undo_do_not_restore_old_alias_binding(self):
        self.seed()
        with self.clients() as (content, client):
            alias = self.blocks(content, False)[0]
            self.generate(client, alias, 'before-edit')
            saved = self.assert_saved_source(alias)
            reader = ReadingContentService(self.docs)
            current = reader.read('book')
            paragraph = self.blocks(content, True)[0]
            reader.mutate('book', action='edit', expected_revision=current['revision'],
                source_token=current['source_token'], mutation_id='edit', block_id=paragraph['block_id'],
                text='Changed paragraph.', discard_interactions=False)
            current = reader.read('book')
            reader.mutate('book', action='undo', expected_revision=current['revision'],
                source_token=current['source_token'], mutation_id='undo')
            self.assertEqual(self.repo.read_batch('book', (saved.key,))[0].view.state, 'stale')
            # The restored full projection has the same ID/text, but a new epoch.
            restored = self.blocks(content, False)[0]
            self.assertEqual(restored['block_id'], alias['block_id'])
            self.assertEqual(restored['content'], alias['content'])
            self.assertEqual(self.read(client, restored), [])
            self.generate(client, restored, 'after-undo')
            self.assertNotEqual(self.assert_saved_source(restored).key, saved.key)

    def test_late_result_after_same_text_reparse_is_not_published(self):
        self.seed()
        with self.clients() as (content, client):
            alias = self.blocks(content, False)[0]
            entered, release = self.block_provider()
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(self.action.generate, doc_name='book', content_block_id=alias['block_id'],
                    source_language='en', target_language='zh', mutation_id='late')
                self.assertTrue(entered.wait(5))
                try:
                    self.seed()
                    after = self.docs.load_document('book').to_dict()
                finally:
                    release.set()
                with self.assertRaisesRegex(TranslationError, 'source_changed|target_missing'):
                    future.result(10)
            self.assertEqual(self.docs.load_document('book').to_dict(), after)
            self.assertEqual(self.read(client, self.blocks(content, False)[0]), [])
            self.assertEqual(len(self.trans_provider.calls), 1)

    def test_overlay_exact_whitespace_matches_content_provider_and_noop(self):
        self.seed()
        with self.clients() as (content, client):
            original = self.blocks(content, True)[0]
            reader = ReadingContentService(self.docs)
            current = reader.read('book')
            text = 'Exact  user text, punctuation. Another sentence.'
            reader.mutate('book', action='edit', expected_revision=current['revision'],
                source_token=current['source_token'], mutation_id='edit-whitespace',
                block_id=original['block_id'], text=text, discard_interactions=False)
            for mode in (None, False, True):
                for route in ('single', 'batch'):
                    block = self.blocks(content, mode, route)[0]
                    self.assertEqual(block['content'], text)
                    self.generate(client, block, f'overlay-{route}-{mode}')
                    self.assert_saved_source(block)
            payload = json.loads(self.trans_provider.calls[-1][0].split('INPUT_JSON:\n')[1])
            self.assertEqual(payload['context'], text)
            self.assertEqual(len(self.trans_provider.calls), 1)
            before = self.docs.load_document('book').to_dict()
            current = reader.read('book')
            reader.mutate('book', action='edit', expected_revision=current['revision'],
                source_token=current['source_token'], mutation_id='same-text',
                block_id=original['block_id'], text=text, discard_interactions=False)
            after = self.docs.load_document('book').to_dict()
            # Explicit no-op commands retain receipts; body/revision/history do not change.
            state = after['parse_provenance']['reading_content']
            self.assertIn('same-text', state['receipts'])
            self.assertIn('same-text', state['no_op_mutations'])
            for document in (before, after):
                state = document['parse_provenance']['reading_content']
                state.pop('receipts', None)
                state.pop('no_op_mutations', None)
            self.assertEqual(after, before)

    def test_concurrent_default_requests_coalesce_and_conflicting_receipt_rejects(self):
        self.seed()
        with self.clients() as (content, client):
            alias = self.blocks(content, False)[0]
            paragraph = self.blocks(content, True)[0]
            entered, release = self.block_provider()
            with ThreadPoolExecutor(max_workers=2) as pool:
                first = pool.submit(self.generate, client, alias, 'one')
                self.assertTrue(entered.wait(5))
                second = pool.submit(self.generate, client, alias, 'two')
                release.set()
                self.assertEqual(first.result(10), second.result(10))
            self.assertEqual(len(self.trans_provider.calls), 1)
            before = self.assert_saved_source(alias)
            response = client.post(BASE + 'generate', json={'doc_name': 'book',
                'content_block_id': paragraph['block_id'], 'source_language': 'en',
                'target_language': 'zh', 'mutation_id': 'one'})
            self.assertEqual(response.status_code, 409, response.text)
            self.assertEqual(self.assert_saved_source(alias), before)
            self.assertEqual(self.read(client, paragraph), [])
            self.assertEqual(len(self.trans_provider.calls), 1)

    def test_failed_refresh_preserves_previous_alias_result(self):
        self.seed()
        with self.clients() as (content, client):
            alias = self.blocks(content, False)[0]
            record = self.generate(client, alias, 'first')
            self.block, = self.repo.resolve_blocks('book', (alias['block_id'],))
            previous = self.assert_saved_source(alias).view.saved
            with self.publication_failure():
                self.request(command_id='refresh', intent='regenerate')
                self.service.shutdown()
            self.assertEqual(self.service.status('book', 'refresh').state, self.failed_publication_state)
            self.assertEqual(self.assert_saved_source(alias).view.saved, previous)
            self.assertEqual(self.read(client, alias), [record])
            self.assertEqual(len(self.trans_provider.calls), 2)

    def test_alias_results_follow_explicit_rename_lifecycle_without_regeneration(self):
        from document_structure.learning_file_lifecycle import FileLearningDocumentLifecycle
        from shared.learning_source import LearningContractError
        if hasattr(self.repo, '_commit'):
            FileLearningDocumentLifecycle(self.docs, self.repo, self.prefs)
        self.seed()
        with self.clients() as (content, client):
            alias = self.blocks(content, False)[0]
            record = self.generate(client, alias, 'before-rename')
            saved = self.assert_saved_source(alias)
            transform = lambda doc, name: replace(doc, document_id=name, title=name)
            self.assertTrue(self.docs.rename_document('book', 'renamed', transform))
            page = self.repo.read_translations('renamed', alias['block_id'])
            self.assertEqual(len(page.records), 1)
            current = page.records[0].value
            self.assertEqual(current.key.resource.source.document_id, 'renamed')
            self.assertEqual(current.key.resource.source.effective_content_hash,
                             saved.key.resource.source.effective_content_hash)
            self.assertEqual([r.translated_text for r in current.view.saved.rows],
                             [r.translated_text for r in saved.view.saved.rows])
            with self.assertRaises(LearningContractError):
                self.repo.read_batch('renamed', (saved.key,))
            self.assertTrue(self.docs.rename_document('renamed', 'book', transform))
            self.assertEqual(self.read(client, alias), [record])
            self.assertEqual(len(self.trans_provider.calls), 1)
            self.assertTrue(all(':compat:' not in b.source.content_block_id
                                for b in self.repo.read_sources('book').blocks))

    def test_legacy_paragraph_translation_alias_remains_readable(self):
        from app.translation_action import TranslationAction
        from section_tasks.translation_contracts import TranslationResult
        from types import SimpleNamespace
        self.seed()
        with self.clients() as (content, client):
            alias = self.blocks(content, False)[0]
            legacy = TranslationAction(self.docs, self.action._profiles, SimpleNamespace(translate=lambda **kw:
                TranslationResult('en', 'zh', 'completed', '舊段落譯文。')))
            app = FastAPI(); app.include_router(router); app.state.translation_action = legacy
            with TestClient(app) as legacy_client:
                self.generate(legacy_client, alias, 'legacy')
            self.assertEqual(self.read(client, alias)[0]['translated_text'], '舊段落譯文。')
            self.generate(client, alias, 'legacy-reuse')
            self.assertFalse(self.trans_provider.calls)
            self.assertEqual(self.read(client, self.blocks(content, True)[0]), [])


class CompatibilityFileTests(CompatibilityCases, unittest.TestCase):
    # Storage failures retain recoverable partial state; reads do not retry.
    failed_publication_state = 'partial'
    @contextmanager
    def publication_failure(self):
        from document_structure.learning_repository import LearningStorageError
        commit = self.repo._commit
        def fail(doc, state):
            if any(op.operation_id == 'refresh' and op.state == 'succeeded' for op in state['operations'].values()):
                raise LearningStorageError('injected before replace')
            return commit(doc, state)
        with patch.object(self.repo, '_commit', side_effect=fail):
            yield

    def setUp(self):
        files.LearningFileRepositoryTests.setUp(self)
        self.setup_shared()


class CompatibilitySqlTests(CompatibilityCases, unittest.TestCase):
    failed_publication_state = 'failed'
    @contextmanager
    def publication_failure(self):
        from db import learning_rows
        persist = learning_rows.persist
        def fail(connection, row_id, scope, doc, state, previous):
            persist(connection, row_id, scope, doc, state, previous)
            if any(op.operation_id == 'refresh' and op.state == 'succeeded' for op in state['operations'].values()):
                raise RuntimeError('injected after SQL writes')
        with patch.object(learning_rows, 'persist', side_effect=fail):
            yield

    setUpClass = classmethod(sql.LearningPostgresTests.setUpClass.__func__)
    drop_schema = classmethod(sql.LearningPostgresTests.drop_schema.__func__)
    def setUp(self):
        sql.LearningPostgresTests.setUp(self)
        self.setup_shared()


if __name__ == '__main__':
    unittest.main()
