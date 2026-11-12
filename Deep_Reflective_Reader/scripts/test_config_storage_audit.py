"""CONFIG-01/02/03: source identity, destination conflicts and pure reads."""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import os
import unittest

from config.faiss_storage_config import FaissStorageConfig
from config.structured_document_storage_config import StructuredDocumentStorageConfig
from document_structure.structured_document_artifact_repository import StructuredDocumentArtifactRepository
from scripts.test_learning_file_repository import fixture


def snapshot(root):
    return {str(p.relative_to(root)): p.read_bytes() if p.is_file() else None
            for p in root.rglob('*')}


class ConfigPathTests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory(); self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        previous = Path.cwd(); os.chdir(self.root); self.addCleanup(os.chdir, previous)

    def test_missing_read_configuration_does_not_create_directories(self):
        before = snapshot(self.root)
        self.assertFalse(FaissStorageConfig('Book.pdf').exists())
        self.assertFalse(StructuredDocumentStorageConfig('Book.pdf').exists())
        self.assertEqual(snapshot(self.root), before)

    def test_existing_qualified_paths_are_not_moved_by_reads(self):
        directory = self.root/'data/faiss/Book.pdf'; directory.mkdir(parents=True)
        (directory/'profile.json').write_text('{}')
        file = self.root/'data/structured/Book.pdf.structured.json'
        file.parent.mkdir(parents=True); file.write_text('{}')
        before = snapshot(self.root)
        self.assertEqual(Path(FaissStorageConfig('Book.pdf').get_raw_profile_path()), Path('data/faiss/Book.pdf/profile.json'))
        self.assertEqual(Path(StructuredDocumentStorageConfig('Book.pdf').get_raw_document_path()), Path('data/structured/Book.pdf.structured.json'))
        self.assertEqual(snapshot(self.root), before)

    def test_same_stem_artifacts_keep_distinct_body(self):
        docs = StructuredDocumentArtifactRepository()
        for name in ('Book.pdf', 'Book.txt'):
            docs.save_reparsed_document(fixture(name), name)
        for name in ('Book.pdf', 'Book.txt'):
            self.assertEqual(docs.load_document(name).document_id, name)
        self.assertEqual(len(list(Path('data/structured').glob('*.structured.json'))), 2)

    def test_layout_content_profile_reads_preserve_legacy_coexisting_and_missing_paths(self):
        from fastapi.testclient import TestClient
        from profile.document_profile_store import DocumentProfileStore
        import main
        from scripts import test_task_unit_content_endpoint as content
        for scenario in ('legacy_only', 'coexisting', 'no_profile_directory'):
            with self.subTest(scenario=scenario):
                # Separate roots make every directory creation observable.
                case = self.root/scenario; case.mkdir(); os.chdir(case)
                docs = StructuredDocumentArtifactRepository()
                original = replace(content._build_cache_valid_document(), document_id='Reader.pdf')
                docs.save_document(original, 'Reader.pdf')
                if scenario != 'no_profile_directory':
                    folder = Path('data/faiss/Reader.pdf'); folder.mkdir(parents=True)
                    (folder/'profile.json').write_text('{"topic":"preserved","language":"en","summary":"old"}')
                if scenario == 'coexisting':
                    docs.save_reparsed_document(replace(original, document_id='Reader'), 'Reader')
                    old = Path('data/faiss/Reader'); old.mkdir()
                    (old/'profile.json').write_text('{"topic":"other","language":"en","summary":"keep"}')
                coordinator, _ = content._build_coordinator(original)
                coordinator.document_artifact_repository = docs
                coordinator.document_preparation_pipeline.loader_factory = None
                coordinator.document_profile_store = DocumentProfileStore()
                before = snapshot(case)
                with patch.object(main, 'section_task_coordinator', coordinator), TestClient(main.app) as client:
                    profile = coordinator._load_existing_document_profile('Reader.pdf')
                    self.assertEqual(profile.topic if profile else None,
                                     None if scenario == 'no_profile_directory' else 'preserved')
                    response = client.post('/documents/task-layout', json={'doc_name':'Reader.pdf'})
                    self.assertEqual(response.status_code, 200, response.text)
                    response = client.get('/documents/Reader.pdf/reading-content', params={'include_blocks':True})
                    self.assertEqual(response.status_code, 200, response.text)
                    response = client.post('/documents/Reader.pdf/task-units/content', json={
                        'task_unit_ids':['task-unit-1','task-unit-2']})
                    self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(snapshot(case), before)
                os.chdir(self.root)

    def test_explicit_profile_write_creates_parent_after_pure_config(self):
        from profile.document_profile import DocumentProfile
        from profile.document_profile_store import DocumentProfileStore
        config = FaissStorageConfig('New.txt')
        self.assertFalse(Path('data').exists())
        profile = DocumentProfile.from_dict({'topic':'test', 'language':'en', 'summary':'summary'})
        DocumentProfileStore.save(profile, config)
        self.assertEqual(DocumentProfileStore.load(config), profile)



from dataclasses import replace
from urllib.parse import quote
import main
from scripts import test_learning_lifecycle as lifecycle
from scripts import test_learning_postgres_repository as postgres
from scripts import test_main_rename_audit as rename
from scripts import test_task_unit_content_endpoint as content
from section_tasks.note_interaction_service import NoteInteractionService


class StorageHttpCases:
    def prepare_http(self):
        rename.RenameCases.http_setup(self)
        if self.docs.lifecycle_storage_identity[0] == 'file':
            self.docs.raw_dir = self.raw
        else:
            self.docs.store.raw_root = self.raw
        coordinator, _ = content._build_coordinator(fixture())
        coordinator.document_artifact_repository = self.docs
        coordinator.document_preparation_pipeline.loader_factory = None
        setting = patch.object(main, 'section_task_coordinator', coordinator)
        setting.start(); self.addCleanup(setting.stop)

    def persist(self, name, text):
        original = rename.different_book(text)
        self.docs.save_reparsed_document(replace(original, document_id=name), name)
        notes = NoteInteractionService(self.docs)
        view = notes.read(name, target_level='document')
        notes.save(name, target_level='document', text='Note: '+text, expected_revision=0,
                   mutation_id='save-'+name, target_token=view['target_token'])

    def body(self, name):
        return self.client.get('/documents/'+quote(name, safe='')+'/reading-content',
                               params={'include_blocks': True})

    def check_source(self, name, text):
        response = self.body(name)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertIn(text, response.text)
        response = self.client.post('/documents/reading-interactions/note/read', json={
            'target': {'doc_name':name, 'target_type':'document'}})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()['note']['text'], 'Note: '+text)

    def upload(self, filename, data=b'text', policy='reject'):
        return self.client.post('/documents/upload', files={'file':(filename, data, 'application/octet-stream')},
                                data={'conflict_policy':policy})

    def storage_snapshot(self):
        if self.docs.lifecycle_storage_identity[0] == 'file':
            return snapshot(self.root)
        # Real SQL rows, including preferences, learning, receipts and revisions.
        # This fixture owns a disposable schema; no production data is inspected.
        import psycopg
        from psycopg import sql
        with psycopg.connect(self.scoped) as connection:
            tables = connection.execute("SELECT tablename FROM pg_tables WHERE schemaname=current_schema()").fetchall()
            return {name: connection.execute(sql.SQL(
                'SELECT row_to_json(t)::text FROM {} AS t ORDER BY 1'
            ).format(sql.Identifier(name))).fetchall() for (name,) in tables}

    def legacy_rename_conflict(self, *, raw_source=True):
        self.persist('Legacy', 'original legacy body')
        self.persist('Source.txt', 'original source body')
        (self.raw/'Legacy.pdf').write_bytes(b'original PDF source')
        if raw_source:
            (self.raw/'Source.txt').write_bytes(b'original TXT source')
        for name in ('Legacy', 'Source.txt'):
            directory = self.derived/name; directory.mkdir()
            (directory/'index.faiss').write_bytes(('index '+name).encode())
            (directory/'profile.json').write_text('{"topic":"preserved"}')
        command = rename.RenameCases.rename_command(self, 'Source.txt', 'Legacy.txt')
        before_files, before_storage = snapshot(self.files), self.storage_snapshot()
        response = self.client.post('/documents/lifecycle/rename', json=command)
        self.assertEqual(response.status_code, 409, response.text)
        self.assertEqual(response.json()['status'], 'conflict')
        self.assertEqual(response.json()['reason'], 'conflict')
        self.assertEqual(snapshot(self.files), before_files)
        self.assertEqual(self.storage_snapshot(), before_storage)
        self.check_source('Legacy', 'original legacy body')
        self.check_source('Legacy.pdf', 'original legacy body')
        self.check_source('Source.txt', 'original source body')
        self.assertEqual(snapshot(self.files), before_files)
        self.assertEqual(self.storage_snapshot(), before_storage)

    def test_legacy_destination_rename_rejects_before_any_mutation(self):
        self.legacy_rename_conflict()

    def test_legacy_destination_repository_only_rename_rejects_before_any_mutation(self):
        self.legacy_rename_conflict(raw_source=False)

    def test_legacy_destination_conflict_over_tcp(self):
        rename.RenameCases.over_tcp(self, self.legacy_rename_conflict)

    def test_qualified_destination_sibling_allows_rename_and_receipt_replay(self):
        for name, text in (('Twin.pdf', 'PDF body'), ('Source.txt', 'TXT body')):
            self.persist(name, text)
            (self.raw/name).write_bytes(text.encode())
        pdf = self.docs.load_document('Twin.pdf').to_dict()
        command = rename.RenameCases.rename_command(self, 'Source.txt', 'Twin.txt')
        result = self.client.post('/documents/lifecycle/rename', json=command)
        self.assertEqual(result.status_code, 200, result.text)
        before_files, before_storage = snapshot(self.files), self.storage_snapshot()
        repeated = self.client.post('/documents/lifecycle/rename', json=command)
        self.assertEqual(repeated.status_code, 200, repeated.text)
        self.assertEqual(repeated.json()['operation_id'], result.json()['operation_id'])
        self.assertEqual(snapshot(self.files), before_files)
        self.assertEqual(self.storage_snapshot(), before_storage)
        self.check_source('Twin.pdf', 'PDF body'); self.check_source('Twin.txt', 'TXT body')
        self.assertEqual(self.docs.load_document('Twin.pdf').to_dict(), pdf)
        self.assertEqual((self.raw/'Twin.pdf').read_bytes(), b'PDF body')
        self.assertEqual((self.raw/'Twin.txt').read_bytes(), b'TXT body')
        self.assertFalse((self.raw/'Source.txt').exists())

    def test_destination_conflict_is_rechecked_under_lock_before_journal(self):
        from contextlib import contextmanager
        from app.document_rename import DocumentRename
        self.persist('Source.txt', 'source body')
        (self.raw/'Source.txt').write_bytes(b'source')
        command = rename.RenameCases.rename_command(self, 'Source.txt', 'Legacy.txt')
        locked = DocumentRename._locked
        before = {}

        @contextmanager
        def occupied_while_waiting(operation):
            with locked(operation):
                # Another owner committed before this command acquired the lock.
                self.persist('Legacy', 'legacy body')
                (self.raw/'Legacy.pdf').write_bytes(b'legacy PDF')
                before['files'] = snapshot(self.files)
                before['storage'] = self.storage_snapshot()
                yield

        with patch.object(DocumentRename, '_locked', occupied_while_waiting), \
                patch.object(DocumentRename, '_save') as journal_write:
            response = self.client.post('/documents/lifecycle/rename', json=command)
            self.assertEqual(response.status_code, 409, response.text)
            journal_write.assert_not_called()
        self.assertEqual(snapshot(self.files), before['files'])
        self.assertEqual(self.storage_snapshot(), before['storage'])
        self.check_source('Legacy', 'legacy body'); self.check_source('Source.txt', 'source body')

    def test_same_command_finishing_during_preflight_reconciles_its_receipt(self):
        from app.document_rename import DocumentRename
        self.persist('Source.txt', 'source body')
        (self.raw/'Source.txt').write_bytes(b'source')
        command = rename.RenameCases.rename_command(self, 'Source.txt', 'Moved.txt')
        check = DocumentRename._check_destination
        completed = []

        def complete_during_check(operation, new, raw_target):
            # Deterministic interleaving: the other copy finishes after the first
            # receipt-existence check, before destination occupancy is read.
            with patch.object(DocumentRename, '_check_destination', check):
                result = self.client.post('/documents/lifecycle/rename', json=command)
            self.assertEqual(result.status_code, 200, result.text)
            completed.append((result.json(), snapshot(self.files), self.storage_snapshot()))
            return check(operation, new, raw_target)

        with patch.object(DocumentRename, '_check_destination', complete_during_check):
            response = self.client.post('/documents/lifecycle/rename', json=command)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(len(completed), 1)
        result, files, storage = completed[0]
        self.assertEqual(response.json()['operation_id'], result['operation_id'])
        self.assertEqual(snapshot(self.files), files)
        self.assertEqual(self.storage_snapshot(), storage)

    def test_same_stem_http_body_notes_index_and_lifecycle_are_isolated(self):
        for name, text in [('Twin.pdf', 'PDF only'), ('Twin.txt', 'TXT only')]:
            (self.raw/name).write_bytes(text.encode())
            self.persist(name, text)
            directory = self.derived/name; directory.mkdir()
            (directory/'profile.json').write_text(text)
        before = snapshot(self.files)
        self.check_source('Twin.pdf', 'PDF only'); self.check_source('Twin.txt', 'TXT only')
        self.assertEqual(snapshot(self.files), before)
        command = rename.RenameCases.rename_command(self, 'Twin.pdf', 'Moved.pdf')
        result = self.client.post('/documents/lifecycle/rename', json=command)
        self.assertEqual(result.status_code, 200, result.text)
        self.assertEqual(self.client.post('/documents/lifecycle/rename', json=command).json()['operation_id'],
                         result.json()['operation_id'])
        self.check_source('Moved.pdf', 'PDF only'); self.check_source('Twin.txt', 'TXT only')
        self.assertFalse((self.raw/'Twin.pdf').exists())
        self.assertEqual((self.raw/'Moved.pdf').read_bytes(), b'PDF only')
        self.assertEqual((self.derived/'Moved.pdf/profile.json').read_text(), 'PDF only')
        result = self.client.post('/documents/lifecycle/delete', json={
            'doc_name':'Moved.pdf', 'action':'delete', 'confirm':True})
        self.assertEqual(result.status_code, 200, result.text)
        self.check_source('Twin.txt', 'TXT only')
        self.assertEqual((self.raw/'Twin.txt').read_bytes(), b'TXT only')
        self.assertEqual((self.derived/'Twin.txt/profile.json').read_text(), 'TXT only')
        self.assertFalse((self.raw/'Moved.pdf').exists())

    def test_upload_distinct_formats_keep_stable_names_and_exact_conflicts(self):
        for name in ('New.txt', 'New.pdf'):
            result = self.upload(name, name.encode())
            self.assertEqual(result.status_code, 201, result.text)
            self.assertEqual(result.json()['doc_name'], name)
        response = self.upload('New.txt', b'bad')
        self.assertEqual(response.status_code, 409, response.text)
        response = self.upload('New.txt', b'changed', 'replace')
        self.assertEqual(response.status_code, 201, response.text)
        self.assertEqual((self.raw/'New.txt').read_bytes(), b'changed')
        self.assertEqual((self.raw/'New.pdf').read_bytes(), b'New.pdf')
        response = self.upload('New.txt', b'renamed', 'create_distinct')
        self.assertEqual(response.json()['doc_name'], 'New (1).txt', response.text)

    def test_legacy_owner_cannot_be_adopted_by_other_format_upload(self):
        self.persist('Legacy', 'old body'); (self.raw/'Legacy.pdf').write_bytes(b'old PDF')
        before = self.docs.load_document('Legacy').to_dict()
        for policy in ('reject', 'replace'):
            response = self.upload('Legacy.txt', b'new TXT', policy)
            self.assertEqual(response.status_code, 409, response.text)
            self.assertFalse((self.raw/'Legacy.txt').exists())
        result = self.upload('Legacy.txt', b'new TXT', 'create_distinct')
        self.assertEqual(result.json()['doc_name'], 'Legacy (1).txt', result.text)
        self.assertEqual(self.docs.load_document('Legacy').to_dict(), before)
        self.check_source('Legacy', 'old body')
        self.check_source('Legacy.pdf', 'old body')

    def test_legacy_source_alias_rename_replay_uses_persisted_identity(self):
        self.persist('Alias', 'old body'); (self.raw/'Alias.txt').write_bytes(b'old source')
        command = rename.RenameCases.rename_command(self, 'Alias.txt', 'Moved.txt')
        first = self.client.post('/documents/lifecycle/rename', json=command)
        self.assertEqual(first.status_code, 200, first.text)
        repeated = self.client.post('/documents/lifecycle/rename', json=command)
        self.assertEqual(repeated.status_code, 200, repeated.text)
        self.assertEqual(first.json()['operation_id'], repeated.json()['operation_id'])
        self.check_source('Moved.txt', 'old body')
        # Historical command cannot adopt a replacement source, even same bytes.
        (self.raw/'Alias.txt').write_bytes(b'old source')
        rejected = self.client.post('/documents/lifecycle/rename', json=command)
        self.assertEqual(rejected.status_code, 409, rejected.text)
        self.assertEqual((self.raw/'Alias.txt').read_bytes(), b'old source')

    def test_ambiguous_legacy_sources_fail_closed_without_mutation(self):
        self.persist('Legacy', 'unknown owner')
        for name in ('Legacy.pdf', 'Legacy.txt'):
            (self.raw/name).write_bytes(name.encode())
        before = snapshot(self.files)
        for name in ('Legacy', 'Legacy.pdf', 'Legacy.txt'):
            self.assertGreaterEqual(self.body(name).status_code, 400)
        response = self.client.post('/documents/lifecycle/delete', json={
            'doc_name':'Legacy', 'action':'delete', 'confirm':True})
        self.assertGreaterEqual(response.status_code, 400, response.text)
        # A rejected delete may acquire its coordination lock; raw/body never change.
        self.assertEqual(snapshot(self.raw), {k.removeprefix('raw/'):v for k,v in before.items() if k.startswith('raw/')})
        self.assertEqual(self.docs.resolve_document_name('Legacy.pdf'), 'Legacy.pdf')

    def test_postgres_or_file_exact_names_ignore_unrelated_derived_alias(self):
        self.persist('Exact.txt', 'new text')
        old = self.derived/'Exact'; old.mkdir(); (old/'profile.json').write_text('unrelated')
        config = FaissStorageConfig('Exact.txt', base_root=self.derived,
                                    identity_resolver=self.docs.resolve_document_name)
        self.assertEqual(config.doc_name, 'Exact.txt')
        self.assertFalse(Path(config.get_raw_profile_path()).exists())
        self.assertEqual((old/'profile.json').read_text(), 'unrelated')

    def test_legacy_index_path_is_read_without_migration_and_explicitly_moved(self):
        self.persist('Old', 'legacy body'); (self.raw/'Old.txt').write_bytes(b'old text')
        legacy = self.derived/'Old.txt'; legacy.mkdir()
        (legacy/'profile.json').write_text('{"topic":"old","language":"en","summary":"keep"}')
        (legacy/'index.faiss').write_bytes(b'old index')
        before = snapshot(self.files)
        from profile.document_profile_store import DocumentProfileStore
        config = FaissStorageConfig('Old.txt', base_root=self.derived,
                                    identity_resolver=self.docs.resolve_document_name)
        self.assertEqual(config.doc_name, 'Old')
        self.assertEqual(config.base_dir, legacy)
        self.assertEqual(DocumentProfileStore.load(config).topic, 'old')
        self.assertEqual(snapshot(self.files), before)
        command = rename.RenameCases.rename_command(self, 'Old.txt', 'New')
        response = self.client.post('/documents/lifecycle/rename', json=command)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertFalse(legacy.exists())
        self.assertEqual((self.derived/'New/index.faiss').read_bytes(), b'old index')
        repeated = self.client.post('/documents/lifecycle/rename', json=command)
        self.assertEqual(repeated.status_code, 200, repeated.text)
        self.assertEqual(repeated.json()['operation_id'], response.json()['operation_id'])
        self.check_source('New', 'legacy body')
        response = self.client.post('/documents/lifecycle/delete', json={
            'doc_name':'New', 'action':'delete', 'confirm':True})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertFalse((self.derived/'New').exists())

    def test_retired_legacy_name_does_not_rebind_to_remaining_qualified_book(self):
        self.persist('Mixed', 'old bare book')
        self.persist('Mixed.txt', 'separate qualified book')
        other = self.docs.load_document('Mixed.txt').to_dict()
        command = rename.RenameCases.rename_command(self, 'Mixed', 'Moved')
        response = self.client.post('/documents/lifecycle/rename', json=command)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(self.body('Mixed').status_code, 404)
        self.check_source('Moved', 'old bare book')
        self.check_source('Mixed.txt', 'separate qualified book')
        self.assertEqual(self.docs.load_document('Mixed.txt').to_dict(), other)
        repeated = self.client.post('/documents/lifecycle/rename', json=command)
        self.assertEqual(repeated.status_code, 200, repeated.text)
        self.assertEqual(repeated.json()['operation_id'], response.json()['operation_id'])

    def test_same_stem_lifecycle_over_tcp(self):
        rename.RenameCases.over_tcp(self, self.test_same_stem_http_body_notes_index_and_lifecycle_are_isolated)



class FileStorageHttpTests(StorageHttpCases, unittest.TestCase):
    def setUp(self):
        lifecycle.FileLifecycleTests.setUp(self)
        self.prepare_http()


class PostgresStorageHttpTests(StorageHttpCases, unittest.TestCase):
    setUpClass = classmethod(postgres.LearningPostgresTests.setUpClass.__func__)
    drop_schema = classmethod(postgres.LearningPostgresTests.drop_schema.__func__)

    def setUp(self):
        postgres.LearningPostgresTests.setUp(self)
        self.prepare_http()


if __name__ == '__main__':
    unittest.main()
