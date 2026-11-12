"""SCHEMA-03: pre-validation files remain usable through the upgraded HTTP API."""
from dataclasses import replace
import unittest
from unittest.mock import patch
from urllib.parse import quote

import main
from document_structure.structured_document_artifact_repository import StructuredDocumentArtifactRepository
from scripts import test_learning_lifecycle as lifecycle
from scripts import test_learning_file_repository as files
from scripts import test_learning_postgres_repository as postgres
from scripts import test_main_rename_audit as rename
from scripts import test_task_unit_content_endpoint as content
from section_tasks.note_interaction_service import NoteInteractionService


class LegacyFileRepository(StructuredDocumentArtifactRepository):
    """Only fixture creation: name resolution used before SCHEMA-01 (440471fa)."""
    def _resolve_document_path(self, doc_name):
        # Freeze the old rule; do not use the upgraded resolver in an upgrade fixture.
        name = doc_name.strip()
        if name.lower().endswith(('.pdf', '.txt')):
            name = name[:-4].strip()
        name = name or 'default'
        return self.base_dir / f'{name}.structured.json'

    _exact_document_path = _resolve_document_path


class LegacyNameCases:
    def test_upgrade_over_tcp(self):
        rename.RenameCases.over_tcp(self, lambda: self.check_upgrade('English: Who am I? <Draft> #100%'))

    def check_upgrade(self, name):
        # Save the book and a real note before switching to the current adapter.
        legacy = self.docs
        if self.docs.lifecycle_storage_identity[0] == 'file':
            self.assertTrue(self.docs.base_dir.resolve().is_relative_to(self.root.resolve()))
            legacy = LegacyFileRepository(base_dir=str(self.docs.base_dir))
        original = replace(files.fixture(name), title=name)
        legacy.save_reparsed_document(original, name)
        service = NoteInteractionService(legacy)
        token = service.read(name, target_level='document')['target_token']
        saved = service.save(name, target_level='document', text='舊版筆記 <keep> ? :\nSecond line.',
                             expected_revision=0, mutation_id='legacy-note', target_token=token)
        persisted = legacy.load_document(name)
        snapshot = persisted.to_dict()
        old_bytes = (legacy.base_dir / f'{name}.structured.json').read_bytes() if legacy is not self.docs else None
        coordinator, _ = content._build_coordinator(original)
        coordinator.document_artifact_repository = self.docs
        coordinator.document_preparation_pipeline.loader_factory = None
        with patch.object(main, 'section_task_coordinator', coordinator):
            self.verify_visibility(name)
            self.verify_body_and_note(name, persisted, saved['note'])
            # Reads/listing do not migrate or silently rewrite persisted artifacts.
            self.assertEqual(self.docs.load_document(name).to_dict(), snapshot)
            if old_bytes is not None:
                self.assertEqual((self.docs.base_dir / f'{name}.structured.json').read_bytes(), old_bytes)

            read = self.note(name)
            updated = self.client.post('/documents/reading-interactions/note/save', json={
                'target': {'doc_name':name, 'target_type':'document'},
                'text':saved['note']['text']+'\nExplicit edit', 'expected_revision':read['note']['revision'],
                'target_token':read['target_token'], 'mutation_id':'after-upgrade'})
            self.assertEqual(updated.status_code, 200, updated.text)
            note = updated.json()['note']
            self.assertEqual(note['note_id'], saved['note']['note_id'])
            self.assertEqual(note['revision'], saved['note']['revision']+1)
            self.verify_body_and_note(name, persisted, note)

            # Preserve legacy raw and derived paths through an explicit lifecycle command.
            raw = original.raw_text.encode()
            (self.raw / f'{name}.txt').write_bytes(raw)
            (self.derived / name).mkdir()
            (self.derived / name / 'profile.json').write_text('{}')
            new = name+' (renamed)'
            command = rename.RenameCases.rename_command(self, name, new)
            response = self.client.post('/documents/lifecycle/rename', json=command)
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.json()['status'], 'accepted')
            replay = self.client.post('/documents/lifecycle/rename', json=command)
            self.assertEqual(replay.status_code, 200, replay.text)
            self.assertEqual(replay.json()['operation_id'], response.json()['operation_id'])
            self.assertFalse((self.raw / f'{name}.txt').exists())
            self.assertEqual((self.raw / f'{new}.txt').read_bytes(), raw)
            self.assertEqual((self.derived / new / 'profile.json').read_text(), '{}')
            self.assertTrue(self.prefs.read_book(name).deleted)
            self.verify_visibility(new)
            self.verify_body_and_note(new, persisted, note)
            with self.assertRaises(FileNotFoundError):
                self.docs.load_document(name)

            # Rename back proves both original and destination can be legacy titles.
            back = rename.RenameCases.rename_command(self, new, name)
            response = self.client.post('/documents/lifecycle/rename', json=back)
            self.assertEqual(response.status_code, 200, response.text)
            self.verify_body_and_note(name, persisted, note)
            deleted = self.client.post('/documents/lifecycle/delete', json={
                'doc_name':name, 'action':'delete', 'confirm':True})
            self.assertEqual(deleted.status_code, 200, deleted.text)
            self.assertEqual(deleted.json()['status'], 'accepted')
            self.assertFalse((self.raw / f'{name}.txt').exists())
            self.assertFalse((self.derived / name).exists())
            self.assertTrue(self.prefs.read_book(name).deleted)
            with self.assertRaises(FileNotFoundError):
                self.docs.load_document(name)

    def verify_visibility(self, name):
        # Assert repository listing too: raw-source merging must not mask hidden artifacts.
        self.assertIn(name, [item.doc_name for item in self.docs.list_documents()])
        for path in ('/documents', '/documents/management'):
            response = self.client.get(path)
            self.assertEqual(response.status_code, 200, response.text)
            self.assertIn(name, [item['doc_name'] for item in response.json()['items']])

    def note(self, name):
        response = self.client.post('/documents/reading-interactions/note/read', json={
            'target':{'doc_name':name, 'target_type':'document'}})
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def verify_body_and_note(self, name, original, note):
        current = self.docs.load_document(name)
        self.assertEqual(current.raw_text, original.raw_text)
        self.assertEqual([c.to_dict() for c in current.chapters], [c.to_dict() for c in original.chapters])
        response = self.client.get(f'/documents/{quote(name, safe="")}/reading-content', params={'include_blocks':True})
        self.assertEqual(response.status_code, 200, response.text)
        expected = [unit.content for chapter in original.chapters for section in chapter.sections for unit in section.task_units]
        self.assertEqual([block['content'] for block in response.json()['blocks']], expected)
        self.assertEqual(self.note(name)['note'], note)


class FileLegacyNameTests(LegacyNameCases, unittest.TestCase):
    def setUp(self):
        lifecycle.FileLifecycleTests.setUp(self)
        rename.RenameCases.http_setup(self)


class PostgresLegacyNameTests(LegacyNameCases, unittest.TestCase):
    setUpClass = classmethod(postgres.LearningPostgresTests.setUpClass.__func__)
    drop_schema = classmethod(postgres.LearningPostgresTests.drop_schema.__func__)

    def setUp(self):
        postgres.LearningPostgresTests.setUp(self)
        rename.RenameCases.http_setup(self)


# Each backend/title has an independent persisted fixture and lifecycle.
for owner in (FileLegacyNameTests, PostgresLegacyNameTests):
    for label, name in [('colon', 'English: A Reader'), ('question', 'Who am I?'),
                        ('angles', 'Book <Draft>'), ('punctuation', 'Reader "Q" | * & # %')]:
        def test(self, name=name):
            self.check_upgrade(name)
        setattr(owner, f'test_upgrade_{label}', test)
del owner

if __name__ == '__main__':
    unittest.main()
