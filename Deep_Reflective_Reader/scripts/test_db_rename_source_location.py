"""DB-02: actual HTTP lifecycle and SQL source-location consistency."""
import hashlib
import os
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import psycopg

from scripts import test_main_rename_audit as rename_tests
from section_tasks.note_interaction_service import NoteInteractionService


@unittest.skipUnless(os.environ.get('DEEP_READER_TEST_POSTGRES_DSN'), 'requires disposable PostgreSQL DSN')
class RenameSourceLocationTests(unittest.TestCase):
    setUpClass = classmethod(rename_tests.PostgresRenameTests.setUpClass.__func__)
    drop_schema = classmethod(rename_tests.PostgresRenameTests.drop_schema.__func__)
    def setUp(self):
        rename_tests.PostgresRenameTests.setUp(self)
        self.docs.store.raw_root = self.raw

    http_setup = rename_tests.RenameCases.http_setup
    rename = rename_tests.RenameCases.rename
    rename_command = rename_tests.RenameCases.rename_command
    save_result = rename_tests.RenameCases.save_result

    def attach_source(self, suffix):
        path = self.raw / ('book' + suffix)
        path.write_bytes(b'isolated source bytes; no extraction is required')
        self.docs.save_document(replace(self.docs.load_document('book'), source_path=str(path)), 'book')
        return path

    def locations(self):
        with psycopg.connect(self.scoped) as connection:
            return connection.execute(
                "SELECT d.metadata_payload->>'source_path', r.source_location, r.original_filename, "
                "s.structured_document_payload->>'source_path' "
                "FROM documents d JOIN raw_source_metadata r ON r.document_id=d.id "
                "JOIN structured_document_snapshots s ON s.document_id=d.id "
                "AND s.source_structure_version=d.current_structure_version WHERE d.id=%s",
                (self.row_id,)).fetchone()

    def check_rename(self, suffix, target, qualified_request=False):
        source = self.attach_source(suffix)
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        key, entry, sentences = self.save_result()
        settings = self.prefs.read_book('book').settings
        notes = NoteInteractionService(self.docs)
        token = notes.read('book', target_level='document')['target_token']
        notes.save('book', target_level='document', text='Keep my note', mutation_id='note',
                   expected_revision=0, target_token=token)
        old = source.name if qualified_request else 'book'
        response = self.rename(target, old=old)
        self.assertEqual(response.status_code, 200, response.text)
        current = self.docs.load_document(target)
        path = Path(current.source_path)
        expected = self.raw / (target if Path(target).suffix.lower() in ('.pdf', '.txt') else target + suffix)
        self.assertEqual(path, expected)
        self.assertTrue(path.is_file())
        self.assertFalse(source.exists())
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)
        self.assertEqual(self.locations(), (str(path), str(path), path.name, str(path)))
        self.assertEqual(self.prefs.read_book(target).settings, settings)
        self.assertEqual(notes.read(target, target_level='document')['note']['text'], 'Keep my note')
        moved_sentences = sentences.with_document_id(target)
        view = self.repo.read_batch(target, (replace(key, resource=moved_sentences.reference),))[0]
        self.assertEqual(view.view.state, 'ready')
        self.assertEqual(view.sentences, moved_sentences)
        self.assertEqual(view.view.saved.context.result_revision, entry.saved.context.result_revision)
        replay = self.rename(target, old=old)
        self.assertEqual(replay.status_code, 200, replay.text)
        self.assertEqual(replay.json()['operation_id'], response.json()['operation_id'])
        self.assertEqual(self.locations(), (str(path), str(path), path.name, str(path)))

    def test_sql_rollback_then_http_recovery_uses_same_destination(self):
        source = self.attach_source('.txt')
        before = self.locations()
        # Owner performs all real statements, then commit fails inside the transaction.
        original = psycopg.Connection.execute
        def fail(connection, query, *args, **kwargs):
            result = original(connection, query, *args, **kwargs)
            if isinstance(query, str) and query.startswith('UPDATE structured_document_snapshots SET structured_document_payload='):
                raise RuntimeError('after snapshot update')
            return result
        with patch.object(psycopg.Connection, 'execute', fail):
            response = self.rename('重新命名.txt')
        self.assertEqual(response.status_code, 503, response.text)
        self.assertTrue(response.json()['recovery_required'])
        self.assertEqual(self.locations(), before)
        self.assertEqual(self.docs.load_document('book').document_id, 'book')
        # SQL is the first effect; rollback leaves the planned move pending.
        self.assertTrue(source.exists())
        self.assertFalse((self.raw / '重新命名.txt').exists())
        response = self.rename('重新命名.txt')
        self.assertEqual(response.status_code, 200, response.text)
        destination = self.raw / '重新命名.txt'
        self.assertEqual(self.locations(), (str(destination), str(destination), destination.name, str(destination)))

    def test_missing_raw_preserves_actual_recorded_locator(self):
        path = self.raw / 'missing.txt'
        self.docs.save_document(replace(self.docs.load_document('book'), source_path=str(path)), 'book')
        response = self.rename('無來源.txt')
        self.assertEqual(response.status_code, 200, response.text)
        self.assertFalse(response.json()['source_affected'])
        self.assertEqual(self.locations(), (str(path), str(path), path.name, str(path)))
        self.assertFalse(path.exists())


# Each case gets a fresh namespace, learning/preferences state and real HTTP client.
for label, suffix, target, qualified in (
    ('txt_bare', '.txt', 'renamed', False),
    ('txt_qualified', '.txt', 'renamed.txt', False),
    ('pdf_bare', '.pdf', 'renamed', False),
    ('pdf_qualified', '.pdf', 'renamed.pdf', True),
    ('uppercase_txt', '.TXT', '重命名.TXT', True),
    ('uppercase_pdf', '.PDF', '中文.PDF', True),
    ('legacy_name', '.txt', 'English: A Reader.txt', False),
):
    def case(self, suffix=suffix, target=target, qualified=qualified):
        self.check_rename(suffix, target, qualified)
    setattr(RenameSourceLocationTests, 'test_' + label, case)


if __name__ == '__main__':
    unittest.main(verbosity=2)
