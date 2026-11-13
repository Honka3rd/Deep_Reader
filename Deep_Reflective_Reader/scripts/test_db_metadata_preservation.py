"""DB-01: real SQL row ownership, concurrent writes and rollback evidence."""
import os
import unittest
from dataclasses import replace
from unittest.mock import patch

import psycopg
from psycopg.types.json import Jsonb

from document_structure.reading_content import ReadingContentService
from scripts import test_atomic_hierarchy_artifacts as atomic
from scripts.artifact_generation_fixtures import bound
from shared.task_artifacts import SummaryArtifact


@unittest.skipUnless(os.environ.get('DEEP_READER_TEST_POSTGRES_DSN'), 'requires disposable PostgreSQL DSN')
class MetadataPreservationTests(unittest.TestCase):
    setUpClass = classmethod(atomic.PostgresAtomicArtifactsTests.setUpClass.__func__)
    drop_schema = classmethod(atomic.PostgresAtomicArtifactsTests.drop_schema.__func__)
    seed = atomic.AtomicArtifactsTests.seed
    section = atomic.AtomicArtifactsTests.section
    overlap = atomic.AtomicArtifactsTests.overlap
    tables = ('chapters', 'sections', 'task_units', 'raw_source_metadata', 'structured_document_snapshots')

    def setUp(self):
        atomic.PostgresAtomicArtifactsTests.setUp(self)
        self.repo.save_document(replace(self.initial, source_path='source.txt'), 'book')
        self.initial = self.repo.load_document('book')
        self.repo.save_reparsed_document(replace(self.initial, document_id='other'), 'other')
        with psycopg.connect(self.scoped) as connection:
            self.row_id = connection.execute(
                'SELECT id FROM documents WHERE namespace=%s AND document_name=%s',
                (self.repo.store.namespace, 'book')).fetchone()[0]
            for table in ('documents', *self.tables):
                key = 'id' if table == 'documents' else 'document_id'
                connection.execute(
                    f'UPDATE {table} SET metadata_payload=metadata_payload || %s WHERE {key}=%s',
                    (Jsonb({'notes': ['legacy note'], 'extension': {'owner': table}}), self.row_id))
        self.before = self.rows()
        self.other_before = self.repo.load_document('other')

    def rows(self):
        with psycopg.connect(self.scoped) as connection:
            result = {'documents': connection.execute(
                'SELECT id, metadata_payload FROM documents WHERE id=%s', (self.row_id,)).fetchall()}
            for table in self.tables:
                result[table] = connection.execute(
                    f'SELECT document_id, metadata_payload FROM {table} WHERE document_id=%s ORDER BY id',
                    (self.row_id,)).fetchall()
            result['snapshot'] = connection.execute(
                'SELECT structured_document_payload FROM structured_document_snapshots WHERE document_id=%s',
                (self.row_id,)).fetchone()[0]
            return result

    def assert_preserved(self):
        rows = self.rows()
        for table in ('documents', *self.tables):
            with self.subTest(table=table):
                self.assertEqual(len(rows[table]), len(self.before[table]))
                for _, metadata in rows[table]:
                    self.assertEqual(metadata.get('notes'), ['legacy note'])
                    self.assertEqual(metadata.get('extension'), {'owner': table})
        self.assertTrue(self.repo.load_enhancement_snapshot('book').includes_user_notes)
        self.assertEqual(self.repo.load_document('other'), self.other_before)
        # Snapshot is a supported DTO projection, not storage for opaque row metadata.
        from document_structure.structured_document import StructuredDocument
        snapshot = StructuredDocument.from_dict(rows['snapshot'])
        current = self.repo.load_document('book')
        self.assertEqual(snapshot.chapters, current.chapters)
        self.assertEqual(snapshot.document_task_artifacts, current.document_task_artifacts)
        self.assertEqual(snapshot.source_path, current.source_path)

    def test_summary_preserves_unselected_nodes_and_projection(self):
        self.repo.update_section_summary_artifact(
            'book', self.sid, bound(SummaryArtifact('new summary'), self.initial, 'section', self.sid))
        self.assertEqual(self.section().task_artifacts.summary.content, 'new summary')
        self.assertIsNone(self.section(1).task_artifacts)
        self.assert_preserved()

    def test_same_section_summary_quiz_concurrency(self):
        atomic.AtomicArtifactsTests.test_same_section_summary_and_quiz(self)
        self.assert_preserved()

    def test_different_section_concurrency(self):
        atomic.AtomicArtifactsTests.test_different_sections(self)
        self.assert_preserved()

    def test_chapter_summary_quiz_concurrency(self):
        atomic.AtomicArtifactsTests.test_chapter_kinds(self)
        self.assert_preserved()

    def test_unit_summary_quiz_concurrency(self):
        atomic.AtomicArtifactsTests.test_unit_kinds(self)
        self.assert_preserved()

    def test_layout_metadata_concurrency(self):
        atomic.AtomicArtifactsTests.test_layout_and_metadata(self)
        self.assert_preserved()

    def test_clear_does_not_resurrect_owned_artifacts_or_reading_state(self):
        atomic.AtomicArtifactsTests.test_same_section_summary_and_quiz(self)
        reader = ReadingContentService(self.repo)
        state = reader.read('book')
        reader.mutate('book', action='edit', expected_revision=state['revision'],
                      source_token=state['source_token'], mutation_id='clear',
                      block_id=state['blocks'][0]['block_id'], text='Changed body.', discard_interactions=True)
        self.assertIsNone(self.section().task_artifacts)
        saved = self.repo.load_document('book')
        self.repo.save_document(saved, 'book')
        self.assertIsNone(self.section().task_artifacts)
        self.assertEqual(reader.read('book')['blocks'][0]['content'], 'Changed body.')
        self.assert_preserved()

    def test_partial_real_save_rolls_back_all_rows_and_snapshot(self):
        original = self.repo.store.save
        def fail(*args, **kwargs):
            self.assertIsNotNone(kwargs.get('_connection'))
            original(*args, **kwargs)
            raise RuntimeError('after real writes')
        with patch.object(self.repo.store, 'save', side_effect=fail):
            with self.assertRaisesRegex(RuntimeError, 'after real writes'):
                self.repo.update_section_summary_artifact(
                    'book', self.sid, bound(SummaryArtifact('rejected'), self.initial, 'section', self.sid))
        self.assertEqual(self.rows(), self.before)
        self.assertEqual(self.repo.load_document('other'), self.other_before)

    def test_layout_reallocated_unit_keeps_metadata_and_block_identity(self):
        from shared.task_unit_model import TaskUnitContentBlock
        section = self.section()
        unit = replace(section.task_units[0], unit_id='candidate-unit', content_blocks=[
            TaskUnitContentBlock('candidate-unit:content:0', section.task_units[0].content)])
        self.repo.update_section_task_units('book', self.sid, [unit])
        saved = self.section().task_units[0]
        self.assertEqual(saved.unit_id, self.uid)
        self.assertEqual(saved.content_blocks[0].block_id, self.uid + ':content:0')
        self.assert_preserved()

    def test_nullable_legacy_metadata_accepts_new_owned_fields(self):
        with psycopg.connect(self.scoped) as connection:
            for table in self.tables:
                connection.execute(f'UPDATE {table} SET metadata_payload=NULL WHERE document_id=%s',
                                   (self.row_id,))
        self.repo.update_section_summary_artifact(
            'book', self.sid, bound(SummaryArtifact('from null'), self.repo.load_document('book'), 'section', self.sid))
        self.assertEqual(self.section().task_artifacts.summary.content, 'from null')
        rows = self.rows()
        for table in self.tables:
            self.assertTrue(all(isinstance(metadata, dict) for _, metadata in rows[table]), table)

    def test_hard_reparse_does_not_restore_old_hierarchy_metadata(self):
        atomic.AtomicArtifactsTests.test_same_section_summary_and_quiz(self)
        self.repo.save_reparsed_document(self.initial, 'book')
        after = self.rows()
        for table in ('chapters', 'sections', 'task_units'):
            self.assertTrue(after[table])
            for _, metadata in after[table]:
                self.assertNotIn('notes', metadata)
                self.assertNotIn('extension', metadata)
                self.assertNotIn('task_artifacts', metadata)
        self.assertNotEqual(self.repo.load_document('book').parse_provenance['interaction_epoch'],
                            self.initial.parse_provenance['interaction_epoch'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
