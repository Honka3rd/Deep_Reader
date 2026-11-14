"""STRUCT-01: real concurrent writes across file and PostgreSQL owners."""
import os
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from document_structure.structured_document_artifact_repository import StructuredDocumentArtifactRepository
from scripts.artifact_generation_fixtures import bound, bound_layout
from shared.task_artifacts import SummaryArtifact, QuizArtifact, TaskArtifacts, DocumentTaskArtifacts
from scripts.test_reading_interaction_artifact_store import _document
from scripts import test_summary_note_postgres as pg_fixture


class AtomicArtifactsTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.repo = StructuredDocumentArtifactRepository(base_dir=temp.name)
        self.seed()

    def seed(self):
        document = replace(_document(), document_id='book')
        chapter = document.chapters[0]
        first = chapter.sections[0]
        second = replace(first, section_id='section-2', section_index=1,
                         char_start=44, char_end=88,
                         task_units=[replace(first.task_units[0], unit_id='unit-2',
                                             parent_section_id='section-2', source_section_ids=['section-2'])])
        self.repo.save_reparsed_document(replace(document, chapters=[replace(chapter, sections=[first, second])]), 'book')
        self.initial = self.repo.load_document('book')
        self.sid = self.initial.chapters[0].sections[0].section_id
        self.other = self.initial.chapters[0].sections[1].section_id
        self.uid = self.initial.chapters[0].sections[0].task_units[0].unit_id
        self.chapter = 'chapter_id::' + self.initial.chapters[0].chapter_id

    def section(self, index=0):
        return self.repo.load_document('book').chapters[0].sections[index]

    def overlap(self, first, second):
        """Pause the first real read; the second writer must block, then merge it."""
        read, release = threading.Event(), threading.Event()
        entered = False
        gate = threading.Lock()
        def wrap(original):
            def paused(*args, **kwargs):
                nonlocal entered
                value = original(*args, **kwargs)
                with gate:
                    pause = not entered
                    entered = True
                if pause:
                    read.set()
                    if not release.wait(5):
                        raise RuntimeError('test writer release timed out')
                return value
            return paused
        with ExitStack() as stack:
            stack.enter_context(patch.object(self.repo, '_load_document_for_write',
                                             side_effect=wrap(self.repo._load_document_for_write)))
            if hasattr(self.repo.store, 'load_enhancement_snapshot'):
                stack.enter_context(patch.object(self.repo.store, 'load_enhancement_snapshot',
                                                 side_effect=wrap(self.repo.store.load_enhancement_snapshot)))
                stack.enter_context(patch.object(self.repo, '_artifact_lock',
                                                 side_effect=AssertionError('SQL used a file lock')))
            with ThreadPoolExecutor(2) as pool:
                a = pool.submit(first)
                try:
                    self.assertTrue(read.wait(5))
                    b = pool.submit(second)
                    with self.assertRaises(TimeoutError):
                        b.result(timeout=.15)
                finally:
                    release.set()
                a.result(timeout=10)
                b.result(timeout=10)

    def test_same_section_summary_and_quiz(self):
        self.overlap(lambda: self.repo.update_section_summary_artifact('book', self.sid, bound(SummaryArtifact('summary'), self.initial, 'section', self.sid)),
                     lambda: self.repo.update_section_quiz_artifact('book', self.sid, bound(QuizArtifact([{'question':'quiz'}]), self.initial, 'section', self.sid)))
        artifacts = self.section().task_artifacts
        self.assertEqual(artifacts.summary.content, 'summary')
        self.assertEqual(artifacts.quiz.items, [{'question':'quiz'}])

    def test_different_sections(self):
        self.overlap(lambda: self.repo.update_section_summary_artifact('book', self.sid, bound(SummaryArtifact('first'), self.initial, 'section', self.sid)),
                     lambda: self.repo.update_section_summary_artifact('book', self.other, bound(SummaryArtifact('second'), self.initial, 'section', self.other)))
        self.assertEqual(self.section().task_artifacts.summary.content, 'first')
        self.assertEqual(self.section(1).task_artifacts.summary.content, 'second')

    def test_chapter_kinds(self):
        self.overlap(lambda: self.repo.update_chapter_summary_artifact('book', self.chapter, bound(SummaryArtifact('chapter'), self.initial, 'chapter', self.chapter.removeprefix('chapter_id::'))),
                     lambda: self.repo.update_chapter_quiz_artifact('book', self.chapter, bound(QuizArtifact([{'question':'quiz'}]), self.initial, 'chapter', self.chapter.removeprefix('chapter_id::'))))
        artifact = self.repo.load_document('book').document_task_artifacts.chapter_artifacts[self.chapter]
        self.assertEqual(artifact.summary.content, 'chapter')
        self.assertEqual(artifact.quiz.items, [{'question':'quiz'}])

    def test_unit_kinds(self):
        self.overlap(lambda: self.repo.update_task_unit_artifacts('book', self.uid, TaskArtifacts(summary=bound(SummaryArtifact('unit'), self.initial, 'task_unit', self.uid))),
                     lambda: self.repo.update_task_unit_artifacts('book', self.uid, TaskArtifacts(quiz=bound(QuizArtifact([{'question':'quiz'}]), self.initial, 'task_unit', self.uid))))
        artifact = self.section().task_units[0].task_artifacts
        self.assertEqual(artifact.summary.content, 'unit')
        self.assertEqual(artifact.quiz.items, [{'question':'quiz'}])

    def test_layout_and_metadata(self):
        units = {s.section_id:s.task_units for c in self.initial.chapters for s in c.sections}
        self.overlap(lambda: self.repo.update_task_layout('book', units, bound_layout(self.initial, {'reading_revision':0, 'marker':'layout'})),
                     lambda: self.repo.update_document_artifacts('book', DocumentTaskArtifacts(metadata={'marker':'metadata'})))
        metadata = self.repo.load_document('book').document_task_artifacts.metadata
        self.assertEqual(metadata['task_layout']['marker'], 'layout')
        self.assertEqual(metadata['marker'], 'metadata')

    def test_layout_cannot_restore_stale_unit_artifacts(self):
        units = {s.section_id:[replace(u, task_artifacts=TaskArtifacts(summary=SummaryArtifact('stale')))
                              for u in s.task_units] for c in self.initial.chapters for s in c.sections}
        self.overlap(lambda: self.repo.update_task_unit_artifacts('book', self.uid, TaskArtifacts(summary=bound(SummaryArtifact('current'), self.initial, 'task_unit', self.uid))),
                     lambda: self.repo.update_task_layout('book', units, bound_layout(self.initial, {'reading_revision':0})))
        self.assertEqual(self.section().task_units[0].task_artifacts.summary.content, 'current')
        self.assertIsNone(self.section(1).task_units[0].task_artifacts)

    def test_combined_section_merge_preserves_other_kind_and_metadata(self):
        self.repo.update_document_artifacts('book', DocumentTaskArtifacts(metadata={'keep':42}))
        self.overlap(lambda: self.repo.update_section_artifacts('book', self.sid, TaskArtifacts(summary=bound(SummaryArtifact('summary'), self.initial, 'section', self.sid))),
                     lambda: self.repo.update_section_artifacts('book', self.sid, TaskArtifacts(quiz=bound(QuizArtifact([{'question':'quiz'}]), self.initial, 'section', self.sid))))
        artifacts = self.section().task_artifacts
        self.assertEqual(artifacts.summary.content, 'summary')
        self.assertEqual(artifacts.quiz.items, [{'question':'quiz'}])
        self.assertEqual(self.repo.load_document('book').document_task_artifacts.metadata['keep'], 42)

    def test_failure_preserves_committed_state(self):
        path = Path(self.repo.base_dir) / 'book.structured.json'
        before = path.read_bytes()
        # Temp-file data has been written; fsync fails before the commit rename.
        with patch('document_structure.structured_document_artifact_repository.os.fsync', side_effect=OSError('injected save failure')):
            with self.assertRaises(OSError):
                self.repo.update_section_summary_artifact('book', self.sid, bound(SummaryArtifact('failed'), self.initial, 'section', self.sid))
        self.assertEqual(path.read_bytes(), before)
        self.assertFalse(list(Path(self.repo.base_dir).glob('*.tmp')))

    def test_unknown_target_does_not_write(self):
        before = self.repo.load_document('book')
        operations = [
            lambda: self.repo.update_section_summary_artifact('book', 'absent', SummaryArtifact('invalid')),
            lambda: self.repo.update_chapter_summary_artifact('book', 'chapter_id::absent', SummaryArtifact('invalid')),
            lambda: self.repo.update_task_unit_artifacts('book', 'absent', TaskArtifacts(summary=SummaryArtifact('invalid'))),
            lambda: self.repo.update_task_layout('book', {'absent': []}, bound_layout(self.initial, {'reading_revision':0})),
        ]
        for operation in operations:
            with self.assertRaises(ValueError):
                operation()
            self.assertEqual(self.repo.load_document('book'), before)


@unittest.skipUnless(os.environ.get('DEEP_READER_TEST_POSTGRES_DSN'), 'requires disposable PostgreSQL DSN')
class PostgresAtomicArtifactsTests(AtomicArtifactsTests):
    setUpClass = classmethod(pg_fixture.PostgresNotes.setUpClass.__func__)
    drop_schema = classmethod(pg_fixture.PostgresNotes.drop_schema.__func__)

    def setUp(self):
        from db.postgres_structured_document_store import PostgresStructuredDocumentStore
        from db.postgres_structured_document_artifact_repository import PostgresStructuredDocumentArtifactRepository
        self.repo = PostgresStructuredDocumentArtifactRepository(
            PostgresStructuredDocumentStore(self.scoped, namespace=uuid4().hex))
        self.seed()

    def test_failure_preserves_committed_state(self):
        before = self.repo.load_document('book')
        save = self.repo.store.save
        def fail_after_save(*args, **kwargs):
            self.assertIsNotNone(kwargs.get('_connection'))
            save(*args, **kwargs)
            raise RuntimeError('after real SQL writes, before transaction commit')
        with patch.object(self.repo.store, 'save', side_effect=fail_after_save):
            with self.assertRaises(RuntimeError):
                self.repo.update_section_summary_artifact('book', self.sid, bound(SummaryArtifact('failed'), self.initial, 'section', self.sid))
        self.assertEqual(self.repo.load_document('book'), before)
        self.repo.update_section_summary_artifact('book', self.sid, bound(SummaryArtifact('retry'), self.initial, 'section', self.sid))
        self.assertEqual(self.section().task_artifacts.summary.content, 'retry')


if __name__ == '__main__':
    unittest.main(verbosity=2)
