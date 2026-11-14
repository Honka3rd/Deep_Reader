"""APP-01: real file/SQL publication races and content endpoint readback."""
import os
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from threading import Event
from unittest.mock import Mock, patch
from uuid import uuid4
from fastapi.testclient import TestClient
import main
from app.section_task_coordinator import SectionTaskCoordinator
from document_structure.reading_content import ReadingContentConflict, ReadingContentService
from document_structure.structured_document_artifact_repository import StructuredDocumentArtifactRepository
from document_structure.task_layout_source import LAYOUT_SOURCE_KEY
from scripts import test_task_layout_persistence_cache as fixture
from scripts import test_summary_note_postgres as sql_fixture
from scripts.artifact_generation_fixtures import bound, bound_layout
from shared.task_artifacts import SummaryArtifact, TaskArtifacts


class LayoutPublicationTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.repo = StructuredDocumentArtifactRepository(base_dir=temp.name)
        self.initialize()

    def initialize(self):
        self.pipeline = Mock()
        self.pipeline.prepare_and_load.side_effect = AssertionError('hidden preparation')
        self.resolver = fixture._FakeTaskUnitResolver()
        self.coordinator = SectionTaskCoordinator(self.pipeline, self.repo, fixture._FakeProfileStore(),
            Mock(), Mock(), self.resolver, fixture._FakeEnhancedParseEvaluator())
        self.repo.save_reparsed_document(fixture._build_base_document(), 'cache-doc')
        self.prepare()
        self.initial = self.repo.load_document('cache-doc')

    def prepare(self, refresh=False):
        self.coordinator.ensure_document_task_layout('cache-doc', refresh_task_units=refresh,
                                                    task_unit_split_mode='progressive')

    def snapshot(self):
        if hasattr(self.repo.store, 'dsn'):
            import psycopg
            with psycopg.connect(self.repo.store.dsn) as connection:
                row = connection.execute('SELECT id FROM documents WHERE namespace=%s AND document_name=%s',
                    (self.repo.store.namespace, 'cache-doc')).fetchone()[0]
                return {table: connection.execute(
                    f'SELECT to_jsonb(t) FROM {table} t WHERE {"id" if table == "documents" else "document_id"}=%s ORDER BY id',
                    (row,)).fetchall() for table in
                    ('documents', 'chapters', 'sections', 'task_units', 'structured_document_snapshots')}
        return (Path(self.repo.base_dir) / 'cache-doc.structured.json').read_bytes()

    def content_readback(self):
        document = self.repo.load_document('cache-doc')
        ids = [u.unit_id for c in document.chapters for s in c.sections for u in s.task_units]
        expected = self.coordinator.get_task_unit_contents(doc_name='cache-doc', task_unit_ids=ids)
        before = self.snapshot()
        with patch.object(main, 'section_task_coordinator', self.coordinator), TestClient(main.app) as client:
            response = client.post('/documents/cache-doc/task-units/content', json={
                'task_unit_ids': ids, 'include_raw_content': True})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual([c['content'] for c in response.json()['contents']], [c.content for c in expected])
        self.assertEqual(self.snapshot(), before)
        self.pipeline.prepare_and_load.assert_not_called()
        return [c.content for c in expected]

    def delayed(self, change, *, rejected=True):
        entered, release = Event(), Event()
        original = self.resolver.resolve_with_options
        def pause(**kwargs):
            result = original(**kwargs)
            if not entered.is_set():
                entered.set()
                if not release.wait(10): raise TimeoutError('layout test release')
            return result
        with patch.object(self.resolver, 'resolve_with_options', side_effect=pause):
            with ThreadPoolExecutor(1) as pool:
                future = pool.submit(self.prepare, True)
                try:
                    self.assertTrue(entered.wait(5))
                    # Real competing writes complete while resolver work is paused.
                    change()
                    before = self.snapshot()
                finally:
                    release.set()
                if rejected:
                    with self.assertRaises((ReadingContentConflict, FileNotFoundError)):
                        future.result(15)
                    self.assertEqual(self.snapshot(), before)
                else: future.result(15)
        self.content_readback()

    def reparse(self, *, same_text=False, reimport=False):
        old = self.repo.load_document('cache-doc')
        new = fixture._build_base_document()
        if not same_text:
            new = replace(new, raw_text=new.raw_text.replace('內容', '新文'), chapters=[
                replace(c, sections=[replace(s, content=s.content.replace('內容', '新文'))
                                     for s in c.sections]) for c in new.chapters])
        if reimport: self.repo.delete_document('cache-doc')
        self.repo.save_reparsed_document(new, 'cache-doc')
        current = self.repo.load_document('cache-doc')
        self.assertNotEqual(current.parse_provenance['interaction_epoch'], old.parse_provenance['interaction_epoch'])
        old_ids = [s.section_id for c in old.chapters for s in c.sections]
        new_ids = [s.section_id for c in current.chapters for s in c.sections]
        if hasattr(self.repo.store, 'dsn'): self.assertNotEqual(old_ids, new_ids)
        else: self.assertEqual(old_ids, new_ids)
        self.prepare()

    def edit(self, *, undo=False, no_op=False):
        reader = ReadingContentService(self.repo)
        state = reader.read('cache-doc'); block = state['blocks'][0]
        reader.mutate('cache-doc', action='edit', expected_revision=state['revision'],
            source_token=state['source_token'], mutation_id=uuid4().hex, block_id=block['block_id'],
            text=block['content'] if no_op else 'Changed body.', discard_interactions=False)
        if undo:
            state = reader.read('cache-doc')
            reader.mutate('cache-doc', action='undo', expected_revision=state['revision'],
                          source_token=state['source_token'], mutation_id=uuid4().hex)

    def test_delayed_layout_after_reparse_keeps_new_body(self):
        self.delayed(self.reparse)
        text = ''.join(self.content_readback())
        self.assertIn('新文', text); self.assertNotIn('內容', text)

    def test_same_text_new_incarnation_rejected(self):
        self.delayed(lambda: self.reparse(same_text=True))

    def test_delete_reimport_same_text_rejected(self):
        self.delayed(lambda: self.reparse(same_text=True, reimport=True))

    def test_edit_rejects_then_explicit_retry_succeeds(self):
        self.delayed(self.edit); self.prepare()
        self.assertIn('Changed body.', ''.join(self.content_readback()))

    def test_edit_undo_does_not_revive_old_layout(self):
        self.delayed(lambda: self.edit(undo=True)); self.prepare()

    def test_noop_source_allows_publication(self):
        self.delayed(lambda: self.edit(no_op=True), rejected=False)

    def test_rename_roundtrip_rejects(self):
        def change():
            transform = lambda document, name: replace(document, document_id=name, title=name)
            self.repo.rename_document('cache-doc', 'other', transform)
            self.repo.rename_document('other', 'cache-doc', transform)
        self.delayed(change)

    def test_source_provenance_change_rejected(self):
        def change():
            current = self.repo.load_document('cache-doc')
            self.repo.save_document(replace(current, source_path='changed-source.txt'), 'cache-doc')
        self.delayed(change)

    def test_content_coordinates_change_rejected(self):
        def change():
            current = self.repo.load_document('cache-doc')
            chapters = list(current.chapters); sections = list(chapters[0].sections)
            units = list(sections[0].task_units); blocks = list(units[0].content_blocks)
            self.assertTrue(blocks)
            blocks[0] = replace(blocks[0], metadata={**(blocks[0].metadata or {}), 'quote_span_start': 1})
            units[0] = replace(units[0], content_blocks=blocks)
            sections[0] = replace(sections[0], task_units=units)
            chapters[0] = replace(chapters[0], sections=sections)
            self.repo.save_document(replace(current, chapters=chapters), 'cache-doc')
        self.delayed(change)

    def test_id_repair_rejects_content_change_and_keeps_artifacts(self):
        current = self.repo.load_document('cache-doc'); section = current.chapters[0].sections[0]
        before = self.snapshot()
        with self.assertRaisesRegex(ReadingContentConflict, 'task_layout_repair_source_changed'):
            self.repo.update_task_layout('cache-doc', {section.section_id: [replace(
                section.task_units[0], content='foreign body')]},
                bound_layout(current, {'reading_revision': 0}), repair_unit_ids=True)
        self.assertEqual(self.snapshot(), before)

    def test_latest_artifacts_preserved_while_resolving(self):
        unit = self.initial.chapters[0].sections[0].task_units[0]
        def save():
            self.repo.update_task_unit_artifacts('cache-doc', unit.unit_id, TaskArtifacts(
                summary=bound(SummaryArtifact('concurrent saved summary'), self.initial, 'task_unit', unit.unit_id)))
        units = [u for c in self.initial.chapters for s in c.sections for u in s.task_units]
        with patch.object(self.resolver, 'resolve_with_options', return_value=units):
            self.delayed(save, rejected=False)
        self.assertEqual(self.repo.load_document('cache-doc').chapters[0].sections[0].task_units[0]
                         .task_artifacts.summary.content, 'concurrent saved summary')

    def test_unbound_write_rejected_without_changes(self):
        before = self.snapshot()
        with self.assertRaisesRegex(ReadingContentConflict, 'task_layout_source_changed'):
            self.repo.update_task_layout('cache-doc', {}, {'reading_revision': 0})
        self.assertEqual(before, self.snapshot())

    def test_final_boundary_rechecks_after_outside_guard(self):
        original = self.repo.update_task_layout
        def interposed(*args, **kwargs):
            with patch.object(self.repo, 'update_task_layout', original): self.reparse(same_text=True)
            before = self.snapshot()
            with self.assertRaises(ReadingContentConflict): original(*args, **kwargs)
            self.assertEqual(before, self.snapshot())
            raise ReadingContentConflict('task_layout_source_changed')
        with patch.object(self.repo, 'update_task_layout', side_effect=interposed):
            with self.assertRaises(ReadingContentConflict): self.prepare(True)
        self.content_readback()

    def test_failure_rolls_back_and_explicit_retry_succeeds(self):
        before = self.snapshot()
        if hasattr(self.repo.store, 'dsn'):
            original = self.repo.store.save
            def fail(*args, **kwargs):
                self.assertIsNotNone(kwargs.get('_connection'))
                original(*args, **kwargs)
                raise OSError('after SQL writes')
            failure = patch.object(self.repo.store, 'save', side_effect=fail)
        else:
            failure = patch('document_structure.structured_document_artifact_repository.os.fsync',
                            side_effect=OSError('before atomic replace'))
        with failure, self.assertRaises(OSError): self.prepare(True)
        self.assertEqual(before, self.snapshot())
        self.prepare(); self.content_readback()

    def test_legacy_cache_read_requires_no_migration_or_generation(self):
        original = self.repo.load_document('cache-doc'); artifacts = original.document_task_artifacts
        metadata = dict(artifacts.metadata)
        metadata['task_layout'] = {k: v for k, v in metadata['task_layout'].items() if k != LAYOUT_SOURCE_KEY}
        self.repo.save_document(replace(original, document_task_artifacts=replace(artifacts, metadata=metadata)), 'cache-doc')
        before = self.snapshot()
        with (patch.object(self.resolver, 'resolve_with_options', side_effect=AssertionError('read generated')),
              patch.object(self.repo, 'update_task_layout', side_effect=AssertionError('read wrote'))):
            self.coordinator.get_document_task_layout('cache-doc', task_unit_split_mode='progressive')
        self.assertEqual(before, self.snapshot())
        self.content_readback()


@unittest.skipUnless(os.environ.get('DEEP_READER_TEST_POSTGRES_DSN'), 'requires disposable PostgreSQL DSN')
class PostgresLayoutPublicationTests(LayoutPublicationTests):
    setUpClass = classmethod(sql_fixture.PostgresNotes.setUpClass.__func__)
    drop_schema = classmethod(sql_fixture.PostgresNotes.drop_schema.__func__)

    def setUp(self):
        from db.postgres_structured_document_store import PostgresStructuredDocumentStore
        from db.postgres_structured_document_artifact_repository import PostgresStructuredDocumentArtifactRepository
        self.repo = PostgresStructuredDocumentArtifactRepository(
            PostgresStructuredDocumentStore(self.scoped, namespace=uuid4().hex))
        self.initialize()


class DuplicateIdPublicationTests(unittest.TestCase):
    def test_explicit_duplicate_repair_cannot_cross_reparse(self):
        case = LayoutPublicationTests()
        case.setUp(); self.addCleanup(case.doCleanups)
        current = case.repo.load_document('cache-doc')
        duplicate = replace(current, chapters=[replace(chapter, sections=[replace(section,
            task_units=[unit.with_unit_id('duplicate-unit') for unit in section.task_units])
            for section in chapter.sections]) for chapter in current.chapters])
        # File-only legacy corruption: SQL's unique keys cannot represent duplicates.
        (Path(case.repo.base_dir) / 'cache-doc.structured.json').write_text(duplicate.to_json())
        normalizer = case.coordinator.task_unit_id_normalizer
        original = normalizer.normalize_document_task_unit_ids
        captured = {}
        def late(**kwargs):
            candidate = original(**kwargs)
            case.repo.save_reparsed_document(fixture._build_base_document(), 'cache-doc')
            captured['current'] = case.snapshot()
            return candidate
        with patch.object(normalizer, 'normalize_document_task_unit_ids', side_effect=late):
            with self.assertRaisesRegex(ReadingContentConflict, 'task_layout_source_changed'):
                case.prepare()
        self.assertEqual(case.snapshot(), captured['current'])
        case.prepare(); case.content_readback()


if __name__ == '__main__':
    unittest.main(verbosity=2)
