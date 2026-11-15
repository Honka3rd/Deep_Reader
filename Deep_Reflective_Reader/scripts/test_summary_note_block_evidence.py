"""TASK-03 optional block evidence vs document version: real file/SQL HTTP."""
from dataclasses import replace
import hashlib
import os
import unittest
from unittest.mock import Mock, patch
from uuid import uuid4

from fastapi.testclient import TestClient
import main
from scripts.test_critical_thinking_publication import CriticalThinkingPublicationTests, PostgresCriticalThinkingPublicationTests
from scripts.test_reading_interaction_artifact_store import _document
from scripts.test_summary_interaction_service import Provider
from section_tasks.summary_interaction_service import SummaryInteractionService
from shared.task_unit_model import TaskUnitContentBlock
from document_structure.reading_content import ReadingContentService

BASE = '/documents/reading-interactions/'


class SummaryNoteBlockEvidenceTests(unittest.TestCase):
    repository = CriticalThinkingPublicationTests.repository
    coordinator = CriticalThinkingPublicationTests.coordinator

    def setUp(self):
        CriticalThinkingPublicationTests.setUp(self)
        self.provider = Provider()
        self.summary = SummaryInteractionService(self.repo, self.provider)
        self.co.summary_interaction_service = self.summary
        self.co.document_preparation_pipeline = Mock()
        self.seed()

    def seed(self, version=37):
        document = _document()
        chapter = document.chapters[0]; section = chapter.sections[0]; unit = section.task_units[0]
        metadata = {'source_hash': hashlib.sha256(unit.content.encode()).hexdigest()}
        if version is not None:
            metadata['source_structure_version'] = version
        block = TaskUnitContentBlock('custom-evidence', unit.content, metadata=metadata)
        document = replace(document, chapters=[replace(chapter, sections=[
            replace(section, task_units=[replace(unit, content_blocks=[block])])])])
        self.repo.save_reparsed_document(document, 'book')
        self.target = self.current_target()
        self.assertNotEqual(self.repo.load_document('book').parse_provenance.get('storage_structure_version'), 37)

    def current_target(self):
        document = self.repo.load_document('book')
        chapter = document.chapters[0]; section = chapter.sections[0]; unit = section.task_units[0]
        return dict(doc_name='book', target_type='content_block', chapter_id=chapter.chapter_id,
                    section_id=section.section_id, task_unit_id=unit.unit_id, content_block_id='custom-evidence')

    def post(self, client, action, target=None, expected=200, **fields):
        response = client.post(BASE + action, json={'target': self.target if target is None else target, **fields})
        self.assertEqual(response.status_code, expected, response.text)
        return response.json()

    def assert_pure_reads(self, client, target):
        before = self.repo.load_document('book')
        with patch.object(self.provider, 'complete_text', side_effect=AssertionError('read generated')), \
                patch.object(self.repo, 'mutate_document_artifacts', side_effect=AssertionError('read wrote')):
            summary = self.post(client, 'summary/read', target)
            note = self.post(client, 'note/read', target)
        self.assertEqual(self.repo.load_document('book'), before)
        self.co.document_preparation_pipeline.prepare_and_load.assert_not_called()
        return summary, note

    def test_http_absent_null_and_distinct_valid_block_version_full_workflows(self):
        for evidence in ({}, {'source_structure_version': None}, {'source_structure_version': 37}):
            with self.subTest(evidence=evidence):
                self.seed()
                target = {**self.target, **evidence}
                with patch.object(main, 'section_task_coordinator', self.co), TestClient(main.app) as client:
                    summary, note = self.assert_pure_reads(client, target)
                    self.assertEqual(summary['envelope']['status'], 'not_generated')
                    generated = self.post(client, 'summary/generate', target)
                    self.assertEqual(generated['envelope']['status'], 'completed')
                    calls = len(self.provider.prompts)
                    snapshot = self.repo.load_document('book')
                    summaries = snapshot.document_task_artifacts.metadata['public_summary'].values()
                    saved = next(a for a in summaries if a['target_level'] == 'content_block')
                    policy = saved['metadata']['interaction_policy']
                    self.assertEqual(policy.get('source_structure_version'), snapshot.parse_provenance.get('storage_structure_version'))
                    self.assertNotEqual(policy.get('source_structure_version'), 37)
                    options = {**target, 'target_level': target['target_type']}
                    options.pop('target_type')
                    resolved = self.co.resolve_reading_target(**options)
                    self.assertEqual(policy['source_sha256'], hashlib.sha256(resolved.content.encode()).hexdigest())
                    self.assertEqual(self.post(client, 'summary/generate', target), generated)
                    self.assertEqual(len(self.provider.prompts), calls)
                    refreshed = self.post(client, 'summary/refresh', target)
                    self.assertEqual(refreshed['envelope']['status'], 'completed')
                    self.assertEqual(len(self.provider.prompts), calls + 1)
                    fields = dict(text='Reader-owned note.', expected_revision=0, mutation_id=uuid4().hex, target_token=note['target_token'])
                    saved_note = self.post(client, 'note/save', target, **fields)
                    self.assertEqual(self.post(client, 'note/save', target, **fields), saved_note)
                    _, reread = self.assert_pure_reads(client, target)
                    self.assertEqual(reread['note'], saved_note['note'])
                    deleted = self.post(client, 'note/delete', target, target_token=reread['target_token'], expected_revision=1)
                    self.assertIsNone(deleted['note'])
                    self.post(client, 'note/save', target, expected=409, **fields)
                    self.assertEqual(self.post(client, 'summary/read', target), refreshed)

    def assert_rejected_all(self, client, bad, opened):
        before = self.repo.load_document('book')
        with patch.object(self.provider, 'complete_text', side_effect=AssertionError('stale generated')):
            for action in ('summary/read', 'summary/generate', 'summary/refresh', 'note/read', 'note/save', 'note/delete'):
                with self.subTest(action=action, evidence=bad):
                    fields = {} if action not in ('note/save', 'note/delete') else dict(target_token=opened['target_token'], expected_revision=1)
                    if action == 'note/save': fields.update(text='Must not replace note.', mutation_id=uuid4().hex)
                    self.post(client, action, bad, expected=409, **fields)
                    self.assertEqual(self.repo.load_document('book'), before)

    def test_wrong_block_version_and_hash_reject_before_generation_or_save(self):
        with patch.object(main, 'section_task_coordinator', self.co), TestClient(main.app) as client:
            opened = self.post(client, 'note/read')
            self.post(client, 'note/save', target_token=opened['target_token'], expected_revision=0, mutation_id=uuid4().hex, text='Keep this note.')
            self.post(client, 'summary/generate')
            for evidence in ({'source_structure_version': 38}, {'source_structure_version': 1}, {'source_hash': 'incorrect'}):
                self.assert_rejected_all(client, {**self.target, **evidence}, opened)

    def test_absent_block_metadata_allows_no_evidence_but_rejects_supplied_version(self):
        self.seed(version=None)
        with patch.object(main, 'section_task_coordinator', self.co), TestClient(main.app) as client:
            for target in (self.target, {**self.target, 'source_structure_version': None}):
                self.assert_pure_reads(client, target)
                self.assertEqual(self.post(client, 'summary/generate', target)['envelope']['status'], 'completed')
            opened = self.post(client, 'note/read')
            self.post(client, 'note/save', target_token=opened['target_token'], expected_revision=0, mutation_id=uuid4().hex, text='Preserved note.')
            self.assert_rejected_all(client, {**self.target, 'source_structure_version': 37}, opened)

    def test_non_block_supplied_document_version_guard_remains(self):
        document = self.repo.load_document('book'); c=document.chapters[0]; s=c.sections[0]; u=s.task_units[0]
        with patch.object(main, 'section_task_coordinator', self.co), TestClient(main.app) as client:
            for level in ('document', 'chapter', 'section', 'task_unit'):
                target = dict(doc_name='book', target_type=level)
                if level != 'document': target['chapter_id'] = c.chapter_id
                if level in ('section', 'task_unit'): target['section_id'] = s.section_id
                if level == 'task_unit': target['task_unit_id'] = u.unit_id
                current_version = document.parse_provenance.get('storage_structure_version')
                target['source_structure_version'] = current_version
                _, opened = self.assert_pure_reads(client, target)
                self.assert_rejected_all(client, {**target, 'source_structure_version': 37}, opened)

    def test_overlay_and_canonical_note_hashes_keep_block_version_contract(self):
        target = {**self.target, 'source_structure_version':37}
        canonical_hash = self.repo.load_document('book').chapters[0].sections[0].task_units[0].content_blocks[0].metadata['source_hash']
        with patch.object(main, 'section_task_coordinator', self.co), TestClient(main.app) as client:
            opened = self.post(client, 'note/read', {**target, 'source_hash': canonical_hash})
            saved = self.post(client, 'note/save', target, target_token=opened['target_token'], expected_revision=0, mutation_id=uuid4().hex, text='Retain across Keep.')
            content = ReadingContentService(self.repo); state=content.read('book')
            content.mutate('book', action='edit', block_id='custom-evidence', text='Changed effective source discussing evidence and explanations.',
                           expected_revision=state['revision'], source_token=state['source_token'], mutation_id=uuid4().hex, discard_interactions=False)
            resolved = self.co.resolve_reading_target(doc_name='book', target_level='content_block',
                task_unit_id=target['task_unit_id'], content_block_id='custom-evidence', source_structure_version=37)
            overlay_hash = resolved.content_block.metadata['source_hash']
            self.assertNotEqual(canonical_hash, overlay_hash)
            for hash_value in (canonical_hash, overlay_hash):
                note = self.post(client, 'note/read', {**target, 'source_hash':hash_value})
                self.assertEqual(note['note'], saved['note'])
            self.post(client, 'note/save', target, expected=409, target_token=opened['target_token'], expected_revision=1, mutation_id=uuid4().hex, text='Old editor')
            current = self.post(client, 'note/read', {**target, 'source_hash':overlay_hash})
            self.post(client, 'note/save', {**target, 'source_hash':overlay_hash}, target_token=current['target_token'], expected_revision=1, mutation_id=uuid4().hex, text='Fresh editor')
            self.post(client, 'summary/read', {**target, 'source_hash':canonical_hash}, expected=409)
            result = self.post(client, 'summary/generate', {**target, 'source_hash':overlay_hash})
            self.assertEqual(result['envelope']['status'], 'completed')
            snapshot = self.repo.load_document('book')
            saved_summary = next(a for a in snapshot.document_task_artifacts.metadata['public_summary'].values() if a['target_level']=='content_block')
            self.assertEqual(saved_summary['metadata']['interaction_policy']['source_sha256'],hashlib.sha256(resolved.content.encode()).hexdigest())
            canonical = {**target, 'source_hash': canonical_hash}
            current = self.post(client, 'note/read', canonical)
            self.post(client, 'note/delete', canonical, target_token=current['target_token'], expected_revision=2)
            self.assertIsNone(self.post(client, 'note/read', {**target, 'source_hash': overlay_hash})['note'])

    def test_late_summary_and_note_editor_cannot_adopt_new_document_generation(self):
        target = {**self.target, 'source_structure_version':37}
        with patch.object(main, 'section_task_coordinator', self.co), TestClient(main.app) as client:
            opened = self.post(client, 'note/read', target)
            self.post(client, 'summary/generate', target)
            snapshots=[]; complete=self.provider.complete_text
            def delayed(prompt):
                self.repo.save_reparsed_document(self.repo.load_document('book'), 'book')
                snapshots.append(self.repo.load_document('book'))
                return complete(prompt)
            with patch.object(self.provider,'complete_text',side_effect=delayed):
                failed=self.post(client,'summary/refresh',target,expected=502)
            self.assertEqual(failed['envelope']['status'],'generation_failed')
            self.assertEqual(self.repo.load_document('book'),snapshots[0])
            fresh_target={**self.current_target(),'source_structure_version':37}
            self.post(client,'note/save',fresh_target,expected=409,target_token=opened['target_token'],expected_revision=0,mutation_id=uuid4().hex,text='Late note')
            self.assertEqual(self.repo.load_document('book'),snapshots[0])
            self.assertEqual(self.post(client,'summary/generate',fresh_target)['envelope']['status'],'completed')

    def test_note_revalidates_block_evidence_inside_mutation(self):
        with patch.object(main,'section_task_coordinator',self.co), TestClient(main.app) as client:
            target={**self.target,'source_structure_version':37}
            opened=self.post(client,'note/read',target)
            original=self.repo.mutate_document_artifacts; snapshots=[]
            def save(name, mutation):
                document=self.repo.load_document(name);c=document.chapters[0];s=c.sections[0];u=s.task_units[0]
                blocks=[replace(b,metadata={**b.metadata,'source_structure_version':38}) for b in u.content_blocks]
                self.repo.save_document(replace(document,chapters=[replace(c,sections=[replace(s,task_units=[replace(u,content_blocks=blocks)])])]),name)
                snapshots.append(self.repo.load_document(name))
                return original(name,mutation)
            with patch.object(self.repo,'mutate_document_artifacts',side_effect=save):
                self.post(client,'note/save',target,expected=409,target_token=opened['target_token'],expected_revision=0,mutation_id=uuid4().hex,text='Late evidence')
            self.assertEqual(self.repo.load_document('book'),snapshots[0])


@unittest.skipUnless(os.environ.get('DEEP_READER_TEST_POSTGRES_DSN'),'requires disposable PostgreSQL DSN')
class PostgresSummaryNoteBlockEvidenceTests(SummaryNoteBlockEvidenceTests):
    setUpClass=classmethod(PostgresCriticalThinkingPublicationTests.setUpClass.__func__)
    drop_schema=classmethod(PostgresCriticalThinkingPublicationTests.drop_schema.__func__)
    repository=PostgresCriticalThinkingPublicationTests.repository

if __name__=='__main__': unittest.main(verbosity=2)
