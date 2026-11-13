"""STRUCT-04: late writers, effective body and lifecycle fences with real owners."""
import os
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Event
from uuid import uuid4

from app.section_task_coordinator import SectionTaskCoordinator, TaskUnitResolveOptions
from document_structure.generated_artifact_source import generation_source_matches
from document_structure.reading_content import ReadingContentService, project_document
from document_structure.structured_document_artifact_repository import StructuredDocumentArtifactRepository
from scripts.test_reading_interaction_artifact_store import _document
from scripts import test_summary_note_postgres as pg_fixture
from scripts.artifact_generation_fixtures import bound, bound_fields
from section_tasks.task_unit_split_mode import TaskUnitSplitMode
from shared.task_artifacts import SummaryArtifact, QuizArtifact, TaskArtifacts, DocumentTaskArtifacts


class GeneratedSourceTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        self.repo = StructuredDocumentArtifactRepository(base_dir=temp.name)
        self.seed()

    def seed(self):
        self.repo.save_reparsed_document(replace(_document(), document_id='book'), 'book')
        self.initial = self.repo.load_document('book')
        self.sid = self.initial.chapters[0].sections[0].section_id
        self.uid = self.initial.chapters[0].sections[0].task_units[0].unit_id
        self.cid = self.initial.chapters[0].chapter_id
        self.service = ReadingContentService(self.repo)

    def change(self, action='edit', **kwargs):
        snapshot = self.service.read('book')
        return self.service.mutate('book', action=action, expected_revision=snapshot['revision'],
            source_token=snapshot['source_token'], mutation_id=uuid4().hex,
            **({'block_id': snapshot['blocks'][0]['block_id'], 'text':'New effective reading body.'}
               if action == 'edit' else {'block_id':snapshot['blocks'][0]['block_id']} if action=='delete' else {}), **kwargs)

    def artifacts(self, document=None, label='old'):
        document = document or self.repo.load_document('book')
        return [(kind, level, target, bound(artifact, document, level, target))
            for kind, artifact in [('summary',SummaryArtifact(label)),('quiz',QuizArtifact([{'answer':label}]))]
            for level,target in [('section',self.sid),('chapter',self.cid),('task_unit',self.uid)]]

    def write(self, entry):
        kind,level,target,artifact = entry
        if level=='task_unit':
            return self.repo.update_task_unit_artifacts('book', target, TaskArtifacts(**{kind:artifact}))
        return getattr(self.repo, f'update_{level}_{kind}_artifact')('book',
            'chapter_id::'+target if level=='chapter' else target, artifact)

    def reject_all(self, entries):
        before = self.repo.load_document('book')
        for entry in entries:
            with self.subTest(kind=entry[:3]):
                with self.assertRaisesRegex(ValueError,'generated_artifact_source_changed|unknown task_unit_id'): self.write(entry)
                self.assertEqual(self.repo.load_document('book'),before)

    def test_edit_keep_clear_and_new_generation(self):
        for keep in [True,False]:
            with self.subTest(keep=keep):
                self.seed(); old=self.artifacts()
                self.change(discard_interactions=not keep)
                self.reject_all(old)
                for entry in self.artifacts(label='new'):self.write(entry)
                current=self.repo.load_document('book')
                self.assertEqual(current.chapters[0].sections[0].task_artifacts.summary.content,'new')

    def test_same_text_noop_keeps_generation_valid(self):
        old=self.artifacts(); snapshot=self.service.read('book')
        self.service.mutate('book',action='edit',expected_revision=snapshot['revision'],
            source_token=snapshot['source_token'],mutation_id=uuid4().hex,
            block_id=snapshot['blocks'][0]['block_id'],text=snapshot['blocks'][0]['content'])
        self.assertEqual(self.service.read('book')['revision'],snapshot['revision'])
        for entry in old:self.write(entry)

    def test_delete_undo_and_restored_text_do_not_revive(self):
        old=self.artifacts(); self.change('delete');self.reject_all(old)
        self.change('undo'); self.reject_all(old)
        for entry in self.artifacts(label='restored'):self.write(entry)

    def test_reparse_and_reimport_reject_old_generation(self):
        for reimport in [False,True]:
            self.seed();old=self.artifacts()
            if reimport:self.repo.delete_document('book')
            self.repo.save_reparsed_document(replace(_document(),document_id='book'),'book')
            self.reject_all(old)
            # SQL hierarchy IDs can change after hard replacement.
            doc=self.repo.load_document('book');self.sid=doc.chapters[0].sections[0].section_id
            self.cid=doc.chapters[0].chapter_id;self.uid=doc.chapters[0].sections[0].task_units[0].unit_id
            for entry in self.artifacts():self.write(entry)

    def test_rename_round_trip_does_not_revive(self):
        old=self.artifacts();snapshot=self.repo.load_document('book')
        transform=lambda document,name:replace(document,document_id=name,title=name)
        self.assertTrue(self.repo.rename_document('book','other',transform))
        self.assertTrue(self.repo.rename_document('other','book',transform))
        self.reject_all(old)
        with self.assertRaisesRegex(ValueError,'artifact_generation_epoch_changed'):
            self.repo.save_document(snapshot,'book')
        for entry in self.artifacts():self.write(entry)

    def test_parallel_old_result_cannot_replace_new_result(self):
        old=self.artifacts()[0]; ready=Event(); release=Event()
        def delayed():
            ready.set();self.assertTrue(release.wait(5))
            with self.assertRaisesRegex(ValueError,'generated_artifact_source_changed'):self.write(old)
        with ThreadPoolExecutor(1) as pool:
            future=pool.submit(delayed);self.assertTrue(ready.wait(5))
            try:
                self.change();self.write(self.artifacts(label='fresh')[0])
            finally:release.set()
            future.result(10)
        self.assertEqual(self.repo.load_document('book').chapters[0].sections[0].task_artifacts.summary.content,'fresh')

    def test_combined_and_document_map_cannot_bypass_binding(self):
        old=bound_fields(TaskArtifacts(summary=SummaryArtifact('old')),self.initial,'section',self.sid)
        chapter=bound_fields(old,self.initial,'chapter',self.cid)
        self.change()
        with self.assertRaises(ValueError):self.repo.update_section_artifacts('book',self.sid,old)
        with self.assertRaises(ValueError):self.repo.update_document_artifacts('book',DocumentTaskArtifacts(chapter_artifacts={'chapter_id::'+self.cid:chapter}))

    def test_unbound_legacy_results_are_not_current_writes(self):
        before=self.repo.load_document('book')
        with self.assertRaises(ValueError):self.repo.update_section_summary_artifact('book',self.sid,SummaryArtifact('legacy'))
        self.assertFalse(generation_source_matches(before,SummaryArtifact('legacy'),'section',self.sid))
        self.assertEqual(self.repo.load_document('book'),before)

    def test_production_builder_and_cache_use_effective_source(self):
        self.change();document=self.repo.load_document('book')
        coordinator=object.__new__(SectionTaskCoordinator)
        options=TaskUnitResolveOptions(TaskUnitSplitMode.SEMANTIC_SAFE,None)
        section=document.chapters[0].sections[0]
        summary=coordinator._build_summary_artifact(content='fresh',target_section=section,document=document,
            document_profile=None,resolve_options=options,prompt_version=coordinator._SECTION_SUMMARY_PROMPT_VERSION,
            scope='section',source_task_unit_id=self.uid,chapter_title=None)
        self.repo.update_section_summary_artifact('book',self.sid,summary)
        document=self.repo.load_document('book');section=document.chapters[0].sections[0]
        self.assertNotEqual(section.content,project_document(document).chapters[0].sections[0].content)
        self.assertTrue(coordinator._is_section_summary_cache_valid(section=section,document=document,
            document_profile=None,resolve_options=options,prompt_version=coordinator._SECTION_SUMMARY_PROMPT_VERSION,
            expected_scope='section'))
        self.change('undo'); self.reject_all([('summary','section',self.sid,summary)])

    def test_keep_retains_quiz_payload_but_not_cache_validity(self):
        from section_tasks.quiz_question import QuizQuestion
        coordinator=object.__new__(SectionTaskCoordinator)
        options=TaskUnitResolveOptions(TaskUnitSplitMode.SEMANTIC_SAFE,None)
        quiz=coordinator._build_quiz_artifact(questions=[QuizQuestion('q1','Question?','Answer')],
            target_section=self.initial.chapters[0].sections[0],document=self.initial,
            document_profile=None,resolve_options=options,prompt_version=coordinator._SECTION_QUIZ_PROMPT_VERSION,
            scope='section',source_task_unit_id=self.uid,chapter_title=None)
        self.repo.update_section_quiz_artifact('book',self.sid,quiz)
        self.change(discard_interactions=False)
        current=self.repo.load_document('book'); section=current.chapters[0].sections[0]
        self.assertEqual(section.task_artifacts.quiz,quiz)
        validity,_=coordinator._validate_section_quiz_artifact(section=section,document=current,
            document_profile=None,resolve_options=options,prompt_version=coordinator._SECTION_QUIZ_PROMPT_VERSION,
            expected_scope='section')
        self.assertFalse(validity.cache_valid)
        self.assertEqual(self.repo.load_document('book'),current)
        self.change('undo')
        restored=self.repo.load_document('book')
        # Even exact source text restored by Undo cannot revive the generation epoch.
        self.assertFalse(generation_source_matches(restored,quiz,'section',self.sid))

    def test_actual_coordinator_delayed_provider_and_effective_body(self):
        from types import SimpleNamespace
        from scripts.test_summary_persistence_cache import (_FakePreparationPipeline,
            _FakeProfileStore, _FakeTaskUnitResolver, _FakeEnhancedParseEvaluator)
        from section_tasks.section_task_result import SectionTaskResult
        from section_tasks.quiz_question import QuizQuestion
        for kind in ['summary', 'quiz']:
            for level in ['section', 'chapter']:
                with self.subTest(kind=kind, level=level):
                    self.seed()
                    calls=[]
                    def invoke(task_unit, **kwargs):
                        calls.append(task_unit.content)
                        if len(calls)==1:
                            self.change()
                        return SectionTaskResult.ok('new summary' if kind=='summary' else [
                            QuizQuestion(question_id='q1',question_text='Question?',answer_text='Answer')])
                    coordinator=SectionTaskCoordinator(document_preparation_pipeline=_FakePreparationPipeline(self.repo),
                        document_artifact_repository=self.repo, document_profile_store=_FakeProfileStore(),
                        chapter_summary_service=SimpleNamespace(summarize_task_unit=invoke),
                        chapter_quiz_service=SimpleNamespace(generate_task_unit_quiz=invoke),
                        task_unit_resolver=_FakeTaskUnitResolver(), enhanced_parse_trigger_evaluator=_FakeEnhancedParseEvaluator())
                    coordinator.ensure_document_task_layout(doc_name='book')
                    current=self.repo.load_document('book')
                    self.sid=current.chapters[0].sections[0].section_id
                    self.cid=current.chapters[0].chapter_id
                    method=getattr(coordinator, ('summarize_' if kind=='summary' else 'generate_')+level+('' if kind=='summary' else '_quiz'))
                    args={'doc_name':'book', level+'_id':self.sid if level=='section' else self.cid}
                    first=method(**args)
                    self.assertFalse(first.success)
                    self.assertIn('generated_artifact_source_changed',str(first.reason))
                    second=method(**args)
                    self.assertTrue(second.success,second.reason)
                    self.assertIn('New effective reading body.',calls[-1])
                    third=method(**args)
                    self.assertTrue(third.success,third.reason)
                    self.assertTrue(third.cache_hit)
                    self.assertEqual(len(calls),2)


@unittest.skipUnless(os.environ.get('DEEP_READER_TEST_POSTGRES_DSN'),'requires disposable PostgreSQL DSN')
class PostgresGeneratedSourceTests(GeneratedSourceTests):
    setUpClass=classmethod(pg_fixture.PostgresNotes.setUpClass.__func__)
    drop_schema=classmethod(pg_fixture.PostgresNotes.drop_schema.__func__)
    def setUp(self):
        from db.postgres_structured_document_store import PostgresStructuredDocumentStore
        from db.postgres_structured_document_artifact_repository import PostgresStructuredDocumentArtifactRepository
        self.repo=PostgresStructuredDocumentArtifactRepository(PostgresStructuredDocumentStore(self.scoped,namespace=uuid4().hex))
        self.seed()


class FileGenerationRecoveryTests(unittest.TestCase):
    def test_rename_journal_recovery_keeps_generation_fence(self):
        from pathlib import Path
        from unittest.mock import patch
        from scripts.test_learning_file_repository import repositories, fixture, configure
        from document_structure import learning_file_lifecycle as implementation
        from document_structure.learning_repository import LearningCommitUnknown
        with tempfile.TemporaryDirectory() as directory:
            docs,prefs,learning=repositories(Path(directory))
            docs.save_document(fixture());configure(prefs)
            lifecycle=implementation.FileLearningDocumentLifecycle(docs,learning,prefs)
            old=docs.load_document('book')
            summary=bound(SummaryArtifact('old'),old,'section','section')
            transform=lambda document,name:replace(document,document_id=name,title=name)
            write=implementation.atomic_write
            def crash(path,payload):
                write(path,payload)
                if path==docs.lifecycle_path('book'):
                    raise OSError('after journal and source removal')
            with patch.object(implementation,'atomic_write',side_effect=crash):
                with self.assertRaises(LearningCommitUnknown):docs.rename_document('book','other',transform)
            self.assertTrue(prefs.lifecycle_journal_path.exists())
            self.assertTrue(lifecycle.recover())
            epoch=docs.load_document('other').parse_provenance['artifact_generation_epoch']
            self.assertFalse(lifecycle.recover())
            self.assertEqual(docs.load_document('other').parse_provenance['artifact_generation_epoch'],epoch)
            docs.rename_document('other','book',transform)
            before=docs.load_document('book')
            with self.assertRaisesRegex(ValueError,'generated_artifact_source_changed'):
                docs.update_section_summary_artifact('book','section',summary)
            self.assertEqual(docs.load_document('book'),before)
            docs.update_section_summary_artifact('book','section',bound(SummaryArtifact('new'),before,'section','section'))
            self.assertEqual(docs.load_document('book').chapters[0].sections[0].task_artifacts.summary.content,'new')

if __name__=='__main__':unittest.main(verbosity=2)
