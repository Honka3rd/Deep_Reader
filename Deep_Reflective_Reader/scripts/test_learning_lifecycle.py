"""LLRN-12 shared file/real-SQL source and identity lifecycle acceptance."""
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from dataclasses import replace
import json
import subprocess
import sys
from pathlib import Path
import unittest
from unittest.mock import patch

from document_structure.learning_file_lifecycle import FileLearningDocumentLifecycle
from document_structure.learning_repository import BatchPayload, OperationWrite, LearningCommitUnknown, LearningStorageError
from document_structure.reading_content import ReadingContentService, ReadingContentConflict
from document_structure import learning_file_codec as codec
from reading_preferences.models import BookSettings, LanguageLearningSettings, PreferenceStorageError, PreferenceConflict
from scripts import test_learning_file_repository as file_tests
from scripts import test_learning_postgres_repository as sql_tests
from section_tasks.learning_cache import HintsKey, TranslationKey
from section_tasks.learning_generation_policy import BatchPlan, GenerationBinding, GenerationCommand
from section_tasks.learning_sentence_segmenter import segment_learning_block
from shared.learning_results import (AlignedTranslation, LearningHints, LearningProficiencyRef,
    GenerationProvenance, LearningResultContext, TranslatedSentence, VocabularyHint)
from shared.learning_source import LearningContractError, SourceSpan


def renamed(document, name):
    return replace(document, document_id=name, title=name)


class LifecycleMatrix:
    def change(self, action='edit', *, text='Changed source.', block=None, interaction_policy='keep', **extra):
        reader=ReadingContentService(self.docs); current=reader.read('book')
        args=dict(action=action, expected_revision=current['revision'], source_token=current['source_token'],
                  mutation_id='mutation-'+str(current['revision'])+'-'+action, **extra)
        if action in ('edit','delete'):
            args.update(block_id=(block or self.block).source.content_block_id, discard_interactions=interaction_policy=='clear')
        if action=='edit': args['text']=text
        return reader.mutate('book',**args)

    def save_result(self, block=None, *, level=None, target='zh', label='saved', running=False):
        block=block or self.block
        workspace=self.prefs.read_workspace()
        if workspace.native_language!=target:
            workspace=self.prefs.save_workspace(target,expected_revision=workspace.revision,mutation_id='native-'+label)
        book=self.prefs.read_book('book')
        if level is not None:
            settings=BookSettings('language_learning',None,(LanguageLearningSettings('en','cefr_reading','1',level),))
            book=self.prefs.save_book('book',settings,expected_revision=book.revision,
                expected_workspace_revision=workspace.revision,resolved_source_language='en',mutation_id='level-'+label)
        sentences=segment_learning_block(source=block.source,effective_text=block.effective_text,source_language='en')
        args=(sentences.reference,'en',target,'policy-1','prompt-'+label)
        key=HintsKey(*args,LearningProficiencyRef('cefr_reading','1',level)) if level else TranslationKey(*args)
        plan=BatchPlan((tuple(s.sentence_id for s in sentences.sentences),))
        binding=GenerationBinding(key,workspace.revision,book.revision)
        entry=self.repo.claim('book',GenerationCommand(label,binding),sentences=sentences,plan=plan,
                              now_ms=10,lease_until_ms=100).next_entry
        def write(action,**kw):
            return self.repo.write_operation('book',OperationWrite(entry.operation.operation_id,entry.revision,
                entry.last_fence,20 if action=='dispatch' else 30,action,**kw))
        entry=write('dispatch')
        if running: return key,entry,sentences
        provenance=GenerationProvenance('test','test-model',key.prompt_version)
        rows=tuple(TranslatedSentence(s.sentence_id,'保存譯文') for s in sentences.sentences)
        first=sentences.sentences[0]
        hints=(VocabularyHint(first.sentence_id,SourceSpan(first.span.start,first.span.end),
                block.effective_text[first.span.start:first.span.end],'en','phrase','測試釋義'),) if level else ()
        batch=BatchPayload('hints' if level else 'translation',provenance,rows=() if level else rows,items=hints)
        entry=write('checkpoint',batch_index=0,batch=batch)
        context=LearningResultContext(key.resource,'en',target,key.policy_version,1,provenance)
        result=LearningHints(context,key.proficiency,hints) if level else AlignedTranslation(context,rows)
        entry=write('commit',result=result)
        return key,entry,sentences

    def test_noop_preserves_all_saved_data_and_source_binding(self):
        key,entry,sentences=self.save_result()
        self.change(text=self.block.effective_text)
        self.assertEqual(self.repo.read_sources('book').blocks[0].source,self.block.source)
        self.assertEqual(self.repo.read_batch('book',(key,))[0].view.saved,entry.saved)
        self.assertEqual(self.repo.read_sentences('book',(sentences.reference,))[0].state,'ready')

    def test_keep_and_clear_invalidate_all_languages_levels_but_preserve_other_block(self):
        for policy in ('keep','clear'):
            with self.subTest(policy=policy):
                # Distinct current source for each pass; no stale fixture reuse.
                self.block=self.repo.read_sources('book').blocks[0]
                keys=[self.save_result(label=policy+'-zh')[0],self.save_result(target='fr',label=policy+'-fr')[0],
                      self.save_result(level='B1',label=policy+'-b1')[0],self.save_result(level='C2',label=policy+'-c2')[0]]
                other=self.repo.read_sources('book').blocks[1]
                other_key,other_entry,_=self.save_result(other,label=policy+'-other')
                self.change(text='New source '+policy+'.',interaction_policy=policy)
                self.assertTrue(all(r.view.state=='stale' and r.view.saved is None for r in self.repo.read_batch('book',tuple(keys))))
                self.assertEqual(self.repo.read_batch('book',(other_key,))[0].view.saved,other_entry.saved)
                self.assertEqual(self.repo.read_sources('book').blocks[1].source,other.source)

    def test_undo_same_text_cannot_revive_old_resource_or_late_writer(self):
        saved=self.save_result()[0]
        key,running,_=self.save_result(label='running',running=True)
        self.change();self.change('undo')
        current=self.repo.read_sources('book').blocks[0]
        self.assertEqual(current.effective_text,self.block.effective_text)
        self.assertNotEqual(current.source,self.block.source)
        self.assertEqual(self.repo.read_batch('book',(saved,))[0].view.state,'stale')
        with self.assertRaises(LearningContractError):
            self.repo.write_operation('book',OperationWrite(running.operation.operation_id,running.revision,
                running.last_fence,40,'renew',lease_until_ms=200))
        self.assertTrue(self.repo.lookup_operation('book','running').outcome_uncertain)

    def test_delete_block_then_undo_keeps_old_outputs_stale(self):
        key,_,_=self.save_result()
        self.change('delete')
        self.assertNotIn(self.block.source.content_block_id,[b.source.content_block_id for b in self.repo.read_sources('book').blocks])
        self.assertEqual(self.repo.read_batch('book',(key,))[0].view.state,'stale')
        self.change('undo')
        self.assertEqual(self.repo.read_batch('book',(key,))[0].view.state,'stale')

    def test_legacy_save_cannot_roll_back_document_epoch_or_content_state(self):
        old=self.docs.load_document('book');key,entry,_=self.save_result()
        self.change();self.change('undo')
        self.docs.save_document(old,'book')
        now=self.docs.load_document('book')
        self.assertEqual(now.parse_provenance['interaction_epoch'],old.parse_provenance['interaction_epoch'])
        self.assertNotEqual(self.repo.read_sources('book').blocks[0].source,self.block.source)
        self.assertEqual(self.repo.read_batch('book',(key,))[0].view.state,'stale')

    def test_metadata_save_without_epoch_is_rejected_and_preserves_cache(self):
        key,entry,_=self.save_result()
        doc=self.docs.load_document('book');provenance=dict(doc.parse_provenance);provenance.pop('interaction_epoch',None)
        from document_structure.reading_content import ReadingContentConflict
        with self.assertRaises(ReadingContentConflict):
            self.docs.save_document(replace(doc,title='New title',parse_provenance=provenance),'book')
        self.assertEqual(self.docs.load_document('book'), doc)
        self.docs.save_document(replace(doc,title='New title'),'book')
        self.assertEqual(self.repo.read_batch('book',(key,))[0].view.saved,entry.saved)

    def test_reparse_same_text_invalidates_every_old_binding(self):
        key,_,sentences=self.save_result()
        before=self.docs.load_document('book')
        self.docs.save_reparsed_document(before,'book')
        self.assertNotEqual(self.docs.load_document('book').parse_provenance['interaction_epoch'],
                            before.parse_provenance['interaction_epoch'])
        self.assertEqual(self.repo.read_batch('book',(key,))[0].view.state,'stale')
        with self.assertRaises(LearningContractError): self.repo.save_sentences('book',sentences)
        self.assertFalse(self.prefs.read_book('book').deleted)

    def test_rename_preserves_saved_output_preferences_and_rebinds_ids(self):
        key,entry,sentences=self.save_result()
        hints,hint_entry,hint_sentences=self.save_result(level='C2',label='hints')
        old_settings=self.prefs.read_book('book').settings
        self.assertTrue(self.docs.rename_document('book','renamed',renamed))
        self.assertTrue(self.prefs.read_book('book').deleted)
        self.assertEqual(self.prefs.read_book('renamed').settings,old_settings)
        for old,old_entry,resource in ((key,entry,sentences),(hints,hint_entry,hint_sentences)):
            new_resource=resource.with_document_id('renamed')
            new_key=replace(old,resource=new_resource.reference)
            result=self.repo.read_batch('renamed',(new_key,))[0]
            self.assertEqual(result.view.state,'ready')
            self.assertEqual(result.view.saved.context.result_revision,old_entry.saved.context.result_revision)
            self.assertEqual(result.sentences,new_resource)
            if type(old) is TranslationKey:
                self.assertEqual([r.translated_text for r in result.view.saved.rows],
                                 [r.translated_text for r in old_entry.saved.rows])
            else:
                self.assertEqual(result.view.saved.items[0].meaning,old_entry.saved.items[0].meaning)
                self.assertEqual(result.view.saved.items[0].span,old_entry.saved.items[0].span)
                self.assertEqual(result.view.saved.items[0].sentence_id,new_resource.sentences[0].sentence_id)
        self.assertEqual(self.repo.lookup_command('renamed','saved').operation_id,'saved')

    def test_rename_fences_inflight_work_and_preserves_uncertain_cost(self):
        key,entry,_=self.save_result(label='running',running=True)
        self.docs.rename_document('book','renamed',renamed)
        op=self.repo.lookup_operation('renamed','running')
        self.assertEqual(op.state,'stale');self.assertTrue(op.outcome_uncertain)
        with self.assertRaises(LearningContractError):
            self.repo.write_operation('renamed',OperationWrite('running',entry.revision,entry.last_fence,40,'renew',lease_until_ms=200))

    def test_rename_destination_conflict_has_no_partial_effect(self):
        key,entry,_=self.save_result()
        self.docs.save_reparsed_document(file_tests.fixture('destination'),'destination')
        before=self.prefs.read_book('book')
        with self.assertRaises(ValueError): self.docs.rename_document('book','destination',renamed)
        self.assertEqual(self.prefs.read_book('book'),before)
        self.assertEqual(self.repo.read_batch('book',(key,))[0].view.saved,entry.saved)

    def test_document_delete_tombstones_preferences_and_recreation_has_fresh_identity(self):
        key,entry,_=self.save_result();settings=self.prefs.read_book('book')
        self.assertTrue(self.docs.delete_document('book'))
        self.assertTrue(self.prefs.read_book('book').deleted)
        self.assertGreater(self.prefs.read_book('book').revision,settings.revision)
        self.assertIsNone(self.repo.lookup_operation('book','saved'))
        self.docs.save_reparsed_document(file_tests.fixture(),'book')
        self.assertNotEqual(self.repo.read_sources('book').blocks[0].source,key.resource.source)
        self.assertEqual(self.repo.read_batch('book',(key,))[0].view.state,'stale')


    def ocr_fixture(self):
        from scripts.test_ocr_cleanup_overlay import CleanupOverlayTests
        from scripts.test_ocr_cleanup_overlay_postgres import PostgresCleanupOverlayTests
        harness=CleanupOverlayTests()
        harness.setUp(); self.addCleanup(harness.doCleanups)
        self.docs.save_reparsed_document(harness.repo.load_document('book'),'book')
        harness.repo=self.docs; harness.namespace=self.repo.scope.namespace
        harness.service=ReadingContentService(self.docs)
        if hasattr(self.docs,'commit_enhancement'):
            harness.snapshot=PostgresCleanupOverlayTests.snapshot.__get__(harness)
            harness.commit=PostgresCleanupOverlayTests.commit.__get__(harness)
        self.block=self.repo.read_sources('book').blocks[0]
        return harness

    def test_ocr_draft_keep_and_ignore_preserve_saved_output(self):
        harness=self.ocr_fixture()
        key,entry,_=self.save_result()
        block_id=self.block.source.content_block_id
        for action,args in (
            ('region_edit',dict(region_id='r1',text='Draft correction')),
            ('review_block_keep',dict(region_id='r1',block_id=block_id)),
        ):
            harness.change(action,**args)
            self.assertEqual(self.repo.read_batch('book',(key,))[0].view.saved,entry.saved)
            self.assertEqual(self.repo.read_sources('book').blocks[0].source,self.block.source)

    def test_ignore_all_regions_preserves_source_and_output(self):
        harness=self.ocr_fixture()
        key,entry,_=self.save_result()
        harness.change('discard_all_regions')
        self.assertEqual(self.repo.read_batch('book',(key,))[0].view.saved,entry.saved)
        self.assertEqual(self.repo.read_sources('book').blocks[0].source,self.block.source)

    def test_ocr_apply_invalidates_all_levels_and_undo_cannot_revive(self):
        harness=self.ocr_fixture()
        keys=[self.save_result(label='ocr-translation')[0],self.save_result(level='B1',label='ocr-hints')[0]]
        other=self.repo.read_sources('book').blocks[1]
        other_key,other_entry,_=self.save_result(other,label='ocr-other')
        harness.change('region_edit',region_id='r1',text='Applied correction')
        harness.change('region_apply',region_id='r1',discard_interactions=False)
        self.assertTrue(all(r.view.state=='stale' for r in self.repo.read_batch('book',tuple(keys))))
        self.assertEqual(self.repo.read_batch('book',(other_key,))[0].view.saved,other_entry.saved)
        harness.change('undo')
        self.assertEqual(self.repo.read_sources('book').blocks[0].effective_text,self.block.effective_text)
        self.assertTrue(all(r.view.state=='stale' for r in self.repo.read_batch('book',tuple(keys))))

    def test_cleanup_removal_restore_history_and_noop_bindings(self):
        harness=self.ocr_fixture()
        key,entry,_=self.save_result()
        harness.commit(*harness.prepare(items=[]))
        self.assertEqual(self.repo.read_batch('book',(key,))[0].view.saved,entry.saved)
        harness.commit(*harness.prepare(operation='remove'))
        self.assertEqual(self.repo.read_batch('book',(key,))[0].view.state,'stale')
        audit=harness.service.read('book',include_cleanup_audit=True)['cleanup_audit']
        record=audit['items'][-1]['deletions'][0]['record_id']
        harness.change('cleanup_edit',cleanup_record_id=record,text='Recovered text')
        harness.change('cleanup_restore',cleanup_record_id=record)
        self.assertEqual(self.repo.read_batch('book',(key,))[0].view.state,'stale')
        self.block=self.repo.read_sources('book').blocks[0]
        current,saved,_=self.save_result(label='restored')
        harness.change('cleanup_edit',cleanup_record_id=record,text='Another draft')
        harness.change('cleanup_delete',cleanup_record_id=record)
        self.assertEqual(self.repo.read_batch('book',(current,))[0].view.saved,saved.saved)
        harness.change('undo')
        self.assertEqual(self.repo.read_batch('book',(current,))[0].view.state,'stale')

    def test_rename_preserves_historical_versions_without_reviving_them(self):
        old,_,sentences=self.save_result(label='old')
        self.change()
        self.block=self.repo.read_sources('book').blocks[0]
        current,_,current_sentences=self.save_result(label='current')
        self.docs.rename_document('book','renamed',renamed)
        old=replace(old,resource=sentences.with_document_id('renamed').reference)
        current=replace(current,resource=current_sentences.with_document_id('renamed').reference)
        self.assertEqual([r.view.state for r in self.repo.read_batch('renamed',(old,current))],['stale','ready'])
        self.assertIsNotNone(self.repo.lookup_command('renamed','old'))
        self.assertIsNotNone(self.repo.lookup_command('renamed','current'))

    def test_concurrent_rename_and_late_writer_have_serial_outcomes(self):
        key,entry,_=self.save_result(label='race',running=True)
        barrier=Barrier(2)
        def rename():
            barrier.wait(timeout=5)
            return self.docs.rename_document('book','renamed',renamed)
        def late():
            barrier.wait(timeout=5)
            try:
                self.repo.write_operation('book',OperationWrite('race',entry.revision,entry.last_fence,40,'renew',lease_until_ms=200))
                return 'before_rename'
            except LearningContractError:
                return 'rejected'
        with ThreadPoolExecutor(2) as pool:
            first=pool.submit(rename);second=pool.submit(late)
            self.assertTrue(first.result(timeout=10))
            self.assertIn(second.result(timeout=10),('before_rename','rejected'))
        self.assertEqual(self.repo.lookup_operation('renamed','race').state,'stale')
        self.assertIsNone(self.repo.lookup_operation('book','race'))


    def test_concurrent_delete_and_preference_save_cannot_resurrect_book_settings(self):
        self.save_result()
        workspace=self.prefs.read_workspace();book=self.prefs.read_book('book')
        barrier=Barrier(2)
        def delete():
            barrier.wait(timeout=5)
            return self.docs.delete_document('book')
        def save():
            barrier.wait(timeout=5)
            try:
                self.prefs.save_book('book',replace(book.settings,active_mode='normal'),expected_revision=book.revision,
                    expected_workspace_revision=workspace.revision,resolved_source_language='en',mutation_id='race-setting')
                return 'before_delete'
            except PreferenceConflict:
                return 'rejected'
        with ThreadPoolExecutor(2) as pool:
            first=pool.submit(delete);second=pool.submit(save)
            self.assertTrue(first.result(timeout=10))
            self.assertIn(second.result(timeout=10),('before_delete','rejected'))
        self.assertTrue(self.prefs.read_book('book').deleted)
        self.assertIsNone(self.repo.lookup_operation('book','saved'))


class FileLifecycleTests(LifecycleMatrix,unittest.TestCase):
    def setUp(self):
        file_tests.LearningFileRepositoryTests.setUp(self)
        self.lifecycle=FileLearningDocumentLifecycle(self.docs,self.repo,self.prefs)

    def test_file_journal_failure_gates_reads_and_explicit_recovery_completes(self):
        key,entry,sentences=self.save_result()
        from document_structure import learning_file_lifecycle as implementation
        original=implementation.atomic_write
        calls=[]
        def fail(path,payload):
            calls.append(path)
            if len(calls)==3: raise OSError('injected after source deletion')
            return original(path,payload)
        with patch.object(implementation,'atomic_write',side_effect=fail):
            with self.assertRaises(LearningCommitUnknown): self.docs.rename_document('book','renamed',renamed)
        for read in (lambda:self.prefs.read_book('book'),lambda:self.docs.load_document('renamed'),
                     lambda:self.repo.lookup_operation('book','saved')):
            with self.assertRaises(PreferenceStorageError): read()
        self.assertTrue(self.lifecycle.recover());self.assertFalse(self.lifecycle.recover())
        migrated=replace(key,resource=sentences.with_document_id('renamed').reference)
        self.assertEqual(self.repo.read_batch('renamed',(migrated,))[0].view.state,'ready')
        self.assertTrue(self.prefs.read_book('book').deleted)

    def test_file_delete_recovery_removes_sidecar_and_fences_preferences(self):
        self.save_result()
        from document_structure import learning_file_lifecycle as implementation
        original=implementation.atomic_write
        def fail(path,payload):
            if path==self.prefs.lifecycle_data_path: raise OSError('preference write failure')
            return original(path,payload)
        with patch.object(implementation,'atomic_write',side_effect=fail):
            with self.assertRaises(LearningCommitUnknown): self.docs.delete_document('book')
        self.assertTrue(self.lifecycle.recover())
        self.assertFalse(self.repo.lifecycle_path('book').exists())
        self.assertTrue(self.prefs.read_book('book').deleted)
        self.assertFalse(self.docs.lifecycle_path('book').exists())


    def test_recovery_rejects_corrupt_payload_and_external_conflict_without_writes(self):
        from document_structure import learning_file_lifecycle as implementation
        self.save_result()
        original=implementation.atomic_write
        def fail(path,payload):
            if path==self.docs.lifecycle_path('book'): raise OSError('stop after journal')
            return original(path,payload)
        with patch.object(implementation,'atomic_write',side_effect=fail):
            with self.assertRaises(LearningCommitUnknown): self.docs.rename_document('book','renamed',renamed)
        journal_path=self.prefs.lifecycle_journal_path
        original_journal=journal_path.read_text()
        journal=json.loads(original_journal)
        journal['writes']['learning_destination']['payload']='{}'
        journal_path.write_text(json.dumps(journal))
        snapshot={p:p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        with self.assertRaises(LearningStorageError): self.lifecycle.recover()
        self.assertEqual({p:p.read_bytes() for p in snapshot},snapshot)
        journal_path.write_text(original_journal)
        self.prefs.lifecycle_data_path.write_text('external conflicting write')
        snapshot={p:p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        with self.assertRaises(LearningStorageError): self.lifecycle.recover()
        self.assertEqual({p:p.read_bytes() for p in snapshot},snapshot)

    def test_process_exit_after_source_removal_recovers_on_new_instance(self):
        key,entry,sentences=self.save_result()
        program = r"""
import os,sys
from scripts.test_learning_file_repository import repositories
from scripts.test_learning_lifecycle import renamed
from document_structure import learning_file_lifecycle as lifecycle
from pathlib import Path
docs,prefs,learning=repositories(Path(sys.argv[1]))
coordinator=lifecycle.FileLearningDocumentLifecycle(docs,learning,prefs)
original=lifecycle.atomic_write
def crash(path,payload):
    original(path,payload)
    if path==docs.lifecycle_path('book'): os._exit(73)
lifecycle.atomic_write=crash
docs.rename_document('book','renamed',renamed)
"""
        result=subprocess.run([sys.executable,'-c',program,str(self.root)],capture_output=True,text=True,timeout=20)
        self.assertEqual(result.returncode,73,result.stderr)
        docs,prefs,repo=file_tests.repositories(self.root)
        coordinator=FileLearningDocumentLifecycle(docs,repo,prefs)
        with self.assertRaises(PreferenceStorageError): docs.load_document('renamed')
        self.assertTrue(coordinator.recover())
        migrated=replace(key,resource=sentences.with_document_id('renamed').reference)
        self.assertEqual(repo.read_batch('renamed',(migrated,))[0].view.state,'ready')
        self.assertTrue(prefs.read_book('book').deleted)

    def test_legacy_raw_source_save_a_b_a_never_revives_old_output(self):
        original=self.docs.load_document('book');key,_,_=self.save_result()
        chapter=original.chapters[0];section=chapter.sections[0];unit=section.task_units[0]
        changed=replace(original,chapters=[replace(chapter,sections=[replace(section,
            task_units=[replace(unit,content='Different raw source.'),*section.task_units[1:]])])])
        self.docs.save_document(changed,'book')
        changed_source=self.repo.read_sources('book').blocks[0].source
        self.docs.save_document(original,'book')
        restored=self.repo.read_sources('book').blocks[0]
        self.assertEqual(restored.effective_text,self.block.effective_text)
        self.assertNotEqual(restored.source,self.block.source)
        self.assertNotEqual(restored.source,changed_source)
        self.assertEqual(self.repo.read_batch('book',(key,))[0].view.state,'stale')


class PostgresLifecycleTests(LifecycleMatrix,unittest.TestCase):
    setUpClass=classmethod(sql_tests.LearningPostgresTests.setUpClass.__func__)
    drop_schema=classmethod(sql_tests.LearningPostgresTests.drop_schema.__func__)
    setUp=sql_tests.LearningPostgresTests.setUp

    def test_sql_rename_failure_rolls_back_documents_preferences_and_all_learning_rows(self):
        key,entry,_=self.save_result()
        prefs=self.prefs.read_book('book');snapshot=sql_tests.LearningPostgresTests.snapshot(self)
        from db import learning_lifecycle
        with patch.object(learning_lifecycle.learning_rows,'persist',side_effect=RuntimeError('fault after deletes')):
            with self.assertRaises(RuntimeError): self.docs.rename_document('book','renamed',renamed)
        self.assertEqual(self.prefs.read_book('book'),prefs)
        self.assertEqual(sql_tests.LearningPostgresTests.snapshot(self),snapshot)
        self.assertEqual(self.repo.read_batch('book',(key,))[0].view.saved,entry.saved)
        self.assertEqual(self.docs.load_document('book').document_id,'book')

    def test_sql_delete_failure_rolls_back_cascades(self):
        key,entry,_=self.save_result();snapshot=sql_tests.LearningPostgresTests.snapshot(self)
        from db.postgres_reading_preferences_repository import PostgresReadingPreferencesRepository
        with patch.object(PostgresReadingPreferencesRepository,'apply_document_lifecycle',side_effect=RuntimeError('fault')):
            with self.assertRaises(RuntimeError): self.docs.delete_document('book')
        self.assertEqual(sql_tests.LearningPostgresTests.snapshot(self),snapshot)
        self.assertFalse(self.prefs.read_book('book').deleted)
        self.assertEqual(self.repo.read_batch('book',(key,))[0].view.saved,entry.saved)


if __name__=='__main__': unittest.main()
