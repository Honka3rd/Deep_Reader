"""LLRN-06 actual POSIX file/process/restart acceptance; no SQL/LLM."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
import multiprocessing
import os
from pathlib import Path
import tempfile
from threading import Event
import unittest
from unittest.mock import patch

from document_structure.learning_file_repository import FileLearningRepository
from document_structure.learning_repository import BatchPayload, LearningCommitUnknown, LearningStorageError, OperationWrite
from document_structure.structured_document_artifact_repository import StructuredDocumentArtifactRepository
from document_structure.structured_document import StructuredChapter, StructuredDocument, StructuredSection
from document_structure.reading_content import ReadingContentService
from reading_preferences.file_repository import FileReadingPreferencesRepository
from reading_preferences.models import BookSettings, LanguageLearningSettings, WorkspaceScope
from section_tasks.learning_cache import HintsKey, TranslationKey
from section_tasks.learning_generation_policy import BatchPlan, GenerationBinding, GenerationCommand
from section_tasks.learning_sentence_segmenter import segment_learning_block
from shared.learning_results import (AlignedTranslation, GenerationProvenance, LearningHints,
    LearningProficiencyRef, LearningResultContext, TranslatedSentence, VocabularyHint)
from shared.learning_source import LearningContractError, SourceSpan, text_hash
from shared.task_unit_model import TaskUnit


def repositories(root, namespace='default'):
    root=Path(root)
    docs=StructuredDocumentArtifactRepository(base_dir=str(root/namespace/'documents'),namespace=namespace)
    prefs=FileReadingPreferencesRepository(root/'preferences',WorkspaceScope(namespace))
    return docs,prefs,FileLearningRepository(root/'learning',docs,prefs)


def fixture(document_id='book'):
    units=[TaskUnit(unit_id=f'unit-{i}',title='Unit',container_title=None,content=f'Hello {i}! Welcome.',
        source_section_ids=['section'],is_fallback_generated=False,parent_section_id='section') for i in range(3)]
    text='\n\n'.join(u.content for u in units)
    section=StructuredSection('section',0,'Section',2,text,0,len(text),parent_chapter_id='chapter',task_units=units)
    return StructuredDocument(document_id,'Book',None,'en',text,
        chapters=[StructuredChapter('chapter','Chapter',1,None,[section])],
        parse_provenance={'storage_structure_version':1})


def configure(prefs, document_id='book'):
    workspace=prefs.read_workspace()
    if workspace.native_language is None:
        workspace=prefs.save_workspace('zh',expected_revision=workspace.revision,mutation_id='native')
    settings=BookSettings('language_learning',None,(LanguageLearningSettings('en','cefr_reading','1','B1'),))
    prefs.save_book(document_id,settings,expected_revision=prefs.read_book(document_id).revision,
        expected_workspace_revision=workspace.revision,resolved_source_language='en',mutation_id='settings')


def claim_worker(root,command,sentences,plan,barrier,queue):
    repo=repositories(root)[2]
    barrier.wait(15)
    try:
        result=repo.claim('book',command,sentences=sentences,plan=plan,now_ms=10,lease_until_ms=100)
        queue.put((result.outcome,result.receipt.operation_id))
    except Exception as error:
        queue.put(('error',type(error).__name__+':'+str(error)))


def dispatch_worker(root,command,barrier,queue):
    repo=repositories(root)[2]
    barrier.wait(15)
    try:
        queue.put(repo.write_operation('book',command).operation.state)
    except LearningContractError:
        queue.put('conflict')


def crash_worker(root,command,sentences,plan,phase):
    repo=repositories(root)[2]
    original=os.replace
    def crash(src,dst):
        if phase=='after': original(src,dst)
        os._exit(71 if phase=='after' else 70)
    with patch('document_structure.learning_file_repository.os.replace',side_effect=crash):
        repo.claim('book',command,sentences=sentences,plan=plan,now_ms=10,lease_until_ms=100)


class LearningFileRepositoryTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root=Path(temp.name)
        self.docs,self.prefs,self.repo=repositories(self.root)
        self.docs.save_document(fixture())
        configure(self.prefs)
        self.block=self.repo.read_sources('book').blocks[0]
        self.sentences=segment_learning_block(source=self.block.source,effective_text=self.block.effective_text,source_language='en')
        self.key=TranslationKey(self.sentences.reference,'en','zh','policy-1','prompt-1')
        self.binding=GenerationBinding(self.key,1,1)
        self.command=GenerationCommand('first',self.binding)
        self.plan=BatchPlan((tuple(s.sentence_id for s in self.sentences.sentences),))
        self.provenance=GenerationProvenance('test','model','prompt-1')

    def claim(self,command=None,**kwargs):
        return self.repo.claim('book',command or self.command,sentences=self.sentences,
            plan=kwargs.pop('plan',self.plan),now_ms=kwargs.pop('now_ms',10),
            lease_until_ms=kwargs.pop('lease_until_ms',100),**kwargs)

    def write(self,entry,action,now=20,**kwargs):
        return self.repo.write_operation('book',OperationWrite(entry.operation.operation_id,
            entry.revision,entry.last_fence,now,action,**kwargs))

    def batch(self,ids=None):
        return BatchPayload('translation',self.provenance,
            rows=tuple(TranslatedSentence(i,'譯文') for i in (ids or self.plan.groups[0])))

    def partial(self,entry=None):
        running=self.write(entry or self.claim().next_entry,'dispatch')
        return self.write(running,'checkpoint',30,batch_index=0,batch=self.batch())

    def result(self,revision=1):
        return AlignedTranslation(LearningResultContext(self.key.resource,'en','zh','policy-1',revision,
            self.provenance),self.batch().rows)

    def snapshot(self):
        return {str(p.relative_to(self.root)):p.read_bytes() for p in self.root.rglob('*') if p.is_file()}

    def edit(self,action='edit'):
        reader=ReadingContentService(self.docs)
        current=reader.read('book')
        return reader.mutate('book',action=action,expected_revision=current['revision'],
            source_token=current['source_token'],mutation_id='edit-'+str(current['revision']),
            block_id=self.block.source.content_block_id,
            **({'text':'Changed content.','discard_interactions':False} if action=='edit' else {}))

    def test_construct_and_missing_reads_create_nothing(self):
        before=self.snapshot()
        fresh=repositories(self.root)[2]
        self.assertIsNone(fresh.lookup_operation('book','none'))
        self.assertIsNone(fresh.lookup_command('book','none'))
        self.assertEqual(fresh.read_batch('book',(self.key,))[0].view.state,'missing')
        self.assertEqual(fresh.read_sentences('book',(self.sentences.reference,))[0].state,'missing')
        self.assertEqual(self.snapshot(),before)
        self.assertFalse((self.root/'learning').exists())

    def test_source_uses_actual_hierarchy_and_effective_overlays(self):
        self.assertEqual((self.block.source.chapter_id,self.block.source.section_id,self.block.source.task_unit_id),
                         ('chapter','section','unit-0'))
        self.edit()
        block=self.repo.read_sources('book').blocks[0]
        self.assertEqual(block.effective_text,'Changed content.')
        self.assertNotEqual(block.source.content_epoch,self.block.source.content_epoch)
        self.assertEqual(block.source.effective_content_hash,text_hash(block.effective_text))
        self.assertEqual(self.docs.load_document('book').chapters[0].sections[0].task_units[0].content,'Hello 0! Welcome.')

    def test_source_pagination_and_stale_cursor(self):
        first=self.repo.read_sources('book',limit=1)
        second=self.repo.read_sources('book',limit=1,cursor=first.next_cursor)
        third=self.repo.read_sources('book',limit=1,cursor=second.next_cursor)
        self.assertEqual(len({p.blocks[0].source.content_block_id for p in (first,second,third)}),3)
        self.assertIsNone(third.next_cursor)
        self.edit()
        with self.assertRaisesRegex(LearningContractError,'cursor_stale'):
            self.repo.read_sources('book',cursor=first.next_cursor)

    def test_query_limits_and_scope_rejected(self):
        for limit in (0,101,True):
            with self.assertRaises(LearningContractError): self.repo.read_sources('book',limit=limit)
        with self.assertRaises(LearningContractError): self.repo.read_batch('book',(self.key,)*101)
        key=replace(self.key,resource=replace(self.key.resource,source=replace(self.key.resource.source,namespace='other')))
        with self.assertRaisesRegex(LearningContractError,'scope_mismatch'): self.repo.read_batch('book',(key,))
        with self.assertRaises(ValueError): self.repo.lookup_operation('../bad','first')

    def test_sentence_save_reload_dedup_and_read_only(self):
        self.repo.save_sentences('book',self.sentences)
        before=self.snapshot()
        fresh=repositories(self.root)[2]
        found=fresh.read_sentences('book',(self.sentences.reference,))[0]
        self.assertEqual(found.sentences,self.sentences)
        self.assertEqual(found.state,'ready')
        fresh.save_sentences('book',self.sentences)
        self.assertEqual(before,self.snapshot())

    def test_forged_parent_and_raw_source_rejected(self):
        forged=replace(self.sentences.source,chapter_id='forged')
        fake=segment_learning_block(source=forged,effective_text=self.block.effective_text,source_language='en')
        with self.assertRaisesRegex(LearningContractError,'source_changed'): self.repo.save_sentences('book',fake)
        forged=replace(self.sentences.source,effective_content_hash=text_hash('Untrusted.'))
        fake=segment_learning_block(source=forged,effective_text='Untrusted.',source_language='en')
        with self.assertRaisesRegex(LearningContractError,'source_changed'): self.repo.save_sentences('book',fake)

    def test_complete_round_trip_source_unchanged(self):
        before={p:p.read_bytes() for p in (self.root/'default'/'documents').glob('*')}
        complete=self.write(self.partial(),'commit',40,result=self.result())
        fresh=repositories(self.root)[2]
        saved=fresh.read_batch('book',(self.key,))[0]
        self.assertEqual(saved.view.saved,self.result())
        self.assertEqual(saved.sentences,self.sentences)
        self.assertEqual(fresh.lookup_operation('book','first'),complete.operation)
        self.assertEqual(fresh.lookup_command('book','first').operation_id,'first')
        self.assertEqual(before,{p:p.read_bytes() for p in before})

    def test_command_replay_conflict_and_history_survive_restart(self):
        self.write(self.partial(),'commit',40,result=self.result())
        self.claim(GenerationCommand('refresh',self.binding,'regenerate'),now_ms=50)
        fresh=repositories(self.root)[2]
        before=self.snapshot()
        replay=fresh.claim('book',self.command,sentences=self.sentences,plan=self.plan,now_ms=60,lease_until_ms=100)
        self.assertTrue(replay.replay)
        self.assertEqual(fresh.lookup_operation('book',replay.receipt.operation_id).state,'succeeded')
        with self.assertRaisesRegex(LearningContractError,'command_conflict'):
            self.claim(replace(self.command,intent='regenerate'))
        self.assertEqual(before,self.snapshot())

    def test_parallel_process_claims_coalesce(self):
        ctx=multiprocessing.get_context('spawn')
        barrier,queue=ctx.Barrier(6),ctx.Queue()
        processes=[ctx.Process(target=claim_worker,args=(str(self.root),replace(self.command,command_id=f'c{i}'),
            self.sentences,self.plan,barrier,queue)) for i in range(6)]
        outputs=self.run_processes(processes,queue)
        self.assertEqual(sum(o[0]=='claimed' for o in outputs),1,outputs)
        self.assertEqual(sum(o[0]=='coalesced' for o in outputs),5,outputs)
        self.assertEqual(len({o[1] for o in outputs}),1)

    def test_parallel_dispatch_receipt_permits_only_one_sender(self):
        entry=self.claim().next_entry
        cmd=OperationWrite('first',entry.revision,entry.last_fence,20,'dispatch')
        ctx=multiprocessing.get_context('spawn')
        barrier,queue=ctx.Barrier(4),ctx.Queue()
        outcomes=self.run_processes([ctx.Process(target=dispatch_worker,args=(str(self.root),cmd,barrier,queue))
                                     for _ in range(4)],queue)
        self.assertEqual(outcomes.count('running'),1)
        self.assertEqual(outcomes.count('conflict'),3)

    def run_processes(self,processes,queue):
        try:
            for process in processes: process.start()
            outputs=[queue.get(timeout=30) for _ in processes]
            for process in processes:
                process.join(30)
                self.assertEqual(process.exitcode,0)
            return outputs
        finally:
            for process in processes:
                if process.is_alive(): process.terminate(); process.join(5)
            queue.close()

    def test_pre_replace_failure_leaves_no_claim(self):
        with patch('document_structure.learning_file_repository.os.fsync',side_effect=OSError('injected')):
            with self.assertRaises(LearningStorageError): self.claim()
        self.assertIsNone(self.repo.lookup_command('book','first'))
        self.assertFalse(list((self.root/'learning').glob('*.tmp')))

    def test_post_replace_failure_is_unknown_and_reconciled_without_reclaim(self):
        with patch.object(self.repo,'_sync_directory',side_effect=OSError('injected')):
            with self.assertRaises(LearningCommitUnknown): self.claim()
        self.assertIsNotNone(self.repo.lookup_command('book','first'))
        before=self.snapshot()
        self.assertTrue(self.claim().replay)
        self.assertEqual(self.snapshot(),before)

    def test_process_death_before_replace_ignores_temp_until_explicit_recovery(self):
        self.crash('before',70)
        self.assertIsNone(self.repo.lookup_operation('book','first'))
        self.assertEqual(len(list((self.root/'learning').glob('*.tmp'))),1)
        before=self.snapshot()
        self.repo.read_batch('book',(self.key,))
        self.assertEqual(before,self.snapshot())
        self.assertEqual(self.repo.recover('book'),1)
        self.assertEqual(self.repo.recover('book'),0)
        self.assertEqual(self.claim().outcome,'claimed')

    def test_process_death_after_replace_retains_durable_receipt(self):
        self.crash('after',71)
        self.assertTrue(self.claim().replay)
        self.assertEqual(self.repo.lookup_operation('book','first').state,'claimed')
        self.assertEqual(self.repo.recover('book'),0)

    def crash(self,phase,code):
        ctx=multiprocessing.get_context('spawn')
        process=ctx.Process(target=crash_worker,args=(str(self.root),self.command,self.sentences,self.plan,phase))
        process.start(); process.join(30)
        if process.is_alive(): process.terminate(); process.join(5)
        self.assertEqual(process.exitcode,code)

    def test_lost_dispatch_ack_never_dispatches_twice(self):
        entry=self.claim().next_entry
        with patch.object(self.repo,'_sync_directory',side_effect=OSError('injected')):
            with self.assertRaises(LearningCommitUnknown): self.write(entry,'dispatch')
        fresh=repositories(self.root)[2]
        self.assertEqual(fresh.lookup_operation('book','first').state,'running')
        with self.assertRaisesRegex(LearningContractError,'revision_conflict'): self.write(entry,'dispatch')
        op=fresh.lookup_operation('book','first')
        expired=fresh.write_operation('book',OperationWrite('first',2,op.fence,100,'expire'))
        self.assertEqual(expired.operation.state,'unknown')
        self.assertEqual(self.claim(replace(self.command,command_id='new'),now_ms=110,lease_until_ms=200).outcome,'reconcile')

    def test_lost_final_commit_ack_keeps_output_and_success_receipt_together(self):
        partial=self.partial()
        with patch.object(self.repo,'_sync_directory',side_effect=OSError('injected')):
            with self.assertRaises(LearningCommitUnknown): self.write(partial,'commit',40,result=self.result())
        fresh=repositories(self.root)[2]
        self.assertEqual(fresh.lookup_operation('book','first').state,'succeeded')
        self.assertEqual(fresh.read_batch('book',(self.key,))[0].view.saved,self.result())
        with self.assertRaisesRegex(LearningContractError,'revision_conflict'):
            self.write(partial,'commit',40,result=self.result())

    def test_failed_refresh_and_commit_rollback_preserve_old_result(self):
        self.write(self.partial(),'commit',40,result=self.result())
        entry=self.claim(GenerationCommand('refresh',self.binding,'regenerate'),now_ms=50).next_entry
        running=self.write(entry,'dispatch',60)
        partial=self.write(running,'checkpoint',70,batch_index=0,batch=self.batch())
        with patch('document_structure.learning_file_repository.os.fsync',side_effect=OSError('injected')):
            with self.assertRaises(LearningStorageError): self.write(partial,'commit',80,result=self.result(2))
        self.assertEqual(self.repo.read_batch('book',(self.key,))[0].view.saved,self.result())
        failed=self.write(partial,'fail',80,reason='worker_stopped',outcome_known=True)
        self.assertEqual(failed.saved,self.result())
        self.assertEqual(failed.operation.state,'failed')

    def test_partial_payload_validation_and_assembly(self):
        ids=self.plan.groups[0]
        entry=self.claim(plan=BatchPlan(tuple((i,) for i in ids))).next_entry
        running=self.write(entry,'dispatch')
        with self.assertRaisesRegex(LearningContractError,'alignment_invalid'):
            self.write(running,'checkpoint',30,batch_index=0,batch=self.batch())
        partial=self.write(running,'checkpoint',30,batch_index=0,batch=self.batch((ids[0],)))
        self.assertEqual(self.repo.read_batch('book',(self.key,))[0].view.state,'partial')
        with self.assertRaises(LearningContractError): self.write(partial,'commit',40,result=self.result())
        running=self.write(partial,'dispatch',40)
        partial=self.write(running,'checkpoint',50,batch_index=1,batch=self.batch((ids[1],)))
        self.assertEqual(len(self.repo.read_checkpoints('book','first',limit=1)),1)
        self.assertEqual(len(self.repo.read_checkpoints('book','first',offset=1,limit=1)),1)
        changed=replace(self.result(),rows=tuple(replace(r,translated_text='different') for r in self.result().rows))
        with self.assertRaisesRegex(LearningContractError,'checkpoint_result_mismatch'):
            self.write(partial,'commit',60,result=changed)
        self.assertEqual(self.write(partial,'commit',60,result=self.result()).operation.state,'succeeded')

    def test_empty_hints_is_persisted_ready_and_translation_independent(self):
        key=HintsKey(self.key.resource,'en','zh','policy-1','prompt-1',LearningProficiencyRef('cefr_reading','1','B1'))
        cmd=GenerationCommand('hints',replace(self.binding,key=key))
        running=self.write(self.claim(cmd).next_entry,'dispatch')
        partial=self.write(running,'checkpoint',30,batch_index=0,batch=BatchPayload('hints',self.provenance))
        result=LearningHints(self.result().context,key.proficiency,())
        self.write(partial,'commit',40,result=result)
        views=self.repo.read_batch('book',(self.key,key))
        self.assertEqual([v.view.state for v in views],['missing','ready'])
        self.assertEqual(views[1].view.saved.items,())
        self.assertEqual(self.claim(replace(cmd,command_id='reuse')).outcome,'reuse')

    def test_hint_assessment_must_match_durable_checkpoint(self):
        from shared.learning_hint_assessment import HintAssessment, HintIssue
        key=HintsKey(self.key.resource,'en','zh','policy-1','prompt-1',LearningProficiencyRef('cefr_reading','1','B1'))
        cmd=GenerationCommand('hints',replace(self.binding,key=key))
        running=self.write(self.claim(cmd).next_entry,'dispatch')
        report=HintAssessment(1,0,0,1,0,(HintIssue('invalid_source',1),))
        partial=self.write(running,'checkpoint',30,batch_index=0,
            batch=BatchPayload('hints',self.provenance,assessment=report))
        result=LearningHints(self.result().context,key.proficiency,(),assessment=report)
        forged=replace(report,candidate_count=2,filtered_count=1)
        with self.assertRaisesRegex(LearningContractError,'assessment_mismatch'):
            self.write(partial,'commit',40,result=replace(result,assessment=forged))
        self.write(partial,'commit',40,result=result)
        self.assertEqual(self.repo.read_batch('book',(key,))[0].view.saved.assessment,report)

    def test_hint_wrong_source_span_rejected_before_checkpoint(self):
        key=HintsKey(self.key.resource,'en','zh','policy-1','prompt-1',LearningProficiencyRef('cefr_reading','1','B1'))
        running=self.write(self.claim(GenerationCommand('hints',replace(self.binding,key=key))).next_entry,'dispatch')
        item=VocabularyHint(self.sentences.sentences[0].sentence_id,SourceSpan(0,5),'wrong','en','word','解釋')
        with self.assertRaisesRegex(LearningContractError,'source_mismatch'):
            self.write(running,'checkpoint',30,batch_index=0,batch=BatchPayload('hints',self.provenance,items=(item,)))
        self.assertEqual(self.repo.read_checkpoints('book','hints'),())

    def test_settings_change_blocks_dispatch(self):
        entry=self.claim().next_entry
        self.prefs.save_workspace('fr',expected_revision=1,mutation_id='changed')
        with self.assertRaisesRegex(LearningContractError,'settings_changed'): self.write(entry,'dispatch')
        self.prefs.save_workspace('zh',expected_revision=2,mutation_id='restored')
        with self.assertRaisesRegex(LearningContractError,'binding_stale'): self.write(entry,'dispatch')

    def test_mode_change_blocks_provider_dispatch(self):
        entry=self.claim().next_entry
        settings=replace(self.prefs.read_book('book').settings,active_mode='normal')
        self.prefs.save_book('book',settings,expected_revision=1,expected_workspace_revision=1,
                             resolved_source_language='en',mutation_id='normal')
        with self.assertRaisesRegex(LearningContractError,'mode_inactive'): self.write(entry,'dispatch')

    def test_source_edit_and_undo_never_accept_late_writer(self):
        partial=self.partial()
        self.edit()
        self.assertEqual(self.repo.read_batch('book',(self.key,))[0].view.state,'stale')
        self.assertEqual(self.repo.read_sentences('book',(self.sentences.reference,))[0].state,'stale')
        with self.assertRaisesRegex(LearningContractError,'source_changed'): self.write(partial,'commit',40,result=self.result())
        reader=ReadingContentService(self.docs)
        current=reader.read('book')
        reader.mutate('book',action='undo',expected_revision=current['revision'],source_token=current['source_token'],mutation_id='undo')
        self.assertEqual(self.repo.read_sources('book').blocks[0].effective_text,self.block.effective_text)
        self.assertEqual(self.repo.read_batch('book',(self.key,))[0].view.state,'stale')

    def test_deleted_block_is_not_resolvable_or_recreated_by_read(self):
        self.repo.save_sentences('book',self.sentences)
        self.edit('delete')
        before=self.snapshot()
        self.assertNotIn(self.block.source.content_block_id,[b.source.content_block_id for b in self.repo.read_sources('book').blocks])
        self.assertEqual(self.repo.read_batch('book',(self.key,))[0].view.state,'stale')
        self.assertEqual(before,self.snapshot())

    def test_cross_book_same_block_id_isolation(self):
        self.claim()
        self.docs.save_document(fixture('other')); configure(self.prefs,'other')
        other=self.repo.read_sources('other').blocks[0]
        self.assertEqual(other.source.content_block_id,self.block.source.content_block_id)
        sentences=segment_learning_block(source=other.source,effective_text=other.effective_text,source_language='en')
        key=replace(self.key,resource=sentences.reference)
        self.assertEqual(self.repo.read_batch('other',(key,))[0].view.state,'missing')
        self.assertIsNone(self.repo.lookup_operation('other','first'))
        with self.assertRaisesRegex(LearningContractError,'scope_mismatch'): self.repo.read_batch('other',(self.key,))

    def test_namespaces_and_copied_snapshot_cannot_leak(self):
        self.claim()
        docs,prefs,other=repositories(self.root,'separate')
        docs.save_document(fixture()); configure(prefs)
        self.assertIsNone(other.lookup_operation('book','first'))
        other._path('book').write_bytes(self.repo._path('book').read_bytes())
        with self.assertRaises(LearningStorageError): other.lookup_operation('book','first')

    def test_corrupt_unknown_schema_duplicate_keys_and_digest_fail_closed(self):
        self.partial()
        path=self.repo._path('book'); valid=path.read_text()
        for damaged in ('{bad',valid.replace('"schema_version":1','"schema_version":9'),
                        valid.replace('"schema_version":1','"schema_version":1,"schema_version":1'),
                        valid.replace('譯文','corrupted')):
            path.write_text(damaged); before=path.read_bytes()
            with self.assertRaises(LearningStorageError): self.repo.read_batch('book',(self.key,))
            self.assertEqual(path.read_bytes(),before)
        path.write_text(valid)

    def test_read_io_failure_is_not_missing(self):
        self.claim()
        with patch.object(Path,'read_text',side_effect=PermissionError('private')):
            with self.assertRaisesRegex(LearningStorageError,'read_failed'): self.repo.lookup_operation('book','first')

    def test_unknown_manual_retry_fences_previous_operation(self):
        running=self.write(self.claim().next_entry,'dispatch')
        unknown=self.write(running,'expire',100)
        newer=self.claim(GenerationCommand('retry',self.binding,'retry','first',True),now_ms=110,lease_until_ms=200).next_entry
        self.assertGreater(newer.last_fence,unknown.last_fence)
        with self.assertRaises(LearningContractError): self.write(unknown,'checkpoint',120,batch_index=0,batch=self.batch())
        self.assertEqual(self.repo.lookup_operation('book','first').state,'unknown')

    def test_lease_renewal_is_explicit_and_durable(self):
        running=self.write(self.claim().next_entry,'dispatch'); before=self.snapshot()
        self.repo.lookup_operation('book','first')
        self.assertEqual(before,self.snapshot())
        renewed=self.write(running,'renew',50,lease_until_ms=200)
        self.assertEqual(repositories(self.root)[2].lookup_operation('book','first').lease_until_ms,200)
        with self.assertRaises(LearningContractError): self.write(renewed,'renew',200,lease_until_ms=300)

    def test_source_and_preference_writers_wait_for_learning_commit(self):
        entered,release=Event(),Event(); original=self.repo._commit
        def held(*args):
            entered.set()
            if not release.wait(10): raise AssertionError('test timed out')
            return original(*args)
        with ThreadPoolExecutor(max_workers=3) as pool,patch.object(self.repo,'_commit',side_effect=held):
            learning=pool.submit(self.claim)
            self.assertTrue(entered.wait(5))
            preferences=pool.submit(self.prefs.save_workspace,'fr',expected_revision=1,mutation_id='changed')
            source=pool.submit(self.edit)
            try:
                with self.assertRaises(TimeoutError): preferences.result(timeout=.05)
                with self.assertRaises(TimeoutError): source.result(timeout=.05)
            finally: release.set()
            self.assertEqual(learning.result(timeout=5).outcome,'claimed')
            preferences.result(timeout=5); source.result(timeout=5)
        self.assertEqual(self.repo.read_batch('book',(self.key,))[0].view.state,'stale')

    def test_checkpoint_read_rejects_stale_source_but_status_remains_readable(self):
        self.partial()
        self.edit()
        with self.assertRaisesRegex(LearningContractError,'source_changed'):
            self.repo.read_checkpoints('book','first')
        self.assertEqual(self.repo.lookup_operation('book','first').state,'partial')

    def test_checkpoint_failure_rolls_back_payload_and_marker_together(self):
        running=self.write(self.claim().next_entry,'dispatch')
        with patch('document_structure.learning_file_repository.os.fsync',side_effect=OSError('injected')):
            with self.assertRaises(LearningStorageError):
                self.write(running,'checkpoint',30,batch_index=0,batch=self.batch())
        self.assertEqual(self.repo.lookup_operation('book','first').state,'running')
        self.assertEqual(self.repo.read_checkpoints('book','first'),())
        self.assertEqual(self.write(running,'checkpoint',30,batch_index=0,batch=self.batch()).operation.state,'partial')

    def test_settings_change_blocks_result_commit_and_preserves_checkpoint(self):
        partial=self.partial()
        self.prefs.save_workspace('fr',expected_revision=1,mutation_id='changed')
        with self.assertRaisesRegex(LearningContractError,'settings_changed'):
            self.write(partial,'commit',40,result=self.result())
        self.assertEqual(self.repo.lookup_operation('book','first').state,'partial')
        self.assertEqual(len(self.repo.read_checkpoints('book','first')),1)

    def test_level_change_reuses_translation_key_but_blocks_old_hints(self):
        self.write(self.partial(),'commit',40,result=self.result())
        settings=self.prefs.read_book('book').settings
        settings=replace(settings,modes=(replace(settings.modes[0],level_id='B2'),))
        self.prefs.save_book('book',settings,expected_revision=1,expected_workspace_revision=1,
                            resolved_source_language='en',mutation_id='level')
        command=GenerationCommand('reuse',replace(self.binding,book_revision=2))
        self.assertEqual(self.claim(command,now_ms=50).outcome,'reuse')
        old=HintsKey(self.key.resource,'en','zh','policy-1','prompt-1',LearningProficiencyRef('cefr_reading','1','B1'))
        with self.assertRaisesRegex(LearningContractError,'settings_changed'):
            self.claim(GenerationCommand('old-level',replace(command.binding,key=old)),now_ms=60)

    def test_invalid_hierarchy_and_projected_document_are_rejected(self):
        document=self.docs.load_document('book')
        chapter=document.chapters[0]
        bad=replace(document,chapters=[chapter,chapter])
        with patch.object(self.docs,'load_document',return_value=bad):
            with self.assertRaisesRegex(LearningContractError,'hierarchy_ambiguous'):
                self.repo.read_sources('book')
        bad=replace(document,chapters=[])
        with patch.object(self.docs,'load_document',return_value=bad):
            with self.assertRaisesRegex(LearningContractError,'document_invalid'):
                self.repo.read_sources('book')
        bad=replace(document,parse_provenance={**document.parse_provenance,'reading_projected':True})
        with patch.object(self.docs,'load_document',return_value=bad):
            with self.assertRaisesRegex(LearningContractError,'document_projected'):
                self.repo.read_sources('book')

    def test_claim_does_not_accept_client_preference_revision_or_language(self):
        with self.assertRaisesRegex(LearningContractError,'binding_stale'):
            self.claim(replace(self.command,binding=replace(self.binding,workspace_revision=99)))
        key=replace(self.key,source_language='fr')
        with self.assertRaisesRegex(LearningContractError,'settings_changed'):
            self.claim(replace(self.command,binding=replace(self.binding,key=key)))
        self.assertIsNone(self.repo.lookup_command('book','first'))

    def test_same_command_id_across_components_is_a_conflict(self):
        self.claim()
        hints=HintsKey(self.key.resource,'en','zh','policy-1','prompt-1',LearningProficiencyRef('cefr_reading','1','B1'))
        with self.assertRaisesRegex(LearningContractError,'command_conflict'):
            self.claim(replace(self.command,binding=replace(self.binding,key=hints)))

    def test_admission_denial_persists_receipt_without_claim(self):
        from section_tasks.learning_generation_policy import AdmissionDecision
        class Deny:
            def evaluate(self,command):
                return AdmissionDecision(False,'test_denied')
        repo=FileLearningRepository(self.root/'learning',self.docs,self.prefs,admission=Deny())
        decision=repo.claim('book',self.command,sentences=self.sentences,plan=self.plan,now_ms=10,lease_until_ms=100)
        self.assertEqual(decision.outcome,'denied')
        self.assertEqual(repo.lookup_command('book','first').admission_reason,'test_denied')
        self.assertIsNone(repo.lookup_operation('book','first'))
        self.assertTrue(self.claim().replay)

    def test_expired_unknown_response_can_be_reconciled_without_new_claim(self):
        running=self.write(self.claim().next_entry,'dispatch')
        unknown=self.write(running,'expire',100)
        recovered=self.write(unknown,'checkpoint',110,batch_index=0,batch=self.batch())
        complete=self.write(recovered,'commit',120,result=self.result())
        self.assertEqual(complete.operation.state,'succeeded')
        self.assertEqual(complete.last_fence,1)

    def test_corrupt_canonical_snapshot_is_not_repaired_from_temp(self):
        self.claim()
        path=self.repo._path('book')
        orphan=path.with_name(path.name+'.abandoned.tmp')
        orphan.write_bytes(path.read_bytes())
        path.write_text('broken')
        with self.assertRaises(LearningStorageError): self.repo.recover('book')
        self.assertEqual(path.read_text(),'broken')
        self.assertTrue(orphan.exists())

    def test_missing_claim_receipt_cannot_silently_reenable_execution(self):
        self.claim()
        path=self.repo._path('book')
        payload=json.loads(path.read_text())
        payload['receipts']={}
        path.write_text(json.dumps(payload))
        with self.assertRaises(LearningStorageError): self.claim()

    def test_committed_output_must_match_durable_checkpoint(self):
        self.write(self.partial(),'commit',40,result=self.result())
        path=self.repo._path('book')
        payload=json.loads(path.read_text())
        entry=next(iter(payload['entries'].values()))
        entry['fields']['saved']['fields']['rows']['tuple'][0]['fields']['translated_text']='altered'
        path.write_text(json.dumps(payload))
        with self.assertRaises(LearningStorageError): self.repo.read_batch('book',(self.key,))

    def test_commit_observability_excludes_source_and_identity(self):
        name='document_structure.learning_file_repository'
        with self.assertLogs(name,level='INFO') as captured:
            self.claim()
        record=captured.records[0]
        self.assertEqual(record.outcome,'committed')
        self.assertEqual(record.stage,'snapshot_commit')
        self.assertEqual(record.entry_count,1)
        self.assertGreaterEqual(record.duration_ms,0)
        self.assertNotIn(self.block.effective_text,str(record.__dict__))
        self.assertNotIn('document_id',record.__dict__)

    def test_action_fields_and_result_mismatch_rejected_without_writes(self):
        with self.assertRaises(LearningContractError): OperationWrite('x',0,1,1,'dispatch',reason='bad')
        partial=self.partial(); before=self.snapshot()
        result=replace(self.result(),context=replace(self.result().context,provenance=replace(self.provenance,model='other')))
        with self.assertRaisesRegex(LearningContractError,'provenance_mismatch'): self.write(partial,'commit',40,result=result)
        self.assertEqual(before,self.snapshot())


if __name__=='__main__':
    unittest.main()
