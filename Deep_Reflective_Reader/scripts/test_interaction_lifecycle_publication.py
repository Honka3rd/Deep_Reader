"""TASK-01: genuine file/SQL mutations, deterministic providers, no live calls."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import os
import json
import unittest
import threading
from unittest.mock import patch
from uuid import uuid4

from fastapi.testclient import TestClient
import main
from document_structure.reading_content import ReadingContentService
from section_tasks.analysis_interaction_orchestrator import AnalysisInteractionOrchestrator
from section_tasks.analysis_interaction_service import AnalysisInteractionService
from section_tasks.quiz_interaction_orchestrator import QuizInteractionOrchestrator
from section_tasks.quiz_interaction_service import QuizInteractionService
from section_tasks.summary_interaction_service import SummaryInteractionService
from section_tasks.note_interaction_service import NoteInteractionService, NoteConflict
from section_tasks.interaction_rename import renamed_interactions
from scripts.test_critical_thinking_publication import (
    CriticalThinkingPublicationTests, PostgresCriticalThinkingPublicationTests,
)
from scripts.test_reading_interaction_artifact_store import _document
from scripts.test_summary_interaction_service import Provider


class InteractionLifecycleTests(unittest.TestCase):
    setUp = CriticalThinkingPublicationTests.setUp
    repository = CriticalThinkingPublicationTests.repository
    coordinator = CriticalThinkingPublicationTests.coordinator

    def seed(self):
        self.repo.delete_document('book')
        self.repo.save_reparsed_document(replace(_document(), document_id='book'), 'book')
        self.provider = Provider()
        self.summary = SummaryInteractionService(self.repo, self.provider)
        self.co.summary_interaction_service = self.summary
        self.notes = NoteInteractionService(self.repo)
        self.content = ReadingContentService(self.repo)
        state = self.notes.read('book', target_level='document')
        self.note = self.notes.save('book', target_level='document', text='Preserved reader note',
            target_token=state['target_token'], expected_revision=0, mutation_id=uuid4().hex)['note']

    def request(self, kind):
        return self.co._build_reading_interaction_request(**self.args, interaction_type=kind)

    def output(self, kind):
        if kind == 'analysis':
            return {'summary':'Incentives shape decisions.', 'reasoning':'The source links assumptions with conclusions.',
                    'explanation':'The argument depends on evidence supporting its assumptions.'}
        return {'items':[{'type':'short_answer','question':'Which assumption supports the conclusion?',
                          'answer':'The evidence supports the stated relationship.'}]}

    def orchestrator(self, kind, callback=None):
        def generate(*args):
            if callback: callback()
            return self.output(kind)
        if kind == 'analysis':
            return AnalysisInteractionOrchestrator(service=AnalysisInteractionService(generate), artifact_store=self.store)
        return QuizInteractionOrchestrator(service=QuizInteractionService(generate), artifact_store=self.store)

    def generate(self, kind, request=None, callback=None):
        request = request or self.request(kind)
        if kind != 'summary':
            return self.orchestrator(kind, callback).generate(request)
        original = self.provider.complete_text
        def complete(prompt):
            if callback: callback()
            return original(prompt)
        with patch.object(self.provider, 'complete_text', side_effect=complete):
            return self.summary.generate(request)

    def edit(self, action='edit', same=False, clear=False):
        state = self.content.read('book')
        fields = {} if action == 'undo' else {'block_id':state['blocks'][0]['block_id']}
        if action == 'edit':
            fields['text'] = state['blocks'][0]['content'] if same else 'Changed reading body explaining evidence and conclusions.'
            fields['discard_interactions'] = clear
        self.content.mutate('book', action=action, expected_revision=state['revision'],
                            source_token=state['source_token'], mutation_id=uuid4().hex, **fields)

    def change(self, transition):
        if transition in ('reparse', 'reimport'):
            if transition == 'reimport': self.repo.delete_document('book')
            self.repo.save_reparsed_document(replace(_document(), document_id='book'), 'book')
        elif transition == 'rename':
            for old, new in [('book','away'),('away','book')]:
                self.repo.rename_document(old, new, renamed_interactions)
        elif transition == 'noop': self.edit(same=True)
        else:
            self.edit(action='delete' if transition == 'delete_undo' else 'edit', clear=transition=='clear')
            if transition in ('undo','delete_undo'): self.edit('undo')

    def test_delayed_generation_matrix_preserves_current_snapshot(self):
        for kind in ('analysis','quiz','summary'):
            for transition in ('reparse','reimport','rename','keep','clear','undo','delete_undo'):
                with self.subTest(kind=kind, transition=transition):
                    self.seed(); snapshots=[]
                    def changed():
                        self.change(transition)
                        snapshots.append(self.repo.load_document('book'))
                    result = None
                    try: result = self.generate(kind, callback=changed)
                    except ValueError as error: self.assertIn('stale',str(error))
                    self.assertEqual(len(snapshots),1)
                    self.assertEqual(self.repo.load_document('book'),snapshots[0])
                    if result: self.assertNotEqual(result.status,'completed')
                    fresh = self.generate(kind)
                    self.assertEqual(fresh.status,'completed',fresh.reason)

    def test_old_request_cannot_reuse_cache_after_rename_roundtrip(self):
        for kind in ('analysis','quiz','summary'):
            with self.subTest(kind=kind):
                self.seed(); request=self.request(kind); self.generate(kind,request)
                self.change('rename');before=self.repo.load_document('book')
                reader=self.summary if kind=='summary' else self.orchestrator(kind)
                self.assertEqual(reader.read(request).status,'stale_target')
                self.assertEqual(self.repo.load_document('book'),before)

    def test_old_summary_target_cannot_adopt_new_epoch_before_dispatch(self):
        for transition in ('reparse','reimport','rename','undo'):
            with self.subTest(transition=transition):
                self.seed(); request=self.request('summary'); self.change(transition)
                before=self.repo.load_document('book')
                result=self.summary.generate(request)
                self.assertEqual(result.status,'stale_target')
                self.assertFalse(self.provider.prompts)
                self.assertEqual(self.repo.load_document('book'),before)

    def test_same_text_noop_reuse_and_refresh_preserve_notes(self):
        for kind in ('analysis','quiz','summary'):
            with self.subTest(kind=kind):
                self.seed(); request=self.request(kind)
                first=self.generate(kind, request, lambda:self.change('noop'))
                self.assertEqual(first.status,'completed',first.reason)
                def forbidden(): self.fail('warm reuse must not call provider')
                self.assertEqual(json.loads(json.dumps(self.generate(kind, self.request(kind), forbidden).to_dict())),
                                 json.loads(json.dumps(first.to_dict())))
                self.assertEqual(self.generate(kind,replace(self.request(kind),refresh=True)).status,'completed')
                self.assertEqual(self.notes.read('book',target_level='document')['note'],self.note)

    def test_late_result_does_not_replace_new_valid_artifacts_or_notes(self):
        for kind in ('analysis','quiz','summary'):
            with self.subTest(kind=kind):
                self.seed(); old=self.request(kind); snapshots=[]
                def publish_new():
                    self.change('rename')
                    if kind=='summary':
                        fresh=SummaryInteractionService(self.repo,Provider()).generate(self.request(kind))
                    else: fresh=self.generate(kind)
                    self.assertEqual(fresh.status,'completed',fresh.reason)
                    snapshots.append(self.repo.load_document('book'))
                try: result=self.generate(kind,old,publish_new)
                except ValueError: result=None
                if result:self.assertNotEqual(result.status,'completed')
                self.assertEqual(self.repo.load_document('book'),snapshots[0])
                self.assertEqual(self.notes.read('book',target_level='document')['note'],self.note)

    def test_parallel_late_writer_preserves_new_generation(self):
        for kind in ('analysis','quiz','summary'):
            with self.subTest(kind=kind):
                self.seed(); ready=threading.Event(); release=threading.Event()
                def wait_for_new():
                    ready.set()
                    self.assertTrue(release.wait(timeout=8))
                def late():
                    try: return self.generate(kind,callback=wait_for_new)
                    except ValueError: return None
                with ThreadPoolExecutor(1) as pool:
                    pending=pool.submit(late)
                    try:
                        self.assertTrue(ready.wait(timeout=5))
                        self.change('rename')
                        current=(SummaryInteractionService(self.repo,Provider()).generate(self.request(kind))
                                 if kind=='summary' else self.generate(kind))
                        self.assertEqual(current.status,'completed',current.reason)
                        before=self.repo.load_document('book')
                    finally: release.set()
                    result=pending.result(timeout=5)
                if result:self.assertNotEqual(result.status,'completed')
                self.assertEqual(self.repo.load_document('book'),before)

    def test_unbound_generated_write_is_rejected(self):
        for kind in ('analysis','quiz'):
            with self.subTest(kind=kind):
                self.seed();artifact=self.generate(kind);before=self.repo.load_document('book')
                unbound=replace(artifact,metadata={'context':{'doc_name':'book'}})
                with self.assertRaisesRegex(ValueError,'stale_interaction_write'):
                    getattr(self.store,'save_'+kind+'_artifact')(unbound)
                self.assertEqual(self.repo.load_document('book'),before)

    def test_note_save_delete_and_receipt_reject_old_editor(self):
        for transition in ('reparse','reimport','rename','keep','clear','undo','delete_undo'):
            with self.subTest(transition=transition):
                self.seed(); opened=self.notes.read('book',target_level='document')
                self.change(transition)
                current=self.notes.read('book',target_level='document')
                if current['note'] is None:
                    self.notes.save('book',target_level='document',target_token=current['target_token'],
                        text='New incarnation note',expected_revision=0,mutation_id='new')
                before=self.repo.load_document('book')
                for action in ('save','delete'):
                    with self.subTest(action=action),self.assertRaises(NoteConflict):
                        args=dict(target_level='document',target_token=opened['target_token'],expected_revision=1)
                        if action=='save': args.update(text='Delayed old edit',mutation_id='old')
                        getattr(self.notes,action)('book',**args)
                    self.assertEqual(self.repo.load_document('book'),before)
                current=self.notes.read('book',target_level='document')
                args=dict(target_level='document',target_token=current['target_token'],expected_revision=current['note']['revision'],
                          text='Fresh editor accepted',mutation_id=uuid4().hex)
                saved=self.notes.save('book',**args)
                self.assertEqual(self.notes.save('book',**args),saved)
                self.assertEqual(saved['note']['text'],'Fresh editor accepted')

    def test_old_note_create_receipt_after_roundtrip_and_noop_control(self):
        self.seed(); state=self.notes.read('book',target_level='document')
        args=dict(target_level='document',target_token=state['target_token'],expected_revision=1,
                  text='Receipt content',mutation_id='receipt')
        saved=self.notes.save('book',**args)
        self.change('noop'); self.assertEqual(self.notes.save('book',**args),saved)
        self.change('rename'); before=self.repo.load_document('book')
        with self.assertRaises(NoteConflict): self.notes.save('book',**args)
        self.assertEqual(self.repo.load_document('book'),before)

    def test_provider_work_has_no_document_lock(self):
        for kind in ('analysis','quiz','summary'):
            with self.subTest(kind=kind):
                self.seed()
                def unlocked():
                    with ThreadPoolExecutor(1) as pool:
                        future=pool.submit(self.repo.mutate_document_artifacts,'book',
                            lambda document:(document.document_task_artifacts,'unlocked'))
                        self.assertEqual(future.result(timeout=3),'unlocked')
                self.assertEqual(self.generate(kind,callback=unlocked).status,'completed')

    def test_publication_revalidates_inside_mutation(self):
        for kind in ('analysis','quiz','summary'):
            with self.subTest(kind=kind):
                self.seed(); original=self.repo.mutate_document_artifacts; snapshots=[]
                def changed(name,mutation):
                    self.change('rename');snapshots.append(self.repo.load_document('book'))
                    return original(name,mutation)
                with patch.object(self.repo,'mutate_document_artifacts',side_effect=changed):
                    try: result=self.generate(kind)
                    except ValueError: result=None
                if result:self.assertNotEqual(result.status,'completed')
                self.assertEqual(self.repo.load_document('book'),snapshots[0])

    def test_storage_failure_rolls_back_existing_outputs(self):
        for kind in ('analysis','quiz','summary'):
            with self.subTest(kind=kind):
                self.seed();self.generate(kind);before=self.repo.load_document('book')
                faults=[]
                if hasattr(self.repo.store,'dsn'):
                    import psycopg
                    execute=psycopg.Connection.execute
                    def fail(connection,query,*args,**kwargs):
                        result=execute(connection,query,*args,**kwargs)
                        if 'UPDATE documents SET metadata_payload=jsonb_set' in str(query):
                            faults.append(True); raise OSError('after update')
                        return result
                    context=patch.object(psycopg.Connection,'execute',fail)
                else:
                    def fail(*args):
                        faults.append(True); raise OSError('before replace')
                    context=patch('document_structure.structured_document_artifact_repository.os.replace',side_effect=fail)
                # Force a genuinely changed result so idempotent equal-output
                # storage is not mistaken for a failed write boundary.
                output=self.output(kind)
                if kind=='analysis': output['explanation'] += ' The assumptions require independent evidence.'
                result = None
                with context, patch.object(self,'output',return_value=output):
                    try: result=self.generate(kind,replace(self.request(kind),refresh=True))
                    except OSError: result=None
                if result:self.assertNotEqual(result.status,'completed')
                self.assertTrue(faults, 'Fault must occur at the real storage boundary')
                self.assertEqual(self.repo.load_document('book'),before)

    def test_unbound_saved_policy_is_stale_without_read_writeback(self):
        for kind in ('analysis','quiz','summary'):
            with self.subTest(kind=kind):
                self.seed(); artifact=self.generate(kind)
                def legacy(document):
                    artifacts=document.document_task_artifacts;metadata=dict(artifacts.metadata)
                    key='public_summary' if kind=='summary' else 'reading_interaction_artifacts'
                    import copy
                    data=copy.deepcopy(metadata[key]); records=data if kind=='summary' else data[kind]
                    for record in records.values():
                        policy=record['metadata'].setdefault('interaction_policy', {})
                        policy.pop('document_incarnation',None);policy.pop('artifact_generation_epoch',None)
                    metadata[key]=data
                    return replace(artifacts,metadata=metadata),None
                self.repo.mutate_document_artifacts('book',legacy);before=self.repo.load_document('book')
                result=(self.summary if kind=='summary' else self.orchestrator(kind)).read(self.request(kind))
                self.assertEqual(result.status,'stale_target')
                self.assertEqual(self.repo.load_document('book'),before)
                self.assertEqual(self.generate(kind).status,'completed')

    def test_http_note_roundtrip_and_summary_generation_reject_late_source(self):
        self.seed(); target={'doc_name':'book','target_type':'document'}
        with patch.object(main,'section_task_coordinator',self.co),TestClient(main.app) as client:
            base='/documents/reading-interactions/'
            opened=client.post(base+'note/read',json={'target':target}).json()
            self.change('rename');before=self.repo.load_document('book')
            for action in ('save','delete'):
                payload={'target':target,'target_token':opened['target_token'],'expected_revision':1}
                if action=='save':payload.update(text='Old HTTP note',mutation_id='old-http')
                response=client.post(base+'note/'+action,json=payload)
                self.assertEqual(response.status_code,409,response.text)
                self.assertEqual(self.repo.load_document('book'),before)
            original=self.provider.complete_text;snapshots=[]
            def delayed(prompt):
                self.change('rename');snapshots.append(self.repo.load_document('book'));return original(prompt)
            with patch.object(self.provider,'complete_text',side_effect=delayed):
                response=client.post(base+'summary/generate',json={'target':target})
            self.assertEqual(response.status_code,502,response.text)
            self.assertEqual(response.json()['envelope']['status'],'generation_failed')
            self.assertEqual(self.repo.load_document('book'),snapshots[0])


@unittest.skipUnless(os.environ.get('DEEP_READER_TEST_POSTGRES_DSN'),'requires disposable PostgreSQL DSN')
class PostgresInteractionLifecycleTests(InteractionLifecycleTests):
    setUpClass=PostgresCriticalThinkingPublicationTests.__dict__['setUpClass']
    drop_schema=PostgresCriticalThinkingPublicationTests.__dict__['drop_schema']
    repository=PostgresCriticalThinkingPublicationTests.repository


if __name__=='__main__':unittest.main(verbosity=2)
