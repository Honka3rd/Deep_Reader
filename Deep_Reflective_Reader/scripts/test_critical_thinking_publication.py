"""APP-03 durable answer and conditional publication; real stores, no live LLM."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch
from uuid import uuid4

from fastapi.testclient import TestClient
import main
from app.section_task_coordinator import SectionTaskCoordinator
from document_structure.structured_document_artifact_repository import StructuredDocumentArtifactRepository
from section_tasks.critical_thinking_session_service import CriticalThinkingSessionService
from section_tasks.reading_interaction_artifact_store import DocumentReadingInteractionArtifactStore
from scripts.test_reading_interaction_artifact_store import _document
from scripts.test_enhancement_postgres_preflight import PostgresPreflightTests

ANSWER = 'The claim assumes that the observed relationship is causal.'
EVALUATION = {'feedback':'The answer identifies the assumption.',
              'strengths':'A clear reason.', 'improvements':'Add evidence.'}


class CriticalThinkingPublicationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.repo = self.repository()
        self.repo.save_reparsed_document(_document(), 'book')
        self.service = CriticalThinkingSessionService(
            lambda request,instruction:{'question':'Which assumption matters most?'},
            lambda session,instruction:dict(EVALUATION))
        self.store = DocumentReadingInteractionArtifactStore(self.repo)
        self.profile = Mock(); self.profile.exists.return_value = False
        self.co = self.coordinator()
        self.args = dict(doc_name='book',target_level='document')
        self.session_id = self.co.generate_critical_thinking_question(**self.args).session_id

    def repository(self):
        return StructuredDocumentArtifactRepository(base_dir=self.temp.name)

    def coordinator(self):
        return SectionTaskCoordinator(Mock(),self.repo,self.profile,Mock(),Mock(),Mock(),Mock(),
            critical_thinking_session_service=self.service,critical_thinking_session_store=self.store)

    def read(self):
        return self.co.read_critical_thinking_session(**self.args,session_id=self.session_id)

    def submit(self):
        return self.co.submit_critical_thinking_answer(**self.args,session_id=self.session_id,answer=ANSWER)

    def retry(self):
        return self.co.retry_critical_thinking_evaluation(**self.args,session_id=self.session_id)

    def request(self):
        return self.co._build_reading_interaction_request(**self.args,interaction_type='critical_thinking_session')

    def artifact(self):
        return self.store.get_critical_thinking_session(self.request(),self.session_id)

    def test_transport_and_invalid_output_preserve_answer_and_retry(self):
        for fault in [TimeoutError('PRIVATE KEY'),ConnectionError('PRIVATE URL'),ValueError('PRIVATE PROVIDER'),'bad json']:
            with self.subTest(fault=type(fault).__name__):
                self.session_id=self.co.generate_critical_thinking_question(**self.args).session_id
                def evaluate(session,instruction):
                    current=self.read()
                    self.assertEqual(current.status,'answer_submitted')
                    self.assertEqual(current.payload['answer'],ANSWER)
                    if isinstance(fault,Exception):raise fault
                    return fault
                with patch.object(self.service,'_evaluation_generator',side_effect=evaluate): result=self.submit()
                self.assertEqual(result.status,'evaluation_failed')
                self.assertEqual(self.read().payload['answer'],ANSWER)
                self.assertNotIn('PRIVATE',repr(result))
                self.assertEqual(result.reason,'evaluation_invalid_output' if isinstance(fault,str) else 'evaluation_provider_failed')
                self.co=self.coordinator()
                self.assertEqual(self.retry().status,'completed')

    def test_interruption_keeps_submitted_answer_for_reloaded_retry(self):
        with patch.object(self.service,'_evaluation_generator',side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):self.submit()
        self.co=self.coordinator()
        self.assertEqual(self.read().status,'answer_submitted')
        self.assertEqual(self.read().payload['answer'],ANSWER)
        self.assertEqual(self.retry().status,'completed')

    def test_provider_has_no_document_write_lock(self):
        def evaluate(session,instruction):
            self.assertEqual(self.read().status,'answer_submitted')
            # Another connection/thread can mutate under the actual owner lock.
            pool = ThreadPoolExecutor(max_workers=1)
            try:
                future=pool.submit(self.repo.mutate_document_artifacts,'book',
                    lambda document:(document.document_task_artifacts,'unlocked'))
                self.assertEqual(future.result(timeout=3),'unlocked')
            finally: pool.shutdown(wait=False)
            return dict(EVALUATION)
        with patch.object(self.service,'_evaluation_generator',side_effect=evaluate):
            self.assertEqual(self.submit().status,'completed')

    def test_submit_save_failure_or_unconfirmed_read_never_evaluates(self):
        before=self.repo.load_document('book'); original=self.repo.mutate_document_artifacts
        for acknowledgement in [False,True]:
            with self.subTest(acknowledgement=acknowledgement):
                def save(name,mutation):
                    if acknowledgement: original(name,mutation)
                    raise OSError('PRIVATE STORAGE')
                with patch.object(self.repo,'mutate_document_artifacts',side_effect=save), \
                     patch.object(self.service,'_evaluation_generator') as provider:
                    if acknowledgement:
                        original_read=self.repo.load_document
                        def read(name):
                            document=original_read(name)
                            if self.store._session_from_document(document,self.session_id).status=='answer_submitted':
                                raise OSError('PRIVATE READ')
                            return document
                        with patch.object(self.repo,'load_document',side_effect=read):
                            with self.assertRaisesRegex(RuntimeError,'save_unconfirmed'):self.submit()
                    else:
                        with self.assertRaisesRegex(RuntimeError,'save_unconfirmed'):self.submit()
                    provider.assert_not_called()
                if not acknowledgement:self.assertEqual(self.repo.load_document('book'),before)
        self.assertEqual(self.read().status,'answer_submitted')
        self.assertEqual(self.retry().status,'completed')

    def test_lost_acknowledgement_reconciles_exact_saved_revision(self):
        original=self.repo.mutate_document_artifacts
        def save(name,mutation):
            original(name,mutation)
            raise OSError('PRIVATE ACK')
        with patch.object(self.repo,'mutate_document_artifacts',side_effect=save), \
             patch.object(self.service,'_evaluation_generator',return_value=dict(EVALUATION)) as provider:
            result=self.submit()
        self.assertEqual(result.status,'completed');self.assertEqual(provider.call_count,1)
        self.assertEqual(self.read().payload['answer'],ANSWER)

    def test_evaluation_save_failure_preserves_accepted_answer(self):
        original=self.repo.mutate_document_artifacts; calls=[]
        def save(name,mutation):
            calls.append(1)
            if len(calls)==2:raise OSError('PRIVATE SAVE')
            return original(name,mutation)
        with patch.object(self.repo,'mutate_document_artifacts',side_effect=save):
            with self.assertRaisesRegex(RuntimeError,'save_unconfirmed'):self.submit()
        self.assertEqual(self.read().status,'answer_submitted')
        self.assertEqual(self.read().payload['answer'],ANSWER)
        self.assertEqual(self.retry().status,'completed')

    def test_concurrent_submissions_only_accept_one_answer(self):
        barrier=threading.Barrier(2); original=self.service.submit_answer
        def submit(session,answer):
            result=original(session,answer);barrier.wait(timeout=4);return result
        def run(answer):
            try:return self.co.submit_critical_thinking_answer(**self.args,session_id=self.session_id,answer=answer)
            except ValueError:return None
        with patch.object(self.service,'submit_answer',side_effect=submit), \
             patch.object(self.service,'_evaluation_generator',return_value=dict(EVALUATION)) as provider, \
             ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(run,answer) for answer in [ANSWER,'A distinct submitted answer.']]
            results=[future.result(timeout=8) for future in futures]
        self.assertEqual(sum(result is not None for result in results),1)
        self.assertEqual(provider.call_count,1)
        self.assertEqual(self.read().payload,next(result.payload for result in results if result))

    def test_late_evaluation_cannot_replace_new_revision_even_same_answer(self):
        with patch.object(self.service,'_evaluation_generator',side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):self.submit()
        old=self.artifact(); late=self.service.evaluate_answer(old)
        failed=replace(old,status='evaluation_failed',reason='evaluation_provider_failed')
        failed=self.store.save_critical_thinking_session(failed,expected_session=old).artifact
        submitted=self.service.submit_answer(failed,ANSWER)
        newer=self.store.save_critical_thinking_session(submitted,expected_session=failed).artifact
        completed=self.service.evaluate_answer(newer)
        self.store.save_critical_thinking_session(completed,expected_session=newer)
        before=self.repo.load_document('book')
        with self.assertRaisesRegex(ValueError,'stale_critical_thinking_session'):
            self.store.save_critical_thinking_session(late,expected_session=old)
        self.assertEqual(self.repo.load_document('book'),before)

    def test_late_evaluation_source_and_lifecycle_changes_preserve_new_snapshot(self):
        for change in ['reparse','reimport','rename','edit','undo']:
            with self.subTest(change=change):
                self.repo.save_reparsed_document(_document(),'book')
                self.session_id=self.co.generate_critical_thinking_question(**self.args).session_id
                after=[]
                def evaluate(session,instruction):
                    if change=='reparse':self.repo.save_reparsed_document(_document(),'book')
                    elif change=='reimport':
                        self.repo.delete_document('book');self.repo.save_reparsed_document(_document(),'book')
                    elif change=='rename':
                        for old,new in [('book','away'),('away','book')]:
                            self.repo.rename_document(old,new,lambda document,name:replace(document,document_id=name))
                    else:
                        from document_structure.reading_content import ReadingContentService
                        content=ReadingContentService(self.repo)
                        layout=content.read('book');block=layout['blocks'][0]
                        content.mutate('book',action='edit',block_id=block['block_id'],text='Changed effective reading source.',
                            expected_revision=layout['revision'],source_token=layout['source_token'],
                            mutation_id=uuid4().hex,discard_interactions=False)
                        if change=='undo':
                            layout=content.read('book')
                            content.mutate('book',action='undo',expected_revision=layout['revision'],
                                source_token=layout['source_token'],mutation_id=uuid4().hex)
                    after.append(self.repo.load_document('book'))
                    return dict(EVALUATION)
                with patch.object(self.service,'_evaluation_generator',side_effect=evaluate):
                    with self.assertRaisesRegex(ValueError,'stale_critical_thinking'):self.submit()
                self.assertEqual(self.repo.load_document('book'),after[0])

    def test_storage_failure_rolls_back_and_keeps_question(self):
        before = self.repo.load_document('book')
        if hasattr(self.repo.store, 'dsn'):
            import psycopg
            execute = psycopg.Connection.execute
            def fail(connection, query, *args, **kwargs):
                result = execute(connection, query, *args, **kwargs)
                if 'UPDATE documents SET metadata_payload=jsonb_set' in str(query):
                    raise OSError('PRIVATE FAILURE AFTER SQL UPDATE')
                return result
            context = patch.object(psycopg.Connection, 'execute', fail)
        else:
            context = patch('document_structure.structured_document_artifact_repository.os.replace',
                            side_effect=OSError('PRIVATE ATOMIC REPLACE FAILURE'))
        with context, patch.object(self.service, '_evaluation_generator') as provider:
            with self.assertRaisesRegex(RuntimeError, 'save_unconfirmed'): self.submit()
            provider.assert_not_called()
        self.assertEqual(self.repo.load_document('book'), before)
        self.assertEqual(self.submit().status, 'completed')

    def test_question_publication_rejects_source_reimport(self):
        after = []
        def question(request, instruction):
            self.repo.delete_document('book')
            self.repo.save_reparsed_document(_document(), 'book')
            after.append(self.repo.load_document('book'))
            return {'question':'A delayed question?'}
        with patch.object(self.service, '_question_generator', side_effect=question):
            with self.assertRaisesRegex(ValueError, 'stale_critical_thinking_source'):
                self.co.generate_critical_thinking_question(**self.args)
        self.assertEqual(self.repo.load_document('book'), after[0])

    def test_unbound_legacy_session_read_is_pure_and_cannot_retry(self):
        def mutation(document):
            import copy
            artifacts = document.document_task_artifacts
            metadata = copy.deepcopy(artifacts.metadata)
            session = metadata['reading_interaction_artifacts']['critical_thinking_session'][self.session_id]
            session['metadata']['interaction_policy'].pop('document_incarnation')
            return replace(artifacts, metadata=metadata), None
        self.repo.mutate_document_artifacts('book', mutation)
        before = self.repo.load_document('book')
        with patch.object(self.service, '_evaluation_generator') as provider:
            self.assertEqual(self.read().status, 'stale_target')
            with self.assertRaisesRegex(ValueError, 'stale_critical_thinking_session'): self.retry()
            provider.assert_not_called()
        self.assertEqual(self.repo.load_document('book'), before)

    def test_http_transport_invalid_output_reload_and_retry(self):
        target={'doc_name':'book','target_type':'document'}
        base='/documents/reading-interactions/critical-thinking'
        with patch.object(main,'section_task_coordinator',self.co),TestClient(main.app) as client:
            for fault in [TimeoutError('PRIVATE PROVIDER'),ConnectionError('PRIVATE URL'),'not-json']:
                with self.subTest(fault=type(fault).__name__):
                    generated=client.post(base+'/generate-question',json={'target':target})
                    self.assertEqual(generated.status_code,200,generated.text)
                    payload={'target':target,'session_id':generated.json()['envelope']['session_id']}
                    def evaluate(*args):
                        if isinstance(fault,Exception):raise fault
                        return fault
                    with patch.object(self.service,'_evaluation_generator',side_effect=evaluate):
                        response=client.post(base+'/submit-answer',json={**payload,'answer':ANSWER})
                    self.assertNotIn('PRIVATE',response.text)
                    self.assertEqual(response.json()['envelope']['status'],'evaluation_failed',response.text)
                    read=client.post(base+'/read',json=payload)
                    self.assertIn(ANSWER,read.text)
                    retry=client.post(base+'/retry-evaluation',json=payload)
                    self.assertEqual(retry.status_code,200,retry.text)
                    self.assertEqual(retry.json()['envelope']['status'],'completed')

    def test_http_reloaded_interrupted_answer_and_stale_publication(self):
        with patch.object(self.service,'_evaluation_generator',side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):self.submit()
        self.co=self.coordinator()
        payload={'target':{'doc_name':'book','target_type':'document'},'session_id':self.session_id}
        after=[]
        def evaluate(*args):
            self.repo.save_reparsed_document(_document(),'book')
            after.append(self.repo.load_document('book'))
            return dict(EVALUATION)
        with patch.object(main,'section_task_coordinator',self.co),TestClient(main.app) as client:
            read=client.post('/documents/reading-interactions/critical-thinking/read',json=payload)
            self.assertEqual(read.json()['envelope']['status'],'answer_submitted')
            self.assertIn(ANSWER,read.text)
            with patch.object(self.service,'_evaluation_generator',side_effect=evaluate):
                result=client.post('/documents/reading-interactions/critical-thinking/retry-evaluation',json=payload)
            self.assertEqual(result.status_code,409,result.text)
            self.assertEqual(self.repo.load_document('book'),after[0])

    def test_http_parallel_submissions_reject_loser_before_evaluation(self):
        barrier=threading.Barrier(2); original=self.service.submit_answer
        def submitted(session,answer):
            result=original(session,answer);barrier.wait(timeout=4);return result
        def call(answer):
            with TestClient(main.app) as client:
                return client.post('/documents/reading-interactions/critical-thinking/submit-answer',json={
                    'target':{'doc_name':'book','target_type':'document'},'session_id':self.session_id,'answer':answer})
        with patch.object(main,'section_task_coordinator',self.co), \
             patch.object(self.service,'submit_answer',side_effect=submitted), \
             patch.object(self.service,'_evaluation_generator',return_value=dict(EVALUATION)) as provider, \
             ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(call,answer) for answer in [ANSWER,'A competing answer.']]
            results=[future.result(timeout=8) for future in futures]
        self.assertEqual(sorted(r.status_code for r in results),[200,409])
        self.assertEqual(provider.call_count,1)
        self.assertEqual(self.read().status,'completed')

    def test_http_unconfirmed_save_is_not_answer_acceptance(self):
        before=self.repo.load_document('book')
        with patch.object(main,'section_task_coordinator',self.co), \
             patch.object(self.repo,'mutate_document_artifacts',side_effect=OSError('PRIVATE STORAGE')), \
             patch.object(self.service,'_evaluation_generator') as provider, TestClient(main.app) as client:
            result=client.post('/documents/reading-interactions/critical-thinking/submit-answer',json={
                'target':{'doc_name':'book','target_type':'document'},'session_id':self.session_id,'answer':ANSWER})
            self.assertEqual(result.status_code,500,result.text)
            self.assertEqual(result.json()['detail'],'critical_thinking_save_unconfirmed')
            self.assertNotIn('PRIVATE',result.text)
            provider.assert_not_called()
        self.assertEqual(self.repo.load_document('book'),before)


@unittest.skipUnless(os.environ.get('DEEP_READER_TEST_POSTGRES_DSN'),'requires disposable PostgreSQL DSN')
class PostgresCriticalThinkingPublicationTests(CriticalThinkingPublicationTests):
    setUpClass=classmethod(PostgresPreflightTests.setUpClass.__func__)
    drop_schema=classmethod(PostgresPreflightTests.drop_schema.__func__)

    def repository(self):
        from db.postgres_structured_document_store import PostgresStructuredDocumentStore
        from db.postgres_structured_document_artifact_repository import PostgresStructuredDocumentArtifactRepository
        return PostgresStructuredDocumentArtifactRepository(PostgresStructuredDocumentStore(self.scoped,namespace=uuid4().hex))

if __name__=='__main__':unittest.main(verbosity=2)
