"""MAIN-01/02/03 actual HTTP, active file / real PostgreSQL owners."""
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient
import main
from scripts import test_learning_lifecycle as life
from scripts import test_learning_postgres_repository as sql_tests
from document_structure.learning_repository import OperationWrite
from shared.learning_source import LearningContractError

class RenameCases:
    save_result = life.LifecycleMatrix.save_result
    def http_setup(self):
        temp=TemporaryDirectory(); self.addCleanup(temp.cleanup)
        self.files=Path(temp.name)
        self.raw=self.files/'raw'; self.raw.mkdir()
        self.derived=self.files/'faiss'; self.derived.mkdir()
        for name,value in [('_RAW_DOCUMENT_BASE_DIR',self.raw),('_DERIVED_DOCUMENT_BASE_DIR',self.derived),('document_artifact_repository',self.docs)]:
            p=patch.object(main,name,value); p.start(); self.addCleanup(p.stop)
        self.client=TestClient(main.app); self.addCleanup(self.client.close)
    def rename(self,new='renamed'):
        return self.client.post('/documents/lifecycle/rename',json={'doc_name':'book','action':'rename','new_doc_name':new})
    def make_files(self):
        (self.raw/'book.txt').write_bytes(b'original source')
        (self.derived/'book').mkdir(); (self.derived/'book'/'profile.json').write_text('{}')
    def test_http_moves_learning_preferences_sentences_receipts_and_fences_writer(self):
        self.make_files()
        key,saved,sentences=self.save_result()
        hints,hint_saved,hint_sentences=self.save_result(level='C2',label='hints')
        _,running,_=self.save_result(label='running',running=True)
        settings=self.prefs.read_book('book').settings
        response=self.rename(); self.assertEqual(response.status_code,200,response.text)
        self.assertTrue(self.prefs.read_book('book').deleted)
        self.assertEqual(self.prefs.read_book('renamed').settings,settings)
        for old_key,entry,resource in ((key,saved,sentences),(hints,hint_saved,hint_sentences)):
            new_resource=resource.with_document_id('renamed')
            view=self.repo.read_batch('renamed',(replace(old_key,resource=new_resource.reference),))[0]
            self.assertEqual(view.view.state,'ready'); self.assertEqual(view.sentences,new_resource)
            self.assertEqual(view.view.saved.context.result_revision,entry.saved.context.result_revision)
        self.assertEqual(self.repo.lookup_command('renamed','saved').operation_id,'saved')
        op=self.repo.lookup_operation('renamed','running')
        self.assertEqual(op.state,'stale'); self.assertTrue(op.outcome_uncertain)
        with self.assertRaises(LearningContractError):
            self.repo.write_operation('book',OperationWrite(running.operation.operation_id,running.revision,running.last_fence,40,'renew',lease_until_ms=200))
        self.assertEqual((self.raw/'renamed.txt').read_bytes(),b'original source')
        self.assertTrue((self.derived/'renamed'/'profile.json').exists())
    def test_repository_only_document_can_rename(self):
        response=self.rename(); self.assertEqual(response.status_code,200,response.text)
        self.assertEqual(self.docs.load_document('renamed').document_id,'renamed')
        self.assertFalse(response.json()['source_affected'])
    def test_invalid_derived_resource_is_rejected_before_any_move(self):
        (self.raw/'book.txt').write_bytes(b'original'); (self.derived/'book').write_text('not directory')
        response=self.rename(); self.assertGreaterEqual(response.status_code,400)
        self.assertTrue((self.raw/'book.txt').exists())
        self.assertEqual(self.docs.load_document('book').document_id,'book')
        self.assertFalse(self.prefs.read_book('book').deleted)

    def test_late_canonical_writer_cannot_recreate_old_name(self):
        old=self.docs.load_document('book')
        response=self.rename();self.assertEqual(response.status_code,200,response.text)
        from document_structure.reading_content import ReadingContentConflict
        with self.assertRaises(ReadingContentConflict):
            self.docs.save_document(old,'book')
        with self.assertRaises(FileNotFoundError): self.docs.load_document('book')

    def test_preference_destination_conflict_does_not_move_files(self):
        from scripts.test_learning_file_repository import configure
        self.make_files();configure(self.prefs,'renamed')
        response=self.rename();self.assertEqual(response.status_code,409,response.text)
        self.assertTrue((self.raw/'book.txt').exists())
        self.assertEqual(self.docs.load_document('book').document_id,'book')
        self.assertFalse(self.prefs.read_book('book').deleted)

    def test_recovery_refuses_changed_files_without_overwrite(self):
        self.make_files();original=Path.rename
        def fail(source,target):
            if source.parent==self.raw: raise OSError('pause files')
            return original(source,target)
        with patch.object(Path,'rename',fail): response=self.rename()
        self.assertEqual(response.status_code,503,response.text)
        (self.raw/'book.txt').write_bytes(b'externally changed')
        response=self.rename();self.assertEqual(response.status_code,503,response.text)
        self.assertTrue(response.json()['recovery_required'])
        self.assertEqual((self.raw/'book.txt').read_bytes(),b'externally changed')
        self.assertFalse((self.raw/'renamed.txt').exists())

    def test_recreated_identity_is_not_replayed_as_completed(self):
        response=self.rename();self.assertEqual(response.status_code,200,response.text)
        from scripts.test_learning_file_repository import fixture
        self.docs.save_reparsed_document(fixture(),'book')
        response=self.rename();self.assertEqual(response.status_code,409,response.text)
        self.assertEqual(self.docs.load_document('book').document_id,'book')


    def test_raw_and_derived_destination_conflicts_are_preflighted(self):
        self.make_files()
        for target in (self.raw/'renamed.txt', self.derived/'renamed'):
            with self.subTest(target=target.name):
                target.write_text('occupied')
                response=self.rename();self.assertEqual(response.status_code,409,response.text)
                self.assertTrue((self.raw/'book.txt').exists())
                self.assertEqual(self.docs.load_document('book').document_id,'book')
                self.assertEqual(target.read_text(),'occupied')
                target.unlink()

    def test_pending_operation_blocks_different_rename(self):
        with patch.object(self.docs,'rename_document',side_effect=OSError('unavailable')):
            response=self.rename()
        self.assertEqual(response.status_code,503,response.text)
        response=self.rename('another');self.assertEqual(response.status_code,409,response.text)
        response=self.rename();self.assertEqual(response.status_code,200,response.text)


    def test_concurrent_duplicate_http_requests_share_one_result(self):
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=2) as pool:
            responses=list(pool.map(lambda _:self.rename(),range(2)))
        self.assertTrue(all(r.status_code==200 for r in responses),[r.text for r in responses])
        self.assertEqual(responses[0].json()['operation_id'],responses[1].json()['operation_id'])
        self.assertEqual(self.prefs.read_book('renamed').revision,1)

    def test_explicit_reverse_and_rename_again_remains_available(self):
        self.make_files()
        response=self.rename();self.assertEqual(response.status_code,200,response.text)
        reverse=self.client.post('/documents/lifecycle/rename',json={
            'doc_name':'renamed','action':'rename','new_doc_name':'book'})
        self.assertEqual(reverse.status_code,200,reverse.text)
        again=self.rename();self.assertEqual(again.status_code,200,again.text)
        self.assertNotEqual(again.json()['operation_id'],response.json()['operation_id'])
        self.assertEqual((self.raw/'renamed.txt').read_bytes(),b'original source')

    def test_corrupt_pending_journal_fails_closed_with_recovery_status(self):
        from app.document_rename import DocumentRename
        with patch.object(self.docs,'rename_document',side_effect=OSError('unavailable')):
            response=self.rename()
        self.assertEqual(response.status_code,503,response.text)
        journal=DocumentRename(self.docs,self.raw,self.derived).pending
        journal.write_text('{broken')
        response=self.rename();self.assertEqual(response.status_code,503,response.text)
        self.assertTrue(response.json()['recovery_required'])
        self.assertEqual(self.docs.load_document('book').document_id,'book')


    def test_real_tcp_http_rename_uses_active_lifecycle(self):
        import socket
        from threading import Thread
        from time import monotonic, sleep
        import httpx
        import uvicorn
        with socket.socket() as listener:
            listener.bind(('127.0.0.1',0)); listener.listen(16)
            address=listener.getsockname()
            server=uvicorn.Server(uvicorn.Config(main.app,lifespan='off',log_level='error'))
            worker=Thread(target=lambda:server.run(sockets=[listener]),daemon=True);worker.start()
            try:
                deadline=monotonic()+5
                while not server.started and worker.is_alive() and monotonic()<deadline: sleep(.01)
                self.assertTrue(server.started)
                original=self.client
                with httpx.Client(base_url=f'http://{address[0]}:{address[1]}') as client:
                    self.client=client
                    try: self.test_http_moves_learning_preferences_sentences_receipts_and_fences_writer()
                    finally: self.client=original
            finally:
                server.should_exit=True;worker.join(timeout=5)
                self.assertFalse(worker.is_alive())


    def test_destination_resources_conflict_even_without_source_files(self):
        for target in (self.raw/'renamed.pdf',self.derived/'renamed'):
            with self.subTest(target=target):
                target.write_bytes(b'existing target')
                response=self.rename();self.assertEqual(response.status_code,409,response.text)
                self.assertEqual(self.docs.load_document('book').document_id,'book')
                self.assertEqual(target.read_bytes(),b'existing target')
                target.unlink()


class FileRenameTests(RenameCases,unittest.TestCase):
    def setUp(self):
        life.FileLifecycleTests.setUp(self); self.http_setup()

class PostgresRenameTests(RenameCases,unittest.TestCase):
    setUpClass=classmethod(sql_tests.LearningPostgresTests.setUpClass.__func__)
    drop_schema=classmethod(sql_tests.LearningPostgresTests.drop_schema.__func__)
    def setUp(self):
        sql_tests.LearningPostgresTests.setUp(self); self.http_setup()
    def test_inactive_structured_files_are_untouched(self):
        inactive=self.files/'inactive'; inactive.mkdir()
        (inactive/'book.structured.json').write_text('inactive original')
        (inactive/'renamed.structured.json').write_text('inactive target')
        self.docs.base_dir=inactive
        response=self.rename(); self.assertEqual(response.status_code,200,response.text)
        self.assertEqual((inactive/'book.structured.json').read_text(),'inactive original')
        self.assertEqual((inactive/'renamed.structured.json').read_text(),'inactive target')

    def test_inactive_json_alone_does_not_establish_existence(self):
        self.docs.delete_document('book')
        inactive=self.files/'inactive';inactive.mkdir();self.docs.base_dir=inactive
        path=inactive/'book.structured.json';path.write_text('inactive original')
        response=self.rename();self.assertEqual(response.status_code,404,response.text)
        self.assertEqual(path.read_text(),'inactive original')


# Each injected stage is a separate unittest with fresh storage and HTTP client.
def recovery_case(stage):
    def test(self):
        from app import document_rename as impl
        self.make_files(); self.save_result()
        original_rename=Path.rename; original_save=impl.DocumentRename._save
        original_remove=impl.atomic_write; original_repository=self.docs.rename_document
        def move(source,target):
            kind='raw' if source.parent==self.raw else 'derived'
            if stage==kind: raise OSError('injected move')
            result=original_rename(source,target)
            if stage==kind+'_ack': raise OSError('injected move acknowledgement')
            return result
        def save(owner,path,record):
            if stage=='journal' and path.name=='pending.json': raise OSError('before journal')
            if stage=='receipt' and path.name!='pending.json': raise OSError('before receipt')
            result=original_save(owner,path,record)
            if stage=='journal_ack' and path.name=='pending.json': raise OSError('journal acknowledgement')
            return result
        def remove(path,payload):
            if stage=='clear' and payload is None: raise OSError('before pending clear')
            return original_remove(path,payload)
        def repository(*args,**kwargs):
            if stage=='repository': raise OSError('before repository')
            result=original_repository(*args,**kwargs)
            if stage=='repository_ack': raise OSError('after repository commit')
            return result
        with patch.object(Path,'rename',move),patch.object(impl.DocumentRename,'_save',save),patch.object(impl,'atomic_write',remove),patch.object(self.docs,'rename_document',repository):
            response=self.rename()
        self.assertEqual(response.status_code,500 if stage=='journal' else 503,response.text)
        self.assertEqual(response.json()['status'],'failed')
        if stage!='journal':
            self.assertTrue(response.json()['recovery_required']); self.assertTrue(response.json()['operation_id'])
        else:
            self.assertEqual(self.docs.load_document('book').document_id,'book')
            self.assertTrue((self.raw/'book.txt').exists())
        retry=self.rename();self.assertEqual(retry.status_code,200,retry.text)
        self.assertTrue(self.prefs.read_book('book').deleted)
        self.assertEqual((self.raw/'renamed.txt').read_bytes(),b'original source')
        self.assertTrue((self.derived/'renamed'/'profile.json').exists())
        revision=self.prefs.read_book('renamed').revision
        again=self.rename();self.assertEqual(again.status_code,200,again.text)
        self.assertEqual(again.json()['operation_id'],retry.json()['operation_id'])
        self.assertEqual(self.prefs.read_book('renamed').revision,revision)
    return test

for owner in (FileRenameTests,PostgresRenameTests):
    for stage in ('journal','journal_ack','repository','repository_ack','raw','raw_ack','derived','derived_ack','receipt','clear'):
        setattr(owner,'test_failure_recovery_'+stage,recovery_case(stage))
del owner


def file_write_case(index):
    def test(self):
        from document_structure import learning_file_lifecycle as impl
        self.make_files();self.save_result()
        original=impl.atomic_write;calls=0
        def fail(path,payload):
            nonlocal calls
            calls+=1
            result=original(path,payload)
            if calls==index: raise OSError('lifecycle lost ack')
            return result
        with patch.object(impl,'atomic_write',fail):
            response=self.rename()
        self.assertEqual(response.status_code,503,response.text)
        self.assertTrue(response.json()['recovery_required'])
        self.assertTrue((self.raw/'book.txt').exists())
        response=self.rename();self.assertEqual(response.status_code,200,response.text)
        self.assertTrue(self.prefs.read_book('book').deleted)
        self.assertEqual(self.repo.lookup_command('renamed','saved').operation_id,'saved')
    return test

for index in range(1,8):
    setattr(FileRenameTests,f'test_file_lifecycle_write_{index}_recovery',file_write_case(index))


def sql_failure_case(statement):
    def test(self):
        import psycopg
        self.make_files();self.save_result()
        original=psycopg.Connection.execute;hit=[]
        def execute(connection,query,*args,**kwargs):
            result=original(connection,query,*args,**kwargs)
            if isinstance(query,str) and query.startswith(statement):
                hit.append(True);raise OSError('injected SQL stage after execute')
            return result
        with patch.object(psycopg.Connection,'execute',execute):
            response=self.rename()
        self.assertTrue(hit);self.assertEqual(response.status_code,503,response.text)
        self.assertEqual(self.docs.load_document('book').document_id,'book')
        self.assertFalse(self.prefs.read_book('book').deleted)
        self.assertTrue((self.raw/'book.txt').exists())
        response=self.rename();self.assertEqual(response.status_code,200,response.text)
        self.assertEqual(self.repo.lookup_command('renamed','saved').operation_id,'saved')
    return test

for index,statement in enumerate(('DELETE FROM learning_checkpoints','DELETE FROM learning_commands',
        'DELETE FROM learning_operations','DELETE FROM learning_cache_entries','DELETE FROM learning_sentences',
        'UPDATE documents SET document_name','UPDATE structured_document_snapshots SET structured_document_payload')):
    setattr(PostgresRenameTests,f'test_sql_rollback_stage_{index}',sql_failure_case(statement))

if __name__=='__main__': unittest.main()
