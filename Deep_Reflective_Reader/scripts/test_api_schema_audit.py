"""SCHEMA-01/02 HTTP and real file boundary regressions; no provider calls."""
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from urllib.parse import quote
import inspect
import unittest
from fastapi.testclient import TestClient
from pydantic import BaseModel, ValidationError
import api_schemas
import api_learning_schemas
import main
from document_structure.structured_document_artifact_repository import StructuredDocumentArtifactRepository
from section_tasks.note_interaction_service import NoteInteractionService
from scripts import test_task_unit_content_endpoint as content

BAD_NAMES = ('', ' ', '.', '..', '../victim', '/tmp/victim', 'a/b', r'a\b',
             'C:book', r'C:\book', 'book\x00', 'book\n', 'book\x7f', 'a'*256,
             '.pdf', '..txt', '...pdf', 'C:', 'c:relative', 'a:b')


class SchemaAuditTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.repo = StructuredDocumentArtifactRepository(base_dir=str(self.root/'inside'))
        self.outside = StructuredDocumentArtifactRepository(base_dir=str(self.root))
        self.document = content._build_cache_valid_document()
        self.outside.save_document(self.document, 'victim')
        (self.root/'victim.structured.lock').unlink()
        self.victim = self.root/'victim.structured.json'; self.original = self.victim.read_bytes()
        self.coordinator, _ = content._build_coordinator(self.document)
        self.coordinator.document_artifact_repository = self.repo
        for name, value in [('section_task_coordinator', self.coordinator), ('document_artifact_repository', self.repo)]:
            p = patch.object(main, name, value); p.start(); self.addCleanup(p.stop)
        self.client = TestClient(main.app); self.addCleanup(self.client.close)

    def test_every_public_request_name_field_rejects_invalid_identity(self):
        models = {m for module in (api_schemas, api_learning_schemas) for name, m in vars(module).items()
                  if name.endswith('Request') and inspect.isclass(m) and issubclass(m, BaseModel)
                  and 'doc_name' in m.model_fields}
        self.assertGreater(len(models), 15)
        for model in models:
            for name in BAD_NAMES:
                with self.subTest(model=model.__name__, name=name):
                    try: model.model_validate({'doc_name': name})
                    except ValidationError as error:
                        self.assertTrue(any(e['loc'] == ('doc_name',) for e in error.errors()))
                    else: self.fail('invalid document name accepted')

    def test_http_traversal_read_and_note_write_leave_outside_file_unchanged(self):
        token = NoteInteractionService(self.outside).read('victim', target_level='document')['target_token']
        for name in ('../victim', str(self.root/'victim'), r'..\victim'):
            with self.subTest(name=name):
                target = {'doc_name': name, 'target_type': 'document'}
                read = self.client.post('/documents/reading-interactions/note/read', json={'target': target})
                write = self.client.post('/documents/reading-interactions/note/save', json={
                    'target': target, 'text': 'outside write', 'expected_revision': 0,
                    'target_token': token, 'mutation_id': 'outside'})
                self.assertEqual((read.status_code, write.status_code), (422, 422))
                self.assertEqual(self.victim.read_bytes(), self.original)
                self.assertFalse((self.root/'victim.structured.lock').exists())

    def test_http_invalid_path_and_query_names_never_reach_services(self):
        with patch.object(self.coordinator, 'get_task_unit_contents') as batch, \
             patch.object(main, '_reading_content_service') as reading, \
             patch.object(main, 'enhancement_preflight_service') as enhancement, \
             patch.object(main, '_validate_document_lifecycle_doc_name') as lifecycle:
            for name in (r'..\victim', 'C:book', 'book\x00', 'book\n', 'a'*256):
                with self.subTest(name=name):
                    encoded = quote(name, safe='')
                    for response in (
                        self.client.post(f'/documents/{encoded}/task-units/content', json={'task_unit_ids':['unit']}),
                        self.client.get(f'/documents/{encoded}/reading-content'),
                        self.client.get('/documents/reparse-structure/status', params={'doc_name':name}),
                        self.client.get('/documents/lifecycle/rename-context', params={'doc_name':name}),
                    ): self.assertEqual(response.status_code, 422, response.text)
            batch.assert_not_called(); reading.assert_not_called()
            enhancement.assert_not_called(); lifecycle.assert_not_called()

    def test_repository_rejects_invalid_names_for_reads_and_writes(self):
        # Keep absolute-path attack fixtures within this disposable test root.
        names = [n for n in BAD_NAMES if n != '/tmp/victim'] + [str(self.root/'victim')]
        for name in names:
            for operation in (self.repo.load_document,
                              lambda n: self.repo.save_reparsed_document(self.document, n),
                              self.repo.delete_document):
                with self.subTest(name=name, operation=operation):
                    with self.assertRaises(ValueError): operation(name)
        self.assertEqual(self.victim.read_bytes(), self.original)

    def test_symlink_escape_and_discovery_do_not_read_or_write_outside(self):
        self.repo.base_dir.mkdir()
        link = self.repo.base_dir/'linked.structured.json'; link.symlink_to(self.victim)
        with patch.object(self.repo.store, 'load', wraps=self.repo.store.load) as load:
            with self.assertRaises(ValueError): self.repo.load_document('linked')
            load.assert_not_called()
        for operation in (lambda: self.repo.save_reparsed_document(self.document, 'linked'),
                          lambda: self.repo.delete_document('linked')):
            with self.assertRaises(ValueError): operation()
        self.assertEqual(self.repo.list_documents(), [])
        self.assertTrue(link.is_symlink()); self.assertEqual(self.victim.read_bytes(), self.original)

    def test_http_symlink_read_and_write_do_not_touch_outside_document(self):
        self.repo.base_dir.mkdir()
        (self.repo.base_dir/'linked.structured.json').symlink_to(self.victim)
        token = NoteInteractionService(self.outside).read('victim', target_level='document')['target_token']
        target = {'doc_name':'linked', 'target_type':'document'}
        with patch.object(self.repo.store, 'load', wraps=self.repo.store.load) as load:
            read = self.client.post('/documents/reading-interactions/note/read', json={'target':target})
            write = self.client.post('/documents/reading-interactions/note/save', json={
                'target':target, 'text':'outside write', 'expected_revision':0,
                'mutation_id':'symlink', 'target_token':token})
            # The name is valid; storage rejects its link using existing 400 mapping.
            self.assertEqual((read.status_code, write.status_code), (400, 400))
            self.assertEqual(read.json()['detail'], 'invalid_document_path')
            self.assertEqual(write.json()['detail'], 'invalid_document_path')
            load.assert_not_called()
        self.assertEqual(self.victim.read_bytes(), self.original)

    def test_upload_uses_name_policy_and_preserves_typed_error_contract(self):
        raw = self.root/'raw'
        with patch.object(main, '_RAW_DOCUMENT_BASE_DIR', raw):
            for name in ('../escape.txt', 'C:book.txt', 'c:relative.txt', '..txt'):
                with self.subTest(name=name):
                    response = self.client.post('/documents/upload', files={'file':(name,b'bytes','text/plain')})
                    self.assertEqual(response.status_code, 400, response.text)
                    self.assertEqual(response.json()['reason'], 'unsafe_filename')
            self.assertFalse(raw.exists())

    def test_rename_destination_and_non_string_names_are_rejected(self):
        from uuid import uuid4
        for name in (*BAD_NAMES, None, 123, True):
            with self.subTest(name=name):
                response = self.client.post('/documents/lifecycle/rename', json={
                    'doc_name':'book', 'action':'rename', 'new_doc_name':name,
                    'command_id':str(uuid4()), 'expected_source_token':'0'*64})
                self.assertEqual(response.status_code, 422, response.text)

    def test_real_tcp_traversal_and_batch_boundary(self):
        import socket
        import threading
        import time
        import httpx
        import uvicorn
        with socket.socket() as listener:
            listener.bind(('127.0.0.1', 0))
            server = uvicorn.Server(uvicorn.Config(main.app, lifespan='off', log_level='error'))
            worker = threading.Thread(target=server.run, kwargs={'sockets':[listener]}, daemon=True)
            worker.start()
            original_client = self.client
            try:
                for _ in range(200):
                    if server.started: break
                    time.sleep(.01)
                self.assertTrue(server.started)
                with httpx.Client(base_url=f'http://127.0.0.1:{listener.getsockname()[1]}') as client:
                    self.client = client
                    self.test_http_traversal_read_and_note_write_leave_outside_file_unchanged()
                    self.test_batch_limit_rejects_before_service_and_accepts_twenty()
            finally:
                self.client = original_client
                server.should_exit = True
                worker.join(timeout=5)
                self.assertFalse(worker.is_alive())

    def test_lock_symlink_escape_is_rejected(self):
        self.repo.base_dir.mkdir()
        (self.repo.base_dir/'book.structured.lock').symlink_to(self.victim)
        with self.assertRaises(ValueError): self.repo.save_reparsed_document(self.document, 'book')
        self.assertEqual(self.victim.read_bytes(), self.original)
        self.assertFalse((self.repo.base_dir/'book.structured.json').exists())

    def test_chinese_and_existing_pdf_txt_aliases_remain_valid_over_http(self):
        name = '中國文化【新千年版】 (1)'
        self.repo.save_document(self.document, name)
        self.repo.raw_dir.mkdir(parents=True)
        # A legacy alias must have one real source, not map every format to one book.
        for suffix in ('.pdf', '.TXT'):
            source = self.repo.raw_dir / (name+suffix)
            source.write_bytes(b'legacy source')
            for alias in (name, name+suffix, ' '+name+' '):
                with self.subTest(alias=alias):
                    result = self.client.post('/documents/reading-interactions/note/read',
                        json={'target':{'doc_name':alias,'target_type':'document'}})
                    self.assertEqual(result.status_code, 200, result.text)
                    self.assertEqual(self.repo.load_document(alias).to_dict(), self.repo.load_document(name).to_dict())
            other = name+('.TXT' if suffix == '.pdf' else '.pdf')
            result = self.client.post('/documents/reading-interactions/note/read',
                json={'target':{'doc_name':other,'target_type':'document'}})
            self.assertEqual(result.status_code, 404, result.text)
            source.unlink()

    def test_batch_limit_rejects_before_service_and_accepts_twenty(self):
        chapter = self.document.chapters[0]; section = chapter.sections[0]; unit = section.task_units[0]
        units = [replace(unit, unit_id=f'unit-{n}') for n in range(20)]
        document = replace(self.document, chapters=[replace(chapter, sections=[replace(section, task_units=units)])])
        coordinator, repository = content._build_coordinator(document)
        with patch.object(main, 'section_task_coordinator', coordinator), \
             patch.object(coordinator, 'get_task_unit_contents', wraps=coordinator.get_task_unit_contents) as service:
            for ids in ([], ['unit-0',' unit-0 '], [f'unit-{n}' for n in range(21)]):
                response = self.client.post('/documents/book/task-units/content', json={'task_unit_ids':ids})
                self.assertEqual(response.status_code, 422, response.text); service.assert_not_called()
            ids = [unit.unit_id for unit in reversed(units)]
            response = self.client.post('/documents/book/task-units/content', json={'task_unit_ids':ids})
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual([item['task_unit_id'] for item in response.json()['contents']], ids)
            self.assertEqual(service.call_count, 1)
            response = self.client.post('/documents/book/task-units/content', json={'task_unit_ids':['unit-0','missing']})
            self.assertEqual(response.status_code, 404, response.text); self.assertNotIn('contents', response.json())
            self.assertEqual(repository.write_calls, 0)
            self.assertEqual(coordinator.document_preparation_pipeline.prepare_calls, 0)


if __name__ == '__main__': unittest.main()
