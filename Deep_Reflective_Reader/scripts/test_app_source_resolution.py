"""APP-02: bounded source identity and real HTTP/SQL execution, no live models."""
from dataclasses import replace
import hashlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
from urllib.parse import quote

from fastapi.testclient import TestClient
import main
from app.cleanup_runtime import build_cleanup_service
from app.enhancement_preflight import EnhancementPreflightService
from app.section_task_coordinator import SectionTaskCoordinator
from config.storage_namespace_helper import StorageNamespaceHelper
from doc_loaders.extraction_identity import ExtractionIdentity
from document_preparation.accepted_source_snapshot import resolve_accepted_source_snapshot
from document_structure.structured_document_artifact_repository import StructuredDocumentArtifactRepository
from scripts import test_accepted_source_snapshot as source_fixture
from scripts import test_enhancement_postgres_preflight as sql_fixture


def filesystem(root):
    return {str(p.relative_to(root)): ('link', os.readlink(p)) if p.is_symlink()
            else ('dir',) if p.is_dir() else ('file', p.read_bytes()) for p in root.rglob('*')}


def document_for(template, name, raw, fmt):
    text = template.raw_text.replace('First', fmt.upper())
    identity = ExtractionIdentity.create(source_sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),
        source_format=fmt, extractor_id='fixture', extraction_version='accepted-v1',
        mode='fast' if fmt == 'pdf' else 'native', settings={})
    section = replace(template.chapters[0].sections[0], content=text, char_end=len(text))
    return replace(template, document_id=name, raw_text=text, source_path='untrusted/elsewhere',
        chapters=[replace(template.chapters[0], sections=[section])],
        parse_provenance={'extraction': identity.record(text)})


class FileSourceResolutionTests(unittest.TestCase):
    def setUp(self):
        fixture = source_fixture.AcceptedSourceTests(); fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.root = Path(fixture.temp.name) / 'raw'; self.root.mkdir()
        self.template = fixture.document
        self.repo = StructuredDocumentArtifactRepository(base_dir=self.root.parent/'structured', raw_dir=self.root)

    def test_qualified_and_unique_legacy_accepted_source_roundtrip(self):
        for name, fmt in [('same.pdf', 'pdf'), ('same.txt', 'txt'), ('中文.PDF', 'pdf'),
                          ('Who am I?.TXT', 'txt'), ('legacy', 'pdf')]:
            with self.subTest(name=name):
                path = self.root / (name if Path(name).suffix else name+'.PDF')
                path.write_bytes(('accepted '+name).encode())
                self.repo.save_reparsed_document(document_for(self.template, name, path, fmt), name)
                current = self.repo.load_document(name); before = filesystem(self.root.parent)
                source = StorageNamespaceHelper.resolve_raw_source(self.root, name)
                accepted = resolve_accepted_source_snapshot(document=current, namespace='test',
                    expected_document_id=name, structure_version=1, source_path=source, accepted=True)
                self.assertEqual(accepted.canonical_text, current.raw_text)
                self.assertEqual(source, path.resolve())
                self.assertEqual(filesystem(self.root.parent), before)

    def test_missing_ambiguous_and_illegal_names_are_pure(self):
        (self.root/'both.pdf').write_bytes(b'pdf'); (self.root/'both.txt').write_bytes(b'txt')
        (self.root/'absent.pdf.pdf').write_bytes(b'wrong identity')
        for name in ['both', 'absent.pdf', '../escape', '/tmp/absolute', 'bad\x00name']:
            with self.subTest(name=repr(name)):
                before = filesystem(self.root.parent)
                with self.assertRaises(ValueError): StorageNamespaceHelper.resolve_raw_source(self.root, name)
                self.assertEqual(filesystem(self.root.parent), before)
        absent = self.root/'missing-root'
        with self.assertRaises(ValueError): StorageNamespaceHelper.resolve_raw_source(absent, 'book')
        self.assertFalse(absent.exists())

    def test_internal_and_external_symlinks(self):
        target = self.root/'target.pdf'; target.write_bytes(b'inside')
        link = self.root/'link.pdf'; link.symlink_to(target)
        self.assertEqual(StorageNamespaceHelper.resolve_raw_source(self.root, 'link.pdf'), target.resolve())
        outside = self.root.parent/'outside.pdf'; outside.write_bytes(b'outside')
        link.unlink(); link.symlink_to(outside)
        before = filesystem(self.root.parent)
        with self.assertRaises(ValueError): StorageNamespaceHelper.resolve_raw_source(self.root, 'link.pdf')
        self.assertEqual(filesystem(self.root.parent), before)

    def test_file_runtime_capability_gate_is_retained(self):
        raw = self.root/'qualified.pdf'; raw.write_bytes(b'accepted')
        self.repo.save_reparsed_document(document_for(self.template, 'qualified.pdf', raw, 'pdf'), 'qualified.pdf')
        co = object.__new__(SectionTaskCoordinator)
        co.document_artifact_repository = self.repo
        co.document_preparation_pipeline = Mock()
        co.document_preparation_pipeline.loader_factory.base_dir = self.root
        service = EnhancementPreflightService(co.read_enhancement_state)
        before = filesystem(self.root.parent)
        with patch.object(main, 'enhancement_preflight_service', service):
            response = TestClient(main.app).post('/documents/reparse-structure/preflight', json={'doc_name':'qualified.pdf'})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()['reason'], 'enhancement_storage_backend_unsupported')
        self.assertFalse(response.json()['ready'])
        self.assertIsNone(build_cleanup_service(namespace='test', coordinator=co, provider=Mock()))
        self.assertEqual(filesystem(self.root.parent), before)
        self.assertEqual(co.document_preparation_pipeline.mock_calls, [])


@unittest.skipUnless(os.environ.get('DEEP_READER_TEST_POSTGRES_DSN'), 'requires disposable PostgreSQL DSN')
class PostgresSourceResolutionTests(unittest.TestCase):
    setUpClass = classmethod(sql_fixture.PostgresPreflightTests.setUpClass.__func__)
    drop_schema = classmethod(sql_fixture.PostgresPreflightTests.drop_schema.__func__)
    enable_execution = sql_fixture.PostgresPreflightTests.enable_execution

    def setUp(self):
        sql_fixture.PostgresPreflightTests.setUp(self)
        self.root = Path(self.fixture.temp.name)
        self.store.raw_root = self.root
        self.name = 'doc'
        self.enable_execution()

    def create(self, name, fmt='pdf', raw_name=None):
        self.name = name
        self.path = self.root/(raw_name or (name if Path(name).suffix.lower() in ('.pdf','.txt') else name+'.'+fmt))
        self.path.write_bytes(('accepted source '+name).encode())
        self.repository.save_reparsed_document(document_for(self.fixture.document, name, self.path, fmt), name)

    def snapshot(self):
        return self.store.load_enhancement_snapshot(self.store.target_for_doc_name(self.name))

    def request(self, path, payload):
        with (patch.object(main, 'section_task_coordinator', self.coordinator),
              patch.object(main, 'enhancement_preflight_service', self.service),
              patch('app.enhancement_preflight.enhancement_preflight_service', self.service)):
            return TestClient(main.app, raise_server_exceptions=False).post(path, json=payload)

    def preflight(self):
        return self.request('/documents/reparse-structure/preflight', {'doc_name':self.name})

    def execute(self, token):
        return self.request('/documents/reparse-structure', {'doc_name':self.name,
            'parser_mode':'llm_enhanced','discard_interactions':True,'confirmation_token':token})

    def test_qualified_matrix_http_preflight_cleanup_and_real_commit(self):
        for name, fmt, raw in [('shared.pdf','pdf',None),('shared.txt','txt',None),
                              ('中文.PDF','pdf',None),('Who am I?.TXT','txt',None),
                              ('Legacy','pdf','Legacy.PDF')]:
            with self.subTest(name=name):
                self.create(name, fmt, raw)
                before = self.snapshot(); files = filesystem(self.root); calls = self.provider.calls
                ready = self.preflight()
                self.assertEqual(ready.status_code, 200, ready.text)
                self.assertTrue(ready.json()['ready'], ready.text)
                cleanup = self.coordinator.read_cleanup_input(name)
                self.assertEqual(cleanup.source.canonical_text, before.document.chapters[0].sections[0].content)
                from smart_cleanup_routes import get_smart_cleanup_service
                flow = build_cleanup_service(namespace=self.store.namespace, coordinator=self.coordinator,
                                             provider=self.provider)
                with patch.dict(main.app.dependency_overrides,{get_smart_cleanup_service:lambda:flow}):
                    cleanup_response = TestClient(main.app).post(
                        '/documents/'+quote(name,safe='')+'/smart-cleanup/preflight',json={})
                self.assertEqual(cleanup_response.status_code,200,cleanup_response.text)
                self.assertEqual(cleanup_response.json()['reason'],cleanup.eligibility().reason)
                self.assertFalse(cleanup_response.json()['execution_available'])  # Fast/TXT are not OCR cleanup targets.
                self.assertEqual(self.snapshot(), before); self.assertEqual(filesystem(self.root), files)
                self.assertEqual(self.provider.calls, calls)
                result = self.execute(ready.json()['confirmation_token'])
                self.assertEqual(result.status_code, 200, result.text)
                self.assertTrue(result.json()['success']); self.assertEqual(self.provider.calls, calls+1)
                self.assertEqual(filesystem(self.root), files)
                self.assertEqual(self.coordinator.read_enhancement_state(name).source.canonical_text, cleanup.source.canonical_text)

    def test_missing_or_replaced_source_rejects_before_generation(self):
        for failure in ['missing','replacement','double_suffix','external_link']:
            with self.subTest(failure=failure):
                self.create('source-'+failure+'.pdf')
                if failure == 'replacement': self.path.write_bytes(b'changed')
                else:
                    original = self.path.read_bytes(); self.path.unlink()
                    if failure == 'double_suffix': Path(str(self.path)+'.pdf').write_bytes(original)
                    if failure == 'external_link':
                        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
                        outside = Path(temp.name)/'source.pdf'; outside.write_bytes(original); self.path.symlink_to(outside)
                before = self.snapshot(); files = filesystem(self.root); calls = self.provider.calls
                response = self.preflight()
                self.assertEqual(response.status_code, 200, response.text)
                self.assertFalse(response.json()['ready'], response.text)
                self.assertIsNone(response.json()['confirmation_token'])
                self.assertEqual(self.provider.calls, calls); self.assertEqual(self.snapshot(), before)
                self.assertEqual(filesystem(self.root), files)

    def test_bare_ambiguity_returns_readiness_not_guess_or_exception(self):
        self.create('legacy')
        (self.root/'legacy.txt').write_bytes(b'other source')
        before = self.snapshot(); files = filesystem(self.root)
        response = self.preflight()
        self.assertEqual(response.status_code, 200, response.text)
        self.assertFalse(response.json()['ready'])
        self.assertEqual(response.json()['reason'], 'source_ambiguous')
        self.assertEqual(self.snapshot(), before); self.assertEqual(filesystem(self.root), files)
        self.assertEqual(self.provider.calls, 0)

    def test_commit_rechecks_current_raw_selection(self):
        for failure in ['missing','replacement','double_suffix','external_link','bare_ambiguity']:
            with self.subTest(failure=failure):
                self.create('during-'+failure if failure == 'bare_ambiguity' else 'during-'+failure+'.pdf')
                ready = self.preflight(); self.assertTrue(ready.json()['ready'], ready.text)
                before = self.snapshot(); captured = {}; original = self.coordinator.document_preparation_pipeline.build_enhancement_candidate
                def generate(**kwargs):
                    candidate = original(**kwargs)
                    if failure == 'bare_ambiguity': (self.root/(self.name+'.txt')).write_bytes(b'other')
                    elif failure == 'replacement': self.path.write_bytes(b'changed')
                    else:
                        content = self.path.read_bytes(); self.path.unlink()
                        if failure == 'double_suffix': Path(str(self.path)+'.pdf').write_bytes(content)
                        if failure == 'external_link':
                            temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
                            outside = Path(temp.name)/'source.pdf'; outside.write_bytes(content); self.path.symlink_to(outside)
                    captured['files'] = filesystem(self.root)
                    return candidate
                with patch.object(self.coordinator.document_preparation_pipeline,'build_enhancement_candidate',side_effect=generate):
                    result = self.execute(ready.json()['confirmation_token'])
                self.assertEqual(result.status_code, 409, result.text)
                self.assertEqual(self.snapshot(), before); self.assertEqual(filesystem(self.root), captured['files'])

    def test_renamed_qualified_source_uses_current_identity(self):
        self.create('before.txt','txt'); old_path = self.path
        new_name = '改名.TXT'; new_path = self.root/new_name
        self.repository.rename_document(self.name,new_name,
            lambda document,name: replace(document,document_id=name,source_path=str(new_path)))
        old_path.rename(new_path); self.name = new_name; self.path = new_path
        ready = self.preflight(); self.assertTrue(ready.json()['ready'], ready.text)
        result = self.execute(ready.json()['confirmation_token'])
        self.assertEqual(result.status_code, 200, result.text)
        self.assertEqual(self.coordinator.read_cleanup_input(new_name).document_id,new_name)

    def test_qualified_alias_of_unique_legacy_owner(self):
        self.create('Alias','txt','Alias.TXT')
        self.name = 'Alias.TXT'
        ready = self.preflight(); self.assertTrue(ready.json()['ready'], ready.text)
        self.assertEqual(self.coordinator.read_cleanup_input(self.name).document_id, 'Alias')
        result = self.execute(ready.json()['confirmation_token'])
        self.assertEqual(result.status_code, 200, result.text)
        self.assertFalse(self.repository.contains_document('Alias.TXT'))

    def test_coexisting_qualified_sources_and_consent_are_isolated(self):
        self.create('Pair.pdf'); pdf = self.snapshot()
        pdf_token = self.preflight().json()['confirmation_token']
        self.create('Pair.txt','txt'); txt = self.snapshot()
        rejected = self.execute(pdf_token)
        self.assertEqual(rejected.status_code, 409, rejected.text)
        self.assertEqual(self.provider.calls, 0); self.assertEqual(self.snapshot(), txt)
        token = self.preflight().json()['confirmation_token']
        self.assertEqual(self.execute(token).status_code, 200)
        self.name = 'Pair.pdf'
        self.assertEqual(self.snapshot(), pdf)
        self.assertEqual(self.coordinator.read_enhancement_state(self.name).source.extraction_identity.source_format, 'pdf')

    def test_reimport_invalidates_preflight_before_provider_dispatch(self):
        self.create('Reimport.pdf')
        token = self.preflight().json()['confirmation_token']
        self.repository.delete_document(self.name)
        self.create('Reimport.pdf')
        before = self.snapshot()
        rejected = self.execute(token)
        self.assertEqual(rejected.status_code,409,rejected.text)
        self.assertEqual(self.provider.calls,0); self.assertEqual(self.snapshot(),before)

    def test_internal_symlink_retarget_during_generation_is_rejected(self):
        self.create('Linked.pdf')
        target = self.root/'target.pdf'; self.path.rename(target); self.path.symlink_to(target)
        ready = self.preflight(); self.assertTrue(ready.json()['ready'], ready.text)
        before = self.snapshot()
        original = self.coordinator.document_preparation_pipeline.build_enhancement_candidate
        def generate(**kwargs):
            candidate = original(**kwargs)
            other = self.root/'other.pdf'; other.write_bytes(b'different source')
            self.path.unlink(); self.path.symlink_to(other)
            return candidate
        with patch.object(self.coordinator.document_preparation_pipeline,'build_enhancement_candidate',side_effect=generate):
            response = self.execute(ready.json()['confirmation_token'])
        self.assertEqual(response.status_code, 409, response.text)
        self.assertEqual(self.snapshot(), before)

    def test_standalone_cleanup_qualified_http_commit_and_source_recheck(self):
        from scripts.test_cleanup_runtime import PostgresCleanupRuntimeTests
        from smart_cleanup_routes import get_smart_cleanup_service
        from document_structure.reading_content import project_document
        for change_source in [False, True, 'during_link']:
            with self.subTest(change_source=change_source):
                case = PostgresCleanupRuntimeTests(); case.scoped = self.scoped
                case.setUp(); self.addCleanup(case.doCleanups)
                repo = case.f.repo; repo.store.raw_root = Path(case.f.temp.name)
                name = '掃描.PDF'; path = Path(case.f.temp.name)/name
                repo.rename_document('book',name,
                    lambda document, new: replace(document,document_id=new,source_path=str(path)))
                case.f.path.rename(path)
                if change_source == 'during_link':
                    original_path = path.with_name('original.pdf')
                    path.rename(original_path); path.symlink_to(original_path)
                case.coordinator.ensure_document_task_layout(name,task_unit_split_mode='progressive')
                before = repo.load_enhancement_snapshot(name)
                url = '/documents/'+quote(name,safe='')+'/smart-cleanup'
                with patch.dict(main.app.dependency_overrides,{get_smart_cleanup_service:lambda:case.flow}):
                    client = TestClient(main.app)
                    ready = client.post(url+'/preflight',json={})
                    self.assertEqual(ready.status_code,200,ready.text)
                    self.assertTrue(ready.json()['execution_available'],ready.text)
                    self.assertEqual(len(case.f.provider.calls),0)
                    self.assertEqual(repo.load_enhancement_snapshot(name),before)
                    request = {key:ready.json()[key] for key in ('operation_id','confirmation_token')}
                    if change_source is True: path.write_bytes(b'changed source before cleanup')
                    complete = case.f.provider.complete_text_with_output_budget
                    def during_plan(*args, **kwargs):
                        result = complete(*args, **kwargs)
                        if change_source == 'during_link':
                            temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
                            outside = Path(temp.name)/'replacement.pdf'; outside.write_bytes(path.read_bytes())
                            path.unlink(); path.symlink_to(outside)
                        return result
                    with patch.object(case.f.provider,'complete_text_with_output_budget',side_effect=during_plan):
                        response = client.post(url,json=request)
                after = repo.load_enhancement_snapshot(name)
                if change_source:
                    self.assertNotEqual(response.json().get('commit'),'committed',response.text)
                    if change_source == 'during_link':
                        self.assertEqual(response.status_code,409,response.text)
                        self.assertEqual(response.json()['commit'],'not_committed')
                        self.assertEqual(response.json()['reason'],'source_changed')
                        self.assertFalse(response.json()['outcome']['reading_content_changed'])
                    self.assertEqual(after,before)
                    self.assertEqual(len(case.f.provider.calls),1 if change_source == 'during_link' else 0)
                else:
                    self.assertEqual(response.status_code,200,response.text)
                    self.assertEqual(response.json()['commit'],'committed',response.text)
                    self.assertEqual(len(case.f.provider.calls),1)
                    effective = project_document(after.document)
                    self.assertNotIn('@@??',''.join(s.content for c in effective.chapters for s in c.sections))
                    self.assertEqual(''.join(s.content for c in after.document.chapters for s in c.sections),case.f.text)


if __name__ == '__main__': unittest.main(verbosity=2)
