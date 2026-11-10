"""LLRN-15 real DI, file/SQL identities, lazy dependencies and resource ownership."""
from dataclasses import replace
from contextlib import chdir
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, Thread
from types import SimpleNamespace
import subprocess
import sys
import unittest
from unittest.mock import Mock, patch
from uuid import uuid4

from dependency_injector import providers
import psycopg

from config.app_DI_config import AppDIConfig, LearningRuntimeConfig
from config.container import ApplicationLookupContainer
from config.faiss_storage_config import FaissStorageConfig
from config.learning_runtime import existing_profile_reader
from config.learning_storage import build_learning_storage
from db.postgres_learning_repository import PostgresLearningRepository
from db.postgres_reading_preferences_repository import PostgresReadingPreferencesRepository
from db.postgres_structured_document_store import PostgresStructuredDocumentStore
from document_structure.learning_repository import LearningStorageError, OperationWrite
from document_structure.structured_document_artifact_repository import StructuredDocumentArtifactRepository
from profile.versioned_document_profile_store import VersionedDocumentProfileStore
from reading_preferences.models import WorkspaceScope, PreferenceStorageError
from scripts.test_learning_file_repository import configure, fixture
from scripts.test_learning_translation_service import Provider
from scripts.test_learning_hints_service import Provider as HintsProvider
from scripts import test_reading_preferences_postgres as preference_tests
from section_tasks.learning_generation_policy import AdmissionDecision, AllowValidGeneration
from section_tasks.learning_translation_service import LearningTranslationError


class OnceProvider(Provider):
    def complete_text_once_with_output_budget(self, prompt, output_tokens):
        if 'proficiency' in json.loads(prompt.split('INPUT_JSON:\n', 1)[1]):
            return HintsProvider.complete_text_with_output_budget(self, prompt, output_tokens)
        return super().complete_text_with_output_budget(prompt, output_tokens)


class CompositionCases:
    def prepare(self):
        self.provider = OnceProvider()
        self.container.llm_provider.override(providers.Object(self.provider))
        self.profile_store = SimpleNamespace(load_source=lambda config: SimpleNamespace(document_language_code='en'))
        self.container.document_profile_store.override(providers.Object(self.profile_store))
        self.addCleanup(self.container.shutdown_resources)
        self.docs = self.container.structured_document_artifact_repository()
        self.prefs = self.container.reading_preferences_repository()
        self.repo = self.container.learning_repository()
        self.docs.save_reparsed_document(fixture(), 'book')
        configure(self.prefs)
        self.service = self.container.learning_service()
        self.block = self.repo.read_sources('book').blocks[0]

    def request(self, command='generate', component='translation'):
        settings = self.container.reading_preferences_service().read_book('book')
        return self.service.request('book', self.block.source.content_block_id, component,
            command_id=command, expected_source=self.block.source,
            expected_workspace_revision=settings.workspace.revision,
            expected_book_revision=settings.book.revision)

    def read(self):
        return self.service.read('book', (self.block.source.content_block_id,))

    def test_same_instances_and_scope(self):
        storage = self.container.learning_storage()
        self.assertIs(storage.documents, self.docs)
        self.assertIs(storage.preferences, self.prefs)
        self.assertIs(storage.learning, self.service.repository)
        self.assertIs(self.service.preferences, self.container.reading_preferences_service())
        self.assertEqual(self.repo.scope, WorkspaceScope(self.namespace))
        self.assertIs(self.repo.preference_repository, self.prefs)
        if storage.lifecycle is not None:
            self.assertIs(self.repo.document_repository, self.docs)
        else:
            self.assertIs(self.repo.document_store, self.docs.store)

    def test_reads_and_startup_never_resolve_provider(self):
        factory = Mock(side_effect=AssertionError('provider resolved on read'))
        self.container.llm_provider.override(providers.Callable(factory))
        self.container.init_resources()
        self.assertEqual([c.view.state for c in self.read().blocks[0].components], ['missing', 'missing'])
        self.assertEqual(self.service.status('book', 'absent').state, 'missing')
        self.assertIsNone(self.service.lookup_command('book', 'absent'))
        self.container.reading_preferences_service().read_workspace()
        factory.assert_not_called()

    def test_missing_provider_capability_and_once_contract_fail_before_claim(self):
        for provider in (None, Provider(), SimpleNamespace(complete_text_once_with_output_budget=lambda *_: 'unused')):
            with self.subTest(provider=type(provider).__name__), self.container.llm_provider.override(providers.Object(provider)):
                with self.assertRaisesRegex(LearningTranslationError, 'capacity_unavailable'):
                    self.request()
                self.assertIsNone(self.service.lookup_command('book', 'generate'))
        factory = Mock(side_effect=ValueError('missing key'))
        with self.container.llm_provider.override(providers.Callable(factory)):
            with self.assertRaisesRegex(LearningTranslationError, 'capacity_unavailable'):
                self.request()
        self.assertIsNone(self.service.lookup_command('book', 'generate'))

    def test_generate_drain_read_and_reuse_without_provider(self):
        self.assertEqual(self.request().outcome, 'claimed')
        self.container.shutdown_resources()
        self.assertEqual(self.read().blocks[0].components[0].view.state, 'ready')
        self.assertEqual(len(self.provider.calls), 1)
        self.assertEqual(self.provider.calls[0][1], self.provider.caps.max_output_tokens)
        # Reinitializing the resource retains storage/lifecycle and starts a new worker.
        self.service = self.container.learning_service()
        with self.container.llm_provider.override(providers.Object(None)):
            self.assertEqual(self.request('reuse').outcome, 'reuse')

    def test_hints_use_same_provider_and_persist_independently(self):
        self.assertEqual(self.request('hints', 'hints').outcome, 'claimed')
        self.container.shutdown_resources()
        components = self.read().blocks[0].components
        self.assertEqual([c.view.state for c in components], ['missing', 'ready'])
        self.assertEqual(len(self.provider.calls), 1)

    def test_shutdown_waits_for_accepted_job(self):
        entered, release, drained = Event(), Event(), Event()
        original = self.provider.complete_text_once_with_output_budget
        def blocked(*args):
            entered.set()
            if not release.wait(10):
                raise TimeoutError('test not released')
            return original(*args)
        self.provider.complete_text_once_with_output_budget = blocked
        self.addCleanup(release.set)
        self.request()
        self.assertTrue(entered.wait(5))
        thread = Thread(target=lambda: (self.container.shutdown_resources(), drained.set()))
        thread.start()
        try:
            self.assertFalse(drained.wait(.1))
        finally:
            release.set()
            thread.join(10)
        self.assertTrue(drained.is_set())
        self.assertEqual(self.service.status('book', 'generate').state, 'succeeded')

    def test_ordinary_document_rename_and_delete_coordinate_preferences(self):
        submission = self.request()
        self.container.shutdown_resources()
        snapshot = self.repo.inspect_operation('book', submission.operation_id)
        self.assertTrue(self.docs.rename_document('book', 'renamed', lambda doc, name: replace(doc, document_id=name)))
        self.assertEqual(self.prefs.read_book('renamed').settings.active_mode, 'language_learning')
        self.assertTrue(self.prefs.read_book('book').deleted)
        self.assertIsNotNone(self.repo.lookup_command('renamed', 'generate'))
        with self.assertRaisesRegex(ValueError, 'learning_document_missing'):
            self.repo.write_operation('book', OperationWrite(submission.operation_id,
                snapshot.entry_revision, snapshot.operation.fence, 1000, 'renew', lease_until_ms=2000))
        self.assertTrue(self.docs.delete_document('renamed'))
        self.assertTrue(self.prefs.read_book('renamed').deleted)


class FileCompositionTests(CompositionCases, unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.namespace = 'file-workspace'
        self.policy = LearningRuntimeConfig(str(self.root/'preferences'), str(self.root/'learning'))
        self.container = ApplicationLookupContainer.build(AppDIConfig(structured_storage_backend='file', learning_runtime=self.policy))
        self.container.file_structured_document_artifact_repository.override(providers.Object(
            StructuredDocumentArtifactRepository(base_dir=str(self.root/'documents'), namespace=self.namespace)))
        self.prepare()

    def test_pending_journal_is_not_repaired_by_read_or_reinit(self):
        path = self.prefs.lifecycle_journal_path
        path.write_text('{}')
        self.container.shutdown_resources()
        self.service = self.container.learning_service()
        with self.assertRaisesRegex(PreferenceStorageError, 'recovery_required'):
            self.read()
        self.assertEqual(path.read_text(), '{}')

    def test_read_does_not_write_any_files(self):
        before = {p: p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        self.read()
        after = {p: p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        self.assertEqual(before, after)

    def test_admission_override_rechecked_at_dispatch(self):
        # Storage must capture the same policy object used for claim and dispatch.
        self.container.shutdown_resources()
        policy = Mock()
        policy.evaluate.side_effect = [AdmissionDecision(True), AdmissionDecision(False, 'restricted')]
        other = ApplicationLookupContainer.build(AppDIConfig(structured_storage_backend='file', learning_runtime=self.policy))
        other.file_structured_document_artifact_repository.override(providers.Object(
            StructuredDocumentArtifactRepository(base_dir=str(self.root/'documents'), namespace=self.namespace)))
        other.learning_admission.override(providers.Object(policy))
        other.document_profile_store.override(providers.Object(self.profile_store))
        other.llm_provider.override(providers.Object(self.provider))
        self.addCleanup(other.shutdown_resources)
        self.container = other
        self.service = other.learning_service()
        self.request()
        other.shutdown_resources()
        self.assertEqual(policy.evaluate.call_count, 2)
        self.assertEqual(self.provider.calls, [])


class PostgresCompositionTests(CompositionCases, unittest.TestCase):
    setUpClass = classmethod(preference_tests.PostgresPreferencesTests.setUpClass.__func__)
    drop_schema = classmethod(preference_tests.PostgresPreferencesTests.drop_schema.__func__)
    isolated_schema = preference_tests.PostgresPreferencesTests.isolated_schema

    def setUp(self):
        self.namespace = uuid4().hex
        self.container = ApplicationLookupContainer.build(AppDIConfig(structured_storage_backend='postgres',
            postgres_dsn=self.scoped, postgres_namespace=self.namespace))
        self.prepare()

    def test_injected_sql_storage_identity_mismatch_is_rejected(self):
        for store, prefs in (
            (PostgresStructuredDocumentStore(self.scoped, 'other'), self.prefs),
            (PostgresStructuredDocumentStore('other', self.namespace), self.prefs),
            (self.docs.store, PostgresReadingPreferencesRepository(self.scoped, WorkspaceScope('other'))),
        ):
            with self.assertRaisesRegex(ValueError, 'storage_identity_mismatch'):
                PostgresLearningRepository(self.scoped, self.repo.scope, documents=store, preferences=prefs)

    def test_missing_migrations_fail_reads_without_creating_tables(self):
        with self.isolated_schema() as dsn:
            container = ApplicationLookupContainer.build(AppDIConfig(structured_storage_backend='postgres',
                postgres_dsn=dsn, postgres_namespace=self.namespace))
            try:
                container.init_resources()
                with self.assertRaises(PreferenceStorageError):
                    container.reading_preferences_service().read_workspace()
                with self.assertRaises(LearningStorageError):
                    container.learning_repository().read_sources('book')
                with psycopg.connect(dsn) as conn:
                    count = conn.execute('SELECT count(*) FROM information_schema.tables WHERE table_schema=current_schema()').fetchone()[0]
                self.assertEqual(count, 0)
            finally:
                container.shutdown_resources()


class WiringBoundaryTests(unittest.TestCase):
    def test_default_startup_does_not_create_paths_or_contact_dependencies(self):
        with TemporaryDirectory() as root, chdir(root):
            policy = LearningRuntimeConfig(root+'/prefs', root+'/learning')
            container = ApplicationLookupContainer.build(AppDIConfig(structured_storage_backend='file', learning_runtime=policy))
            factory = Mock(side_effect=AssertionError('provider startup'))
            container.llm_provider.override(providers.Callable(factory))
            try:
                container.init_resources()
                container.reading_preferences_service().read_workspace()
                self.assertEqual(list(Path(root).rglob('*')), [])
                factory.assert_not_called()
            finally:
                container.shutdown_resources()

    def test_clean_import_and_startup_without_keys_network_or_files(self):
        code = '''
from unittest.mock import patch
from pathlib import Path
with patch('socket.socket.connect', side_effect=AssertionError('network')):
    from config.app_DI_config import AppDIConfig
    from config.container import ApplicationLookupContainer
    container = ApplicationLookupContainer.build(AppDIConfig(structured_storage_backend='file'))
    container.init_resources()
    container.reading_preferences_service().read_workspace()
    container.shutdown_resources()
assert not list(Path('.').rglob('*'))
'''
        env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]), PYTHONDONTWRITEBYTECODE='1')
        env.pop('OPENAI_API_KEY', None)
        with TemporaryDirectory() as root:
            result = subprocess.run([sys.executable, '-c', code], cwd=root, env=env,
                                    capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_sql_construction_does_not_connect_or_migrate(self):
        with patch('psycopg.connect', side_effect=AssertionError('startup SQL')):
            container = ApplicationLookupContainer.build(AppDIConfig(structured_storage_backend='postgres',
                postgres_dsn='host=unconfigured', postgres_namespace='example'))
            try:
                container.init_resources()
                self.assertEqual(container.learning_repository().scope.namespace, 'example')
            finally:
                container.shutdown_resources()

    def test_missing_dsn_is_explicit_not_file_fallback(self):
        container = ApplicationLookupContainer.build(AppDIConfig(structured_storage_backend='postgres', postgres_dsn=''))
        with self.assertRaisesRegex(ValueError, 'storage_unavailable'):
            container.learning_service()

    def test_backend_mismatch_is_explicit(self):
        with self.assertRaisesRegex(ValueError, 'backend_mismatch'):
            build_learning_storage(backend='postgres', documents=StructuredDocumentArtifactRepository(),
                policy=LearningRuntimeConfig(), dsn='unused', namespace='default', admission=AllowValidGeneration())

    def test_unsafe_public_repository_override_is_rejected_for_learning(self):
        container = ApplicationLookupContainer.build(AppDIConfig(structured_storage_backend='file'))
        container.structured_document_artifact_repository.override(providers.Object(StructuredDocumentArtifactRepository()))
        with self.assertRaisesRegex(ValueError, 'document_repository_mismatch'):
            container.learning_service()

    def test_source_profile_is_read_only_and_does_not_reenter_repository(self):
        with TemporaryDirectory() as root:
            path = Path(root)/'profile.json'
            path.write_text(json.dumps({'document_language': 'en'}))
            config = SimpleNamespace(doc_name='book', get_raw_profile_path=lambda: str(path))
            factory = Mock(return_value=config)
            repo = Mock()
            repo.profile_structure_invalidated.side_effect = AssertionError('transaction reentry')
            store = VersionedDocumentProfileStore(repo)
            scope = WorkspaceScope('default')
            reader = existing_profile_reader(scope, store, factory)
            before = path.read_bytes()
            self.assertEqual(reader.read(scope, 'book').code, 'en')
            factory.assert_called_once_with('book', read_only=True)
            repo.profile_structure_invalidated.assert_not_called()
            self.assertEqual(path.read_bytes(), before)

    def test_missing_profile_reader_is_unavailable_and_aliases_invalid(self):
        scope = WorkspaceScope('default')
        config = SimpleNamespace(doc_name='book')
        reader = existing_profile_reader(scope, object(), lambda *args, **kw: config)
        self.assertEqual(reader.read(scope, 'book').problem, 'profile_unavailable')
        self.assertEqual(reader.read(scope, 'book.pdf').problem, 'profile_invalid')

    def test_profile_path_resolution_has_no_mkdir_or_migration(self):
        scope = WorkspaceScope('default')
        reader = existing_profile_reader(scope, VersionedDocumentProfileStore(), FaissStorageConfig)
        with (patch.object(Path, 'mkdir', side_effect=AssertionError('mkdir')),
              patch.object(FaissStorageConfig, '_migrate_legacy_namespace_dir', side_effect=AssertionError('migration'))):
            self.assertEqual(reader.read(scope, 'missing-'+uuid4().hex).problem, 'profile_missing')

    def test_invalid_operational_limits_rejected(self):
        for values in ({'max_workers': 0}, {'max_workers': True}, {'lease_ms': -1}, {'results_base_dir': ''}):
            with self.assertRaises(ValueError):
                LearningRuntimeConfig(**values)


if __name__ == '__main__':
    unittest.main()
