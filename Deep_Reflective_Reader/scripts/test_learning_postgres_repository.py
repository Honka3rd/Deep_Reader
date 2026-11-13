"""LLRN-08: disposable PostgreSQL contracts, faults, races and migration tests."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace
import multiprocessing
import os
from pathlib import Path
from threading import Event
import time
import unittest
from unittest.mock import patch
from uuid import uuid4

import psycopg
from psycopg import sql
from psycopg.conninfo import make_conninfo
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from db import learning_rows
from db.postgres_learning_repository import PostgresLearningRepository
from db.postgres_reading_preferences_repository import PostgresReadingPreferencesRepository
from db.postgres_structured_document_store import PostgresStructuredDocumentStore
from db.postgres_structured_document_artifact_repository import PostgresStructuredDocumentArtifactRepository
from document_structure import learning_file_codec as codec
from document_structure.learning_repository import LearningCommitUnknown, LearningStorageError, OperationWrite
from reading_preferences.models import WorkspaceScope
from section_tasks.learning_cache import TranslationKey
from section_tasks.learning_generation_policy import BatchPlan, GenerationBinding, GenerationCommand, AdmissionDecision
from section_tasks.learning_sentence_segmenter import segment_learning_block
from shared.learning_results import GenerationProvenance
from shared.learning_source import LearningContractError
from scripts import test_learning_file_repository as file_tests
from scripts import test_reading_preferences_postgres as preference_tests

TABLES = ('learning_sentences','learning_cache_entries','learning_operations','learning_commands','learning_checkpoints')
MIGRATIONS = Path(__file__).resolve().parents[1] / 'db/migrations'


def repository(dsn, namespace):
    return PostgresLearningRepository(dsn, WorkspaceScope(namespace))


def process_call(dsn, namespace, action, command, sentences, plan, barrier, output):
    repo = repository(dsn, namespace)
    barrier.wait(timeout=20)
    try:
        if action == 'claim':
            result = repo.claim('book', command, sentences=sentences, plan=plan, now_ms=10, lease_until_ms=100)
            output.put((result.outcome, result.receipt.operation_id, result.replay))
        else:
            output.put(repo.write_operation('book', command).operation.state)
    except LearningContractError:
        output.put('conflict')


def crash_call(dsn, namespace, action, command, sentences, plan, after):
    repo = repository(dsn, namespace)
    real = repo._connect

    class Crash:
        def __init__(self): self.connection = real()
        def __getattr__(self, name): return getattr(self.connection, name)
        def commit(self):
            if after: self.connection.commit()
            os._exit(72 if after else 71)

    repo._connect = Crash
    if action == 'claim':
        repo.claim('book', command, sentences=sentences, plan=plan, now_ms=10, lease_until_ms=100)
    else:
        repo.write_operation('book', command)


class LearningPostgresTests(unittest.TestCase):
    setUpClass = classmethod(preference_tests.PostgresPreferencesTests.setUpClass.__func__)
    drop_schema = classmethod(preference_tests.PostgresPreferencesTests.drop_schema.__func__)
    isolated_schema = preference_tests.PostgresPreferencesTests.isolated_schema

    def setUp(self):
        self.namespace = uuid4().hex
        self.docs = PostgresStructuredDocumentArtifactRepository(PostgresStructuredDocumentStore(self.scoped, self.namespace))
        self.prefs = PostgresReadingPreferencesRepository(self.scoped, WorkspaceScope(self.namespace))
        self.repo = repository(self.scoped, self.namespace)
        self.docs.save_reparsed_document(file_tests.fixture(), 'book')
        file_tests.configure(self.prefs)
        self.block = self.repo.read_sources('book').blocks[0]
        self.sentences = segment_learning_block(source=self.block.source, effective_text=self.block.effective_text, source_language='en')
        self.key = TranslationKey(self.sentences.reference, 'en', 'zh', 'policy-1', 'prompt-1')
        self.binding = GenerationBinding(self.key, 1, 1)
        self.command = GenerationCommand('first', self.binding)
        self.plan = BatchPlan((tuple(s.sentence_id for s in self.sentences.sentences),))
        self.provenance = GenerationProvenance('test', 'model', 'prompt-1')
        with psycopg.connect(self.scoped) as conn:
            self.row_id = conn.execute('SELECT id FROM documents WHERE namespace=%s AND document_name=%s',
                                       (self.namespace, 'book')).fetchone()[0]

    claim = file_tests.LearningFileRepositoryTests.claim
    write = file_tests.LearningFileRepositoryTests.write
    batch = file_tests.LearningFileRepositoryTests.batch
    partial = file_tests.LearningFileRepositoryTests.partial
    result = file_tests.LearningFileRepositoryTests.result
    test_hint_assessment_must_match_durable_checkpoint = file_tests.LearningFileRepositoryTests.test_hint_assessment_must_match_durable_checkpoint
    edit = file_tests.LearningFileRepositoryTests.edit
    test_source_pagination_and_stale_cursor = file_tests.LearningFileRepositoryTests.test_source_pagination_and_stale_cursor
    test_query_limits_and_scope_rejected = file_tests.LearningFileRepositoryTests.test_query_limits_and_scope_rejected
    test_forged_parent_and_raw_source_rejected = file_tests.LearningFileRepositoryTests.test_forged_parent_and_raw_source_rejected
    test_partial_payload_validation_and_assembly = file_tests.LearningFileRepositoryTests.test_partial_payload_validation_and_assembly
    test_empty_hints_is_persisted_ready_and_translation_independent = file_tests.LearningFileRepositoryTests.test_empty_hints_is_persisted_ready_and_translation_independent
    test_hint_wrong_source_span_rejected_before_checkpoint = file_tests.LearningFileRepositoryTests.test_hint_wrong_source_span_rejected_before_checkpoint
    test_settings_change_blocks_dispatch = file_tests.LearningFileRepositoryTests.test_settings_change_blocks_dispatch
    test_mode_change_blocks_provider_dispatch = file_tests.LearningFileRepositoryTests.test_mode_change_blocks_provider_dispatch
    test_source_edit_and_undo_never_accept_late_writer = file_tests.LearningFileRepositoryTests.test_source_edit_and_undo_never_accept_late_writer
    test_deleted_block_is_not_resolvable_or_recreated_by_read = file_tests.LearningFileRepositoryTests.test_deleted_block_is_not_resolvable_or_recreated_by_read
    test_unknown_manual_retry_fences_previous_operation = file_tests.LearningFileRepositoryTests.test_unknown_manual_retry_fences_previous_operation
    test_checkpoint_read_rejects_stale_source_but_status_remains_readable = file_tests.LearningFileRepositoryTests.test_checkpoint_read_rejects_stale_source_but_status_remains_readable
    test_settings_change_blocks_result_commit_and_preserves_checkpoint = file_tests.LearningFileRepositoryTests.test_settings_change_blocks_result_commit_and_preserves_checkpoint
    test_level_change_reuses_translation_key_but_blocks_old_hints = file_tests.LearningFileRepositoryTests.test_level_change_reuses_translation_key_but_blocks_old_hints
    test_claim_does_not_accept_client_preference_revision_or_language = file_tests.LearningFileRepositoryTests.test_claim_does_not_accept_client_preference_revision_or_language
    test_same_command_id_across_components_is_a_conflict = file_tests.LearningFileRepositoryTests.test_same_command_id_across_components_is_a_conflict
    test_expired_unknown_response_can_be_reconciled_without_new_claim = file_tests.LearningFileRepositoryTests.test_expired_unknown_response_can_be_reconciled_without_new_claim
    test_action_fields_and_result_mismatch_rejected_without_writes = file_tests.LearningFileRepositoryTests.test_action_fields_and_result_mismatch_rejected_without_writes

    def snapshot(self):
        with psycopg.connect(self.scoped) as conn:
            return {table: conn.execute(sql.SQL('SELECT row_to_json(t) FROM {} t WHERE document_row_id=%s ORDER BY row_to_json(t)::text').format(sql.Identifier(table)),
                                        (self.row_id,)).fetchall() for table in TABLES}

    def test_missing_reads_and_constructor_do_not_write(self):
        before = self.snapshot()
        repo = repository(self.scoped, self.namespace)
        self.assertIsNone(repo.lookup_command('book', 'missing'))
        self.assertIsNone(repo.lookup_operation('book', 'missing'))
        self.assertEqual(repo.read_batch('book', (self.key,))[0].view.state, 'missing')
        self.assertEqual(repo.read_sentences('book', (self.sentences.reference,))[0].state, 'missing')
        self.assertEqual(before, self.snapshot())
        self.assertTrue(all(not rows for rows in before.values()))

    def test_source_uses_relational_parent_ids_and_effective_overlay(self):
        doc = self.docs.load_document('book')
        chapter = doc.chapters[0]; section = chapter.sections[0]; unit = section.task_units[0]
        self.assertEqual((self.block.source.chapter_id,self.block.source.section_id,self.block.source.task_unit_id),
                         (chapter.chapter_id, section.section_id, unit.unit_id))
        self.edit()
        self.assertEqual(self.repo.read_sources('book').blocks[0].effective_text, 'Changed content.')
        self.assertEqual(self.docs.load_document('book').chapters[0].sections[0].task_units[0].content, unit.content)

    def test_sentence_dedup_and_complete_readback_after_reconnect(self):
        doc = self.docs.load_document('book')
        self.repo.save_sentences('book', self.sentences)
        before = self.snapshot()
        self.repo.save_sentences('book', self.sentences)
        self.assertEqual(before, self.snapshot())
        complete = self.write(self.partial(), 'commit', 40, result=self.result())
        repo = repository(self.scoped, self.namespace)
        view = repo.read_batch('book', (self.key,))[0]
        self.assertEqual(view.view.saved, self.result())
        self.assertEqual(view.sentences, self.sentences)
        self.assertEqual(repo.lookup_operation('book', 'first'), complete.operation)
        self.assertEqual(repo.lookup_command('book', 'first').operation_id, 'first')
        self.assertEqual(self.docs.load_document('book'), doc)

    def test_same_command_replay_and_old_history_survive_new_claim(self):
        first = self.claim()
        self.assertTrue(self.claim().replay)
        with self.assertRaisesRegex(LearningContractError, 'command_conflict'):
            self.claim(replace(self.command, intent='regenerate'))
        failed = self.write(first.next_entry, 'fail', reason='worker_stopped', outcome_known=True)
        newer = self.claim(GenerationCommand('retry', self.binding, 'retry', 'first'), now_ms=30).next_entry
        self.assertGreater(newer.last_fence, failed.last_fence)
        self.assertEqual(self.repo.lookup_operation('book', 'first').state, 'failed')
        self.assertTrue(self.claim().replay)
        self.assertEqual(self.repo.lookup_command('book', 'first'), first.receipt)

    def test_admission_denied_receipt_without_entry_or_operation(self):
        class Deny:
            def evaluate(self, command): return AdmissionDecision(False, 'test_denied')
        repo = PostgresLearningRepository(self.scoped, self.repo.scope, admission=Deny())
        decision = repo.claim('book', self.command, sentences=self.sentences, plan=self.plan, now_ms=10, lease_until_ms=100)
        self.assertEqual(decision.outcome, 'denied')
        self.assertEqual(repo.lookup_command('book', 'first').admission_reason, 'test_denied')
        self.assertIsNone(repo.lookup_operation('book', 'first'))
        self.assertTrue(self.claim().replay)
        self.assertFalse(self.snapshot()['learning_cache_entries'])

    def processes(self, action, commands):
        ctx = multiprocessing.get_context('spawn')
        barrier, output = ctx.Barrier(len(commands)+1), ctx.Queue()
        processes = [ctx.Process(target=process_call, args=(self.scoped, self.namespace, action, command,
                     self.sentences, self.plan, barrier, output)) for command in commands]
        try:
            for process in processes: process.start()
            barrier.wait(timeout=20)
            results = [output.get(timeout=20) for _ in processes]
            for process in processes:
                process.join(timeout=15)
                self.assertEqual(process.exitcode, 0)
            return results
        finally:
            for process in processes:
                if process.is_alive(): process.terminate()
                process.join(timeout=5)
            output.close()

    def test_six_process_claims_same_key_have_one_winner(self):
        results = self.processes('claim', [replace(self.command, command_id=f'claim-{i}') for i in range(6)])
        self.assertCountEqual([r[0] for r in results], ['claimed']+['coalesced']*5)
        self.assertEqual(len({r[1] for r in results}), 1)
        self.assertEqual(len(self.snapshot()['learning_commands']), 6)
        self.assertEqual(len(self.snapshot()['learning_operations']), 1)

    def test_same_command_concurrency_is_durable_replay(self):
        results = self.processes('claim', [self.command]*3)
        self.assertEqual(sum(not r[2] for r in results), 1)
        self.assertEqual(len(self.snapshot()['learning_commands']), 1)

    def test_different_keys_commit_without_lost_updates(self):
        commands = [GenerationCommand(f'key-{i}', replace(self.binding, key=replace(self.key, policy_version=f'p-{i}'))) for i in range(3)]
        self.assertEqual([r[0] for r in self.processes('claim', commands)], ['claimed']*3)
        self.assertEqual(len(self.snapshot()['learning_cache_entries']), 3)
        self.assertEqual(len(self.snapshot()['learning_sentences']), 1)

    def test_four_process_dispatches_allow_only_one_sender(self):
        entry = self.claim().next_entry
        command = OperationWrite('first', entry.revision, entry.last_fence, 20, 'dispatch')
        self.assertCountEqual(self.processes('write', [command]*4), ['running']+['conflict']*3)

    @contextmanager
    def reject(self, table, *, deferred=False, condition='TRUE'):
        with psycopg.connect(self.scoped) as conn:
            conn.execute("CREATE OR REPLACE FUNCTION reject_learning_write() RETURNS trigger LANGUAGE plpgsql AS $$ "
                         "BEGIN RAISE EXCEPTION 'injected private payload' USING ERRCODE='23514'; END $$")
            timing = 'CREATE CONSTRAINT TRIGGER reject_learning AFTER' if deferred else 'CREATE TRIGGER reject_learning BEFORE'
            deferral = 'DEFERRABLE INITIALLY DEFERRED' if deferred else ''
            conn.execute(sql.SQL(f'{timing} INSERT OR UPDATE ON {{}} {deferral} FOR EACH ROW WHEN ({condition}) '
                                 'EXECUTE FUNCTION reject_learning_write()').format(sql.Identifier(table)))
        try: yield
        finally:
            with psycopg.connect(self.scoped) as conn:
                conn.execute(sql.SQL('DROP TRIGGER reject_learning ON {}').format(sql.Identifier(table)))

    def test_receipt_insert_failure_rolls_back_sentence_entry_operation(self):
        before = self.snapshot()
        with self.reject('learning_commands'), self.assertRaises(LearningStorageError): self.claim()
        self.assertEqual(before, self.snapshot())
        self.assertIsNone(self.repo.lookup_command('book', 'first'))
        self.assertEqual(self.claim().outcome, 'claimed')

    def test_checkpoint_insert_failure_rolls_back_marker_and_operation(self):
        entry = self.write(self.claim().next_entry, 'dispatch')
        before = self.snapshot()
        with self.reject('learning_checkpoints'), self.assertRaises(LearningStorageError):
            self.write(entry, 'checkpoint', 30, batch_index=0, batch=self.batch())
        self.assertEqual(before, self.snapshot())
        self.assertEqual(self.repo.read_checkpoints('book', 'first'), ())
        self.assertEqual(self.repo.lookup_operation('book', 'first').state, 'running')

    def test_failed_refresh_commit_preserves_previous_output_and_other_key(self):
        self.write(self.partial(), 'commit', 40, result=self.result())
        other = replace(self.key, policy_version='other')
        self.claim(GenerationCommand('other', replace(self.binding, key=other)))
        entry = self.claim(GenerationCommand('refresh', self.binding, 'regenerate'), now_ms=50).next_entry
        running = self.write(entry, 'dispatch', 60)
        partial = self.write(running, 'checkpoint', 70, batch_index=0, batch=self.batch())
        before = self.snapshot()
        with self.reject('learning_operations', deferred=True, condition='NEW.result_revision IS NOT NULL'), self.assertRaises(LearningStorageError) as error:
            self.write(partial, 'commit', 80, result=self.result(2))
        self.assertNotIsInstance(error.exception, LearningCommitUnknown)
        self.assertEqual(self.snapshot(), before)
        views = self.repo.read_batch('book', (self.key, other))
        self.assertEqual(views[0].view.saved, self.result())
        self.assertEqual(views[1].view.state, 'claimed')
        self.write(partial, 'fail', 80, reason='worker_stopped', outcome_known=True)
        self.assertEqual(self.repo.read_batch('book', (self.key,))[0].view.saved, self.result())

    @contextmanager
    def lost_ack(self):
        original = self.repo._connect
        class Lost:
            def __init__(self): self.connection = original()
            def __getattr__(self, name): return getattr(self.connection, name)
            def commit(self):
                self.connection.commit()
                raise psycopg.OperationalError('lost private acknowledgement')
        with patch.object(self.repo, '_connect', Lost): yield

    def test_lost_claim_dispatch_and_final_ack_reconcile_without_resend(self):
        with self.lost_ack(), self.assertRaises(LearningCommitUnknown): self.claim()
        replay = self.claim()
        self.assertTrue(replay.replay)
        entry = self.repo.read_batch('book', (self.key,))[0]
        op = self.repo.lookup_operation('book', 'first')
        command = OperationWrite('first', 1, op.fence, 20, 'dispatch')
        with self.lost_ack(), self.assertRaises(LearningCommitUnknown): self.repo.write_operation('book', command)
        self.assertEqual(self.repo.lookup_operation('book', 'first').state, 'running')
        with self.assertRaises(LearningContractError): self.repo.write_operation('book', command)
        partial = self.repo.write_operation('book', OperationWrite('first', 2, op.fence, 30, 'checkpoint', batch_index=0, batch=self.batch()))
        with self.lost_ack(), self.assertRaises(LearningCommitUnknown): self.write(partial, 'commit', 40, result=self.result())
        self.assertEqual(self.repo.read_batch('book', (self.key,))[0].view.saved, self.result())
        self.assertEqual(self.repo.lookup_operation('book', 'first').state, 'succeeded')

    def crash(self, action, command, after):
        ctx = multiprocessing.get_context('spawn')
        process = ctx.Process(target=crash_call, args=(self.scoped, self.namespace, action, command,
                              self.sentences, self.plan, after))
        process.start(); process.join(timeout=20)
        if process.is_alive():
            process.terminate(); process.join(timeout=5)
            self.fail('crash fixture stuck')
        self.assertEqual(process.exitcode, 72 if after else 71)

    def test_process_death_before_and_after_claim_commit(self):
        self.crash('claim', self.command, False)
        self.assertIsNone(self.repo.lookup_command('book', 'first'))
        self.assertTrue(all(not rows for rows in self.snapshot().values()))
        self.crash('claim', self.command, True)
        self.assertTrue(self.claim().replay)
        self.assertEqual(self.repo.lookup_operation('book', 'first').state, 'claimed')

    def test_process_death_before_and_after_dispatch_commit(self):
        entry = self.claim().next_entry
        command = OperationWrite('first', entry.revision, entry.last_fence, 20, 'dispatch')
        self.crash('write', command, False)
        self.assertEqual(self.repo.lookup_operation('book', 'first').state, 'claimed')
        self.crash('write', command, True)
        self.assertEqual(self.repo.lookup_operation('book', 'first').state, 'running')
        with self.assertRaises(LearningContractError): self.repo.write_operation('book', command)

    def test_process_death_before_and_after_result_commit(self):
        partial = self.partial()
        command = OperationWrite('first', partial.revision, partial.last_fence, 40, 'commit', result=self.result())
        self.crash('write', command, False)
        self.assertIsNone(self.repo.read_batch('book', (self.key,))[0].view.saved)
        self.crash('write', command, True)
        self.assertEqual(self.repo.read_batch('book', (self.key,))[0].view.saved, self.result())
        self.assertEqual(self.repo.lookup_operation('book', 'first').state, 'succeeded')

    def test_sql_source_and_preference_writers_wait_for_learning_commit(self):
        reached, release = Event(), Event()
        original = learning_rows.persist
        def paused(*args, **kwargs):
            original(*args, **kwargs)
            reached.set()
            if not release.wait(timeout=15): raise RuntimeError('fixture timed out')
        with ThreadPoolExecutor(max_workers=3) as pool:
            with patch.object(learning_rows, 'persist', paused):
                claim = pool.submit(self.claim)
                self.assertTrue(reached.wait(timeout=10))
                edit = pool.submit(self.edit)
                native = pool.submit(self.prefs.save_workspace, 'fr', expected_revision=1, mutation_id='native-change')
                try:
                    deadline = time.monotonic()+10
                    with psycopg.connect(self.scoped, autocommit=True) as conn:
                        while time.monotonic() < deadline:
                            waits = conn.execute("SELECT count(*) FROM pg_stat_activity WHERE wait_event_type='Lock' "
                                "AND (query LIKE 'SELECT id,metadata_payload FROM documents%%' "
                                "OR query LIKE 'SELECT scope_kind,%%')").fetchone()[0]
                            if waits >= 2: break
                            time.sleep(.01)
                        self.assertGreaterEqual(waits, 2)
                    self.assertFalse(edit.done()); self.assertFalse(native.done())
                finally: release.set()
                self.assertEqual(claim.result(timeout=10).outcome, 'claimed')
                edit.result(timeout=10); native.result(timeout=10)
        self.assertEqual(self.repo.read_batch('book', (self.key,))[0].view.state, 'stale')

    def test_other_namespace_and_document_do_not_share_history(self):
        self.claim()
        self.docs.save_reparsed_document(file_tests.fixture('other'), 'other')
        self.assertIsNone(self.repo.lookup_operation('other', 'first'))
        other = repository(self.scoped, uuid4().hex)
        self.assertIsNone(other.lookup_command('book', 'first'))
        with self.assertRaises(LearningContractError): self.repo.read_batch('other', (self.key,))

    def test_corrupt_checkpoint_or_missing_receipt_is_not_missing(self):
        self.partial()
        with psycopg.connect(self.scoped) as conn:
            saved = conn.execute('SELECT payload FROM learning_checkpoints WHERE document_row_id=%s', (self.row_id,)).fetchone()[0]
            corrupt = codec.encode(replace(self.batch(), rows=tuple(replace(r, translated_text='corrupt') for r in self.batch().rows)))
            conn.execute('UPDATE learning_checkpoints SET payload=%s WHERE document_row_id=%s', (Jsonb(corrupt), self.row_id))
        for read in (lambda: self.repo.read_batch('book',(self.key,)), lambda: self.repo.read_checkpoints('book','first')):
            with self.assertRaises(LearningStorageError): read()
        with psycopg.connect(self.scoped) as conn:
            conn.execute('UPDATE learning_checkpoints SET payload=%s WHERE document_row_id=%s', (Jsonb(saved), self.row_id))
            conn.execute('DELETE FROM learning_commands WHERE document_row_id=%s', (self.row_id,))
        with self.assertRaises(LearningStorageError): self.repo.lookup_operation('book', 'first')
        with self.assertRaises(LearningStorageError): self.claim()

    def test_target_batch_does_not_load_unrelated_history_and_queries_are_constant(self):
        self.write(self.partial(), 'commit', 40, result=self.result())
        # Add many invalid unrelated receipts: only requested keys/history may be read.
        with psycopg.connect(self.scoped) as conn:
            conn.execute("INSERT INTO learning_commands(document_row_id,command_id,payload) "
                         "SELECT %s,'unrelated-'||n,'{}'::jsonb FROM generate_series(1,200) n", (self.row_id,))
        queries = []
        real = self.repo._connect
        class Trace:
            def __init__(self): self.connection = real()
            def __getattr__(self, name): return getattr(self.connection, name)
            def execute(self, query, params=None):
                queries.append(str(query))
                return self.connection.execute(query, params)
        with patch.object(self.repo, '_connect', Trace):
            self.assertEqual(self.repo.read_batch('book',(self.key,))[0].view.state,'ready')
            one = len(queries); queries.clear()
            keys = (self.key, *(replace(self.key, policy_version=f'absent-{i}') for i in range(99)))
            views = self.repo.read_batch('book', keys)
            self.assertEqual([view.view.state for view in views], ['ready'] + ['missing'] * 99)
            self.assertEqual(len(queries),one)
            queries.clear()
            self.assertEqual(self.repo.read_checkpoints('book','first',limit=1), (self.batch(),))
            cp = [q for q in queries if 'FROM learning_checkpoints' in q]
            self.assertEqual(len(cp),1)
            self.assertIn('LIMIT %s',cp[0])

    def test_explicit_lease_renewal_persists(self):
        entry=self.write(self.claim().next_entry,'dispatch')
        before=self.snapshot()
        self.repo.lookup_operation('book','first')
        self.assertEqual(before,self.snapshot())
        self.write(entry,'renew',50,lease_until_ms=200)
        self.assertEqual(repository(self.scoped,self.namespace).lookup_operation('book','first').lease_until_ms,200)

    def test_migration_upgrade_reapply_downgrade_and_rollback(self):
        with self.isolated_schema() as dsn:
            with psycopg.connect(dsn) as conn:
                for path in sorted(MIGRATIONS.glob('00[1-4]*.sql')): conn.execute(path.read_text())
                conn.execute("INSERT INTO documents(namespace,document_name,current_structure_version) VALUES ('n','book',1)")
            repo=repository(dsn,'n')
            with self.assertRaises(LearningStorageError): repo.lookup_operation('book','missing')
            with self.assertRaises(psycopg.Error), psycopg.connect(dsn) as conn:
                conn.execute((MIGRATIONS/'005_learning_persistence.sql').read_text()); conn.execute('SELECT 1/0')
            with psycopg.connect(dsn) as conn:
                self.assertIsNone(conn.execute("SELECT to_regclass('learning_sentences')").fetchone()[0])
                conn.execute((MIGRATIONS/'005_learning_persistence.sql').read_text())
                conn.execute((MIGRATIONS/'005_learning_persistence.sql').read_text())
            self.assertIsNone(repo.lookup_operation('book','missing'))
            with self.assertRaises(psycopg.Error), psycopg.connect(dsn) as conn:
                conn.execute((MIGRATIONS/'rollback/005_learning_persistence.sql').read_text()); conn.execute('SELECT 1/0')
            self.assertIsNone(repo.lookup_operation('book','missing'))
            with psycopg.connect(dsn) as conn:
                conn.execute((MIGRATIONS/'rollback/005_learning_persistence.sql').read_text())
                self.assertEqual(conn.execute('SELECT count(*) FROM documents').fetchone()[0],1)
                self.assertIsNone(conn.execute("SELECT to_regclass('learning_sentences')").fetchone()[0])
            with self.assertRaises(LearningStorageError): repo.lookup_operation('book','missing')

    def test_transaction_logging_excludes_private_content(self):
        with self.assertLogs('db.postgres_learning_repository',level='INFO') as logs:
            self.claim()
        text=' '.join(str(r.__dict__) for r in logs.records)
        for value in (self.namespace,self.block.effective_text,self.scoped,self.key.cache_id): self.assertNotIn(value,text)
        self.assertTrue(any(r.__dict__.get('outcome')=='committed' for r in logs.records))
        self.assertTrue(any(r.__dict__.get('operation_count')==1 for r in logs.records))

    def test_file_and_sql_match_same_source_commands_and_results(self):
        import tempfile
        with tempfile.TemporaryDirectory() as root:
            docs, prefs, file_repo = file_tests.repositories(root, self.namespace)
            # Freeze only the file fixture's initial epoch allocator to the SQL
            # fixture's legacy epoch, so both adapters receive identical source identity.
            with patch('document_structure.structured_document_artifact_repository.uuid4') as allocator:
                allocator.return_value.hex = self.block.source.document_epoch
                docs.save_reparsed_document(self.docs.load_document('book'))
            file_tests.configure(prefs)
            self.assertEqual(file_repo.read_sources('book'), self.repo.read_sources('book'))
            results=[]
            for repo in (file_repo, self.repo):
                first=repo.claim('book',self.command,sentences=self.sentences,plan=self.plan,now_ms=10,lease_until_ms=100)
                coalesced=repo.claim('book',replace(self.command,command_id='also'),sentences=self.sentences,
                                     plan=self.plan,now_ms=11,lease_until_ms=100)
                running=repo.write_operation('book',OperationWrite('first',first.next_entry.revision,1,20,'dispatch'))
                partial=repo.write_operation('book',OperationWrite('first',running.revision,1,30,'checkpoint',batch_index=0,batch=self.batch()))
                complete=repo.write_operation('book',OperationWrite('first',partial.revision,1,40,'commit',result=self.result()))
                results.append((first,coalesced,complete,repo.read_checkpoints('book','first'),
                                repo.lookup_command('book','also'),repo.read_batch('book',(self.key,))))
            self.assertEqual(*results)

    def test_old_epoch_ordinary_save_cannot_initialize_missing_file_document(self):
        import tempfile
        from document_structure.reading_content import ReadingContentConflict
        with tempfile.TemporaryDirectory() as root:
            docs, _, _ = file_tests.repositories(root, self.namespace)
            with self.assertRaisesRegex(ReadingContentConflict, 'reading_document_missing_reload_required'):
                docs.save_document(self.docs.load_document('book'))
            self.assertFalse(docs.contains_document('book'))

    def test_paid_results_backup_downgrade_reupgrade_restore(self):
        with self.isolated_schema() as dsn:
            with psycopg.connect(dsn) as conn:
                for path in sorted(MIGRATIONS.glob('*.sql')): conn.execute(path.read_text())
            fixture=LearningPostgresTests('test_missing_reads_and_constructor_do_not_write')
            fixture.scoped=dsn; fixture.setUp()
            fixture.write(fixture.partial(),'commit',40,result=fixture.result())
            saved=fixture.repo.read_batch('book',(fixture.key,))
            backup=fixture.snapshot()
            with psycopg.connect(dsn) as conn:
                conn.execute((MIGRATIONS/'rollback/005_learning_persistence.sql').read_text())
                self.assertEqual(conn.execute('SELECT count(*) FROM documents').fetchone()[0],1)
                self.assertEqual(conn.execute('SELECT count(*) FROM reading_book_preferences').fetchone()[0],1)
                conn.execute((MIGRATIONS/'005_learning_persistence.sql').read_text())
                for table in TABLES:
                    for (row,) in backup[table]:
                        conn.execute(sql.SQL('INSERT INTO {} SELECT * FROM jsonb_populate_record(NULL::{},%s)').format(
                                     sql.Identifier(table),sql.Identifier(table)),(Jsonb(row),))
            self.assertEqual(fixture.repo.read_batch('book',(fixture.key,)),saved)
            self.assertTrue(fixture.claim().replay)

    def test_database_identity_version_fence_and_parent_constraints(self):
        self.claim()
        with psycopg.connect(self.scoped) as conn:
            op=conn.execute('SELECT payload FROM learning_operations WHERE document_row_id=%s',(self.row_id,)).fetchone()[0]
        cases=[
            ('UPDATE learning_sentences SET schema_version=9 WHERE document_row_id=%s',(self.row_id,)),
            ('UPDATE learning_cache_entries SET revision=-1 WHERE document_row_id=%s',(self.row_id,)),
            ('INSERT INTO learning_operations(document_row_id,operation_id,cache_id,fence,payload) VALUES (%s,%s,%s,1,%s)',
             (self.row_id,'other',self.key.cache_id,Jsonb(op))),
            ('INSERT INTO learning_checkpoints(document_row_id,operation_id,batch_index,payload) VALUES (%s,%s,0,%s)',
             (self.row_id,'missing',Jsonb(codec.encode(self.batch())))),
        ]
        before=self.snapshot()
        for query,params in cases:
            with self.subTest(query=query),self.assertRaises(psycopg.IntegrityError),psycopg.connect(self.scoped) as conn:
                conn.execute(query,params)
        self.assertEqual(before,self.snapshot())

    def test_read_snapshot_remains_consistent_during_source_edit(self):
        self.write(self.partial(),'commit',40,result=self.result())
        reached,release=Event(),Event()
        original=learning_rows.load_state
        def pause(*args,**kwargs):
            reached.set()
            if not release.wait(timeout=10): raise RuntimeError('fixture timeout')
            return original(*args,**kwargs)
        with ThreadPoolExecutor(max_workers=1) as pool:
            with patch.object(learning_rows,'load_state',pause):
                future=pool.submit(self.repo.read_batch,'book',(self.key,))
                try:
                    self.assertTrue(reached.wait(timeout=10))
                    self.edit()
                finally: release.set()
                self.assertEqual(future.result(timeout=10)[0].view.state,'ready')
        self.assertEqual(self.repo.read_batch('book',(self.key,))[0].view.state,'stale')

    def test_missing_document_never_recreated_by_learning_write(self):
        self.claim()
        self.docs.delete_document('book')
        with self.assertRaisesRegex(LearningContractError,'document_missing'): self.claim()
        self.assertIsNone(self.repo.lookup_operation('book','first'))
        self.docs.save_reparsed_document(file_tests.fixture(),'book')
        self.assertIsNone(self.repo.lookup_operation('book','first'))
        with self.assertRaisesRegex(LearningContractError,'source_changed'): self.claim()

    def test_transaction_helpers_require_caller_owned_active_transaction(self):
        with psycopg.connect(self.scoped,autocommit=True,row_factory=dict_row) as conn:
            with self.assertRaisesRegex(ValueError,'transaction_required'):
                self.prefs.lock_snapshot('book',connection=conn)
            with self.assertRaisesRegex(ValueError,'transaction_required'):
                self.docs.store.load_in_transaction('book',connection=conn)
        with psycopg.connect(self.scoped,row_factory=dict_row) as conn:
            conn.execute('BEGIN')
            before=conn.info.transaction_status
            self.prefs.lock_snapshot('book',connection=conn)
            self.docs.store.load_in_transaction('book',connection=conn)
            self.assertEqual(conn.info.transaction_status,before)
            conn.rollback()



if __name__=='__main__': unittest.main()
