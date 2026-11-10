"""LLRN-07: real disposable PostgreSQL migration, parity and transaction tests.

Requires DEEP_READER_TEST_POSTGRES_DSN. Creates/drops only random test schemas.
The DSN must point at a disposable test database, never the running library.
"""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace
import json
import multiprocessing
import os
from pathlib import Path
import tempfile
from threading import Barrier
import unittest
from unittest.mock import patch
from uuid import uuid4

import psycopg
from psycopg import sql
from psycopg.conninfo import make_conninfo
from psycopg.types.json import Jsonb

from db.postgres_reading_preferences_repository import PostgresReadingPreferencesRepository as Repository
from reading_preferences.file_repository import FileReadingPreferencesRepository
from reading_preferences.models import (
    BookSettings, PreferenceCommitUnknown, PreferenceConflict, PreferenceError,
    PreferenceStorageError, PreservedModeSettings, WorkspaceScope,
)
from reading_preferences.serialization import encode_settings
from scripts import test_reading_preferences as file_tests

MIGRATIONS = Path(__file__).resolve().parents[1] / "db/migrations"
UP = MIGRATIONS / "004_reading_preferences.sql"
DOWN = MIGRATIONS / "rollback/004_reading_preferences.sql"


def process_write(dsn, namespace, barrier, output, index, kind):
    repo = Repository(dsn, WorkspaceScope(namespace))
    barrier.wait(timeout=20)
    try:
        if kind == "workspace":
            value = repo.save_workspace(("en", "fr")[index], expected_revision=0,
                                        mutation_id=f"write-{index}")
        else:
            value = repo.save_book(f"book-{index}" if kind == "different" else "book",
                                  BookSettings(source_language_override="en"), expected_revision=0,
                                  expected_workspace_revision=0, resolved_source_language=None,
                                  mutation_id="same" if kind == "duplicate" else f"write-{index}")
        output.put(("saved", value.revision))
    except PreferenceConflict:
        output.put(("conflict", None))


def crash_write(dsn, namespace, after):
    repo = Repository(dsn, WorkspaceScope(namespace))
    original = repo._connect

    class CrashConnection:
        def __init__(self):
            self.connection = original()

        def __getattr__(self, name):
            return getattr(self.connection, name)

        def commit(self):
            if after:
                self.connection.commit()
            os._exit(72 if after else 71)

    repo._connect = CrashConnection
    repo.save_workspace("fr", expected_revision=1, mutation_id="crash")


class PostgresPreferencesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dsn = os.environ["DEEP_READER_TEST_POSTGRES_DSN"]
        cls.schema = "llrn07_" + uuid4().hex
        with psycopg.connect(cls.dsn, autocommit=True) as conn:
            conn.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(cls.schema)))
        cls.addClassCleanup(cls.drop_schema)
        cls.scoped = make_conninfo(cls.dsn, options=f"-c search_path={cls.schema}")
        with psycopg.connect(cls.scoped) as conn:
            for path in sorted(MIGRATIONS.glob("*.sql")):
                conn.execute(path.read_text())

    @classmethod
    def drop_schema(cls):
        with psycopg.connect(cls.dsn, autocommit=True) as conn:
            conn.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(cls.schema)))

    def setUp(self):
        self.repo = Repository(self.scoped, WorkspaceScope(uuid4().hex))

    save = file_tests.ReadingPreferencesTests.save
    native = file_tests.ReadingPreferencesTests.native
    test_normal_mode_needs_no_learning_fields = file_tests.ReadingPreferencesTests.test_normal_mode_needs_no_learning_fields
    test_switching_mode_retains_saved_learning_configuration = file_tests.ReadingPreferencesTests.test_switching_mode_retains_saved_learning_configuration
    test_same_language_and_wrong_proficiency_binding_rejected = file_tests.ReadingPreferencesTests.test_same_language_and_wrong_proficiency_binding_rejected
    test_whole_settings_write_is_atomic_on_validation_failure = file_tests.ReadingPreferencesTests.test_whole_settings_write_is_atomic_on_validation_failure
    test_same_value_preserves_revision_and_replay = file_tests.ReadingPreferencesTests.test_same_value_preserves_revision_and_replay
    test_cas_and_mutation_id_reuse = file_tests.ReadingPreferencesTests.test_cas_and_mutation_id_reuse
    test_native_revision_checked_inside_book_transaction = file_tests.ReadingPreferencesTests.test_native_revision_checked_inside_book_transaction
    test_native_change_does_not_rewrite_book_preferences = file_tests.ReadingPreferencesTests.test_native_change_does_not_rewrite_book_preferences
    test_delete_tombstone_prevents_stale_resurrection_and_preserves_workspace = file_tests.ReadingPreferencesTests.test_delete_tombstone_prevents_stale_resurrection_and_preserves_workspace
    test_delete_missing_book_fences_inflight_initial_save = file_tests.ReadingPreferencesTests.test_delete_missing_book_fences_inflight_initial_save
    test_rename_is_atomic_and_replayable_with_old_identity_fenced = file_tests.ReadingPreferencesTests.test_rename_is_atomic_and_replayable_with_old_identity_fenced

    def snapshot(self):
        with psycopg.connect(self.scoped) as conn:
            workspace = conn.execute("SELECT * FROM reading_workspace_preferences WHERE namespace=%s",
                                     (self.repo.scope.namespace,)).fetchall()
            books = conn.execute("SELECT * FROM reading_book_preferences WHERE namespace=%s ORDER BY document_id",
                                 (self.repo.scope.namespace,)).fetchall()
            return workspace, books

    def test_read_defaults_no_rows_or_physical_updates(self):
        self.assertEqual(self.repo.read_workspace().revision, 0)
        self.assertIsNone(self.repo.read_workspace().native_language)
        self.assertEqual(self.repo.read_book("中文書").settings, BookSettings())
        self.assertEqual(self.snapshot(), ([], []))
        self.native()
        self.save()
        with psycopg.connect(self.scoped) as conn:
            before = conn.execute("SELECT xmin::text,ctid::text FROM reading_book_preferences WHERE namespace=%s",
                                  (self.repo.scope.namespace,)).fetchall()
        for _ in range(3):
            self.repo.read_workspace()
            self.repo.read_book("book")
        with psycopg.connect(self.scoped) as conn:
            after = conn.execute("SELECT xmin::text,ctid::text FROM reading_book_preferences WHERE namespace=%s",
                                 (self.repo.scope.namespace,)).fetchall()
        self.assertEqual(before, after)

    def test_reconnect_reads_same_typed_settings(self):
        self.native()
        saved = self.save()
        fresh = Repository(self.scoped, self.repo.scope)
        self.assertEqual(fresh.read_book("book"), saved)
        self.assertEqual(fresh.read_workspace(), self.repo.read_workspace())

    def test_registry_validation_rolls_back_new_workspace(self):
        for settings, source in ((BookSettings("language_learning"), "en"),
                                 (BookSettings("language_learning", modes=(file_tests.learning(),)), "en")):
            with self.assertRaises(PreferenceError):
                self.save(settings, source=source)
            self.assertEqual(self.snapshot(), ([], []))
        self.native()
        before = self.snapshot()
        for entry in (replace(file_tests.learning(), level_id="N5"),
                      replace(file_tests.learning(), version="9")):
            with self.assertRaises(ValueError):
                self.save(BookSettings("language_learning", modes=(entry,)))
        for code in ("unknown", "EN", "en-US", True, {}):
            with self.assertRaises(PreferenceError):
                self.native(code)
        self.assertEqual(self.snapshot(), before)

    def test_namespace_and_exact_identity_isolation(self):
        self.native()
        saved = self.save(doc="Book.pdf")
        other = Repository(self.scoped, WorkspaceScope(uuid4().hex))
        self.assertEqual(other.read_book("Book.pdf").revision, 0)
        other.save_workspace("ja", expected_revision=0, mutation_id="native")
        self.assertEqual(self.repo.read_book("Book").revision, 0)
        self.assertEqual(self.repo.read_book("book.pdf").revision, 0)
        self.assertEqual(self.repo.read_book("Book.pdf"), saved)
        self.assertEqual(self.repo.read_workspace().native_language, "zh")
        for name in ("é", "e\u0301", "x'; DROP TABLE documents;--"):
            self.save(BookSettings(), doc=name, command=uuid4().hex)
        with psycopg.connect(self.scoped) as conn:
            self.assertEqual(conn.execute("SELECT count(*) FROM reading_book_preferences WHERE namespace=%s",
                                          (self.repo.scope.namespace,)).fetchone()[0], 4)

    def test_future_settings_preserved_and_bad_payload_is_not_missing(self):
        self.save(BookSettings())
        settings = {"active_mode": "future", "source_language_override": None,
                    "modes": {"future": {"version": "7", "payload": {"nested": [1, "值"]}}}}
        with psycopg.connect(self.scoped) as conn:
            conn.execute("UPDATE reading_book_preferences SET settings=%s WHERE namespace=%s",
                         (Jsonb(settings), self.repo.scope.namespace))
        before = self.snapshot()
        saved = self.repo.read_book("book")
        self.assertIsInstance(saved.settings.modes[0], PreservedModeSettings)
        self.assertEqual(before, self.snapshot())
        with self.assertRaises(PreferenceError):
            self.save(saved.settings, command="activate")
        self.save(replace(saved.settings, active_mode="normal"), command="normal")
        for payload in ({}, {"mutation_id": "x", "signature": "bad"}):
            with psycopg.connect(self.scoped) as conn:
                conn.execute("UPDATE reading_book_preferences SET receipt=%s WHERE namespace=%s",
                             (Jsonb(payload), self.repo.scope.namespace))
            before = self.snapshot()
            with self.assertRaises(PreferenceStorageError):
                self.repo.read_book("book")
            with self.assertRaises(PreferenceStorageError):
                self.repo.delete_book("book", expected_revision=1, mutation_id="repair")
            self.assertEqual(before, self.snapshot())

    @contextmanager
    def fail_book_write(self, *, deferred=False):
        with psycopg.connect(self.scoped) as conn:
            conn.execute("CREATE OR REPLACE FUNCTION reject_preference_write() RETURNS trigger "
                         "LANGUAGE plpgsql AS $$ BEGIN RAISE EXCEPTION 'private payload failure' "
                         "USING ERRCODE='23514'; END $$")
            if deferred:
                conn.execute("CREATE CONSTRAINT TRIGGER reject_preferences AFTER INSERT OR UPDATE "
                             "ON reading_book_preferences DEFERRABLE INITIALLY DEFERRED "
                             "FOR EACH ROW EXECUTE FUNCTION reject_preference_write()")
            else:
                conn.execute("CREATE TRIGGER reject_preferences BEFORE INSERT OR UPDATE "
                             "ON reading_book_preferences FOR EACH ROW "
                             "WHEN (NEW.document_id = 'destination') EXECUTE FUNCTION reject_preference_write()")
        try:
            yield
        finally:
            with psycopg.connect(self.scoped) as conn:
                conn.execute("DROP TRIGGER reject_preferences ON reading_book_preferences")

    def test_rename_second_write_failure_rolls_back_both_rows_and_receipts(self):
        self.save(BookSettings(source_language_override="en"))
        before = self.snapshot()
        with self.fail_book_write(), self.assertRaises(PreferenceStorageError) as error:
            self.repo.rename_book("book", "destination", expected_revision=1,
                                  expected_destination_revision=0, mutation_id="rename")
        self.assertNotIsInstance(error.exception, PreferenceCommitUnknown)
        self.assertEqual(before, self.snapshot())
        self.assertEqual(self.repo.rename_book("book", "destination", expected_revision=1,
                                              expected_destination_revision=0, mutation_id="rename").revision, 1)

    def test_commit_constraint_failure_is_known_rollback_including_workspace(self):
        with self.fail_book_write(deferred=True), self.assertRaises(PreferenceStorageError) as error:
            self.save(BookSettings())
        self.assertNotIsInstance(error.exception, PreferenceCommitUnknown)
        self.assertNotIn("private", str(error.exception))
        self.assertEqual(self.snapshot(), ([], []))

    def test_lost_commit_ack_reconciles_without_duplicate_revision(self):
        self.native()
        real_connect = self.repo._connect

        class LostAck:
            def __init__(self):
                self.connection = real_connect()

            def __getattr__(self, name):
                return getattr(self.connection, name)

            def commit(self):
                self.connection.commit()
                raise psycopg.OperationalError("lost private acknowledgement")

        with patch.object(self.repo, "_connect", LostAck), self.assertRaises(PreferenceCommitUnknown):
            self.repo.save_workspace("fr", expected_revision=1, mutation_id="unknown")
        fresh = Repository(self.scoped, self.repo.scope)
        self.assertEqual(fresh.read_workspace().native_language, "fr")
        self.assertEqual(fresh.save_workspace("fr", expected_revision=1, mutation_id="unknown").revision, 2)

    def process_results(self, kind):
        ctx = multiprocessing.get_context("spawn")
        barrier, output = ctx.Barrier(3), ctx.Queue()
        processes = [ctx.Process(target=process_write,
                                args=(self.scoped, self.repo.scope.namespace, barrier, output, i, kind))
                     for i in range(2)]
        try:
            for process in processes:
                process.start()
            barrier.wait(timeout=20)
            results = [output.get(timeout=20) for _ in processes]
            for process in processes:
                process.join(timeout=15)
                self.assertEqual(process.exitcode, 0)
            return results
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate()
                process.join(timeout=5)
            output.close()

    def test_concurrent_first_workspace_writers_cas(self):
        self.assertCountEqual(self.process_results("workspace"), [("saved", 1), ("conflict", None)])

    def test_concurrent_first_book_writers_cas(self):
        self.assertCountEqual(self.process_results("book"), [("saved", 1), ("conflict", None)])

    def test_concurrent_duplicate_command_replays(self):
        self.assertEqual(self.process_results("duplicate"), [("saved", 1), ("saved", 1)])
        self.assertEqual(self.repo.read_book("book").revision, 1)

    def test_concurrent_different_books_preserve_both(self):
        self.assertEqual(self.process_results("different"), [("saved", 1), ("saved", 1)])
        self.assertEqual(len(self.snapshot()[1]), 2)

    def test_workspace_revision_race_blocks_late_book_write(self):
        self.native()
        # Hold the real workspace row lock while another connection tries a book
        # write. Observe its PostgreSQL lock wait before committing the new native.
        with psycopg.connect(self.scoped) as conn, ThreadPoolExecutor(max_workers=1) as pool:
            conn.execute("UPDATE reading_workspace_preferences SET revision=2,native_language='fr' "
                         "WHERE namespace=%s", (self.repo.scope.namespace,))
            future = pool.submit(self.repo.save_book, "book", BookSettings(), expected_revision=0,
                                 expected_workspace_revision=1, resolved_source_language=None,
                                 mutation_id="late")
            try:
                self.wait_for_lock()
            finally:
                conn.commit()
            with self.assertRaises(PreferenceConflict):
                future.result(timeout=10)
        self.assertEqual(self.repo.read_book("book").revision, 0)

    def wait_for_lock(self):
        import time
        deadline = time.monotonic() + 10
        with psycopg.connect(self.scoped, autocommit=True) as probe:
            while time.monotonic() < deadline:
                if probe.execute("SELECT EXISTS(SELECT 1 FROM pg_stat_activity WHERE "
                                 "pid <> pg_backend_pid() AND wait_event_type='Lock' AND "
                                 "query LIKE 'INSERT INTO reading_workspace_preferences%')").fetchone()[0]:
                    return
                time.sleep(.01)
        self.fail("preference writer did not wait for workspace lock")

    def test_opposing_renames_conflict_without_deadlock(self):
        for name in ("a", "b"):
            self.save(BookSettings(source_language_override="en"), doc=name, command=name)
        barrier = Barrier(2)

        def rename(source, destination):
            barrier.wait(timeout=10)
            try:
                self.repo.rename_book(source, destination, expected_revision=1,
                                      expected_destination_revision=1, mutation_id="rename-" + source)
            except PreferenceConflict as error:
                return str(error)

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(rename, "a", "b"), pool.submit(rename, "b", "a")]
            self.assertEqual([f.result(timeout=10) for f in futures],
                             ["preference_destination_exists"] * 2)
        self.assertFalse(self.repo.read_book("a").deleted)
        self.assertFalse(self.repo.read_book("b").deleted)

    def test_process_exit_before_and_after_commit(self):
        self.native()
        ctx = multiprocessing.get_context("spawn")
        for after in (False, True):
            with self.subTest(after=after):
                process = ctx.Process(target=crash_write, args=(self.scoped, self.repo.scope.namespace, after))
                process.start()
                process.join(timeout=15)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=5)
                    self.fail("crash fixture did not terminate")
                self.assertEqual(process.exitcode, 72 if after else 71)
                self.assertEqual(self.repo.read_workspace().native_language, "fr" if after else "zh")
        self.assertEqual(self.repo.save_workspace("fr", expected_revision=1, mutation_id="crash").revision, 2)

    def test_file_sql_parity_for_receipts_and_tombstones(self):
        with tempfile.TemporaryDirectory() as root:
            file_repo = FileReadingPreferencesRepository(Path(root), self.repo.scope)
            actions = [
                ("save_workspace", ("zh",), dict(expected_revision=0, mutation_id="native")),
                ("save_book", ("book", BookSettings("language_learning", modes=(file_tests.learning(),))),
                 dict(expected_revision=0, expected_workspace_revision=1, resolved_source_language="en", mutation_id="save")),
                ("rename_book", ("book", "新書"), dict(expected_revision=1, expected_destination_revision=0, mutation_id="rename")),
                ("rename_book", ("book", "新書"), dict(expected_revision=1, expected_destination_revision=0, mutation_id="rename")),
                ("delete_book", ("新書",), dict(expected_revision=1, mutation_id="delete")),
                ("delete_book", ("新書",), dict(expected_revision=1, mutation_id="delete")),
                ("save_book", ("新書", BookSettings()), dict(expected_revision=2, expected_workspace_revision=1,
                                                           resolved_source_language=None, mutation_id="revive")),
            ]
            for method, args, kwargs in actions:
                self.assertEqual(getattr(file_repo, method)(*args, **kwargs), getattr(self.repo, method)(*args, **kwargs))
                with psycopg.connect(self.scoped) as conn:
                    sql_receipts = dict(conn.execute("SELECT document_id,receipt FROM reading_book_preferences "
                                                    "WHERE namespace=%s", (self.repo.scope.namespace,)).fetchall())
                payload = json.loads(file_repo._path.read_text())
                self.assertEqual(sql_receipts, {key: value["receipt"] for key, value in payload["books"].items()})
                for name in ("book", "新書"):
                    self.assertEqual(file_repo.read_book(name), self.repo.read_book(name))

    def test_only_target_book_loaded_with_bounded_queries(self):
        self.native()
        saved = self.save()
        # Unrelated malformed rows must not require loading a workspace-wide map.
        with psycopg.connect(self.scoped) as conn:
            conn.execute("INSERT INTO reading_book_preferences "
                         "(namespace,document_id,revision,settings,deleted) "
                         "SELECT %s,'unrelated-'||n,0,'{}'::jsonb,false FROM generate_series(1,200) n",
                         (self.repo.scope.namespace,))
        queries = []
        real_connect = self.repo._connect

        class TracedConnection:
            def __init__(self):
                self.connection = real_connect()

            def __getattr__(self, name):
                return getattr(self.connection, name)

            def execute(self, query, params=None):
                queries.append(query)
                return self.connection.execute(query, params)

        with patch.object(self.repo, "_connect", TracedConnection):
            self.assertEqual(self.repo.read_book("book"), saved)
            self.save(replace(saved.settings, active_mode="normal"), command="normal")
        selects = [q for q in queries if q.startswith("SELECT")]
        self.assertEqual(len(selects), 5)  # read, helper book/workspace reads, write's two reads
        self.assertTrue(all("WHERE namespace = %s" in q for q in selects))
        self.assertTrue(all("document_id = %s" in q for q in selects if "reading_book_preferences" in q))

    def test_unknown_learning_version_retained_but_activation_rejected(self):
        self.native()
        self.save()
        with psycopg.connect(self.scoped) as conn:
            conn.execute("UPDATE reading_book_preferences SET settings=jsonb_set(settings, "
                         "'{modes,language_learning,version}', '\"9\"'::jsonb) WHERE namespace=%s",
                         (self.repo.scope.namespace,))
        future = self.repo.read_book("book")
        self.assertIsInstance(future.settings.modes[0], PreservedModeSettings)
        with self.assertRaises(PreferenceError):
            self.save(future.settings, command="future")
        saved = self.save(replace(future.settings, active_mode="normal"), command="normal")
        self.assertEqual(saved.settings.modes, future.settings.modes)

    def test_concurrent_rename_and_delete_cannot_resurrect_source(self):
        self.save(BookSettings(source_language_override="en"))
        barrier = Barrier(2)

        def change(rename):
            barrier.wait(timeout=10)
            try:
                if rename:
                    self.repo.rename_book("book", "new", expected_revision=1,
                                          expected_destination_revision=0, mutation_id="rename")
                else:
                    self.repo.delete_book("book", expected_revision=1, mutation_id="delete")
                return "saved"
            except PreferenceConflict:
                return "conflict"

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(change, value) for value in (False, True)]
            self.assertCountEqual([f.result(timeout=10) for f in futures], ["saved", "conflict"])
        self.assertTrue(self.repo.read_book("book").deleted)
        self.assertEqual(self.repo.read_book("book").revision, 2)

    def test_rename_into_tombstone_and_missing_source_matches_file(self):
        with tempfile.TemporaryDirectory() as root:
            file_repo = FileReadingPreferencesRepository(Path(root), self.repo.scope)
            for repo in (file_repo, self.repo):
                repo.delete_book("destination", expected_revision=0, mutation_id="delete")
                result = repo.rename_book("missing", "destination", expected_revision=0,
                                          expected_destination_revision=1, mutation_id="rename")
                self.assertEqual(result.revision, 2)
                self.assertFalse(result.deleted)
                self.assertTrue(repo.read_book("missing").deleted)
            self.assertEqual(file_repo.read_book("destination"), self.repo.read_book("destination"))

    def test_logs_do_not_expose_namespace_document_language_or_dsn(self):
        with self.assertLogs("db.postgres_reading_preferences_repository", level="INFO") as logs:
            self.native()
            self.save(BookSettings(), doc="private-book")
        output = " ".join(logs.output)
        self.assertIn("operation=save_book outcome=committed", output)
        self.assertIn("duration_ms=", output)
        for private in (self.repo.scope.namespace, "private-book", self.scoped):
            self.assertNotIn(private, output)

    @contextmanager
    def isolated_schema(self):
        name = "llrn07_migration_" + uuid4().hex
        with psycopg.connect(self.dsn, autocommit=True) as conn:
            conn.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(name)))
        scoped = make_conninfo(self.dsn, options=f"-c search_path={name}")
        try:
            yield scoped
        finally:
            with psycopg.connect(self.dsn, autocommit=True) as conn:
                conn.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(name)))

    def test_missing_migration_is_explicit_error_and_reads_never_create_tables(self):
        with self.isolated_schema() as dsn:
            repo = Repository(dsn, self.repo.scope)
            for read in (repo.read_workspace, lambda: repo.read_book("book")):
                with self.assertRaises(PreferenceStorageError):
                    read()
            with psycopg.connect(dsn) as conn:
                self.assertIsNone(conn.execute("SELECT to_regclass('reading_workspace_preferences')").fetchone()[0])
                self.assertIsNone(conn.execute("SELECT to_regclass('reading_book_preferences')").fetchone()[0])

    def test_upgrade_reapply_backup_downgrade_restore_preserves_existing_documents(self):
        with self.isolated_schema() as dsn:
            with psycopg.connect(dsn) as conn:
                for name in ("001_phase_1_core_hierarchy.sql", "002_ocr_runs.sql", "003_interaction_version_guards.sql"):
                    conn.execute((MIGRATIONS / name).read_text())
                conn.execute("INSERT INTO documents(namespace,document_name,current_structure_version,metadata_payload) "
                             "VALUES ('original','book',1,'{\"keep\":true}')")
                before = conn.execute("SELECT row_to_json(documents) FROM documents").fetchall()
                conn.execute(UP.read_text())
            repo = Repository(dsn, self.repo.scope)
            repo.save_workspace("zh", expected_revision=0, mutation_id="native")
            saved = repo.save_book("book", BookSettings(source_language_override="en"), expected_revision=0,
                                   expected_workspace_revision=1, resolved_source_language=None, mutation_id="save")
            with psycopg.connect(dsn) as conn:
                backup_workspace = conn.execute("SELECT * FROM reading_workspace_preferences").fetchall()
                backup_books = conn.execute("SELECT * FROM reading_book_preferences").fetchall()
                conn.execute(UP.read_text())
            self.assertEqual(repo.read_book("book"), saved)
            with psycopg.connect(dsn) as conn:
                conn.execute(DOWN.read_text())
                self.assertIsNone(conn.execute("SELECT to_regclass('reading_book_preferences')").fetchone()[0])
                self.assertEqual(conn.execute("SELECT row_to_json(documents) FROM documents").fetchall(), before)
            with self.assertRaises(PreferenceStorageError):
                repo.read_book("book")
            with psycopg.connect(dsn) as conn:
                conn.execute(UP.read_text())
                for row in backup_workspace:
                    conn.execute("INSERT INTO reading_workspace_preferences VALUES (%s,%s,%s,%s,%s,%s)",
                                 (*row[:-1], Jsonb(row[-1])))
                for row in backup_books:
                    conn.execute("INSERT INTO reading_book_preferences VALUES (%s,%s,%s,%s,%s,%s,%s)",
                                 (*row[:4], Jsonb(row[4]), row[5], Jsonb(row[6])))
            self.assertEqual(repo.read_book("book"), saved)

    def test_failed_upgrade_and_downgrade_are_transactional(self):
        with self.isolated_schema() as dsn:
            with self.assertRaises(psycopg.Error), psycopg.connect(dsn) as conn:
                conn.execute(UP.read_text())
                conn.execute("SELECT 1/0")
            with psycopg.connect(dsn) as conn:
                self.assertIsNone(conn.execute("SELECT to_regclass('reading_workspace_preferences')").fetchone()[0])
                conn.execute(UP.read_text())
            repo = Repository(dsn, self.repo.scope)
            saved = repo.save_workspace("zh", expected_revision=0, mutation_id="native")
            with self.assertRaises(psycopg.Error), psycopg.connect(dsn) as conn:
                conn.execute(DOWN.read_text())
                conn.execute("SELECT 1/0")
            self.assertEqual(repo.read_workspace(), saved)

    def test_database_constraints(self):
        self.native()
        cases = [
            ("UPDATE reading_workspace_preferences SET revision=-1 WHERE namespace=%s", (self.repo.scope.namespace,)),
            ("UPDATE reading_workspace_preferences SET scope_kind='user' WHERE namespace=%s", (self.repo.scope.namespace,)),
            ("UPDATE reading_workspace_preferences SET schema_version=2 WHERE namespace=%s", (self.repo.scope.namespace,)),
            ("INSERT INTO reading_workspace_preferences(namespace) VALUES (%s)", (self.repo.scope.namespace,)),
            ("INSERT INTO reading_book_preferences VALUES (%s,'book',1,0,%s,false,NULL)",
             ("missing-parent", Jsonb(encode_settings(BookSettings())))),
            ("INSERT INTO reading_book_preferences VALUES (%s,'book',1,1,%s,true,NULL)",
             (self.repo.scope.namespace, Jsonb(encode_settings(BookSettings(source_language_override="en"))))),
        ]
        before = self.snapshot()
        for query, args in cases:
            with self.subTest(query=query), self.assertRaises(psycopg.IntegrityError), psycopg.connect(self.scoped) as conn:
                conn.execute(query, args)
        self.assertEqual(before, self.snapshot())


if __name__ == "__main__":
    unittest.main()
