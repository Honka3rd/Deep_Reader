"""LLRN-09 conformance: file and disposable PostgreSQL, no provider calls.

SQL cases require DEEP_READER_TEST_POSTGRES_DSN pointing at a disposable DB.
"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, replace
import json
from pathlib import Path
import tempfile
from threading import Barrier
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from uuid import uuid4

from language.language_profile_registry import LanguageProfileRegistry
from language.language_proficiency_registry import LanguageProficiencyRegistry
from reading_preferences.effective import PreferenceValidationError
from reading_preferences.file_repository import FileReadingPreferencesRepository
from reading_preferences.models import (
    BookSettings, PreferenceCommitUnknown, PreferenceConflict, PreferenceError,
    PreferenceStorageError, PreservedModeSettings, WorkspaceScope,
)
from reading_preferences.profile_source import ExistingProfileLanguageReader
from reading_preferences.service import ReadingPreferencesService
from scripts import test_reading_preferences as foundation


def profile(code="en"):
    return SimpleNamespace(document_language_code=code)


def settings(code="en", level="B1", override=None):
    return BookSettings("language_learning", override, (foundation.learning(code, level),))


class ServiceCases:
    def init_service(self):
        self.load = Mock(return_value=profile())
        self.service = ReadingPreferencesService(self.repo, ExistingProfileLanguageReader(self.load))

    def native(self, code="zh"):
        return self.service.save_workspace(code, expected_revision=self.repo.read_workspace().revision,
                                           mutation_id=uuid4().hex)

    def save(self, value=None, **kw):
        return self.service.save_book("book", value or settings(),
            expected_revision=kw.get("rev", self.repo.read_book("book").revision),
            expected_workspace_revision=kw.get("workspace_rev", self.repo.read_workspace().revision),
            mutation_id=kw.get("command", uuid4().hex))

    def codes(self, result):
        mode = next(m for m in result.modes if m.mode_id == "language_learning")
        return {r.code for r in mode.reasons}

    def test_pure_defaults_catalog_and_immutable_projection(self):
        before = self.snapshot()
        self.assertEqual([m.mode_id for m in self.service.catalog()], ["normal", "language_learning"])
        self.load.assert_not_called()
        result = self.service.read_book("book")
        self.assertEqual(result.effective_mode, "normal")
        self.assertEqual(self.codes(result), {"native_language_required", "proficiency_required"})
        self.assertEqual(before, self.snapshot())
        with self.assertRaises(FrozenInstanceError):
            result.book.revision = 1

    def test_same_language_rejected_atomically_with_readable_reason(self):
        self.native()
        self.load.return_value = profile("zh")
        before = self.snapshot()
        with self.assertRaises(PreferenceValidationError) as caught:
            self.save(settings("zh", "1"))
        self.assertIn("same_language", {r.code for r in caught.exception.reasons})
        self.assertTrue(all(r.message and r.field for r in caught.exception.reasons))
        self.assertEqual(before, self.snapshot())
        self.assertIn("same_language", self.codes(self.service.read_book("book")))

    def test_english_book_resolves_profile_and_workspace(self):
        self.native()
        result = self.save()
        self.assertEqual(result.effective_mode, "language_learning")
        self.assertEqual((result.source.code, result.source.origin), ("en", "profile"))
        self.assertEqual(result.workspace.native_language, "zh")
        self.assertEqual(result.proficiency.level.level_id, "B1")
        self.load.assert_called_once_with(self.repo.scope, "book")
        self.assertEqual(result, self.service.read_book("book"))

    def test_profile_missing_corrupt_unknown_and_unavailable_are_distinct(self):
        self.native()
        for effect, value, expected in (
            (None, None, "profile_missing"),
            (FileNotFoundError(), None, "profile_missing"),
            (ValueError("private data"), None, "profile_invalid"),
            (PermissionError("private path"), None, "profile_unavailable"),
            (None, profile("unknown"), "source_language_required"),
            (None, profile("xx"), "source_language_invalid"),
            (None, profile("en-US"), "source_language_invalid"),
        ):
            with self.subTest(expected=expected, value=value):
                self.load.side_effect, self.load.return_value = effect, value
                before = self.snapshot()
                result = self.service.read_book("book")
                self.assertIn(expected, self.codes(result))
                self.assertEqual(result.effective_mode, "normal")
                with self.assertRaisesRegex(PreferenceValidationError, expected):
                    self.save()
                self.assertEqual(before, self.snapshot())
                self.assertNotIn("private", repr(result))

    def test_unexpected_loader_failure_propagates(self):
        self.load.side_effect = RuntimeError("database failed")
        before = self.snapshot()
        with self.assertRaisesRegex(RuntimeError, "database failed"):
            self.service.read_book("book")
        self.assertEqual(before, self.snapshot())

    def test_override_bypasses_unneeded_profile_reads(self):
        self.native()
        for failure in (FileNotFoundError(), ValueError(), OSError(), RuntimeError()):
            self.load.side_effect = failure
            result = self.save(settings(override="en"))
            self.assertEqual(result.source.origin, "override")
            self.assertTrue(result.active.eligible)
            self.assertEqual(result, self.service.read_book("book"))
        self.load.assert_not_called()

    def test_wrong_profile_correction_and_explicit_override_clear(self):
        self.native()
        self.load.return_value = profile("zh")
        result = self.save(settings(override="en"))
        before = self.snapshot()
        with self.assertRaises(PreferenceValidationError):
            self.save(replace(result.book.settings, source_language_override=None))
        self.assertEqual(before, self.snapshot())
        normal = self.save(replace(result.book.settings, active_mode="normal", source_language_override=None))
        self.assertEqual((normal.source.code, normal.effective_mode), ("zh", "normal"))

    def test_source_change_requires_matching_proficiency(self):
        self.native()
        saved = self.save()
        before = self.snapshot()
        with self.assertRaisesRegex(PreferenceValidationError, "proficiency_language_mismatch"):
            self.save(replace(saved.book.settings, source_language_override="ja"))
        self.assertEqual(before, self.snapshot())
        normal = self.save(replace(saved.book.settings, active_mode="normal", source_language_override="ja"))
        self.assertIsNone(normal.proficiency)
        self.assertEqual(normal.proficiency_scheme.scheme_id, "jlpt_reading")
        self.assertIn("proficiency_language_mismatch", self.codes(normal))
        new = self.save(settings("ja", "N4", override="ja"))
        self.assertEqual(new.proficiency.level.level_id, "N4")

    def test_profile_change_reprojects_without_mutation(self):
        self.native()
        saved = self.save()
        before = self.snapshot()
        self.load.return_value = profile("ja")
        current = self.service.read_book("book")
        self.assertEqual(current.book, saved.book)
        self.assertIsNone(current.effective_mode)
        self.assertIsNone(current.proficiency)
        self.assertIn("proficiency_language_mismatch", self.codes(current))
        self.assertEqual(before, self.snapshot())

    def test_native_change_pauses_and_resumes_without_book_writes(self):
        self.native()
        saved = self.save()
        self.native("en")
        before = self.snapshot()
        current = self.service.read_book("book")
        self.assertEqual(current.book, saved.book)
        self.assertIsNone(current.effective_mode)
        self.assertIn("same_language", self.codes(current))
        self.assertEqual(before, self.snapshot())
        self.native("zh")
        self.assertEqual(self.service.read_book("book").effective_mode, "language_learning")
        self.assertEqual(self.repo.read_book("book"), saved.book)

    def test_clearing_native_pauses_without_guessing_a_default(self):
        self.native()
        saved = self.save()
        self.native(None)
        current = self.service.read_book("book")
        self.assertIsNone(current.effective_mode)
        self.assertIn("native_language_required", self.codes(current))
        self.assertEqual(current.book, saved.book)

    def test_normal_mode_is_independent_of_learning_requirements(self):
        self.load.side_effect = OSError("unavailable")
        result = self.save(BookSettings())
        self.assertEqual(result.effective_mode, "normal")
        self.assertIn("profile_unavailable", self.codes(result))
        self.assertIsNone(result.workspace.native_language)

    def test_mode_roundtrip_preserves_saved_settings(self):
        self.native()
        saved = self.save()
        normal = self.save(replace(saved.book.settings, active_mode="normal"))
        self.assertEqual(normal.book.settings.modes, saved.book.settings.modes)
        restored = self.save(saved.book.settings)
        self.assertEqual(restored.proficiency, saved.proficiency)
        with self.assertRaisesRegex(PreferenceError, "retention_required"):
            self.save(BookSettings())

    def test_invalid_registry_scheme_level_version_and_mode_are_atomic(self):
        self.native()
        before = self.snapshot()
        entry = foundation.learning()
        invalid = [BookSettings("future"), BookSettings("language_learning"), settings(override="xx"),
                   BookSettings(modes=("arbitrary",)), BookSettings(modes=(entry, entry)),
                   BookSettings("language_learning", modes=(PreservedModeSettings("language_learning", "9", "{}"),))]
        for field, value in (("language_code", "xx"), ("scheme_id", "jlpt_reading"),
                             ("scheme_version", "99"), ("level_id", "N5"), ("version", "9")):
            invalid.append(BookSettings("language_learning", modes=(replace(entry, **{field: value}),)))
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.save(value)
            self.assertEqual(before, self.snapshot())
        for native in ("xx", "EN", "unknown", True):
            with self.assertRaises(PreferenceError):
                self.native(native)
            self.assertEqual(before, self.snapshot())

    def test_all_supported_languages_use_registry_proficiency(self):
        for code in LanguageProfileRegistry.get_supported_language_codes():
            if code == "unknown":
                continue
            self.native("en" if code == "zh" else "zh")
            scheme = LanguageProficiencyRegistry.get_scheme(code)
            result = self.save(settings(code, scheme.levels[0].level_id, override=code))
            self.assertTrue(result.active.eligible)
            self.assertEqual(result.proficiency.scheme, scheme)

    def test_unknown_stored_mode_preserved_and_can_return_to_normal(self):
        self.seed_settings({"active_mode": "future", "source_language_override": None,
                            "modes": {"future": {"version": "9", "payload": {"value": "keep"}}}})
        before = self.snapshot()
        current = self.service.read_book("book")
        self.assertIsNone(current.effective_mode)
        self.assertEqual(current.active.reasons[0].code, "mode_unknown")
        self.assertEqual(before, self.snapshot())
        with self.assertRaises(PreferenceError):
            self.save(current.book.settings)
        normal = self.save(replace(current.book.settings, active_mode="normal"))
        self.assertEqual(normal.book.settings.modes, current.book.settings.modes)
        self.assertEqual(normal.effective_mode, "normal")

    def test_future_learning_version_cannot_activate(self):
        self.native()
        self.seed_settings({"active_mode": "language_learning", "source_language_override": None,
                            "modes": {"language_learning": {"version": "9", "payload": {"x": 1}}}})
        before = self.snapshot()
        result = self.service.read_book("book")
        self.assertIn("settings_unsupported", self.codes(result))
        with self.assertRaises(PreferenceValidationError):
            self.save(result.book.settings)
        self.assertEqual(before, self.snapshot())
        self.assertEqual(self.save(replace(result.book.settings, active_mode="normal")).effective_mode, "normal")

    def test_unknown_stored_source_is_not_normalized_or_repaired(self):
        self.seed_settings({"active_mode": "normal", "source_language_override": "future-language", "modes": {}})
        before = self.snapshot()
        self.load.reset_mock()
        result = self.service.read_book("book")
        self.assertIn("source_language_invalid", self.codes(result))
        self.assertEqual(result.book.settings.source_language_override, "future-language")
        self.assertEqual(before, self.snapshot())
        self.load.assert_not_called()

    def test_outdated_stored_proficiency_is_unavailable_and_retained(self):
        self.native()
        self.seed_settings({"active_mode": "language_learning", "source_language_override": None,
            "modes": {"language_learning": {"version": "1", "payload": {
                "language_code": "en", "scheme_id": "cefr_reading", "scheme_version": "old", "level_id": "B1"}}}})
        before = self.snapshot()
        current = self.service.read_book("book")
        self.assertIsNone(current.effective_mode)
        self.assertIn("proficiency_invalid", self.codes(current))
        self.assertEqual(before, self.snapshot())
        normal = self.save(replace(current.book.settings, active_mode="normal"))
        self.assertEqual(normal.effective_mode, "normal")
        self.assertEqual(normal.book.settings.modes, current.book.settings.modes)

    def test_profile_logs_only_operation_outcome_and_duration(self):
        self.load.side_effect = OSError("secret book text and private path")
        with self.assertLogs("reading_preferences.profile_source", level="INFO") as captured:
            self.service.read_book("private-book-id")
        log = "\n".join(captured.output)
        self.assertIn("outcome=profile_unavailable", log)
        self.assertIn("duration_ms=", log)
        self.assertNotIn("secret", log)
        self.assertNotIn("private", log)

    def test_tombstone_is_unavailable_and_stale_save_conflicts(self):
        self.native()
        saved = self.save()
        self.repo.delete_book("book", expected_revision=saved.book.revision, mutation_id="delete")
        current = self.service.read_book("book")
        self.assertIsNone(current.effective_mode)
        self.assertIn("book_deleted", {r.code for r in current.active.reasons})
        with self.assertRaises(PreferenceConflict):
            self.save(saved.book.settings, rev=saved.book.revision)

    def test_duplicate_and_same_value_save_preserve_revision(self):
        self.native()
        saved = self.save(command="apply", rev=0)
        self.assertEqual(saved, self.save(command="apply", rev=0))
        self.assertEqual(self.save(command="another").book.revision, saved.book.revision)
        with self.assertRaises(PreferenceConflict):
            self.save(settings(level="B2"), command="another")

    def test_workspace_race_rejects_apply_inside_repository(self):
        self.native()
        def load(*_):
            self.native("en")
            return profile()
        self.load.side_effect = load
        with self.assertRaises(PreferenceConflict):
            self.save(workspace_rev=1)
        self.assertEqual(self.repo.read_book("book").revision, 0)

    def test_book_race_rejects_apply_inside_repository(self):
        self.native()
        def load(*_):
            self.repo.save_book("book", BookSettings(source_language_override="fr"), expected_revision=0,
                expected_workspace_revision=1, resolved_source_language=None, mutation_id="competing")
            return profile()
        self.load.side_effect = load
        with self.assertRaises(PreferenceConflict):
            self.save(rev=0)
        self.assertEqual(self.repo.read_book("book").settings, BookSettings(source_language_override="fr"))

    def test_read_rejects_mixed_preferences_snapshots(self):
        self.native()
        for kind in ("workspace", "book"):
            def load(*_):
                if kind == "workspace":
                    self.native("fr")
                else:
                    self.repo.save_book("book", BookSettings(source_language_override="fr"), expected_revision=0,
                        expected_workspace_revision=self.repo.read_workspace().revision,
                        resolved_source_language=None, mutation_id="competing")
                return profile()
            self.load.side_effect = load
            with self.subTest(kind=kind), self.assertRaisesRegex(PreferenceConflict, "snapshot_changed"):
                self.service.read_book("book")

    def test_concurrent_apply_has_one_winner(self):
        self.native()
        barrier = Barrier(2)
        def load(*_):
            barrier.wait(timeout=15)
            return profile()
        self.load.side_effect = load
        def apply(level):
            try:
                return self.save(settings(level=level), rev=0, workspace_rev=1)
            except PreferenceConflict:
                return None
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(apply, ["B1", "B2"]))
        self.assertEqual(sum(r is not None for r in results), 1)
        self.assertEqual(self.repo.read_book("book"), next(r.book for r in results if r))

    def test_concurrent_duplicate_apply_replays(self):
        self.native()
        barrier = Barrier(2)
        def load(*_):
            barrier.wait(timeout=15)
            return profile()
        self.load.side_effect = load
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda _: self.save(rev=0, workspace_rev=1, command="same"), range(2)))
        self.assertEqual(results[0], results[1])
        self.assertEqual(self.repo.read_book("book").revision, 1)

    def test_unknown_commit_propagates_then_reconciles_without_auto_retry(self):
        self.native()
        original, calls = self.repo.save_book, []
        def commit(*args, **kw):
            calls.append(original(*args, **kw))
            raise PreferenceCommitUnknown("preference_commit_unknown")
        with patch.object(self.repo, "save_book", side_effect=commit):
            with self.assertRaises(PreferenceCommitUnknown):
                self.save(command="lost-ack", rev=0)
        self.assertEqual(len(calls), 1)
        self.assertEqual(self.service.read_book("book").book, calls[0])
        self.assertEqual(self.save(command="lost-ack", rev=0).book, calls[0])

    def test_failed_write_does_not_report_success(self):
        self.native()
        before = self.snapshot()
        with patch.object(self.repo, "save_book", side_effect=PreferenceStorageError("failed")):
            with self.assertRaises(PreferenceStorageError):
                self.save()
        self.assertEqual(before, self.snapshot())

    def test_no_io_after_successful_commit(self):
        self.native()
        original = self.repo.save_book
        def save(*args, **kw):
            result = original(*args, **kw)
            self.load.side_effect = AssertionError("profile read after commit")
            self.repo.read_workspace = Mock(side_effect=AssertionError("workspace read after commit"))
            self.repo.read_book = Mock(side_effect=AssertionError("book read after commit"))
            return result
        with patch.object(self.repo, "save_book", side_effect=save):
            self.assertTrue(self.save().active.eligible)

    def test_server_scope_and_exact_identity(self):
        self.service.read_book("Book.pdf")
        self.load.assert_called_once_with(self.repo.scope, "Book.pdf")
        for name in ("../book", " x", "", "a/b"):
            with self.assertRaises(PreferenceError):
                self.service.read_book(name)
        self.assertEqual(self.load.call_count, 1)


class FileServiceTests(ServiceCases, unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "prefs"
        self.repo = FileReadingPreferencesRepository(self.root, WorkspaceScope("test"))
        self.init_service()

    def snapshot(self):
        return {str(p.relative_to(self.root)): p.read_bytes() for p in self.root.rglob("*") if p.is_file()}

    def seed_settings(self, value):
        self.save(BookSettings())
        data = json.loads(self.repo._path.read_text())
        data["books"]["book"]["settings"] = value
        self.repo._path.write_text(json.dumps(data))

    def test_existing_profile_store_read_and_override_preserve_bytes(self):
        from profile.document_profile_store import DocumentProfileStore
        path = Path(self.tmp.name) / "profile.json"
        path.write_text(json.dumps({"topic": "topic", "summary": "summary", "document_language": "en"}))
        config = SimpleNamespace(get_raw_profile_path=lambda: str(path))
        load = Mock(side_effect=lambda scope, doc: DocumentProfileStore.load(config))
        self.service = ReadingPreferencesService(self.repo, ExistingProfileLanguageReader(load))
        before = path.read_bytes()
        with patch.object(DocumentProfileStore, "save", side_effect=AssertionError("profile mutation")), \
             patch.object(DocumentProfileStore, "clear", side_effect=AssertionError("profile deletion")):
            self.native()
            self.save()
            self.service.read_book("book")
            self.save(settings("ja", "N5", override="ja"))
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(load.call_count, 2)
        path.write_text("{broken")
        result = self.save(BookSettings("normal", modes=(foundation.learning("ja", "N5"),)))
        self.assertIn("profile_invalid", self.codes(result))
        self.assertEqual(path.read_text(), "{broken")

    def test_existing_profile_malformed_shapes_are_reported_without_repair(self):
        from profile.document_profile_store import DocumentProfileStore
        path = Path(self.tmp.name) / "profile.json"
        config = SimpleNamespace(get_raw_profile_path=lambda: str(path))
        self.service = ReadingPreferencesService(self.repo, ExistingProfileLanguageReader(
            lambda scope, doc: DocumentProfileStore.load(config)))
        for value in ([], 42, {"document_language": 42}):
            path.write_text(json.dumps(value))
            before = path.read_bytes()
            self.assertIn("profile_invalid", self.codes(self.service.read_book("book")))
            self.assertEqual(before, path.read_bytes())


class PostgresServiceTests(ServiceCases, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from scripts import test_reading_preferences_postgres as sql_tests
        sql_tests.PostgresPreferencesTests.setUpClass.__func__(cls)

    @classmethod
    def drop_schema(cls):
        from scripts import test_reading_preferences_postgres as sql_tests
        sql_tests.PostgresPreferencesTests.drop_schema.__func__(cls)

    def setUp(self):
        from db.postgres_reading_preferences_repository import PostgresReadingPreferencesRepository
        self.repo = PostgresReadingPreferencesRepository(self.scoped, WorkspaceScope(uuid4().hex))
        self.init_service()

    def snapshot(self):
        from scripts import test_reading_preferences_postgres as sql_tests
        return sql_tests.PostgresPreferencesTests.snapshot(self)

    def seed_settings(self, value):
        import psycopg
        from psycopg.types.json import Jsonb
        self.save(BookSettings())
        with psycopg.connect(self.scoped) as conn:
            conn.execute("UPDATE reading_book_preferences SET settings=%s WHERE namespace=%s",
                         (Jsonb(value), self.repo.scope.namespace))


if __name__ == "__main__":
    unittest.main()
