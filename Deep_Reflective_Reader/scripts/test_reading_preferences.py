"""LLRN-02 file contract, failures, lifecycle and real process concurrency."""

from dataclasses import FrozenInstanceError, replace
import json
import multiprocessing
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from language.language_proficiency_registry import LanguageProficiencyRegistry
from reading_preferences import file_repository as storage
from reading_preferences.file_repository import FileReadingPreferencesRepository as Repository
from reading_preferences.models import (
    BookSettings, LanguageLearningSettings, PreservedModeSettings, PreferenceError,
    PreferenceConflict, PreferenceCommitUnknown, PreferenceStorageError, WorkspaceScope, list_modes,
)


def learning(code="en", level="B1"):
    scheme = LanguageProficiencyRegistry.get_scheme(code)
    return LanguageLearningSettings(code, scheme.scheme_id, scheme.version, level)


def concurrent_write(root, barrier, output, index, separate_books):
    repo = Repository(Path(root), WorkspaceScope("default"))
    barrier.wait(timeout=15)
    try:
        if separate_books:
            result = repo.save_book(f"book{index}", BookSettings(source_language_override="en"),
                                    expected_revision=0, expected_workspace_revision=0,
                                    resolved_source_language=None, mutation_id=f"write{index}")
        else:
            result = repo.save_workspace(("en", "fr")[index], expected_revision=0,
                                         mutation_id=f"write{index}")
        output.put(("saved", result.revision))
    except PreferenceConflict:
        output.put(("conflict", None))


def crash_during_replace(root, after):
    repo = Repository(Path(root), WorkspaceScope("default"))
    real_replace = os.replace

    def interrupt(source, destination):
        if after:
            real_replace(source, destination)
        os._exit(72 if after else 71)

    with patch.object(storage.os, "replace", side_effect=interrupt):
        repo.save_workspace("fr", expected_revision=1, mutation_id="crash-command")


class ReadingPreferencesTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "preferences"
        self.repo = Repository(self.root, WorkspaceScope("default"))

    def save(self, settings=None, doc="book", rev=None, source="en", command="book-save"):
        if settings is None:
            settings = BookSettings("language_learning", modes=(learning(),))
        if rev is None:
            rev = self.repo.read_book(doc).revision
        return self.repo.save_book(doc, settings, expected_revision=rev,
                                  expected_workspace_revision=self.repo.read_workspace().revision,
                                  resolved_source_language=source, mutation_id=command)

    def native(self, code="zh", command="native"):
        return self.repo.save_workspace(code, expected_revision=self.repo.read_workspace().revision,
                                        mutation_id=command)

    def payload(self):
        return json.loads(self.repo._path.read_text())

    def store_fixture(self, data):
        self.repo._path.write_text(json.dumps(data, ensure_ascii=False))

    def test_reads_and_catalog_do_not_create_directories_or_defaults(self):
        self.assertEqual([m.mode_id for m in list_modes()], ["normal", "language_learning"])
        self.assertIsNone(self.repo.read_workspace().native_language)
        self.assertEqual(self.repo.read_book("未設定的書").settings, BookSettings())
        self.assertEqual(self.repo.read_book("book").revision, 0)
        self.assertFalse(self.root.exists())

    def test_save_reload_and_scope_are_immutable(self):
        self.native()
        saved = self.save()
        again = Repository(self.root, WorkspaceScope("default"))
        self.assertEqual(again.read_book("book"), saved)
        self.assertEqual(again.read_workspace(), self.repo.read_workspace())
        with self.assertRaises(FrozenInstanceError):
            saved.settings.active_mode = "normal"
        self.assertEqual(saved.scope.kind, "workspace")

    def test_normal_mode_needs_no_learning_fields(self):
        saved = self.save(BookSettings(), source=None)
        self.assertEqual(saved.settings.active_mode, "normal")
        self.assertEqual(saved.revision, 0)
        self.assertIsNone(self.repo.read_workspace().native_language)

    def test_switching_mode_retains_saved_learning_configuration(self):
        self.native()
        saved = self.save()
        normal = self.save(replace(saved.settings, active_mode="normal"), command="normal")
        self.assertEqual(normal.settings.modes, saved.settings.modes)
        self.assertEqual(self.save(saved.settings, command="learning").settings, saved.settings)
        with self.assertRaisesRegex(PreferenceError, "retention_required"):
            self.save(BookSettings(), command="drop")

    def test_learning_cannot_activate_without_native_source_or_level(self):
        for settings, source in ((BookSettings("language_learning"), "en"),
                                 (BookSettings("language_learning", modes=(learning(),)), None),
                                 (BookSettings("language_learning", modes=(learning(),)), "en")):
            with self.subTest(settings=settings, source=source), self.assertRaises(PreferenceError):
                self.save(settings, source=source)
        self.assertEqual(self.repo.read_book("book").revision, 0)
        self.assertFalse(self.repo._path.exists())

    def test_missing_source_after_native_is_configured_does_not_activate(self):
        self.native()
        before = self.repo._path.read_bytes()
        with self.assertRaisesRegex(PreferenceError, "languages_required"):
            self.save(source=None)
        self.assertEqual(self.repo._path.read_bytes(), before)

    def test_preserved_payload_cannot_bypass_typed_settings_submission(self):
        self.native()
        before = self.repo._path.read_bytes()
        valid_payload = json.dumps({"language_code": "en", "scheme_id": learning().scheme_id,
                                    "scheme_version": "1", "level_id": "B1"})
        for payload in (valid_payload, "{bad", '{"value": NaN}', '{"value":1,"value":2}'):
            with self.subTest(payload=payload), self.assertRaises(PreferenceError):
                self.save(BookSettings("language_learning", modes=(
                    PreservedModeSettings("language_learning", "1", payload),)))
        self.assertEqual(self.repo._path.read_bytes(), before)

    def test_same_language_and_wrong_proficiency_binding_rejected(self):
        self.native("en")
        with self.assertRaisesRegex(PreferenceError, "same_language"):
            self.save()
        self.native("zh", command="native-change")
        for settings, source in ((BookSettings("language_learning", "fr", (learning(),)), "en"),
                                 (BookSettings("language_learning", modes=(learning(),)), "fr")):
            with self.subTest(source=source), self.assertRaisesRegex(PreferenceError, "source_mismatch"):
                self.save(settings, source=source)

    def test_registry_and_level_rejections_do_not_write(self):
        self.native()
        before = self.repo._path.read_bytes()
        for item in (replace(learning(), level_id="N5"), replace(learning(), scheme_version="0"),
                     replace(learning(), scheme_id="jlpt_reading"), replace(learning(), version="2")):
            with self.subTest(item=item), self.assertRaises(ValueError):
                self.save(BookSettings("language_learning", modes=(item,)))
        for code in ("unknown", "en-US", "EN", "", 1, True, {}):
            with self.subTest(code=code), self.assertRaises(PreferenceError):
                self.native(code)
        self.assertEqual(self.repo._path.read_bytes(), before)

    def test_whole_settings_write_is_atomic_on_validation_failure(self):
        self.native()
        original = self.save()
        with self.assertRaises(ValueError):
            self.save(replace(original.settings, source_language_override="ja"), command="bad")
        self.assertEqual(self.repo.read_book("book"), original)

    def test_same_value_preserves_revision_and_replay(self):
        first = self.native()
        self.assertEqual(self.native(command="same").revision, first.revision)
        saved = self.save()
        unchanged = self.save(saved.settings, command="same-book")
        self.assertEqual(unchanged.revision, saved.revision)
        self.assertEqual(self.save(saved.settings, rev=saved.revision, command="same-book"), unchanged)

    def test_cas_and_mutation_id_reuse(self):
        self.native()
        saved = self.save()
        self.assertEqual(self.save(rev=0), saved)
        with self.assertRaisesRegex(PreferenceConflict, "mutation_conflict"):
            self.save(replace(saved.settings, active_mode="normal"), rev=0)
        with self.assertRaisesRegex(PreferenceConflict, "revision_changed"):
            self.save(rev=0, command="other")
        self.save(replace(saved.settings, active_mode="normal"), command="next")
        with self.assertRaises(PreferenceConflict):
            self.save(rev=0)

    def test_native_revision_checked_inside_book_transaction(self):
        self.native()
        settings = BookSettings("language_learning", modes=(learning(),))
        self.native("fr", command="change")
        with self.assertRaises(PreferenceConflict):
            self.repo.save_book("book", settings, expected_revision=0, expected_workspace_revision=1,
                                resolved_source_language="en", mutation_id="stale")
        self.assertEqual(self.repo.read_book("book").revision, 0)

    def test_native_change_does_not_rewrite_book_preferences(self):
        self.native()
        original = self.save()
        self.native("en", command="equal-now")
        self.assertEqual(self.repo.read_book("book"), original)
        # Effective eligibility is LLRN-09; future activation still cannot bypass same-language validation.
        with self.assertRaisesRegex(PreferenceError, "same_language"):
            self.save(command="reapply")

    def test_namespaces_and_book_names_do_not_alias(self):
        self.native()
        original = self.save(doc="Book.pdf")
        self.assertEqual(self.repo.read_book("Book").revision, 0)
        other = Repository(self.root, WorkspaceScope("another"))
        self.assertIsNone(other.read_workspace().native_language)
        other.save_workspace("ja", expected_revision=0, mutation_id="other")
        self.assertEqual(self.repo.read_book("Book.pdf"), original)
        self.assertEqual(self.repo.read_workspace().native_language, "zh")

    def test_scope_and_revision_input_validation(self):
        for value in ("../bad", " ", "x/y", "x\\y", "..", "x\n", None, 3):
            with self.subTest(value=value), self.assertRaises(PreferenceError):
                WorkspaceScope(value)
        for value in (-1, True, "0", 0.0):
            with self.subTest(value=value), self.assertRaises(PreferenceError):
                self.repo.save_workspace("en", expected_revision=value, mutation_id="test")
        self.assertFalse(self.root.exists())

    def test_unknown_active_mode_and_settings_round_trip_without_repair(self):
        self.save(BookSettings(), source=None)
        payload = self.payload()
        payload["books"]["book"]["settings"] = {
            "active_mode": "future", "source_language_override": None,
            "modes": {"future": {"version": "7", "payload": {"nested": [1, {"x": "值"}]}}},
        }
        self.store_fixture(payload)
        before = self.repo._path.read_bytes()
        saved = self.repo.read_book("book")
        self.assertEqual(saved.settings.active_mode, "future")
        self.assertIsInstance(saved.settings.modes[0], PreservedModeSettings)
        self.assertEqual(self.repo._path.read_bytes(), before)
        with self.assertRaisesRegex(PreferenceError, "mode_unknown"):
            self.save(saved.settings, source=None, command="unsupported")
        normal = self.save(replace(saved.settings, active_mode="normal"), source=None, command="normal")
        self.assertEqual(normal.settings.modes, saved.settings.modes)
        fabricated = PreservedModeSettings("other", "1", "{}")
        with self.assertRaisesRegex(PreferenceError, "unsupported"):
            self.save(replace(normal.settings, modes=(*normal.settings.modes, fabricated)), command="fake")

    def test_unknown_learning_version_is_preserved_but_cannot_activate(self):
        self.native()
        saved = self.save()
        payload = self.payload()
        payload["books"]["book"]["settings"]["modes"]["language_learning"]["version"] = "2"
        self.store_fixture(payload)
        future = self.repo.read_book("book")
        self.assertIsInstance(future.settings.modes[0], PreservedModeSettings)
        with self.assertRaisesRegex(PreferenceError, "unsupported"):
            self.save(future.settings, command="future")
        normal = self.save(replace(future.settings, active_mode="normal"), command="normal")
        self.assertEqual(normal.settings.modes, future.settings.modes)
        self.assertEqual(saved.revision + 1, normal.revision)

    def test_corrupt_incompatible_and_wrong_scope_data_are_not_missing(self):
        self.native()
        original = self.payload()
        for field, value in (("schema_version", 2), ("schema_version", True),
                             ("scope", {"kind": "user", "namespace": "default"}),
                             ("books", []), ("workspace", {"revision": True})):
            with self.subTest(field=field, value=value):
                self.store_fixture(original | {field: value})
                before = self.repo._path.read_bytes()
                with self.assertRaises(PreferenceStorageError):
                    self.repo.read_workspace()
                with self.assertRaises(PreferenceStorageError):
                    self.repo.save_workspace("fr", expected_revision=1, mutation_id="repair")
                self.assertEqual(before, self.repo._path.read_bytes())
        for raw in (b'{"x":1,"x":2}', b'not json', b'\xff'):
            self.repo._path.write_bytes(raw)
            with self.assertRaises(PreferenceStorageError):
                self.repo.read_workspace()

    def test_io_failure_is_not_a_default(self):
        self.native()
        with patch.object(Path, "read_text", side_effect=PermissionError("private path")):
            with self.assertRaisesRegex(PreferenceStorageError, "preference_read_failed") as error:
                self.repo.read_workspace()
            self.assertNotIn("private path", str(error.exception))

    def test_delete_tombstone_prevents_stale_resurrection_and_preserves_workspace(self):
        workspace = self.native()
        saved = self.save()
        deleted = self.repo.delete_book("book", expected_revision=saved.revision, mutation_id="delete")
        self.assertTrue(deleted.deleted)
        self.assertEqual(deleted.settings, BookSettings())
        self.assertEqual(self.repo.delete_book("book", expected_revision=saved.revision, mutation_id="delete"), deleted)
        with self.assertRaises(PreferenceConflict):
            self.save(saved.settings, rev=saved.revision, command="old")
        self.assertEqual(self.repo.read_workspace(), workspace)
        recreated = self.save(BookSettings(), rev=deleted.revision, command="recreate")
        self.assertFalse(recreated.deleted)
        self.assertGreater(recreated.revision, deleted.revision)

    def test_delete_missing_book_fences_inflight_initial_save(self):
        deleted = self.repo.delete_book("missing", expected_revision=0, mutation_id="delete")
        self.assertEqual(deleted.revision, 1)
        with self.assertRaises(PreferenceConflict):
            self.save(BookSettings(), doc="missing", rev=0, command="late")

    def test_rename_is_atomic_and_replayable_with_old_identity_fenced(self):
        self.native()
        saved = self.save()
        renamed = self.repo.rename_book("book", "新書", expected_revision=1,
                                        expected_destination_revision=0, mutation_id="rename")
        self.assertEqual(renamed.settings, saved.settings)
        self.assertTrue(self.repo.read_book("book").deleted)
        self.assertEqual(self.repo.rename_book("book", "新書", expected_revision=1,
                                              expected_destination_revision=0, mutation_id="rename"), renamed)
        with self.assertRaises(PreferenceConflict):
            self.save(rev=1, command="late")
        self.save(replace(renamed.settings, active_mode="normal"), doc="新書", command="next")
        with self.assertRaisesRegex(PreferenceConflict, "superseded"):
            self.repo.rename_book("book", "新書", expected_revision=1,
                                  expected_destination_revision=0, mutation_id="rename")

    def test_rename_collision_and_failed_commit_preserve_both_keys(self):
        self.save(BookSettings(source_language_override="en"))
        self.save(BookSettings(), doc="existing", command="existing")
        before = self.repo._path.read_bytes()
        with self.assertRaisesRegex(PreferenceConflict, "destination_exists"):
            self.repo.rename_book("book", "existing", expected_revision=1,
                                  expected_destination_revision=0, mutation_id="collision")
        with patch.object(storage.os, "fsync", side_effect=OSError("fail before replace")):
            with self.assertRaises(PreferenceStorageError) as raised:
                self.repo.rename_book("book", "new", expected_revision=1,
                                      expected_destination_revision=0, mutation_id="failure")
            self.assertNotIsInstance(raised.exception, PreferenceCommitUnknown)
        self.assertEqual(self.repo._path.read_bytes(), before)
        self.assertFalse(list(self.root.glob("*.tmp")))

    def test_failure_after_replace_is_unknown_and_reconciles_by_same_command(self):
        self.native()
        real_fsync = os.fsync
        calls = 0

        def fail_directory(fd):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("directory sync failed")
            return real_fsync(fd)

        with patch.object(storage.os, "fsync", side_effect=fail_directory):
            with self.assertRaises(PreferenceCommitUnknown):
                self.repo.save_workspace("fr", expected_revision=1, mutation_id="unknown")
        restarted = Repository(self.root, self.repo.scope)
        self.assertEqual(restarted.read_workspace().native_language, "fr")
        self.assertEqual(restarted.save_workspace("fr", expected_revision=1, mutation_id="unknown").revision, 2)

    def processes(self, separate_books):
        ctx = multiprocessing.get_context("spawn")
        barrier = ctx.Barrier(3)
        output = ctx.Queue()
        processes = [ctx.Process(target=concurrent_write,
                                args=(str(self.root), barrier, output, index, separate_books)) for index in range(2)]
        try:
            for process in processes:
                process.start()
            barrier.wait(timeout=15)
            results = [output.get(timeout=15) for _ in processes]
            for process in processes:
                process.join(timeout=10)
                self.assertEqual(process.exitcode, 0)
            return results
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate()
                process.join(timeout=5)
            output.close()

    def test_real_process_cas_allows_one_writer(self):
        self.assertCountEqual([result[0] for result in self.processes(False)], ["saved", "conflict"])
        self.assertEqual(self.repo.read_workspace().revision, 1)

    def test_real_process_different_books_do_not_lose_updates(self):
        self.assertEqual([result[0] for result in self.processes(True)], ["saved", "saved"])
        self.assertEqual(self.repo.read_book("book0").revision, 1)
        self.assertEqual(self.repo.read_book("book1").revision, 1)

    def test_process_crash_before_and_after_replace(self):
        self.native()
        ctx = multiprocessing.get_context("spawn")
        for after in (False, True):
            with self.subTest(after=after):
                process = ctx.Process(target=crash_during_replace, args=(str(self.root), after))
                process.start()
                process.join(timeout=15)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=5)
                    self.fail("crash fixture did not terminate")
                self.assertEqual(process.exitcode, 72 if after else 71)
                restarted = Repository(self.root, self.repo.scope)
                self.assertEqual(restarted.read_workspace().native_language, "fr" if after else "zh")
        self.assertEqual(self.repo.save_workspace("fr", expected_revision=1, mutation_id="crash-command").revision, 2)


if __name__ == "__main__":
    unittest.main()
