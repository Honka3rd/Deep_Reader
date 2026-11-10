"""POSIX file adapter: one small preferences snapshot per server workspace.

Only preference metadata is held here, never book text/generated artifacts.
Atomic replacement gives consistent unlocked reads; a stable flock serializes
all workspace/book writes, including rename of two preference keys. This is not
an atomic transaction with document files or PostgreSQL (LLRN-07/12/13).
"""

from contextlib import contextmanager, suppress
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile

from reading_preferences.commands import command_receipt, is_replay, check_revision
from reading_preferences.models import (
    BookPreferences, BookSettings, PreferenceCommitUnknown, PreferenceConflict,
    PreferenceError, PreferenceStorageError, WorkspacePreferences, WorkspaceScope,
    identifier, language, revision, validate_book_write,
)
from reading_preferences.serialization import (
    canonical_json, decode_book, decode_settings, decode_workspace, empty_book,
    empty_state, encode_settings, reject_constant, strict_object, validate_state,
)


class FileReadingPreferencesRepository:
    def __init__(self, base_dir: Path, scope: WorkspaceScope):
        if type(scope) is not WorkspaceScope:
            raise PreferenceError("preference_scope_invalid")
        self._scope = scope
        scope_key = hashlib.sha256(canonical_json([scope.kind, scope.namespace]).encode()).hexdigest()
        # No directories, defaults, locks or migration are created by construction/read.
        self._path = Path(base_dir) / f"{scope_key}.preferences.json"

    @property
    def scope(self) -> WorkspaceScope:
        return self._scope

    def _load(self) -> dict:
        self.assert_lifecycle_ready()
        try:
            raw = self._path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return empty_state(self.scope)
        except UnicodeError:
            raise PreferenceStorageError("preference_data_invalid") from None
        except OSError:
            raise PreferenceStorageError("preference_read_failed") from None
        try:
            data = json.loads(raw, object_pairs_hook=strict_object,
                              parse_constant=reject_constant)
            validate_state(self.scope, data)
        except (ValueError, TypeError, KeyError, UnicodeError):
            raise PreferenceStorageError("preference_data_invalid") from None
        return data

    @contextmanager
    def _lock(self):
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            lock = self._path.with_suffix(".lock").open("a")
        except OSError:
            raise PreferenceStorageError("preference_lock_failed") from None
        with lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX)
            except OSError:
                raise PreferenceStorageError("preference_lock_failed") from None
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def _commit(self, state):
        validate_state(self.scope, state)
        temporary = None
        replacing = False
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self._path.parent,
                                             prefix=self._path.name + ".", suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(canonical_json(state))
                stream.flush()
                os.fsync(stream.fileno())
            replacing = True
            os.replace(temporary, self._path)
            descriptor = os.open(self._path.parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        except OSError:
            if replacing:
                raise PreferenceCommitUnknown("preference_commit_unknown") from None
            raise PreferenceStorageError("preference_write_failed") from None
        finally:
            if temporary is not None:
                with suppress(OSError):
                    temporary.unlink(missing_ok=True)

    @contextmanager
    def locked_snapshot(self, document_id: str):
        """Internal coordination for learning writes; no preference mutation.

        Hold this before the document/learning locks. Never call preference writes
        in the context. Ordinary reads remain lock-free and create no files.
        """
        identifier(document_id)
        with self._lock():
            state = self._load()
            yield (decode_workspace(self.scope, state["workspace"]),
                   decode_book(self.scope, document_id, state["books"].get(document_id, empty_book())))

    @property
    def lifecycle_data_path(self):
        return self._path

    @property
    def lifecycle_journal_path(self):
        return self._path.with_suffix('.lifecycle.json')

    def assert_lifecycle_ready(self):
        if self.lifecycle_journal_path.exists():
            raise PreferenceStorageError('document_lifecycle_recovery_required')

    @contextmanager
    def lifecycle_lock(self):
        """Workspace coordination, including explicit interrupted-write recovery."""
        with self._lock():
            yield

    def stage_document_lifecycle(self, document_id, *, new_document_id=None, mutation_id):
        """Caller must hold lifecycle_lock; return a validated write, never commit."""
        from reading_preferences.lifecycle import PreferenceFileWrite, stage_document_lifecycle
        state = self._load()
        source, destination = stage_document_lifecycle(self.scope,
            state['books'].get(document_id, empty_book()), state['books'].get(new_document_id),
            document_id=document_id, new_document_id=new_document_id, mutation_id=mutation_id)
        state['books'][document_id] = source
        if new_document_id is not None:
            state['books'][new_document_id] = destination
        validate_state(self.scope, state)
        return PreferenceFileWrite(self._path, canonical_json(state))

    def read_workspace(self) -> WorkspacePreferences:
        return decode_workspace(self.scope, self._load()["workspace"])

    def read_book(self, document_id: str) -> BookPreferences:
        identifier(document_id)
        return decode_book(self.scope, document_id, self._load()["books"].get(document_id, empty_book()))

    def save_workspace(self, native_language: str | None, *, expected_revision: int,
                       mutation_id: str) -> WorkspacePreferences:
        language(native_language)
        revision(expected_revision)
        receipt = command_receipt(mutation_id, "workspace", expected_revision, native_language)
        with self._lock():
            state = self._load()
            current = state["workspace"]
            if not is_replay(current, receipt):
                check_revision(current, expected_revision)
                current["revision"] += int(current["native_language"] != native_language)
                current.update(native_language=native_language, receipt=receipt)
                self._commit(state)
            return decode_workspace(self.scope, current)

    def save_book(self, document_id: str, settings: BookSettings, *, expected_revision: int,
                  expected_workspace_revision: int, resolved_source_language: str | None,
                  mutation_id: str) -> BookPreferences:
        identifier(document_id)
        revision(expected_revision)
        revision(expected_workspace_revision)
        encoded = encode_settings(settings)
        # Canonicalize order and validate shape before locking; language/version
        # semantics are checked under the same lock as the current workspace.
        decode_settings(encoded)
        receipt = command_receipt(mutation_id, "book", document_id, expected_revision,
                                expected_workspace_revision, resolved_source_language, encoded)
        with self._lock():
            state = self._load()
            check_revision(state["workspace"], expected_workspace_revision)
            current = state["books"].get(document_id, empty_book())
            if not is_replay(current, receipt):
                check_revision(current, expected_revision)
                validate_book_write(settings, decode_settings(current["settings"]),
                                    state["workspace"]["native_language"], resolved_source_language)
                current["revision"] += int(current["settings"] != encoded or current["deleted"])
                current.update(settings=encoded, deleted=False, receipt=receipt)
                state["books"][document_id] = current
                self._commit(state)
            return decode_book(self.scope, document_id, current)

    def delete_book(self, document_id: str, *, expected_revision: int,
                    mutation_id: str) -> BookPreferences:
        identifier(document_id)
        revision(expected_revision)
        receipt = command_receipt(mutation_id, "delete", document_id, expected_revision)
        with self._lock():
            state = self._load()
            current = state["books"].get(document_id, empty_book())
            if not is_replay(current, receipt):
                check_revision(current, expected_revision)
                current["revision"] += int(not current["deleted"])
                current.update(settings=encode_settings(BookSettings()), deleted=True, receipt=receipt)
                state["books"][document_id] = current
                self._commit(state)
            return decode_book(self.scope, document_id, current)

    def rename_book(self, document_id: str, new_document_id: str, *, expected_revision: int,
                    expected_destination_revision: int, mutation_id: str) -> BookPreferences:
        identifier(document_id)
        identifier(new_document_id)
        revision(expected_revision)
        revision(expected_destination_revision)
        if document_id == new_document_id:
            raise PreferenceError("preference_rename_same_identity")
        receipt = command_receipt(mutation_id, "rename", document_id, new_document_id,
                                expected_revision, expected_destination_revision)
        with self._lock():
            state = self._load()
            source = state["books"].get(document_id, empty_book())
            destination = state["books"].get(new_document_id, empty_book())
            if is_replay(source, receipt):
                if not is_replay(destination, receipt):
                    raise PreferenceConflict("preference_rename_superseded")
                return decode_book(self.scope, new_document_id, destination)
            check_revision(source, expected_revision)
            check_revision(destination, expected_destination_revision)
            if source["deleted"]:
                raise PreferenceConflict("preference_source_deleted")
            if new_document_id in state["books"] and not destination["deleted"]:
                raise PreferenceConflict("preference_destination_exists")
            destination.update(settings=source["settings"], deleted=False, receipt=receipt,
                               revision=destination["revision"] + 1)
            source.update(settings=encode_settings(BookSettings()), deleted=True, receipt=receipt,
                          revision=source["revision"] + 1)
            state["books"][document_id] = source
            state["books"][new_document_id] = destination
            self._commit(state)
            return decode_book(self.scope, new_document_id, destination)
