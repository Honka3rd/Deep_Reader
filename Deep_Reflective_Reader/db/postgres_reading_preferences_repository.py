"""Bounded PostgreSQL implementation of the reading-preferences port.

Explicit migration 004 is required. Writes serialize on a workspace row before
reading any book; INSERT ON CONFLICT covers an absent workspace's first writer.
One transaction contains settings, revisions, tombstones and last-command receipts.
This adapter does not orchestrate actual document lifecycle or effective settings.
"""

from contextlib import contextmanager
import logging
from time import monotonic

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from reading_preferences.commands import check_revision, command_receipt, is_replay
from reading_preferences.models import (
    BookPreferences, BookSettings, PreferenceCommitUnknown, PreferenceConflict,
    PreferenceError, PreferenceStorageError, WorkspacePreferences, WorkspaceScope,
    identifier, language, revision, validate_book_write,
)
from reading_preferences.serialization import (
    decode_book, decode_settings, decode_workspace, empty_book, empty_state, encode_settings,
)

logger = logging.getLogger(__name__)


class PostgresReadingPreferencesRepository:
    def __init__(self, dsn: str, scope: WorkspaceScope):
        if type(scope) is not WorkspaceScope:
            raise PreferenceError("preference_scope_invalid")
        self._dsn = dsn
        self._scope = scope

    @property
    def scope(self) -> WorkspaceScope:
        return self._scope

    @property
    def dsn(self):
        """Connection identity for composition; never expose in transport/logs."""
        return self._dsn

    def _connect(self):
        return psycopg.connect(self._dsn, row_factory=dict_row)

    @contextmanager
    def _transaction(self, operation: str, *, write: bool = False):
        started = monotonic()
        outcome = "failed"
        committing = False
        connection = None
        try:
            connection = self._connect()
            # Explicitly prohibit incidental writes on the ordinary read path.
            if not write:
                connection.execute("SET TRANSACTION READ ONLY")
            yield connection
            committing = write
            connection.commit()
            outcome = "committed" if write else "read"
        except PreferenceConflict:
            outcome = "conflict"
            raise
        except psycopg.Error as error:
            # A transport loss during COMMIT cannot establish whether it committed.
            # Server-reported constraint/serialization failures prove rejection.
            if committing and (error.sqlstate is None or error.sqlstate.startswith("08")
                               or error.sqlstate == "40003"
                               or (isinstance(error, psycopg.OperationalError)
                                   and error.sqlstate not in {"40001", "40P01"})):
                outcome = "unknown"
                raise PreferenceCommitUnknown("preference_commit_unknown") from None
            raise PreferenceStorageError("preference_write_failed" if write else
                                         "preference_read_failed") from None
        finally:
            # Closing an uncommitted connection rolls back; never mask an unknown
            # outcome by attempting recovery/replay here.
            if connection is not None:
                connection.close()
            logger.info("reading_preferences operation=%s outcome=%s duration_ms=%.2f",
                        operation, outcome, (monotonic() - started) * 1000)

    def _workspace(self, connection, *, lock: bool = False):
        if lock:
            connection.execute(
                "INSERT INTO reading_workspace_preferences(namespace) VALUES (%s) "
                "ON CONFLICT(namespace) DO NOTHING", (self.scope.namespace,))
        row = connection.execute(
            "SELECT scope_kind, schema_version, revision, native_language, receipt "
            "FROM reading_workspace_preferences WHERE namespace = %s" +
            (" FOR UPDATE" if lock else ""), (self.scope.namespace,)).fetchone()
        if row is None:
            return empty_state(self.scope)["workspace"]
        if row.pop("scope_kind") != self.scope.kind or row.pop("schema_version") != 1:
            raise PreferenceStorageError("preference_storage_version_unsupported")
        try:
            decode_workspace(self.scope, row)
        except (ValueError, TypeError, KeyError):
            raise PreferenceStorageError("preference_data_invalid") from None
        return row

    def _book(self, connection, document_id):
        row = connection.execute(
            "SELECT schema_version, revision, settings, deleted, receipt "
            "FROM reading_book_preferences WHERE namespace = %s AND document_id = %s",
            (self.scope.namespace, document_id)).fetchone()
        if row is None:
            return empty_book(), False
        if row.pop("schema_version") != 1:
            raise PreferenceStorageError("preference_storage_version_unsupported")
        try:
            decode_book(self.scope, document_id, row)
        except (ValueError, TypeError, KeyError):
            raise PreferenceStorageError("preference_data_invalid") from None
        return row, True

    def _write_book(self, connection, document_id, data):
        # Called only after the workspace lock, with the caller's connection.
        decode_book(self.scope, document_id, data)
        connection.execute(
            "INSERT INTO reading_book_preferences "
            "(namespace, document_id, revision, settings, deleted, receipt) "
            "VALUES (%s, %s, %s, %s, %s, %s) "
            "ON CONFLICT(namespace, document_id) DO UPDATE SET "
            "revision = EXCLUDED.revision, settings = EXCLUDED.settings, "
            "deleted = EXCLUDED.deleted, receipt = EXCLUDED.receipt",
            (self.scope.namespace, document_id, data["revision"], Jsonb(data["settings"]),
             data["deleted"], Jsonb(data["receipt"])))

    def lock_snapshot(self, document_id: str, *, connection):
        """Coordinate a learning write inside its caller-owned transaction.

        Acquire this before document/learning locks. Never open/commit a second
        connection; the caller also owns rollback. Ordinary reads remain pure.
        """
        identifier(document_id)
        if connection.info.transaction_status != psycopg.pq.TransactionStatus.INTRANS:
            raise PreferenceError("preference_transaction_required")
        workspace = self._workspace(connection, lock=True)
        book, _ = self._book(connection, document_id)
        return decode_workspace(self.scope, workspace), decode_book(self.scope, document_id, book)

    def apply_document_lifecycle(self, document_id, *, connection, new_document_id=None, mutation_id):
        """Same-transaction preference migration, after workspace then document locks."""
        from reading_preferences.lifecycle import stage_document_lifecycle
        self.lock_snapshot(document_id, connection=connection)
        source, _ = self._book(connection, document_id)
        destination = None
        if new_document_id is not None:
            destination, exists = self._book(connection, new_document_id)
            if not exists:
                destination = None
        source, destination = stage_document_lifecycle(self.scope, source, destination,
            document_id=document_id, new_document_id=new_document_id, mutation_id=mutation_id)
        self._write_book(connection, document_id, source)
        if new_document_id is not None:
            self._write_book(connection, new_document_id, destination)

    def read_workspace(self) -> WorkspacePreferences:
        with self._transaction("read_workspace") as connection:
            result = decode_workspace(self.scope, self._workspace(connection))
        return result

    def read_book(self, document_id: str) -> BookPreferences:
        identifier(document_id)
        with self._transaction("read_book") as connection:
            current, _ = self._book(connection, document_id)
            result = decode_book(self.scope, document_id, current)
        return result

    def save_workspace(self, native_language: str | None, *, expected_revision: int,
                       mutation_id: str) -> WorkspacePreferences:
        language(native_language)
        revision(expected_revision)
        receipt = command_receipt(mutation_id, "workspace", expected_revision, native_language)
        with self._transaction("save_workspace", write=True) as connection:
            current = self._workspace(connection, lock=True)
            if not is_replay(current, receipt):
                check_revision(current, expected_revision)
                current["revision"] += int(current["native_language"] != native_language)
                current.update(native_language=native_language, receipt=receipt)
                connection.execute(
                    "UPDATE reading_workspace_preferences SET revision = %s, "
                    "native_language = %s, receipt = %s WHERE namespace = %s",
                    (current["revision"], native_language, Jsonb(receipt), self.scope.namespace))
            result = decode_workspace(self.scope, current)
        return result

    def save_book(self, document_id: str, settings: BookSettings, *, expected_revision: int,
                  expected_workspace_revision: int, resolved_source_language: str | None,
                  mutation_id: str) -> BookPreferences:
        identifier(document_id)
        revision(expected_revision)
        revision(expected_workspace_revision)
        encoded = encode_settings(settings)
        decode_settings(encoded)
        receipt = command_receipt(mutation_id, "book", document_id, expected_revision,
                                  expected_workspace_revision, resolved_source_language, encoded)
        with self._transaction("save_book", write=True) as connection:
            workspace = self._workspace(connection, lock=True)
            check_revision(workspace, expected_workspace_revision)
            current, _ = self._book(connection, document_id)
            if not is_replay(current, receipt):
                check_revision(current, expected_revision)
                validate_book_write(settings, decode_settings(current["settings"]),
                                    workspace["native_language"], resolved_source_language)
                current["revision"] += int(current["settings"] != encoded or current["deleted"])
                current.update(settings=encoded, deleted=False, receipt=receipt)
                self._write_book(connection, document_id, current)
            result = decode_book(self.scope, document_id, current)
        return result

    def delete_book(self, document_id: str, *, expected_revision: int,
                    mutation_id: str) -> BookPreferences:
        identifier(document_id)
        revision(expected_revision)
        receipt = command_receipt(mutation_id, "delete", document_id, expected_revision)
        with self._transaction("delete_book", write=True) as connection:
            self._workspace(connection, lock=True)
            current, _ = self._book(connection, document_id)
            if not is_replay(current, receipt):
                check_revision(current, expected_revision)
                current["revision"] += int(not current["deleted"])
                current.update(settings=encode_settings(BookSettings()), deleted=True, receipt=receipt)
                self._write_book(connection, document_id, current)
            result = decode_book(self.scope, document_id, current)
        return result

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
        with self._transaction("rename_book", write=True) as connection:
            self._workspace(connection, lock=True)
            source, _ = self._book(connection, document_id)
            destination, destination_exists = self._book(connection, new_document_id)
            if is_replay(source, receipt):
                if not is_replay(destination, receipt):
                    raise PreferenceConflict("preference_rename_superseded")
            else:
                check_revision(source, expected_revision)
                check_revision(destination, expected_destination_revision)
                if source["deleted"]:
                    raise PreferenceConflict("preference_source_deleted")
                if destination_exists and not destination["deleted"]:
                    raise PreferenceConflict("preference_destination_exists")
                destination.update(settings=source["settings"], deleted=False, receipt=receipt,
                                   revision=destination["revision"] + 1)
                source.update(settings=encode_settings(BookSettings()), deleted=True, receipt=receipt,
                              revision=source["revision"] + 1)
                self._write_book(connection, document_id, source)
                self._write_book(connection, new_document_id, destination)
            result = decode_book(self.scope, new_document_id, destination)
        return result
