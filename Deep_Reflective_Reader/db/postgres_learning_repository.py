"""Learning repository with short PostgreSQL transactions and explicit migrations.

Lock order: workspace preferences -> documents row -> learning rows. Source and
preference writers cannot pass between revalidation and commit. Reads use a
single REPEATABLE READ READ ONLY snapshot and never repair, expire or generate.
"""
from contextlib import contextmanager
import logging
from time import monotonic

import psycopg
from psycopg.rows import dict_row

from db import learning_rows
from db.postgres_reading_preferences_repository import PostgresReadingPreferencesRepository
from db.postgres_structured_document_store import PostgresStructuredDocumentStore, PostgresStructuredDocumentTarget
from document_structure import learning_file_codec as codec
from document_structure.learning_repository import LearningStorageError, LearningCommitUnknown, BatchPayload
from document_structure.learning_repository_behavior import LearningRepositoryBehavior
from document_structure.learning_source_resolver import learning_blocks
from reading_preferences.models import WorkspaceScope, identifier
from shared.learning_source import require, token

logger = logging.getLogger(__name__)


class PostgresLearningRepository(LearningRepositoryBehavior):
    def __init__(self, dsn: str, scope: WorkspaceScope, *, admission=None,
                 documents=None, preferences=None):
        require(type(scope) is WorkspaceScope, 'learning_scope_invalid')
        self._dsn = dsn
        self._scope = scope
        self._admission = admission
        self._documents = documents if documents is not None else PostgresStructuredDocumentStore(dsn, namespace=scope.namespace)
        self._preferences = preferences if preferences is not None else PostgresReadingPreferencesRepository(dsn, scope)
        require(self._documents.dsn == dsn and self._documents.namespace == scope.namespace
                and self._preferences.dsn == dsn and self._preferences.scope == scope,
                'learning_storage_identity_mismatch')

    @property
    def scope(self):
        return self._scope

    @property
    def document_store(self):
        return self._documents

    @property
    def preference_repository(self):
        return self._preferences

    def _connect(self):
        return psycopg.connect(self._dsn, row_factory=dict_row)

    @contextmanager
    def _transaction(self, *, write=False):
        connection = None
        committing = False
        started = monotonic()
        outcome = 'failed'
        try:
            connection = self._connect()
            if not write:
                connection.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY')
            else:
                connection.execute('SET TRANSACTION ISOLATION LEVEL READ COMMITTED')
            yield connection
            committing = write
            connection.commit()
            outcome = 'committed' if write else 'read'
        except psycopg.Error as error:
            if committing and (error.sqlstate is None or error.sqlstate.startswith('08')
                or error.sqlstate == '40003' or (isinstance(error, psycopg.OperationalError)
                and error.sqlstate not in {'40001', '40P01'})):
                outcome = 'unknown'
                raise LearningCommitUnknown('learning_commit_unknown') from None
            raise LearningStorageError('learning_write_failed' if write else 'learning_read_failed') from None
        finally:
            if connection is not None:
                connection.close()
            logger.info('learning_sql_transaction', extra={'stage': 'write' if write else 'read',
                'outcome': outcome, 'duration_ms': round((monotonic() - started) * 1000, 2)})

    def _identity(self, connection, document_id, *, lock=False):
        identifier(document_id)
        row = connection.execute('SELECT id FROM documents WHERE namespace=%s AND document_name=%s' +
                                 (' FOR UPDATE' if lock else ''), (self.scope.namespace, document_id)).fetchone()
        return row['id'] if row is not None else None

    def _document(self, connection, document_id):
        document = self._documents.load_in_transaction(
            PostgresStructuredDocumentTarget(self.scope.namespace, document_id), connection=connection)
        require(document.document_id == document_id, 'learning_document_identity_mismatch')
        return document

    @contextmanager
    def _read_context(self, document_id, *, include_document=True, source_only=False, **selection):
        with self._transaction() as connection:
            row_id = self._identity(connection, document_id)
            document = self._document(connection, document_id) if include_document else None
            state = (codec.empty_state(self.scope, document_id) if source_only or row_id is None else
                     learning_rows.load_state(connection, row_id, self.scope, document_id, **selection))
            yield document, state

    @contextmanager
    def _write_context(self, document_id, **selection):
        identifier(document_id)
        with self._transaction(write=True) as connection:
            workspace, book = self._preferences.lock_snapshot(document_id, connection=connection)
            row_id = self._identity(connection, document_id, lock=True)
            require(row_id is not None, 'learning_document_missing')
            document = self._document(connection, document_id)
            blocks = {b.source.content_block_id: b for b in learning_blocks(document, self.scope, document_id)}
            state = learning_rows.load_state(connection, row_id, self.scope, document_id, **selection)
            before = {key: dict(value) if type(value) is dict else value for key, value in state.items()}

            def persist(updated):
                learning_rows.persist(connection, row_id, self.scope, document_id, updated, before)
                logger.info('learning_sql_write', extra={'stage': 'persist', 'outcome': 'staged',
                    'entry_count': len(updated['entries']), 'operation_count': len(updated['operations'])})

            yield workspace, book, blocks, state, persist

    def read_checkpoints(self, document_id, operation_id, *, offset=0, limit=50):
        """Page checkpoint rows without loading earlier/later payloads or history."""
        token(operation_id)
        self._page(offset, limit)
        with self._transaction() as connection:
            row_id = self._identity(connection, document_id)
            row = connection.execute('SELECT * FROM learning_operations WHERE document_row_id=%s AND operation_id=%s',
                                     (row_id, operation_id)).fetchone()
            require(row is not None, 'learning_operation_missing')
            op = learning_rows.decoded(row, 'operations')
            receipt_row = connection.execute('SELECT * FROM learning_commands WHERE document_row_id=%s AND command_id=%s',
                                             (row_id, operation_id)).fetchone()
            if receipt_row is None:
                raise LearningStorageError('learning_claim_receipt_missing')
            receipt = learning_rows.decoded(receipt_row, 'receipts')
            if receipt.command.binding != op.binding or receipt.operation_id != operation_id or receipt.outcome != 'claimed':
                raise LearningStorageError('learning_claim_receipt_invalid')
            self._scope_key(document_id, op.binding.key)
            document = self._document(connection, document_id)
            blocks = {b.source.content_block_id: b for b in learning_blocks(document, self.scope, document_id)}
            block = self._block(blocks, op.binding.key.resource.source)
            row = connection.execute('SELECT * FROM learning_sentences WHERE document_row_id=%s AND resource_id=%s',
                                     (row_id, op.binding.key.resource.resource_id)).fetchone()
            if row is None:
                raise LearningStorageError('learning_resource_missing')
            sentences = learning_rows.decoded(row, 'sentences')
            sentences.validate_source(block.source, block.effective_text)
            require(sentences.reference == op.binding.key.resource, 'learning_reference_mismatch')
            op.plan.validate_against(sentences)
            rows = connection.execute('SELECT * FROM learning_checkpoints WHERE document_row_id=%s AND operation_id=%s '
                'AND batch_index >= %s ORDER BY batch_index LIMIT %s', (row_id, operation_id, offset, limit)).fetchall()
            expected = min(limit, max(0, len(op.checkpoints) - offset))
            if len(rows) != expected:
                raise LearningStorageError('learning_checkpoint_count_invalid')
            parts = []
            for index, row in enumerate(rows, offset):
                part = learning_rows.decoded(row, 'checkpoints')
                if (row['batch_index'] != index or type(part) is not BatchPayload
                    or codec.payload_digest(part) != op.checkpoints[index]):
                    raise LearningStorageError('learning_checkpoint_digest_mismatch')
                part.validate(op, index, sentences, block.effective_text)
                parts.append(part)
            return tuple(parts)
