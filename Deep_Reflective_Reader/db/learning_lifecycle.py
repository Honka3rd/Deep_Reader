"""Transaction-local document lifecycle integration; no commits or DDL."""
from time import time
from uuid import uuid4

from db import learning_rows
from db.postgres_reading_preferences_repository import PostgresReadingPreferencesRepository
from document_structure import learning_file_codec as codec
from document_structure.learning_lifecycle import rename_learning_state
from reading_preferences.models import WorkspaceScope


def lock_preferences(connection, dsn, namespace, document_id):
    # Compatibility with explicitly unmigrated stores. If partially migrated,
    # subsequent SQL fails closed; never repair/create schema during a mutation.
    row = connection.execute("SELECT to_regclass('reading_workspace_preferences') AS preferences, "
                             "to_regclass('learning_sentences') AS learning").fetchone()
    if row['preferences'] is None:
        if row['learning'] is not None:
            raise ValueError('learning_preferences_schema_missing')
        return None
    preferences = PostgresReadingPreferencesRepository(dsn, WorkspaceScope(namespace))
    preferences.lock_snapshot(document_id, connection=connection)
    return preferences


def rename(connection, row_id, preferences, old_name, new_name):
    if preferences is None:
        return
    preferences.apply_document_lifecycle(old_name, connection=connection,
        new_document_id=new_name, mutation_id=uuid4().hex)
    if connection.execute("SELECT to_regclass('learning_sentences') AS table_name").fetchone()['table_name'] is None:
        return
    scope = preferences.scope
    state = codec.empty_state(scope, old_name)
    # Explicit whole-document identity migration, five set queries, not the saved
    # read path. All levels/history must move; no N+1 or per-row commits.
    for table, field, identity in (
        ('learning_sentences','sentences','resource_id'),
        ('learning_cache_entries','entries','cache_id'),
        ('learning_operations','operations','operation_id'),
        ('learning_commands','receipts','command_id')):
        for row in connection.execute(f'SELECT * FROM {table} WHERE document_row_id=%s', (row_id,)):
            state[field][row[identity]] = learning_rows.decoded(row, field)
    for row in connection.execute('SELECT * FROM learning_checkpoints WHERE document_row_id=%s '
                                  'ORDER BY operation_id,batch_index', (row_id,)):
        ident = row['operation_id']
        parts = state['checkpoints'].get(ident, ())
        if row['batch_index'] != len(parts):
            raise ValueError('learning_checkpoint_index_invalid')
        state['checkpoints'][ident] = (*parts, learning_rows.decoded(row,'checkpoints'))
    migrated = rename_learning_state(state, scope, old_name, new_name, now_ms=int(time()*1000))
    # Child-first replacement under the same document lock and transaction.
    # Logical identity changes all key/resource hashes; payload text is preserved.
    for table in ('learning_checkpoints','learning_commands','learning_operations',
                  'learning_cache_entries','learning_sentences'):
        connection.execute(f'DELETE FROM {table} WHERE document_row_id=%s', (row_id,))
    learning_rows.persist(connection, row_id, scope, new_name, migrated, codec.empty_state(scope,new_name))


def delete_preferences(connection, preferences, document_id):
    if preferences is not None:
        preferences.apply_document_lifecycle(document_id, connection=connection, mutation_id=uuid4().hex)
    # Document FK cascades remove all learning rows in the caller's transaction.
