"""Transaction-local typed learning rows; bounded dependency loading, no commits.

Only requested cache entries and their current/saved/historically requested
operations are loaded. No scan of all commands, versions, levels or book history.
"""
from psycopg.types.json import Jsonb

from document_structure import learning_file_codec as codec
from document_structure.learning_repository import LearningStorageError
from shared.learning_source import require


def decoded(row, identity):
    try:
        require(row['schema_version'] == 1, 'learning_schema_unsupported')
        value = codec.decode(row['payload'])
        if identity == 'sentences':
            require(value.reference.resource_id == row['resource_id'], 'learning_resource_invalid')
        elif identity == 'entries':
            require((value.key.cache_id, value.key.resource.resource_id, value.revision,
                     value.last_fence, value.result_revision) ==
                    (row['cache_id'], row['resource_id'], row['revision'], row['last_fence'],
                     row['result_revision']), 'learning_entry_invalid')
        elif identity == 'operations':
            require((value.operation_id, value.binding.key.cache_id, value.fence, value.result_revision) ==
                    (row['operation_id'], row['cache_id'], row['fence'], row['result_revision']),
                    'learning_operation_invalid')
        elif identity == 'receipts':
            require((value.command.command_id, value.operation_id) ==
                    (row['command_id'], row['operation_id']), 'learning_receipt_invalid')
        return value
    except (ValueError, TypeError, KeyError, AttributeError, RecursionError):
        raise LearningStorageError('learning_data_invalid_or_unsupported') from None


def load_state(connection, row_id, scope, document_id, *, keys=(), references=(),
               operation_id=None, command_id=None, translation_block=None,
               translation_after='', translation_limit=50, **unused):
    state = codec.empty_state(scope, document_id)
    cache_ids = {key.cache_id for key in keys}
    if translation_block is not None:
        # Only one block's complete translation keys, never its hints or whole history.
        rows = connection.execute("SELECT cache_id FROM learning_cache_entries WHERE document_row_id=%s "
            "AND cache_id > %s AND result_revision > 0 AND payload #>> '{fields,key,type}' = 'TranslationKey' "
            "AND payload #>> '{fields,key,fields,resource,fields,source,fields,content_block_id}' = %s "
            "ORDER BY cache_id LIMIT %s", (row_id, translation_after, translation_block, translation_limit)).fetchall()
        cache_ids.update(row['cache_id'] for row in rows)
    op_ids = set()
    if command_id is not None:
        row = connection.execute('SELECT * FROM learning_commands WHERE document_row_id=%s AND command_id=%s',
                                 (row_id, command_id)).fetchone()
        if row is not None:
            receipt = decoded(row, 'receipts')
            state['receipts'][command_id] = receipt
            cache_ids.add(receipt.command.binding.key.cache_id)
            if receipt.operation_id is not None:
                op_ids.add(receipt.operation_id)
    if operation_id is not None:
        op_ids.add(operation_id)
    if op_ids:
        rows = connection.execute('SELECT * FROM learning_operations WHERE document_row_id=%s AND operation_id=ANY(%s)',
                                  (row_id, list(op_ids))).fetchall()
        for row in rows:
            op = decoded(row, 'operations')
            state['operations'][op.operation_id] = op
            cache_ids.add(op.binding.key.cache_id)
    if cache_ids:
        rows = connection.execute('SELECT * FROM learning_cache_entries WHERE document_row_id=%s AND cache_id=ANY(%s)',
                                  (row_id, list(cache_ids))).fetchall()
        for row in rows:
            entry = decoded(row, 'entries')
            state['entries'][entry.key.cache_id] = entry
            if entry.operation is not None:
                op_ids.add(entry.operation.operation_id)
        # At most current operation + successful saved-result operation per key,
        # plus the one historical operation explicitly requested by the caller.
        rows = connection.execute('SELECT o.* FROM learning_operations o '
            'WHERE o.document_row_id=%s AND o.operation_id=ANY(%s) UNION ALL '
            'SELECT o.* FROM learning_cache_entries e JOIN learning_operations o '
            'ON o.document_row_id=e.document_row_id AND o.cache_id=e.cache_id '
            'AND o.result_revision=e.result_revision '
            'WHERE e.document_row_id=%s AND e.cache_id=ANY(%s) AND NOT o.operation_id=ANY(%s)',
            (row_id, list(op_ids), row_id, list(cache_ids), list(op_ids))).fetchall()
        for row in rows:
            state['operations'][row['operation_id']] = decoded(row, 'operations')
    resource_ids = {ref.resource_id for ref in references}
    resource_ids.update(key.resource.resource_id for key in keys)
    resource_ids.update(entry.key.resource.resource_id for entry in state['entries'].values())
    resource_ids.update(op.binding.key.resource.resource_id for op in state['operations'].values())
    if resource_ids:
        rows = connection.execute('SELECT * FROM learning_sentences WHERE document_row_id=%s AND resource_id=ANY(%s)',
                                  (row_id, list(resource_ids))).fetchall()
        for row in rows:
            state['sentences'][row['resource_id']] = decoded(row, 'sentences')
    if state['operations']:
        ids = list(state['operations'])
        for row in connection.execute('SELECT * FROM learning_commands WHERE document_row_id=%s AND command_id=ANY(%s)',
                                      (row_id, ids)).fetchall():
            state['receipts'][row['command_id']] = decoded(row, 'receipts')
        for row in connection.execute('SELECT * FROM learning_checkpoints WHERE document_row_id=%s '
                                      'AND operation_id=ANY(%s) ORDER BY operation_id,batch_index', (row_id, ids)).fetchall():
            parts = state['checkpoints'].get(row['operation_id'], ())
            if row['batch_index'] != len(parts):
                raise LearningStorageError('learning_checkpoint_index_invalid')
            state['checkpoints'][row['operation_id']] = (*parts, decoded(row, 'checkpoints'))
    try:
        codec.validate_state(state, scope, document_id)
    except (ValueError, TypeError, KeyError, AttributeError, RecursionError):
        raise LearningStorageError('learning_data_invalid_or_unsupported') from None
    return state


def persist(connection, row_id, scope, document_id, state, before):
    """Write only changed rows using the caller's document/preference locks."""
    codec.validate_state(state, scope, document_id)
    for ident, value in state['sentences'].items():
        if before['sentences'].get(ident) != value:
            connection.execute('INSERT INTO learning_sentences(document_row_id,resource_id,payload) VALUES (%s,%s,%s)',
                               (row_id, ident, Jsonb(codec.encode(value))))
    for ident, value in state['entries'].items():
        if before['entries'].get(ident) != value:
            connection.execute('INSERT INTO learning_cache_entries '
                '(document_row_id,cache_id,resource_id,revision,last_fence,result_revision,payload) VALUES (%s,%s,%s,%s,%s,%s,%s) '
                'ON CONFLICT(document_row_id,cache_id) DO UPDATE SET revision=EXCLUDED.revision, '
                'last_fence=EXCLUDED.last_fence,result_revision=EXCLUDED.result_revision,payload=EXCLUDED.payload',
                (row_id, ident, value.key.resource.resource_id, value.revision, value.last_fence,
                 value.result_revision, Jsonb(codec.encode(value))))
    for ident, value in state['operations'].items():
        if before['operations'].get(ident) != value:
            connection.execute('INSERT INTO learning_operations '
                '(document_row_id,operation_id,cache_id,fence,result_revision,payload) VALUES (%s,%s,%s,%s,%s,%s) '
                'ON CONFLICT(document_row_id,operation_id) DO UPDATE SET '
                'result_revision=EXCLUDED.result_revision,payload=EXCLUDED.payload',
                (row_id, ident, value.binding.key.cache_id, value.fence, value.result_revision, Jsonb(codec.encode(value))))
    for ident, value in state['receipts'].items():
        if before['receipts'].get(ident) != value:
            connection.execute('INSERT INTO learning_commands(document_row_id,command_id,operation_id,payload) '
                               'VALUES (%s,%s,%s,%s)', (row_id, ident, value.operation_id, Jsonb(codec.encode(value))))
    for ident, parts in state['checkpoints'].items():
        previous = before['checkpoints'].get(ident, ())
        require(parts[:len(previous)] == previous, 'learning_checkpoint_immutable')
        for index in range(len(previous), len(parts)):
            connection.execute('INSERT INTO learning_checkpoints(document_row_id,operation_id,batch_index,payload) '
                               'VALUES (%s,%s,%s,%s)', (row_id, ident, index, Jsonb(codec.encode(parts[index]))))
