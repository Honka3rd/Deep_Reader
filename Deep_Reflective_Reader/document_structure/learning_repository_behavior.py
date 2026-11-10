"""Shared learning repository behavior; storage adapters supply atomic contexts.

_read_context(document_id, **selection) yields (document, validated state).
_write_context yields (workspace, book, blocks, state, persist), holding source and
settings guards until persist and the surrounding commit complete. Selections
bound SQL history reads; file storage may load its existing document sidecar.
No transactions, provider work, SQL or filesystem access live here.
"""
from hashlib import sha256
from reading_preferences.models import PreferenceError
from document_structure import learning_file_codec as codec
from document_structure.learning_repository import (MAX_READ_ITEMS, OperationWrite, OperationSnapshot, SavedLearning, SavedTranslation, TranslationPage,
    SentenceRead, SourceCursor, SourcePage)
from document_structure.learning_source_resolver import current_binding, learning_blocks
from document_structure.reading_content import source_token_for, state_for
from section_tasks.learning_cache import HintsKey, TranslationKey, validate_complete_result, validate_result_key
from section_tasks import learning_generation_policy as policy
from shared.learning_source import LearningContractError, SentenceResource, SentenceResourceRef, integer, require, token


class LearningRepositoryBehavior:

    def _scope_key(self, document_id, key):
        require(type(key) in (TranslationKey, HintsKey), 'learning_key_invalid')
        self._scope_ref(document_id, key.resource)

    def _scope_ref(self, document_id, reference):
        require(type(reference) is SentenceResourceRef, 'learning_reference_invalid')
        source = reference.source
        require((source.scope_kind, source.namespace, source.document_id)
                == (self.scope.kind, self.scope.namespace, document_id), 'learning_scope_mismatch')

    @staticmethod
    def _bound(values):
        require(type(values) is tuple and len(values) <= MAX_READ_ITEMS, 'learning_read_batch_invalid')

    @staticmethod
    def _page(offset, limit):
        integer(offset)
        integer(limit, minimum=1)
        require(limit <= MAX_READ_ITEMS, 'learning_read_limit_invalid')

    @staticmethod
    def _block(blocks, source):
        block = blocks.get(source.content_block_id)
        require(block is not None and block.source == source, 'learning_source_changed')
        return block

    def read_sources(self, document_id, *, limit=50, cursor=None):
        self._page(0, limit)
        with self._read_context(document_id, source_only=True) as (document, state):
            source_token = sha256(codec.canonical([self.scope.namespace, source_token_for(document),
                state_for(document).get('revision', 0)]).encode()).hexdigest()
            offset = 0
            if cursor is not None:
                require(type(cursor) is SourceCursor and cursor.document_id == document_id
                        and cursor.source_token == source_token, 'learning_cursor_stale')
                integer(cursor.offset)
                offset = cursor.offset
            blocks = learning_blocks(document, self.scope, document_id)
            require(offset <= len(blocks), 'learning_cursor_invalid')
            next_offset = offset + limit
            return SourcePage(blocks[offset:next_offset], SourceCursor(document_id, source_token, next_offset)
                              if next_offset < len(blocks) else None)

    def resolve_blocks(self, document_id, block_ids):
        """One canonical snapshot for a bounded current-page selection; no pagination scan."""
        self._bound(block_ids)
        require(len(set(block_ids)) == len(block_ids), 'learning_blocks_duplicate')
        for ident in block_ids:
            token(ident)
        with self._read_context(document_id, source_only=True) as (document, _):
            blocks = {b.source.content_block_id: b for b in learning_blocks(document, self.scope, document_id)}
            require(all(ident in blocks for ident in block_ids), 'learning_block_missing')
            return tuple(blocks[ident] for ident in block_ids)

    def inspect_operation(self, document_id, operation_id):
        token(operation_id)
        with self._read_context(document_id, operation_id=operation_id, include_document=False) as (_, state):
            op = state['operations'].get(operation_id)
            if op is None:
                return None
            entry = state['entries'].get(op.binding.key.cache_id)
            current = entry is not None and entry.operation is not None and entry.operation.operation_id == operation_id
            return OperationSnapshot(op, entry.revision if current else None)

    def read_sentences(self, document_id, references):
        self._bound(references)
        for reference in references:
            self._scope_ref(document_id, reference)
        with self._read_context(document_id, references=references) as (document, state):
            blocks = {b.source.content_block_id: b for b in learning_blocks(document, self.scope, document_id)}
            result = []
            for reference in references:
                block = blocks.get(reference.source.content_block_id)
                if block is None or block.source != reference.source:
                    result.append(SentenceRead('stale', None))
                    continue
                sentences = state['sentences'].get(reference.resource_id)
                if sentences is not None:
                    require(sentences.reference == reference, 'learning_reference_mismatch')
                    sentences.validate_source(block.source, block.effective_text)
                result.append(SentenceRead('ready' if sentences is not None else 'missing', sentences))
            return tuple(result)

    def read_batch(self, document_id, keys):
        self._bound(keys)
        for key in keys:
            self._scope_key(document_id, key)
        with self._read_context(document_id, keys=keys) as (document, state):
            blocks = {b.source.content_block_id: b for b in learning_blocks(document, self.scope, document_id)}
            result = []
            for key in keys:
                entry = state['entries'].get(key.cache_id, policy.CacheEntry(key))
                block = blocks.get(key.resource.source.content_block_id)
                if block is None or block.source != key.resource.source:
                    result.append(SavedLearning(key, policy.read_entry(entry, evidence='stale'), None))
                    continue
                sentences = state['sentences'].get(key.resource.resource_id)
                if sentences is not None:
                    sentences.validate_source(block.source, block.effective_text)
                    require(sentences.reference == key.resource, 'learning_reference_mismatch')
                if entry.saved is not None:
                    validate_complete_result(key, entry.saved, sentences, block.effective_text)
                result.append(SavedLearning(key, policy.read_entry(entry), sentences))
            return tuple(result)

    def save_sentences(self, document_id, sentences):
        require(type(sentences) is SentenceResource, 'learning_resource_invalid')
        self._scope_ref(document_id, sentences.reference)
        with self._write_context(document_id, references=(sentences.reference,)) as (_, _, blocks, state, persist):
            block = self._block(blocks, sentences.source)
            sentences.validate_source(block.source, block.effective_text)
            resource_id = sentences.reference.resource_id
            old = state['sentences'].get(resource_id)
            require(old is None or old == sentences, 'learning_resource_conflict')
            if old is None:
                state['sentences'][resource_id] = sentences
                persist(state)
            return sentences

    def read_translations(self, document_id, block_id, *, after='', limit=50):
        """Bounded complete translations; no preference, provider or write dependency."""
        token(block_id)
        require(type(after) is str, 'learning_cursor_invalid')
        self._page(0, limit)
        with self._read_context(document_id, translation_block=block_id,
                                translation_after=after, translation_limit=limit) as (document, state):
            blocks = {b.source.content_block_id: b for b in learning_blocks(document, self.scope, document_id)}
            require(block_id in blocks, 'learning_block_missing')
            block = blocks[block_id]
            entries = sorted((entry for ident, entry in state['entries'].items()
                if ident > after and type(entry.key) is TranslationKey and entry.saved is not None
                and entry.key.resource.source.content_block_id == block_id), key=lambda e: e.key.cache_id)[:limit]
            records = []
            for entry in entries:
                key = entry.key
                if key.resource.source != block.source:
                    continue
                sentences = state['sentences'].get(key.resource.resource_id)
                validate_complete_result(key, entry.saved, sentences, block.effective_text)
                operation = next((op for op in state['operations'].values()
                    if op.binding.key == key and op.state == 'succeeded'
                    and op.result_revision == entry.result_revision), None)
                require(operation is not None, 'learning_receipt_invalid')
                records.append(SavedTranslation(SavedLearning(key, policy.read_entry(entry), sentences), operation))
            return TranslationPage(tuple(records), entries[-1].key.cache_id if len(entries) == limit else None)

    def claim(self, document_id, command, *, sentences, plan, now_ms, lease_until_ms, validate_binding=None):
        require(type(command) is policy.GenerationCommand, 'learning_command_invalid')
        self._scope_key(document_id, command.binding.key)
        with self._write_context(document_id, keys=(command.binding.key,), command_id=command.command_id) as (workspace, book, blocks, state, persist):
            key = command.binding.key
            entry = state['entries'].get(key.cache_id, policy.CacheEntry(key))
            receipt = state['receipts'].get(command.command_id)
            if receipt is not None:
                require(receipt.command == command, 'learning_command_conflict')
                return policy.RequestDecision(receipt.outcome, entry.revision, None, receipt, replay=True)
            block = self._block(blocks, key.resource.source)
            binding = current_binding(key, workspace, book, purpose=command.binding.active_mode)
            if validate_binding is not None:
                validate_binding(command.binding, workspace, book)
            decision = policy.plan_request(entry, command, current=binding, sentences=sentences,
                effective_text=block.effective_text, plan=plan, now_ms=now_ms, lease_until_ms=lease_until_ms,
                admission=self._admission)
            if decision.next_entry is not None:
                state['sentences'][sentences.reference.resource_id] = sentences
                state['entries'][key.cache_id] = decision.next_entry
                state['operations'][decision.next_entry.operation.operation_id] = decision.next_entry.operation
            state['receipts'][command.command_id] = decision.receipt
            persist(state)
            return decision

    def write_operation(self, document_id, command, *, validate_binding=None):
        require(type(command) is OperationWrite, 'learning_operation_command_invalid')
        with self._write_context(document_id, operation_id=command.operation_id) as (workspace, book, blocks, state, persist):
            op = state['operations'].get(command.operation_id)
            require(op is not None, 'learning_operation_missing')
            key = op.binding.key
            entry = state['entries'][key.cache_id]
            require(entry.revision == command.expected_entry_revision, 'learning_entry_revision_conflict')
            require(entry.operation.operation_id == command.operation_id and entry.last_fence == command.fence,
                    'learning_fence_conflict')
            args = dict(operation_id=command.operation_id, fence=command.fence, now_ms=command.now_ms)
            # Expiry/failure record observations even if settings/source became stale;
            # they neither authorize provider IO nor install new generated results.
            if command.action == 'invalidate':
                is_current = True
                try:
                    self._block(blocks, key.resource.source)
                    require(current_binding(key, workspace, book, purpose=op.binding.active_mode) == op.binding, 'learning_binding_stale')
                    if validate_binding is not None:
                        validate_binding(op.binding, workspace, book)
                except (LearningContractError, PreferenceError):
                    is_current = False
                transition = policy.mark_binding_stale(entry, binding_is_current=is_current, **args)
            elif command.action == 'expire':
                transition = policy.expire_operation(entry, **args)
            elif command.action == 'fail':
                transition = policy.finish_failure(entry, reason=command.reason,
                    outcome_known=command.outcome_known, **args)
            else:
                block = self._block(blocks, key.resource.source)
                binding = current_binding(key, workspace, book, purpose=op.binding.active_mode)
                sentences = state['sentences'][key.resource.resource_id]
                sentences.validate_source(block.source, block.effective_text)
                if validate_binding is not None:
                    validate_binding(op.binding, workspace, book)
                args['current'] = binding
                if command.action == 'dispatch':
                    # Recheck the same centralized restriction policy before every
                    # paid batch, using its original durable command (no new quota).
                    command_receipt = state['receipts'][op.operation_id]
                    decision = (self._admission or policy.AllowValidGeneration()).evaluate(command_receipt.command)
                    require(type(decision) is policy.AdmissionDecision, 'learning_admission_invalid')
                    if not decision.allowed:
                        raise policy.LearningAdmissionDenied(decision.reason)
                    transition = policy.dispatch_batch(entry, **args)
                elif command.action == 'renew':
                    transition = policy.renew_lease(entry, lease_until_ms=command.lease_until_ms, **args)
                elif command.action == 'checkpoint':
                    command.batch.validate(op, command.batch_index, sentences, block.effective_text)
                    transition = policy.checkpoint_batch(entry, batch_index=command.batch_index,
                        checkpoint_digest=codec.payload_digest(command.batch), **args)
                    state['checkpoints'][op.operation_id] = (*state['checkpoints'].get(op.operation_id, ()), command.batch)
                elif command.action == 'commit':
                    validate_result_key(key, command.result)
                    parts = state['checkpoints'].get(op.operation_id, ())
                    require(all(p.provenance == command.result.context.provenance for p in parts),
                            'learning_result_provenance_mismatch')
                    expected = tuple(row for p in parts for row in (p.items if type(key) is HintsKey else p.rows))
                    actual = command.result.items if type(key) is HintsKey else command.result.rows
                    if type(key) is HintsKey:
                        from shared.learning_hint_assessment import combine_assessments
                        require(command.result.assessment == combine_assessments(p.assessment for p in parts),
                                'learning_result_assessment_mismatch')
                    require(actual == expected, 'learning_checkpoint_result_mismatch')
                    transition = policy.commit_result(entry, result=command.result, sentences=sentences,
                        effective_text=block.effective_text, **args)
            updated = transition.next_entry
            state['entries'][key.cache_id] = updated
            state['operations'][op.operation_id] = updated.operation
            persist(state)
            return updated

    def lookup_operation(self, document_id, operation_id):
        token(operation_id)
        with self._read_context(document_id, operation_id=operation_id, include_document=False) as (_, state):
            return state["operations"].get(operation_id)

    def lookup_command(self, document_id, command_id):
        token(command_id)
        with self._read_context(document_id, command_id=command_id, include_document=False) as (_, state):
            return state["receipts"].get(command_id)

    def read_checkpoints(self, document_id, operation_id, *, offset=0, limit=50):
        token(operation_id)
        self._page(offset, limit)
        with self._read_context(document_id, operation_id=operation_id) as (document, state):
            require(operation_id in state['operations'], 'learning_operation_missing')
            op = state['operations'][operation_id]
            blocks = {b.source.content_block_id: b for b in learning_blocks(document, self.scope, document_id)}
            block = self._block(blocks, op.binding.key.resource.source)
            sentences = state['sentences'][op.binding.key.resource.resource_id]
            sentences.validate_source(block.source, block.effective_text)
            parts = state['checkpoints'].get(operation_id, ())[offset:offset + limit]
            for index, part in enumerate(parts, offset):
                part.validate(op, index, sentences, block.effective_text)
            return parts
