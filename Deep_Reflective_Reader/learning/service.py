"""LLRN-13 explicit learning use cases. No routes, prompts, SQL or implicit reads."""
from time import time
import asyncio

from .context import LearningContext, LearningSubmission
from .worker import LearningWorker
from document_structure.learning_repository import LearningCommitUnknown, OperationSnapshot
from section_tasks.learning_generation_policy import GenerationBinding, GenerationCommand
from section_tasks.learning_sentence_segmenter import segment_learning_block
from shared.learning_source import LearningContractError, require, integer


class LearningService:
    def __init__(self, repository, preferences, translation, hints, *, lease_ms, max_workers=4,
                 clock=None):
        self.context = LearningContext(repository, preferences, translation, hints)
        self.repository, self.preferences = repository, preferences
        self.clock = clock or (lambda: int(time()*1000))
        self.worker = LearningWorker(self.context, clock=self.clock, lease_ms=lease_ms, max_workers=max_workers)

    def shutdown(self, *, wait=True):
        """Application lifecycle, not browser lifecycle. Accepted jobs are drained."""
        self.worker.shutdown(wait=wait)

    def record_start_failure(self, document_id, snapshot):
        """Record an acknowledged claim rejected before executor handoff; never retry."""
        self.worker._failure(document_id, snapshot, reason='worker_stopped', known=True)

    def read(self, document_id, block_ids):
        return self.context.read(document_id, block_ids)

    def read_book_preferences(self, document_id):
        """Require a canonical existing book before exposing its settings."""
        self.repository.read_sources(document_id, limit=1)
        return self.preferences.read_book(document_id)

    def source_binding(self, document_id, block_id):
        """Server-resolved identity for comparison with opaque transport tokens."""
        block, = self.repository.resolve_blocks(document_id, (block_id,))
        return block.source

    def status(self, document_id, operation_id):
        """Read-only status: no expiry, repair, scheduling or provider calls."""
        return self.context.status(document_id, operation_id)

    async def wait_for_completion(self, document_id, operation_id, *, timeout=120):
        """At most two storage reads; notification/timeout never executes recovery.

        Subscribe before reading to close the completion race. Other processes
        and pre-restart jobs have no local notifier; the bounded wait still ends
        with current durable state, without pretending the job failed.
        """
        completion = self.worker.completion
        future = completion.subscribe(document_id, operation_id)
        try:
            current = await asyncio.to_thread(self.status, document_id, operation_id)
            if (current.state not in ('claimed', 'running', 'partial')
                    or not current.source_current or not current.settings_current
                    or current.snapshot is None or current.snapshot.entry_revision is None):
                return current
            try:
                await asyncio.wait_for(asyncio.shield(asyncio.wrap_future(future)), timeout)
            except asyncio.TimeoutError:
                pass
            return await asyncio.to_thread(self.status, document_id, operation_id)
        finally:
            completion.unsubscribe(document_id, operation_id, future)

    def lookup_command(self, document_id, command_id):
        """Read-only reconciliation after a response/claim acknowledgement was lost."""
        return self.repository.lookup_command(document_id, command_id)

    @staticmethod
    def _submission(decision):
        receipt = decision.receipt
        return LearningSubmission(decision.outcome, receipt.command.command_id, receipt.operation_id,
                                  decision.replay, receipt.admission_reason)

    def request(self, document_id, block_id, component, *, command_id, expected_source,
                expected_workspace_revision, expected_book_revision, intent='get_or_generate',
                retry_operation_id=None, acknowledge_possible_duplicate_cost=False):
        """Accept one explicit component command, owned by the backend after claim.

        Client expectations are compared with actual source/settings; they cannot
        supply languages, raw text, key versions or hierarchy ownership.
        """
        integer(expected_workspace_revision)
        integer(expected_book_revision)
        settings, block, sentences = self.context.selection(document_id, block_id)
        require(block.source == expected_source, 'learning_source_changed')
        require((settings.workspace.revision, settings.book.revision)
                == (expected_workspace_revision, expected_book_revision), 'learning_settings_changed')
        key = self.context.key(component, settings, sentences)
        binding = GenerationBinding(key, settings.workspace.revision, settings.book.revision)
        command = GenerationCommand(command_id, binding, intent, retry_operation_id, acknowledge_possible_duplicate_cost)
        prior = self.repository.lookup_command(document_id, command_id)
        if prior is not None:
            require(prior.command == command, 'learning_command_conflict')
            return LearningSubmission(prior.outcome, command_id, prior.operation_id, True, prior.admission_reason)
        saved, = self.repository.read_batch(document_id, (key,))
        # A stale operation can be explicitly retried against the current
        # binding; stale source evidence cannot authorize any generation.
        require(saved.view.state not in ('read_error', 'unsupported', 'unavailable')
                and (saved.view.state != 'stale' or saved.view.operation_state == 'stale'),
                'learning_read_not_verified')
        reuse = intent == 'get_or_generate' and saved.view.saved is not None
        reserved, handed_off = False, False
        try:
            # Coalesced/failed/unknown work must remain inspectable even when
            # all local worker slots are busy or the provider is unavailable.
            if saved.view.operation_state in ('claimed', 'running', 'partial', 'unknown', 'failed', 'stale') and intent != 'retry':
                now = self.clock()
                try:
                    decision = self.repository.claim(document_id, command, sentences=sentences, plan=None,
                        now_ms=now, lease_until_ms=now+self.worker.lease_ms,
                        validate_binding=self.context.validate_binding)
                    return self._submission(decision)
                except LearningContractError as error:
                    # A concurrent worker completed a refresh between snapshots;
                    # regenerate now needs preparation. No claim was written.
                    if str(error) != 'learning_batches_invalid':
                        raise
            prepared = None
            if not reuse and sentences.sentences:
                self.worker.reserve()
                reserved = True
                prepared = self.context.prepare(component, settings, block, sentences)
            now = self.clock()
            decision = self.repository.claim(document_id, command, sentences=sentences,
                plan=prepared.plan if prepared else None, now_ms=now, lease_until_ms=now+self.worker.lease_ms,
                validate_binding=self.context.validate_binding)
            if decision.next_entry is not None:
                entry = decision.next_entry
                snapshot = OperationSnapshot(entry.operation, entry.revision)
                try:
                    self.worker.start(document_id, prepared, snapshot)
                    handed_off = True
                except Exception:
                    # Claim committed, but no executor accepted it. Record this
                    # known pre-dispatch failure; do not silently lose ownership.
                    self.record_start_failure(document_id, snapshot)
                    raise
            return self._submission(decision)
        except LearningCommitUnknown:
            # Even if a read finds claimed/running, it is not dispatch permission.
            return LearningSubmission('unknown', command_id, None)
        finally:
            if reserved and not handed_off:
                self.worker.release()

    def reconcile(self, document_id, operation_id):
        """Explicit recovery; never re-send an outstanding/unknown paid batch.

        Safe claimed/partial work may resume via a NEW acknowledged dispatch CAS.
        Completed durable checkpoints may assemble/commit without a provider call.
        """
        status = self.status(document_id, operation_id)
        snapshot = status.snapshot
        if snapshot is None:
            return LearningSubmission('missing', operation_id, None)
        op = snapshot.operation
        if snapshot.entry_revision is None:
            return LearningSubmission('superseded', operation_id, operation_id)
        try:
            if status.state == 'stale' and op.state in ('claimed', 'running', 'partial', 'unknown'):
                self.worker.write(document_id, snapshot, 'invalidate')
                return LearningSubmission('stale', operation_id, operation_id)
            if op.state not in ('claimed', 'running', 'partial'):
                return LearningSubmission(status.state, operation_id, operation_id)
            complete = len(op.checkpoints) == len(op.plan.groups)
            if not complete and self.clock() >= op.lease_until_ms:
                updated = self.worker.write(document_id, snapshot, 'expire')
                return LearningSubmission(updated.operation.state, operation_id, operation_id)
            if op.state == 'running':
                return LearningSubmission('reconcile', operation_id, operation_id)
            prepared = None
            if not complete:
                key = op.binding.key
                if op.binding.active_mode == 'translation':
                    block, = self.repository.resolve_blocks(document_id, (key.resource.source.content_block_id,))
                    sentences = segment_learning_block(source=block.source, effective_text=block.effective_text,
                                                       source_language=key.source_language)
                    prepared = self.context.services['translation'].prepare(sentences=sentences,
                        effective_text=block.effective_text, source_language=key.source_language,
                        explanation_language=key.explanation_language)
                else:
                    settings, block, sentences = self.context.selection(document_id, key.resource.source.content_block_id)
                    prepared = self.context.prepare(key.component, settings, block, sentences)
                require(block.source == op.binding.key.resource.source, 'learning_source_changed')
                require(prepared is not None and prepared.key == op.binding.key and prepared.plan == op.plan,
                        'learning_recovery_plan_changed')
            self.worker.reserve()
            try:
                self.worker.start(document_id, prepared, snapshot)
            except Exception:
                self.worker.release()
                raise
            return LearningSubmission('resuming', operation_id, operation_id)
        except LearningCommitUnknown:
            return LearningSubmission('unknown', operation_id, operation_id)

    def read_partial(self, document_id, operation_id, *, offset=0, limit=50):
        """Bounded validated durable batch payloads, never a fabricated complete result."""
        return self.read_partial_snapshot(document_id, operation_id, offset=offset, limit=limit)[1]

    def read_partial_snapshot(self, document_id, operation_id, *, offset=0, limit=50):
        """Return coherent status/batches, rejecting concurrent checkpoint/settings changes.

        Both observations are read-only; no locks across requests or repair/retry.
        This is optimistic consistency, not a cross-profile/storage transaction.
        """
        status = self.status(document_id, operation_id)
        require(status.snapshot is not None and status.source_current and status.settings_current,
                'learning_binding_stale')
        batches = self.repository.read_checkpoints(document_id, operation_id, offset=offset, limit=limit)
        require(self.status(document_id, operation_id) == status, 'learning_operation_conflict')
        return status, batches
