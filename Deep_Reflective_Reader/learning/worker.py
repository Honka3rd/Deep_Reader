"""Backend-owned execution; durable dispatch CAS is the only provider permission."""
from concurrent.futures import ThreadPoolExecutor
import logging
from threading import BoundedSemaphore, Lock
from time import monotonic
from .completion import LearningCompletion

from document_structure.learning_repository import (
    LearningCommitUnknown, LearningStorageError, OperationSnapshot, OperationWrite, MAX_READ_ITEMS,
)
from section_tasks.learning_generation_policy import LearningAdmissionDenied
from section_tasks.learning_assembly import assemble_checkpoints
from section_tasks.learning_translation_service import LearningTranslationError
from section_tasks.learning_hints_output import LearningHintsError
from shared.learning_source import LearningContractError, integer

logger = logging.getLogger(__name__)


class LearningBusy(RuntimeError):
    pass


class LearningWorker:
    def __init__(self, context, *, clock, lease_ms, max_workers):
        integer(lease_ms, minimum=1)
        integer(max_workers, minimum=1)
        self.context, self.clock, self.lease_ms = context, clock, lease_ms
        self._slots = BoundedSemaphore(max_workers)
        self._lock = Lock()
        self._closed = False
        self.completion = LearningCompletion()
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix='learning')

    def reserve(self):
        with self._lock:
            if self._closed or not self._slots.acquire(blocking=False):
                raise LearningBusy('learning_worker_busy')

    def release(self):
        self._slots.release()

    def start(self, document_id, prepared, snapshot):
        # Caller owns the reservation until successful submission. Futures are
        # deliberately private: HTTP cancellation cannot cancel accepted work.
        with self._lock:
            if self._closed:
                raise LearningBusy('learning_worker_closed')
            self._pool.submit(self._run, document_id, prepared, snapshot)

    def shutdown(self, *, wait=True):
        with self._lock:
            self._closed = True
        self._pool.shutdown(wait=wait, cancel_futures=False)

    def write(self, document_id, snapshot, action, **fields):
        op = snapshot.operation
        entry = self.context.repository.write_operation(document_id, OperationWrite(
            op.operation_id, snapshot.entry_revision, op.fence, self.clock(), action, **fields),
            validate_binding=self.context.validate_binding)
        return OperationSnapshot(entry.operation, entry.revision)

    def _failure(self, document_id, snapshot, *, reason, known):
        # Best-effort observation. Losing CAS never overwrites another worker;
        # storage errors remain uncertain in durable state, not false success.
        try:
            self.write(document_id, snapshot, 'fail', reason=reason, outcome_known=known)
        except Exception:
            logger.warning('learning_worker stage=record_failure outcome=unconfirmed')

    def _batches(self, document_id, op):
        parts = []
        while len(parts) < len(op.checkpoints):
            page = self.context.repository.read_checkpoints(document_id, op.operation_id,
                offset=len(parts), limit=min(MAX_READ_ITEMS, len(op.checkpoints)-len(parts)))
            if not page:
                raise LearningContractError('learning_checkpoint_missing')
            parts.extend(page)
        return tuple(parts)

    def _run(self, document_id, prepared, snapshot):
        started, outcome = monotonic(), 'stopped'
        try:
            service = self.context.services[snapshot.operation.binding.key.component]
            while True:
                op = snapshot.operation
                if len(op.checkpoints) == len(op.plan.groups):
                    batches = self._batches(document_id, op)
                    if prepared is None:
                        key = op.binding.key
                        block, = self.context.repository.resolve_blocks(document_id, (key.resource.source.content_block_id,))
                        stored, = self.context.repository.read_sentences(document_id, (key.resource,))
                        if stored.state != 'ready':
                            raise LearningContractError('learning_source_changed')
                        result = assemble_checkpoints(op, stored.sentences, block.effective_text, batches)
                    else:
                        result = service.assemble(prepared, op, batches)
                    snapshot = self.write(document_id, snapshot, 'commit', result=result)
                    outcome = 'succeeded'
                    return
                now = self.clock()
                if now >= op.lease_until_ms:
                    self.write(document_id, snapshot, 'expire')
                    outcome = 'expired'
                    return
                if now + self.lease_ms > op.lease_until_ms:
                    snapshot = self.write(document_id, snapshot, 'renew', lease_until_ms=now+self.lease_ms)
                snapshot = self.write(document_id, snapshot, 'dispatch')
                try:
                    batch = service.generate_batch(prepared, snapshot.operation, now_ms=self.clock())
                except (LearningTranslationError, LearningHintsError) as error:
                    known = error.outcome_known
                    reason = ('invalid_output' if 'invalid_output' in error.code else 'provider_rejected') if known else 'provider_outcome_unknown'
                    self._failure(document_id, snapshot, reason=reason, known=known)
                    outcome = 'failed' if known else 'unknown'
                    return
                snapshot = self.write(document_id, snapshot, 'checkpoint',
                    batch_index=len(snapshot.operation.checkpoints), batch=batch)
        except (LearningCommitUnknown, LearningStorageError):
            # Never replay a dispatch or paid call after an unconfirmed write.
            outcome = 'storage_unconfirmed'
        except LearningAdmissionDenied:
            self._failure(document_id, snapshot, reason='worker_stopped', known=True)
            outcome = 'denied'
        except LearningContractError:
            try:
                self.write(document_id, snapshot, 'invalidate')
                outcome = 'stale'
            except Exception:
                # CAS winner or a still-current binding; no inferred takeover.
                outcome = 'conflict'
        except Exception:
            # Unknown failures after dispatch cannot establish provider outcome.
            running = snapshot.operation.state == 'running'
            self._failure(document_id, snapshot,
                reason='provider_outcome_unknown' if running else 'worker_stopped', known=not running)
            outcome = 'unknown' if running else 'failed'
        finally:
            self.release()
            self.completion.notify(document_id, snapshot.operation.operation_id)
            logger.info('learning_worker stage=run outcome=%s duration_ms=%.1f', outcome, (monotonic()-started)*1000)
