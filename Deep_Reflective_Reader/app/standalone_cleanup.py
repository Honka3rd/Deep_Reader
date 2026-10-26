"""Independent cleanup orchestration; no prompt, matching or storage policy here."""

from dataclasses import dataclass
from contextlib import ExitStack
import hashlib
import hmac
import logging
import secrets
from threading import RLock
from time import monotonic, perf_counter
from typing import Callable, ContextManager

from app.cleanup_preflight_logging import preflight_stage, safe_reason
from document_preparation.accepted_source_snapshot import SourceNotReady
from document_preparation.cleanup_input_adapter import CleanupInputUnavailable
from document_structure.cleanup_input import CleanupInput
from document_structure.cleanup_operation import (
    CleanupCommitCommand, CleanupOperationBinding, CleanupOperationState, CleanupRepository, StagedCleanup,
)
from document_structure import ocr_cleanup_planner as engine
from document_structure.ocr_cleanup_contract import CleanupOutcome
from document_structure.ocr_cleanup_validation import validate_cleanup_input
from document_structure.plan_prompt_budget import EnhancementPromptBudgetError
from document_structure.reading_content import ReadingContentConflict, pending_review_regions


LOGGER = logging.getLogger(__name__)


class CleanupUnavailable(ValueError):
    """Safe use-case reason, not a provider/storage exception message."""


@dataclass(frozen=True)
class CleanupPreflight:
    ready: bool
    reason: str | None = None
    operation_id: str | None = None
    confirmation_token: str | None = None
    expires_in_seconds: int | None = None


@dataclass(frozen=True)
class CleanupResult:
    operation_id: str
    state: CleanupOperationState | None
    reason: str | None = None
    durability: str = "process_local"
    pending_review_count: int | None = None

    @property
    def phase(self):
        return self.state.phase if self.state else "unknown"

    @property
    def retry(self):
        return self.state.retry if self.state else "reconcile"


@dataclass
class _Attempt:
    binding: CleanupOperationBinding
    token_hash: str
    expires: float
    updated: float
    result: CleanupResult
    started: bool = False
    active: bool = False


def _hash(value):
    return hashlib.sha256(value.encode()).hexdigest()


class StandaloneCleanupService:
    """Namespace-scoped, injected use case, not a durable precommit job runner.

    read_input must verify an accepted source and adapt ONE coherent snapshot.
    lease must be the SAME cross-process document lease used by enhancement,
    without a long transaction. No permissive fallback is provided.
    layout_ready is read-only; prepare_layout is an explicit, source/revision-bound
    continuation, never permission to re-extract, re-split or regenerate a profile.
    Runtime composition belongs to SCLEAN-07; HTTP mapping belongs to SCLEAN-06.
    """

    def __init__(self, *, namespace: str, repository: CleanupRepository,
                 read_input: Callable[[str], CleanupInput], provider,
                 lease: Callable[[str], ContextManager],
                 layout_ready: Callable[[CleanupInput], bool],
                 prepare_layout: Callable[[CleanupInput], None],
                 clock=monotonic, limit=1024, retention=3600):
        self.namespace, self.repository = namespace, repository
        self.read_input, self.provider, self.lease = read_input, provider, lease
        self.layout_ready, self.prepare_layout = layout_ready, prepare_layout
        self.clock, self.limit, self.retention = clock, limit, retention
        self._attempts = {}
        self._lock = RLock()

    def _read(self, document_id):
        snapshot = self.read_input(document_id)
        if snapshot.namespace != self.namespace or snapshot.document_id != document_id:
            raise CleanupUnavailable("cleanup_identity_changed")
        return snapshot

    def _prune(self):
        now = self.clock()
        self._attempts = {key: value for key, value in self._attempts.items()
                          if value.active or now - value.updated < self.retention}

    def _entry(self, document_id, operation_id):
        with self._lock:
            self._prune()
            entry = self._attempts.get(operation_id)
            return entry if entry and entry.binding.document_id == document_id else None

    @staticmethod
    def _log(operation_id, stage, started, result=None):
        LOGGER.info("cleanup_use_case origin=standalone operation=%s stage=%s phase=%s reason=%s elapsed_ms=%.1f",
                    _hash(operation_id)[:16], stage, result.phase if result else None,
                    result.reason if result else None, (perf_counter() - started) * 1000)

    @staticmethod
    def _capacity(snapshot, provider):
        try:
            return engine.cleanup_input_capacity(snapshot, provider)
        except EnhancementPromptBudgetError as error:
            raise CleanupUnavailable("capacity_exceeded" if str(error) == "enhancement_prompt_budget_exceeded"
                                     else "capacity_unavailable") from None
        except Exception:
            raise CleanupUnavailable("capacity_unavailable") from None

    def preflight(self, document_id: str) -> CleanupPreflight:
        """Issues a short-lived opaque consent handle; no generation or writes."""
        with preflight_stage(LOGGER, "overall") as outcome:
            try:
                result = self._preflight(document_id)
            except BlockingIOError:
                result = CleanupPreflight(False, "cleanup_in_progress")
            except CleanupUnavailable as error:
                result = CleanupPreflight(False, str(error))
            except FileNotFoundError:
                result = CleanupPreflight(False, "cleanup_document_missing")
            except (SourceNotReady, CleanupInputUnavailable):
                result = CleanupPreflight(False, "cleanup_source_not_ready")
            except Exception:
                result = CleanupPreflight(False, "cleanup_preflight_unavailable")
            outcome.update(status="ready" if result.ready else "unavailable",
                           reason=safe_reason(result.reason), execution_available=result.ready)
            return result

    def _preflight(self, document_id):
        with ExitStack() as stack:
            # Time acquisition separately from work performed while holding the lease.
            with preflight_stage(LOGGER, "lease"):
                stack.enter_context(self.lease(document_id))
            with preflight_stage(LOGGER, "reconciliation") as reconciliation:
                with self._lock:
                    uncertain = [entry for entry in self._attempts.values()
                                 if entry.binding.document_id == document_id and entry.result.phase == "unknown"]
                for entry in uncertain:
                    recovered = self.lookup(document_id, entry.binding.operation_id)
                    if recovered.reason == "cleanup_reconciliation_unavailable":
                        reconciliation.update(status="unavailable", reason=safe_reason(recovered.reason))
                        return CleanupPreflight(False, recovered.reason)
            with preflight_stage(LOGGER, "input") as input_stage:
                snapshot = self._read(document_id)
                eligible = snapshot.eligibility()
                if eligible.status != "eligible":
                    input_stage.update(status="unavailable", reason=safe_reason(eligible.reason))
                    return CleanupPreflight(False, eligible.reason)
            with preflight_stage(LOGGER, "capacity"):
                capacity = self._capacity(snapshot, self.provider)
            with preflight_stage(LOGGER, "consent") as consent:
                token = secrets.token_urlsafe(32)
                operation = _hash(token)[:32]
                binding = CleanupOperationBinding(operation_id=operation, namespace=self.namespace,
                    document_id=document_id, input_fingerprint=snapshot.fingerprint(),
                    expected_reading_revision=snapshot.reading_revision,
                    prompt_version=engine.REMAINING_CLEANUP_PROMPT_VERSION,
                    policy_version=engine.REMAINING_CLEANUP_POLICY_VERSION,
                    capacity_fingerprint=capacity, consent_scope="standalone_ocr_cleanup_v1")
                with self._lock:
                    self._prune()
                    if len(self._attempts) >= self.limit:
                        consent.update(status="unavailable", reason="cleanup_operation_capacity")
                        return CleanupPreflight(False, "cleanup_operation_capacity")
                    now = self.clock()
                    self._attempts[operation] = _Attempt(binding, _hash(token), now + 300, now,
                        self._state(binding, "planning", "not_started"))
                consent["execution_available"] = True
            return CleanupPreflight(True, operation_id=operation, confirmation_token=token,
                                    expires_in_seconds=300)

    @staticmethod
    def _state(binding, phase, commit, *, outcome=None, retry="none", reason=None, durable=False,
               pending_review_count=None):
        return CleanupResult(binding.operation_id, CleanupOperationState(binding=binding, phase=phase,
            commit=commit, outcome=outcome, retry=retry), reason,
            "persisted_commit" if durable else "process_local", pending_review_count)

    def _unknown(self, operation_id, entry=None, reason="cleanup_outcome_unknown"):
        if entry:
            return self._state(entry.binding, "unknown", "unknown", retry="reconcile", reason=reason)
        return CleanupResult(operation_id, None, reason)

    @staticmethod
    def _retry(outcome):
        return "regenerate_cleanup" if outcome.reason in {
            "provider_failed", "invalid_plan", "some_items_rejected", "no_safe_items", "cancelled",
        } else "none"

    def _committed(self, receipt):
        binding = receipt.binding
        try:
            snapshot = self._read(binding.document_id)
        except SourceNotReady as error:
            if error.reason == "source_unreadable":
                raise
            return self._state(binding, "superseded", "committed", outcome=receipt.outcome,
                               reason="cleanup_source_changed", durable=True)
        except (CleanupInputUnavailable, CleanupUnavailable):
            return self._state(binding, "superseded", "committed", outcome=receipt.outcome,
                               reason="cleanup_source_changed", durable=True)
        review_count = len(pending_review_regions(snapshot.document()))
        if (receipt.source_structure_fingerprint is None
                or snapshot.source_structure_fingerprint() != receipt.source_structure_fingerprint
                or snapshot.reading_revision != receipt.outcome.reading_revision):
            return self._state(binding, "superseded", "committed", outcome=receipt.outcome,
                               reason="cleanup_operation_superseded", durable=True, pending_review_count=review_count)
        if not self.layout_ready(snapshot):
            return self._state(binding, "layout", "committed", outcome=receipt.outcome,
                               retry="resume_layout", durable=True, pending_review_count=review_count)
        return self._state(binding, "finished", "committed", outcome=receipt.outcome,
            retry=self._retry(receipt.outcome), reason=receipt.outcome.reason, durable=True,
            pending_review_count=review_count)

    def lookup(self, document_id: str, operation_id: str) -> CleanupResult:
        """A read failure is unknown, NEVER a missing receipt or permission to retry."""
        started = perf_counter()
        entry = self._entry(document_id, operation_id)
        try:
            receipt = self.repository.lookup_cleanup(namespace=self.namespace,
                document_id=document_id, operation_id=operation_id)
            if receipt is not None:
                if (receipt.binding.namespace != self.namespace or receipt.binding.document_id != document_id
                        or receipt.binding.operation_id != operation_id):
                    raise ValueError("receipt identity mismatch")
                result = self._committed(receipt)
            else:
                if entry is None:
                    # Establish document existence for unknown handles without
                    # treating a failed source/storage read as a missing receipt.
                    self._read(document_id)
                result = entry.result if entry else self._unknown(operation_id)
                if result.state and result.state.commit == "committed":
                    result = self._state(result.state.binding, "superseded", "committed",
                        outcome=result.state.outcome, reason="cleanup_operation_superseded")
        except FileNotFoundError:
            raise
        except (SourceNotReady, CleanupInputUnavailable):
            result = self._unknown(operation_id, entry, "cleanup_source_not_ready")
        except Exception:
            result = self._unknown(operation_id, entry, "cleanup_reconciliation_unavailable")
        self._log(operation_id, "lookup", started, result)
        return result

    def _terminal(self, binding, reason, *, status="skipped"):
        outcome = CleanupOutcome(status=status, reason=reason, reading_revision=binding.expected_reading_revision,
                                 proposed_count=None, reading_content_changed=False, applied_count=0, skipped_count=0)
        return self._state(binding, "finished", "not_committed", outcome=outcome,
                           reason=reason, retry=self._retry(outcome))

    def execute(self, document_id: str, operation_id: str, *, confirmation_token: str,
                confirmed: bool) -> CleanupResult:
        if confirmed is not True or not isinstance(confirmation_token, str) or not confirmation_token:
            raise CleanupUnavailable("cleanup_confirmation_required")
        # The public handle is a one-way token commitment, so even post-restart
        # layout continuation can verify consent without persisting the secret.
        if len(confirmation_token) > 1024 or not hmac.compare_digest(_hash(confirmation_token)[:32], operation_id):
            raise CleanupUnavailable("cleanup_confirmation_invalid")
        started = perf_counter()
        try:
            with self.lease(document_id):
                prior = self.lookup(document_id, operation_id)
                if prior.phase == "layout":
                    return self._finish_layout(document_id, operation_id)
                if prior.phase != "planning":
                    return prior
                entry = self._entry(document_id, operation_id)
                with self._lock:
                    if entry is None or entry.started:
                        return prior
                    if len(confirmation_token) > 1024 or not hmac.compare_digest(entry.token_hash, _hash(confirmation_token)):
                        raise CleanupUnavailable("cleanup_confirmation_invalid")
                    if self.clock() >= entry.expires:
                        raise CleanupUnavailable("cleanup_confirmation_expired")
                    entry.started = entry.active = True
                try:
                    result = self._run(entry)
                except Exception:
                    result = self._unknown(operation_id, entry, "cleanup_execution_unavailable")
                except BaseException:
                    entry.result = self._unknown(operation_id, entry)
                    raise
                finally:
                    with self._lock:
                        entry.active = False
                        entry.updated = self.clock()
                entry.result = result
                self._log(operation_id, "execute", started, result)
                return result
        except BlockingIOError:
            raise CleanupUnavailable("cleanup_in_progress") from None

    def _run(self, entry):
        binding = entry.binding
        try:
            snapshot = self._read(binding.document_id)
        except (SourceNotReady, CleanupInputUnavailable, CleanupUnavailable):
            return self._terminal(binding, "source_changed")
        if not binding.matches(snapshot):
            return self._terminal(binding, "reading_revision_changed")
        try:
            capacity = self._capacity(snapshot, self.provider)
        except CleanupUnavailable as error:
            return self._terminal(binding, str(error))
        if (capacity != binding.capacity_fingerprint
                or binding.prompt_version != engine.REMAINING_CLEANUP_PROMPT_VERSION
                or binding.policy_version != engine.REMAINING_CLEANUP_POLICY_VERSION):
            return self._terminal(binding, "cleanup_policy_changed")
        planned = engine.plan_cleanup_input(snapshot=snapshot, provider=self.provider,
            operation_id=binding.operation_id, expected_capacity_fingerprint=binding.capacity_fingerprint)
        if planned.outcome is not None:
            return self._state(binding, "finished", "not_committed", outcome=planned.outcome,
                retry=self._retry(planned.outcome), reason=planned.outcome.reason)
        entry.result = self._state(binding, "staging", "not_started")
        resolved = validate_cleanup_input(planned.proposal, snapshot=snapshot,
                                         expected_fingerprint=planned.input_fingerprint)
        command = CleanupCommitCommand(snapshot=snapshot, staged=StagedCleanup(binding=binding,
            proposal=planned.proposal, deletions=resolved.deletions, rejections=resolved.rejections))
        try:
            if self._capacity(snapshot, self.provider) != binding.capacity_fingerprint:
                return self._terminal(binding, "cleanup_policy_changed")
        except CleanupUnavailable as error:
            return self._terminal(binding, str(error))
        entry.result = self._state(binding, "committing", "unknown")
        try:
            self.repository.commit_cleanup(command)
        except Exception as error:
            recovered = self.lookup(binding.document_id, binding.operation_id)
            if recovered.state and recovered.state.commit == "committed":
                return self._finish_layout(binding.document_id, binding.operation_id)
            if recovered.reason == "cleanup_reconciliation_unavailable":
                return recovered
            if isinstance(error, ReadingContentConflict):
                reasons = {
                    "cleanup_source_changed": "source_changed",
                    "cleanup_confirmation_stale": "reading_revision_changed",
                    "cleanup_reading_mapping_changed": "reading_revision_changed",
                    "cleanup_operation_superseded": "reading_revision_changed",
                    "cleanup_policy_changed": "cleanup_policy_changed",
                    "cleanup_source_evidence_changed": "source_changed",
                }
                reason = reasons.get(str(error), "storage_failed")
                diagnostic = str(error) if str(error) in {*reasons, "cleanup_source_unavailable"} else "cleanup_commit_conflict"
                LOGGER.warning("cleanup_use_case stage=commit_rejected operation=%s reason=%s outcome=%s",
                               _hash(binding.operation_id)[:16], diagnostic, reason)
                return self._terminal(binding, reason,
                                      status="failed" if reason == "storage_failed" else "skipped")
            return self._unknown(binding.operation_id, entry)
        return self._finish_layout(binding.document_id, binding.operation_id)

    def _finish_layout(self, document_id, operation_id):
        result = self.lookup(document_id, operation_id)
        if result.phase == "committing":
            return self._unknown(operation_id, self._entry(document_id, operation_id))
        if result.phase != "layout":
            return result
        started = perf_counter()
        try:
            receipt = self.repository.lookup_cleanup(namespace=self.namespace,
                document_id=document_id, operation_id=operation_id)
            snapshot = self._read(document_id)
            # The continuation port must also CAS this snapshot before its write.
            if (receipt is None or snapshot.reading_revision != result.state.outcome.reading_revision
                    or snapshot.source_structure_fingerprint() != receipt.source_structure_fingerprint):
                return self.lookup(document_id, operation_id)
            self.prepare_layout(snapshot)
        except Exception:
            checked = self.lookup(document_id, operation_id)
            if checked.phase == "layout":
                checked = CleanupResult(operation_id, checked.state, "cleanup_layout_failed",
                                        checked.durability, checked.pending_review_count)
            self._log(operation_id, "layout", started, checked)
            return checked
        checked = self.lookup(document_id, operation_id)
        self._log(operation_id, "layout", started, checked)
        return checked

    def resume_layout(self, document_id: str, operation_id: str) -> CleanupResult:
        """Explicit continuation, including after restart; never repeats generation."""
        try:
            with self.lease(document_id):
                return self._finish_layout(document_id, operation_id)
        except BlockingIOError:
            raise CleanupUnavailable("cleanup_in_progress") from None
