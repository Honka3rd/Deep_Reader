"""Pure LLRN-05 decisions: no IO, locks, durable claims or provider calls.

Adapters MUST atomically compare entry revision/current binding, write the next
entry AND receipt/history, then acknowledge commit. Only a committed dispatch
transition authorizes ONE batch call outside the transaction. Lost acks require
lookup, never dispatch replay. Actual adapters belong to LLRN-06/08.
"""
from dataclasses import dataclass, replace
from typing import Protocol

from section_tasks.learning_cache import (CacheKey, HintsKey, LearningResult, TranslationKey,
    validate_complete_result, validate_result_key)
from shared.learning_source import SentenceResource, digest_value, integer, require, token


@dataclass(frozen=True, slots=True)
class GenerationBinding:
    key: CacheKey
    workspace_revision: int
    book_revision: int
    active_mode: str = "language_learning"

    def __post_init__(self):
        require(type(self.key) in (TranslationKey, HintsKey), "learning_key_invalid")
        integer(self.workspace_revision)
        integer(self.book_revision)
        token(self.active_mode)


def active_generation_binding(binding):
    """Ordinary aligned translation has no native-language/proficiency prerequisite."""
    return binding.active_mode == 'language_learning' or (
        type(binding.key) is TranslationKey and binding.active_mode == 'translation'
        and binding.workspace_revision == 0 and binding.book_revision == 0)


@dataclass(frozen=True, slots=True)
class GenerationCommand:
    command_id: str
    binding: GenerationBinding
    intent: str = "get_or_generate"
    retry_operation_id: str | None = None
    acknowledge_possible_duplicate_cost: bool = False

    def __post_init__(self):
        token(self.command_id)
        require(type(self.binding) is GenerationBinding, "learning_binding_invalid")
        require(self.intent in ("get_or_generate", "regenerate", "retry"), "learning_intent_invalid")
        require(type(self.acknowledge_possible_duplicate_cost) is bool, "learning_ack_invalid")
        if self.intent == "retry":
            token(self.retry_operation_id)
        else:
            require(self.retry_operation_id is None and not self.acknowledge_possible_duplicate_cost,
                    "learning_retry_fields_invalid")


@dataclass(frozen=True, slots=True)
class BatchPlan:
    """Ordered sentence-ID partitions selected using configured model capabilities.

    Checkpoints reference separately persisted, validated partial payloads. They
    cannot substitute for complete results. No token/character estimate here.
    """
    groups: tuple[tuple[str, ...], ...]

    def __post_init__(self):
        require(type(self.groups) is tuple and bool(self.groups), "learning_batches_invalid")
        ids = []
        for group in self.groups:
            require(type(group) is tuple and bool(group), "learning_batch_empty")
            for sentence_id in group:
                token(sentence_id)
            ids.extend(group)
        require(len(set(ids)) == len(ids), "learning_batch_duplicate")

    def validate_against(self, sentences: SentenceResource) -> None:
        require(tuple(x for group in self.groups for x in group)
                == tuple(s.sentence_id for s in sentences.sentences), "learning_batch_coverage_invalid")


_STATES = ("claimed", "running", "partial", "succeeded", "failed", "unknown", "stale")
_ACTIVE = ("claimed", "running", "partial")


@dataclass(frozen=True, slots=True)
class GenerationOperation:
    operation_id: str
    binding: GenerationBinding
    fence: int
    base_result_revision: int
    plan: BatchPlan
    state: str
    created_at_ms: int
    updated_at_ms: int
    lease_until_ms: int
    checkpoints: tuple[str, ...] = ()
    failure_code: str | None = None
    result_revision: int | None = None
    outcome_uncertain: bool = False

    def __post_init__(self):
        token(self.operation_id)
        require(type(self.binding) is GenerationBinding and type(self.plan) is BatchPlan,
                "learning_operation_invalid")
        integer(self.fence, minimum=1)
        integer(self.base_result_revision)
        for value in (self.created_at_ms, self.updated_at_ms, self.lease_until_ms):
            integer(value)
        require(self.created_at_ms <= self.updated_at_ms and self.created_at_ms < self.lease_until_ms,
                "learning_operation_time_invalid")
        require(self.state in _STATES, "learning_operation_state_invalid")
        require(type(self.outcome_uncertain) is bool, "learning_outcome_invalid")
        if self.state != "stale":
            require(self.outcome_uncertain == (self.state in ("running", "unknown")),
                    "learning_outcome_state_invalid")
        require(type(self.checkpoints) is tuple and len(self.checkpoints) <= len(self.plan.groups),
                "learning_checkpoints_invalid")
        for checkpoint in self.checkpoints:
            digest_value(checkpoint)
        require(self.state != "claimed" or not self.checkpoints, "learning_checkpoint_state_invalid")
        require(self.state != "partial" or bool(self.checkpoints), "learning_checkpoint_state_invalid")
        require(self.state != "running" or len(self.checkpoints) < len(self.plan.groups),
                "learning_batch_already_complete")
        if self.failure_code is not None:
            require(self.failure_code in ("provider_rejected", "invalid_output", "worker_stopped",
                "lease_expired", "provider_outcome_unknown", "binding_changed"),
                "learning_failure_code_invalid")
        if self.state == "succeeded":
            require(type(self.result_revision) is int and self.result_revision == self.base_result_revision + 1
                    and len(self.checkpoints) == len(self.plan.groups), "learning_receipt_invalid")
        else:
            require(self.result_revision is None, "learning_receipt_invalid")


@dataclass(frozen=True, slots=True)
class CacheEntry:
    key: CacheKey
    revision: int = 0
    last_fence: int = 0
    saved: LearningResult | None = None
    operation: GenerationOperation | None = None

    def __post_init__(self):
        require(type(self.key) in (TranslationKey, HintsKey), "learning_key_invalid")
        integer(self.revision)
        integer(self.last_fence)
        if self.saved is not None:
            validate_result_key(self.key, self.saved)
        require(self.operation is not None or self.last_fence == 0, "learning_operation_missing")
        if self.operation is not None:
            require(type(self.operation) is GenerationOperation and self.operation.binding.key == self.key
                    and self.operation.fence == self.last_fence, "learning_operation_key_mismatch")

    @property
    def result_revision(self) -> int:
        return self.saved.context.result_revision if self.saved is not None else 0


@dataclass(frozen=True, slots=True)
class ReadView:
    state: str
    saved: LearningResult | None
    operation_state: str | None
    operation_id: str | None


def read_entry(entry: CacheEntry, *, evidence: str = "verified") -> ReadView:
    """Verified requires current source/version/payload validation by the reader.

    Read failures cannot be converted to an empty entry. Ready saved data and a
    refresh operation's state are independent; reads do not expire operations.
    """
    require(evidence in ("verified", "read_error", "unsupported", "stale"), "learning_read_evidence_invalid")
    op = entry.operation
    if evidence != "verified":
        return ReadView(evidence, None, None, None)
    state = "ready" if entry.saved is not None else (op.state if op else "missing")
    if state == "succeeded":  # Receipt without output is corruption, never missing.
        state = "unavailable"
    return ReadView(state, entry.saved, op.state if op else None, op.operation_id if op else None)


@dataclass(frozen=True, slots=True)
class AdmissionDecision:
    allowed: bool
    reason: str = "allowed"

    def __post_init__(self):
        require(type(self.allowed) is bool, "learning_admission_invalid")
        token(self.reason)


class LearningAdmissionDenied(ValueError):
    def __init__(self, reason):
        self.reason = reason
        super().__init__("learning_admission_denied")


class GenerationAdmission(Protocol):
    """Pure synchronous restriction check; no remote IO or fabricated user/quota."""
    def evaluate(self, command: GenerationCommand) -> AdmissionDecision: ...


class AllowValidGeneration:
    def evaluate(self, command: GenerationCommand) -> AdmissionDecision:
        return AdmissionDecision(True)


@dataclass(frozen=True, slots=True)
class CommandReceipt:
    """UNIQUE(scope, document, command_id), across component keys.

    Adapters retain command identity/operation history even after a newer op
    takes the slot, or explicit tombstones that prevent old-command execution.
    """
    command: GenerationCommand
    outcome: str
    operation_id: str | None
    result_revision: int | None
    admission_reason: str | None = None


@dataclass(frozen=True, slots=True)
class RequestDecision:
    outcome: str
    expected_entry_revision: int
    next_entry: CacheEntry | None
    receipt: CommandReceipt
    replay: bool = False


@dataclass(frozen=True, slots=True)
class Transition:
    expected_entry_revision: int
    next_entry: CacheEntry


def plan_request(entry: CacheEntry, command: GenerationCommand, *, current: GenerationBinding,
                 sentences: SentenceResource, effective_text: str,
                 now_ms: int, lease_until_ms: int, plan: BatchPlan | None = None,
                 receipt: CommandReceipt | None = None, evidence: str = "verified",
                 admission: GenerationAdmission | None = None) -> RequestDecision:
    """Plan in a short repository transaction; returned claims are NOT durable."""
    require(entry.key == command.binding.key, "learning_command_key_mismatch")
    if receipt is not None:
        require(receipt.command == command, "learning_command_conflict")
        return RequestDecision(receipt.outcome, entry.revision, None, receipt, replay=True)
    require(evidence == "verified", "learning_read_not_verified")
    require(command.binding == current and current.key == entry.key
            and active_generation_binding(current), "learning_binding_stale")
    current.key.validate_selection()
    sentences.validate_source(current.key.resource.source, effective_text)
    require(sentences.reference == current.key.resource, "learning_reference_mismatch")
    if entry.saved is not None:
        validate_complete_result(entry.key, entry.saved, sentences, effective_text)
    integer(now_ms)
    integer(lease_until_ms)
    require(lease_until_ms > now_ms, "learning_lease_invalid")
    op = entry.operation

    def decision(outcome, next_entry=None, operation_id=None, result_revision=None, admission_reason=None):
        return RequestDecision(outcome, entry.revision, next_entry,
            CommandReceipt(command, outcome, operation_id, result_revision, admission_reason))

    if command.intent == "get_or_generate" and entry.saved is not None:
        return decision("reuse", result_revision=entry.result_revision)
    if not sentences.sentences:
        return decision("no_content")
    if op and op.state in _ACTIVE:
        if op.binding != current or now_ms >= op.lease_until_ms:
            return decision("reconcile", operation_id=op.operation_id)
        return decision("coalesced", operation_id=op.operation_id)
    if op and op.outcome_uncertain:
        if not (command.intent == "retry" and command.retry_operation_id == op.operation_id
                and command.acknowledge_possible_duplicate_cost):
            return decision("reconcile", operation_id=op.operation_id)
    if command.intent == "retry":
        require(op is not None and op.operation_id == command.retry_operation_id
                and op.state in ("failed", "stale", "unknown"), "learning_retry_target_invalid")
    elif op is not None and op.state in ("failed", "stale"):
        return decision("retry_required", operation_id=op.operation_id)
    elif op is not None and op.state == "succeeded" and entry.saved is None:
        return decision("unavailable", operation_id=op.operation_id)
    admission_result = (admission or AllowValidGeneration()).evaluate(command)
    require(type(admission_result) is AdmissionDecision, "learning_admission_invalid")
    if not admission_result.allowed:
        return decision("denied", admission_reason=admission_result.reason)
    require(op is None or now_ms >= op.updated_at_ms, "learning_clock_reversed")
    require(type(plan) is BatchPlan, "learning_batches_invalid")
    plan.validate_against(sentences)
    new_op = GenerationOperation(command.command_id, current, entry.last_fence + 1,
        entry.result_revision, plan, "claimed", now_ms, now_ms, lease_until_ms)
    proposed = replace(entry, revision=entry.revision + 1, last_fence=new_op.fence, operation=new_op)
    return decision("claimed", proposed, new_op.operation_id)


def _operation(entry, operation_id, fence, now_ms):
    integer(fence, minimum=1)
    integer(now_ms)
    op = entry.operation
    require(op is not None and op.operation_id == operation_id and op.fence == fence,
            "learning_fence_conflict")
    require(now_ms >= op.updated_at_ms, "learning_clock_reversed")
    return op


def _transition(entry, op, now_ms, **changes):
    updated = replace(op, updated_at_ms=now_ms, **changes)
    return Transition(entry.revision, replace(entry, revision=entry.revision + 1, operation=updated))


def dispatch_batch(entry: CacheEntry, *, operation_id: str, fence: int,
                   current: GenerationBinding, now_ms: int) -> Transition:
    """Persist running before ONE next batch; dispatch transitions cannot replay."""
    op = _operation(entry, operation_id, fence, now_ms)
    require(op.binding == current and active_generation_binding(current), "learning_binding_stale")
    current.key.validate_selection()
    require(op.state in ("claimed", "partial") and now_ms < op.lease_until_ms
            and len(op.checkpoints) < len(op.plan.groups), "learning_dispatch_forbidden")
    return _transition(entry, op, now_ms, state="running", outcome_uncertain=True)



def renew_lease(entry: CacheEntry, *, operation_id: str, fence: int,
                current: GenerationBinding, now_ms: int, lease_until_ms: int) -> Transition:
    """Explicit fenced worker heartbeat, not read-time takeover/recovery."""
    op = _operation(entry, operation_id, fence, now_ms)
    integer(lease_until_ms)
    require(op.binding == current and active_generation_binding(current), "learning_binding_stale")
    require(op.state in _ACTIVE and now_ms < op.lease_until_ms
            and lease_until_ms > op.lease_until_ms, "learning_lease_renewal_forbidden")
    return _transition(entry, op, now_ms, lease_until_ms=lease_until_ms)


def checkpoint_batch(entry: CacheEntry, *, operation_id: str, fence: int,
                     current: GenerationBinding, now_ms: int, batch_index: int,
                     checkpoint_digest: str) -> Transition:
    """Validate batch payload before calling; store payload + marker atomically.

    Unknown accepts a recovered response for the SAME outstanding batch/fence.
    Reconciliation is not permission for a new provider call.
    """
    op = _operation(entry, operation_id, fence, now_ms)
    integer(batch_index)
    digest_value(checkpoint_digest)
    require(op.binding == current and active_generation_binding(current), "learning_binding_stale")
    require(op.state in ("running", "unknown") and batch_index == len(op.checkpoints),
            "learning_checkpoint_forbidden")
    return _transition(entry, op, now_ms, state="partial", failure_code=None, outcome_uncertain=False,
                       checkpoints=(*op.checkpoints, checkpoint_digest))


def finish_failure(entry: CacheEntry, *, operation_id: str, fence: int, now_ms: int,
                   reason: str, outcome_known: bool) -> Transition:
    """Known means definitive response/no in-flight request, never timeout alone."""
    op = _operation(entry, operation_id, fence, now_ms)
    require(type(outcome_known) is bool, "learning_outcome_invalid")
    require(op.state in (*_ACTIVE, "unknown"), "learning_failure_forbidden")
    require((outcome_known and reason in ("provider_rejected", "invalid_output", "worker_stopped"))
            or (not outcome_known and reason == "provider_outcome_unknown" and op.state == "running"),
            "learning_failure_evidence_invalid")
    require(reason != "worker_stopped" or op.state in ("claimed", "partial"),
            "learning_failure_evidence_invalid")
    return _transition(entry, op, now_ms, state="failed" if outcome_known else "unknown",
                       failure_code=reason, outcome_uncertain=not outcome_known)


def expire_operation(entry: CacheEntry, *, operation_id: str, fence: int, now_ms: int) -> Transition:
    op = _operation(entry, operation_id, fence, now_ms)
    require(op.state in _ACTIVE and now_ms >= op.lease_until_ms, "learning_expiry_forbidden")
    return _transition(entry, op, now_ms, state="unknown" if op.state == "running" else "failed",
                       failure_code="lease_expired")


def invalidate_operation(entry: CacheEntry, *, operation_id: str, fence: int,
                         current: GenerationBinding, now_ms: int) -> Transition:
    """Lifecycle owner separately invalidates data. This does not cancel IO.

    Retain same-key saved data; changed-source data is unusable under a new key.
    """
    op = _operation(entry, operation_id, fence, now_ms)
    require(op.binding != current and op.state in (*_ACTIVE, "unknown"), "learning_invalidation_forbidden")
    return _transition(entry, op, now_ms, state="stale", failure_code="binding_changed")


def mark_binding_stale(entry: CacheEntry, *, operation_id: str, fence: int,
                       now_ms: int, binding_is_current: bool) -> Transition:
    """Repository has revalidated the actual source/settings under its write locks."""
    op = _operation(entry, operation_id, fence, now_ms)
    require(binding_is_current is False and op.state in (*_ACTIVE, "unknown"),
            "learning_invalidation_forbidden")
    return _transition(entry, op, now_ms, state="stale", failure_code="binding_changed")


def commit_result(entry: CacheEntry, *, operation_id: str, fence: int,
                  current: GenerationBinding, now_ms: int, result: LearningResult,
                  sentences: SentenceResource, effective_text: str) -> Transition:
    """Output + succeeded operation/history receipt MUST commit together."""
    op = _operation(entry, operation_id, fence, now_ms)
    require(op.binding == current and active_generation_binding(current), "learning_binding_stale")
    current.key.validate_selection()
    require(op.state == "partial" and len(op.checkpoints) == len(op.plan.groups),
            "learning_result_incomplete")
    require(entry.result_revision == op.base_result_revision, "learning_result_cas_conflict")
    validate_complete_result(entry.key, result, sentences, effective_text)
    require(result.context.result_revision == op.base_result_revision + 1, "learning_result_revision_invalid")
    transition = _transition(entry, op, now_ms, state="succeeded", result_revision=result.context.result_revision)
    return replace(transition, next_entry=replace(transition.next_entry, saved=result))
