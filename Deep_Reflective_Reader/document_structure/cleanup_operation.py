"""Internal standalone cleanup commands/outcomes, separate from enhancement DTOs."""

from typing import Literal, Protocol

from pydantic import Field, model_validator

from document_structure.cleanup_input import CleanupInput
from document_structure.ocr_cleanup_contract import (
    CleanupContract, CleanupOutcome, CleanupProposal, Digest, Identifier,
)
from document_structure.ocr_cleanup_validation import CleanupItemRejection, ValidatedDeletion


class CleanupOperationBinding(CleanupContract):
    """App-issued consent binding; not a user-selectable authorization scope."""

    operation_id: Identifier
    namespace: Identifier
    document_id: Identifier
    input_fingerprint: Digest
    expected_reading_revision: int = Field(ge=0)
    prompt_version: Identifier
    policy_version: Identifier
    capacity_fingerprint: Digest
    consent_scope: Literal["standalone_ocr_cleanup_v1"]

    def matches(self, snapshot: CleanupInput) -> bool:
        return (self.namespace == snapshot.namespace and self.document_id == snapshot.document_id
                and self.expected_reading_revision == snapshot.reading_revision
                and self.input_fingerprint == snapshot.fingerprint())


class StagedCleanup(CleanupContract):
    """Pure validator output, not a commit receipt or permission to skip guards."""

    binding: CleanupOperationBinding
    proposal: CleanupProposal
    deletions: tuple[ValidatedDeletion, ...]
    rejections: tuple[CleanupItemRejection, ...]

    @model_validator(mode="after")
    def accounted(self):
        indices = [item.item_index for item in (*self.deletions, *self.rejections)]
        if (any(type(i) is not int for i in indices)
                or sorted(indices) != list(range(len(self.proposal.items)))):
            raise ValueError("cleanup_stage_count_mismatch")
        end = 0
        for deletion in self.deletions:
            item = self.proposal.items[deletion.item_index]
            if (type(deletion.char_start) is not int or type(deletion.char_end) is not int
                    or not end <= deletion.char_start < deletion.char_end
                    or deletion.quote != item.quote or deletion.reason != item.reason):
                raise ValueError("cleanup_stage_deletion_invalid")
            end = deletion.char_end
        return self


class CleanupCommitCommand(CleanupContract):
    """Content-only repository command; no hierarchy replacement or raw HTTP plan.

    Repository revalidates source, hierarchy, reading revision and exact deletion
    safety under its write lock. It must preserve source/IDs/profile/notes/manual
    edits and atomically persist overlays, invalidation, audit, Undo and receipt.
    Stage construction does not establish semantic safety or authorization.
    """

    snapshot: CleanupInput
    staged: StagedCleanup

    @model_validator(mode="after")
    def bound(self):
        if not self.staged.binding.matches(self.snapshot):
            raise ValueError("cleanup_input_binding_mismatch")
        eligibility = self.snapshot.eligibility()
        if eligibility.status != "eligible":
            raise ValueError(eligibility.reason)
        for deletion in self.staged.deletions:
            if not any(not part.manual and part.char_start <= deletion.char_start
                       < deletion.char_end <= part.char_end for part in self.snapshot.parts):
                raise ValueError("cleanup_deletion_outside_view")
            if self.snapshot.source.canonical_text[deletion.char_start:deletion.char_end] != deletion.quote:
                raise ValueError("cleanup_deletion_text_mismatch")
            if any(span.char_start < deletion.char_end and deletion.char_start < span.char_end
                   for span in self.snapshot.protected_ranges):
                raise ValueError("cleanup_deletion_protected")
        return self


class CleanupCommitResult(CleanupContract):
    binding: CleanupOperationBinding
    outcome: CleanupOutcome
    replayed: bool = False
    reading_layout_required: bool
    # Older receipts without lineage cannot authorize post-restart continuation.
    source_structure_fingerprint: Digest | None = None

    @model_validator(mode="after")
    def revision(self):
        expected = self.binding.expected_reading_revision + int(self.outcome.reading_content_changed)
        if self.outcome.reading_revision != expected:
            raise ValueError("cleanup_commit_revision_mismatch")
        return self


class CleanupOperationState(CleanupContract):
    """Operation certainty and recovery are not terminal CleanupOutcome statuses."""

    binding: CleanupOperationBinding
    phase: Literal["planning", "staging", "committing", "layout", "finished", "unknown", "superseded"]
    commit: Literal["not_started", "not_committed", "committed", "unknown"]
    outcome: CleanupOutcome | None
    retry: Literal["none", "reconcile", "regenerate_cleanup", "resume_layout"]

    @model_validator(mode="after")
    def coherent(self):
        allowed = {
            "planning": {"not_started"}, "staging": {"not_started"},
            "committing": {"unknown"}, "layout": {"committed"},
            "finished": {"not_committed", "committed"},
            "unknown": {"unknown"}, "superseded": {"not_committed", "committed", "unknown"},
        }
        if self.commit not in allowed[self.phase]:
            raise ValueError("cleanup_phase_commit_mismatch")
        if self.phase == "unknown" and self.retry != "reconcile":
            raise ValueError("cleanup_unknown_requires_reconciliation")
        retries = {
            "planning": {"none"}, "staging": {"none"}, "committing": {"none"},
            "layout": {"none", "resume_layout"}, "finished": {"none", "regenerate_cleanup"},
            "unknown": {"reconcile"}, "superseded": {"none"},
        }
        if self.retry not in retries[self.phase]:
            raise ValueError("cleanup_retry_phase_mismatch")
        if self.phase in {"planning", "staging", "committing", "unknown"}:
            if self.outcome is not None:
                raise ValueError("cleanup_unconfirmed_outcome")
        elif self.phase != "superseded" and self.outcome is None:
            raise ValueError("cleanup_terminal_outcome_missing")
        if self.outcome is not None:
            if self.outcome.reading_content_changed and self.commit != "committed":
                raise ValueError("cleanup_change_not_committed")
            expected = self.binding.expected_reading_revision + int(self.outcome.reading_content_changed)
            if self.outcome.reading_revision != expected:
                raise ValueError("cleanup_outcome_revision_mismatch")
        if self.retry == "regenerate_cleanup" and self.outcome.reason in {
            "native_source", "ocr_evidence_missing", "ocr_evidence_mismatch",
            "extraction_mode_unsupported", "capacity_exceeded", "capacity_unavailable",
        }:
            raise ValueError("cleanup_retry_not_available")
        return self


class CleanupRepository(Protocol):
    """Optional internal port; implementations arrive in SCLEAN-04.

    Same operation + same binding replays a durable receipt without model calls
    or another revision. Reused IDs with different bindings conflict. A fresh
    attempt requires new consent/ID. No-op writes may persist a receipt only;
    they never advance reading revision or clear interactions. Storage failures
    propagate, and unknown commits must be reconciled before any fresh attempt.
    """

    def commit_cleanup(self, command: CleanupCommitCommand) -> CleanupCommitResult: ...

    def lookup_cleanup(self, *, namespace: str, document_id: str,
                       operation_id: str) -> CleanupCommitResult | None:
        """Read-only receipt lookup; None means absent, never a swallowed failure.

        A historical receipt does not prove current state or safe retry: the app
        must compare current revisions and reconcile in-flight ownership too.
        """
        ...
