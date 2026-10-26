"""Side-effect-free cleanup contracts; proposals never authorize source mutation."""

import json
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from doc_loaders.extraction_identity import ExtractionIdentity, ExtractionResult


CLEANUP_SCHEMA_VERSION = "ocr_cleanup_plan_v1"
CLEANUP_POLICY_VERSION = "ocr_cleanup_policy_v1"
CLEANUP_CONSENT_SCOPE = "ocr_cleanup_reading_overlay_v1"
MAX_CONTEXT_CHARS = 128
MAX_RESPONSE_BYTES = 1024 * 1024

Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Identifier = Annotated[str, Field(min_length=1, max_length=200, pattern=r"\S")]
CleanupReason = Literal["ocr_garbage", "recognition_fragment"]
OutcomeReason = Literal[
    "native_source", "ocr_evidence_missing", "ocr_evidence_mismatch",
    "extraction_mode_unsupported", "capacity_exceeded", "provider_failed",
    "invalid_plan", "no_safe_items", "some_items_rejected", "source_changed",
    "reading_revision_changed", "storage_failed", "cancelled",
    "structure_not_ready", "cleanup_policy_changed", "capacity_unavailable",
]


class CleanupContract(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class CleanupItem(CleanupContract):
    action: Literal["delete"]
    quote: str = Field(min_length=1)
    prefix_quote: str | None = Field(default=None, min_length=1, max_length=MAX_CONTEXT_CHARS)
    suffix_quote: str | None = Field(default=None, min_length=1, max_length=MAX_CONTEXT_CHARS)
    reason: CleanupReason

    @model_validator(mode="after")
    def nonblank_quote(self):
        if not self.quote.strip():
            raise ValueError("cleanup_quote_is_blank")
        for value in (self.quote, self.prefix_quote, self.suffix_quote):
            if value is not None:
                value.encode("utf-8")
        return self


class CleanupProposal(CleanupContract):
    schema_version: Literal["ocr_cleanup_plan_v1"]
    items: tuple[CleanupItem, ...]


def cleanup_response_schema():
    """Provider-facing schema excludes all backend identities and positions."""
    schema = CleanupProposal.model_json_schema()

    def strict(node):
        if isinstance(node, dict):
            node.pop("default", None)
            if node.get("type") == "object":
                node["required"] = list(node.get("properties", {}))
                node["additionalProperties"] = False
            for value in node.values():
                strict(value)
        elif isinstance(node, list):
            for value in node:
                strict(value)

    strict(schema)
    return schema


def parse_cleanup_proposal(payload: str) -> CleanupProposal:
    """Reject over-limit or ambiguous JSON without exposing model/source text."""
    if not isinstance(payload, str):
        raise ValueError("cleanup_plan_invalid")
    try:
        if len(payload.encode("utf-8")) > MAX_RESPONSE_BYTES:
            raise ValueError("cleanup_response_too_large")
    except UnicodeError:
        raise ValueError("cleanup_plan_invalid") from None

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate_key")
            result[key] = value
        return result

    try:
        json.loads(payload, object_pairs_hook=unique_object)
        return CleanupProposal.model_validate_json(payload)
    except (ValueError, RecursionError):
        raise ValueError("cleanup_plan_invalid") from None


class CanonicalOcrSpan(CleanupContract):
    """Backend evidence only, never a model-selected deletion interval."""

    char_start: int = Field(ge=0)
    char_end: int = Field(gt=0)
    page_index: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def ordered(self):
        if self.char_end <= self.char_start:
            raise ValueError("invalid_ocr_span")
        return self


class ActualOcrEvidence(CleanupContract):
    """Normalized persisted extraction evidence; not inferred from source wording."""

    schema_version: Literal["cleanup_ocr_evidence_v1"]
    extraction_identity: ExtractionIdentity
    text_sha256: Digest
    spans: tuple[CanonicalOcrSpan, ...]


class CleanupEligibility(CleanupContract):
    status: Literal["eligible", "not_applicable", "skipped"]
    reason: Literal[
        "native_source", "ocr_evidence_missing", "ocr_evidence_mismatch",
        "extraction_mode_unsupported",
    ] | None

    @model_validator(mode="after")
    def coherent(self):
        allowed = {
            "eligible": {None}, "not_applicable": {"native_source"},
            "skipped": {"ocr_evidence_missing", "ocr_evidence_mismatch", "extraction_mode_unsupported"},
        }
        if self.reason not in allowed[self.status]:
            raise ValueError("invalid_cleanup_eligibility")
        return self


def cleanup_eligibility(source: ExtractionResult, evidence: ActualOcrEvidence | None = None):
    """Inspect an accepted snapshot only; no loader, model or storage calls."""
    mode = source.identity.mode
    if mode == "ocr":
        return CleanupEligibility(status="eligible", reason=None)
    if mode in {"fast", "native"}:
        return CleanupEligibility(status="not_applicable", reason="native_source")
    if mode != "auto":
        return CleanupEligibility(status="skipped", reason="extraction_mode_unsupported")
    if evidence is None:
        return CleanupEligibility(status="skipped", reason="ocr_evidence_missing")
    if not isinstance(evidence, ActualOcrEvidence):
        return CleanupEligibility(status="skipped", reason="ocr_evidence_mismatch")
    previous_end = 0
    matches = evidence.extraction_identity == source.identity and evidence.text_sha256 == source.text_sha256
    for span in evidence.spans:
        if span.char_start < previous_end or span.char_end > source.text_length:
            matches = False
        previous_end = span.char_end
    if not matches:
        return CleanupEligibility(status="skipped", reason="ocr_evidence_mismatch")
    if not evidence.spans:
        return CleanupEligibility(status="not_applicable", reason="native_source")
    return CleanupEligibility(status="eligible", reason=None)


class CleanupBinding(CleanupContract):
    """Backend preconditions; construction alone is not consent or commit authority."""

    operation_id: Identifier
    document_id: Identifier
    namespace: Identifier
    extraction_identity: ExtractionIdentity
    text_sha256: Digest
    text_length: int = Field(gt=0)
    structure_version: int = Field(ge=1)
    structure_fingerprint: Digest
    expected_reading_revision: int = Field(ge=0)
    source_representation: Literal["canonical_text_v1", "readable_full_source_v1"]
    prompt_version: Identifier
    policy_version: Literal["ocr_cleanup_policy_v1"]
    consent_scope: Literal["ocr_cleanup_reading_overlay_v1"]

    def matches_source(self, source: ExtractionResult) -> bool:
        return (self.extraction_identity == source.identity
                and self.text_sha256 == source.text_sha256
                and self.text_length == source.text_length)


class BoundCleanupProposal(CleanupContract):
    binding: CleanupBinding
    proposal: CleanupProposal

    @classmethod
    def bind(cls, *, source: ExtractionResult, binding: CleanupBinding, proposal: CleanupProposal):
        if not binding.matches_source(source):
            raise ValueError("cleanup_source_mismatch")
        return cls(binding=binding, proposal=proposal)


class CleanupOutcome(CleanupContract):
    """Terminal cleanup result, independent of structural change_kind/success."""

    status: Literal["not_applicable", "no_changes", "completed", "partial", "skipped", "failed"]
    proposed_count: int | None = Field(ge=0)
    applied_count: int = Field(ge=0)
    skipped_count: int = Field(ge=0)
    reason: OutcomeReason | None
    reading_content_changed: bool
    reading_revision: int = Field(ge=0)

    @model_validator(mode="after")
    def coherent(self):
        if self.proposed_count is None:
            if self.applied_count or self.skipped_count:
                raise ValueError("unknown_cleanup_count")
        elif self.proposed_count != self.applied_count + self.skipped_count:
            raise ValueError("cleanup_count_mismatch")
        changed = self.status in {"completed", "partial"}
        if self.reading_content_changed != changed or bool(self.applied_count) != changed:
            raise ValueError("cleanup_change_mismatch")
        if changed and self.reading_revision == 0:
            raise ValueError("cleanup_revision_not_advanced")
        allowed = {
            "not_applicable": {"native_source"}, "no_changes": {None},
            "completed": {None}, "partial": {"some_items_rejected"},
            "skipped": {"ocr_evidence_missing", "ocr_evidence_mismatch", "extraction_mode_unsupported",
                        "capacity_exceeded", "no_safe_items", "source_changed", "reading_revision_changed", "cancelled",
                        "structure_not_ready", "cleanup_policy_changed", "capacity_unavailable"},
            "failed": {"provider_failed", "invalid_plan", "storage_failed"},
        }
        if self.reason not in allowed[self.status]:
            raise ValueError("cleanup_reason_mismatch")
        if self.status == "no_changes" and self.proposed_count != 0:
            raise ValueError("cleanup_no_changes_requires_empty_plan")
        if self.status == "not_applicable" and self.proposed_count is not None:
            raise ValueError("cleanup_not_requested")
        if self.status == "completed" and self.skipped_count:
            raise ValueError("cleanup_completed_with_skips")
        if self.status == "partial" and not self.skipped_count:
            raise ValueError("cleanup_partial_without_skips")
        return self
