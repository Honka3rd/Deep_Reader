"""Explicit cleanup workflow and read-only reconciliation of committed operations."""

from dataclasses import replace
import logging
from time import perf_counter

from app.enhancement_operations import EnhancementOperations
from doc_loaders.extraction_identity import ExtractionResult, read_extraction_record
from document_structure.ocr_cleanup_contract import (
    CLEANUP_CONSENT_SCOPE, CLEANUP_POLICY_VERSION, BoundCleanupProposal, CleanupBinding,
    CleanupProposal, CleanupOutcome, cleanup_eligibility,
)
from document_structure.ocr_cleanup_planner import REMAINING_CLEANUP_PROMPT_VERSION
from document_structure.reading_content import pending_review_regions, state_for
from section_tasks.reparse_document_structure_result import ReparseDocumentStructureResult

LOGGER = logging.getLogger(__name__)


def cleanup_preflight(state, enabled):
    source = state.source
    context = source.cleanup_context
    eligible = cleanup_eligibility(ExtractionResult(source.extraction_identity, source.canonical_text),
                                  context.evidence if context else None)
    return dict(enabled=enabled, eligibility=eligible.status, reason=eligible.reason,
                scope=CLEANUP_CONSENT_SCOPE, policy_version=CLEANUP_POLICY_VERSION,
                reading_revision=state.reading_revision,
                invalidation="generated_interactions_only_if_reading_content_changed",
                preserves_notes_when_structure_unchanged=True)


def prepare_cleanup(coordinator, request, state, candidate):
    started = perf_counter()
    source = state.source
    binding = CleanupBinding(operation_id=EnhancementOperations._key(request),
        document_id=source.document_id, namespace=source.namespace,
        extraction_identity=source.extraction_identity, text_sha256=source.text_sha256,
        text_length=source.text_length, structure_version=source.structure_version,
        structure_fingerprint=state.interaction_revision,
        expected_reading_revision=state.reading_revision, source_representation="readable_full_source_v1",
        prompt_version=REMAINING_CLEANUP_PROMPT_VERSION, policy_version=CLEANUP_POLICY_VERSION,
        consent_scope=CLEANUP_CONSENT_SCOPE)
    from document_preparation.cleanup_input_adapter import adapt_document_cleanup_input, CleanupInputUnavailable
    from document_preparation.cleanup_handoff import PreparedCleanup
    try:
        baseline = adapt_document_cleanup_input(document=state.document, snapshot=source,
            expected_reading_revision=state.reading_revision)
        prepared = coordinator.document_preparation_pipeline.prepare_enhancement_cleanup(
            candidate=candidate, binding=binding, baseline=baseline)
    except CleanupInputUnavailable:
        prepared = PreparedCleanup(application=None, outcome=CleanupOutcome(
            status="failed", reason="invalid_plan", proposed_count=None, applied_count=0,
            skipped_count=0, reading_content_changed=False, reading_revision=state.reading_revision))
    if prepared.application is None:
        # Still carry existing reading overlays through a changed hierarchy. This
        # empty application authorizes no deletion; retain the planner's warning.
        from document_structure.ocr_cleanup_overlay import CleanupApplication
        context = source.cleanup_context
        application = CleanupApplication(BoundCleanupProposal(binding=binding,
            proposal=CleanupProposal(schema_version="ocr_cleanup_plan_v1", items=())),
            evidence=context.evidence if context else None, pages=context.pages if context else ())
        prepared = PreparedCleanup(application=application, outcome=prepared.outcome)
    LOGGER.info("enhancement_stage operation=%s stage=cleanup status=%s reason=%s elapsed_ms=%.1f",
                binding.operation_id, prepared.outcome.status if prepared.outcome else "proposed",
                prepared.outcome.reason if prepared.outcome else None, (perf_counter() - started) * 1000)
    return prepared


def read_committed_operation(coordinator, request):
    """A receipt proves commit, never generation-in-progress or layout readiness."""
    if not request.get("allow_ocr_cleanup"):
        return None
    key = EnhancementOperations._key(request)
    try:
        stored = coordinator.document_artifact_repository.load_enhancement_snapshot(request["doc_name"])
    except FileNotFoundError:
        return None
    document = stored.document
    receipt = document.parse_provenance.get("enhancement_receipts", {}).get(key)
    if receipt is None:
        return None
    LOGGER.info("enhancement_stage operation=%s stage=reconcile status=committed", key)
    result = ReparseDocumentStructureResult.ok(doc_name=request["doc_name"], parser_mode="llm_enhanced",
        structured_document_path=None, structure_version=receipt["structure_version"],
        change_kind=receipt["change_kind"], section_count=receipt["section_count"])
    current_revision = state_for(document).get("revision", 0)
    identity, extraction = read_extraction_record(document.parse_provenance)
    current = (stored.structure_version == receipt["structure_version"]
               and current_revision == receipt["reading_revision"]
               and identity.to_dict() == receipt["extraction_identity"]
               and extraction["text_sha256"] == receipt["text_sha256"])
    count = len(pending_review_regions(document))
    result = replace(result, cleanup=receipt["cleanup"], pending_review_count=count, review_required=bool(count))
    if not current:
        return replace(result, phase="superseded", reading_layout_required=None,
                       layout_error="enhancement_result_superseded")
    ready = coordinator.reading_layout_ready(document)
    return replace(result, phase="complete" if ready else "layout", reading_layout_required=not ready,
                   retryable=not ready, retry_stage=None if ready else "layout")


def finish_layout(coordinator, request):
    """Explicit POST continuation. Reads/result polling never invoke this function."""
    from app.enhancement_preflight import EnhancementConflict
    from document_preparation.accepted_source_snapshot import SourceNotReady
    from document_preparation.cleanup_input_adapter import CleanupInputUnavailable
    try:
        with coordinator.enhancement_lease(request["doc_name"]):
            result = read_committed_operation(coordinator, request)
            if result is None:
                raise EnhancementConflict("enhancement_outcome_unknown")
            if result.phase != "layout":
                return result
            # Verify the trusted source file before allowing post-restart mutation.
            try:
                snapshot = coordinator.read_cleanup_input(request["doc_name"])
            except SourceNotReady:
                raise EnhancementConflict("source_changed") from None
            except CleanupInputUnavailable:
                return replace(result, layout_error="enhancement_layout_failed")
            try:
                started = perf_counter()
                coordinator.prepare_cleanup_layout(snapshot)
            except Exception:
                LOGGER.warning("enhancement_stage operation=%s stage=layout status=failed",
                               EnhancementOperations._key(request))
                return replace(result, layout_error="enhancement_layout_failed")
            LOGGER.info("enhancement_stage operation=%s stage=layout status=completed elapsed_ms=%.1f",
                        EnhancementOperations._key(request), (perf_counter() - started) * 1000)
            return read_committed_operation(coordinator, request)
    except BlockingIOError:
        raise EnhancementConflict("enhancement_in_progress") from None
