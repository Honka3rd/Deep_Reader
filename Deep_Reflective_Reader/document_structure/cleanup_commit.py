"""Shared locked-state checks and staging for content-only cleanup repositories."""

from dataclasses import replace
import json
import logging
from pathlib import Path
from time import perf_counter

from doc_loaders.extraction_identity import extraction_matches
from document_structure.cleanup_operation import CleanupCommitResult
from document_structure.cleanup_reading_view import cleanup_reading_view
from document_structure.ocr_cleanup_contract import ActualOcrEvidence, CleanupOutcome
from document_structure.ocr_cleanup_overlay import canonical_text, stage_validated_cleanup
from document_structure.ocr_cleanup_planner import REMAINING_CLEANUP_POLICY_VERSION, REMAINING_CLEANUP_PROMPT_VERSION
from document_structure.ocr_cleanup_validation import validate_cleanup_input
from document_structure.reading_content import ReadingContentConflict, digest, state_for, advance_layout_revision


RECEIPTS = "standalone_cleanup_receipts"


def log_cleanup_commit(binding, stage, started):
    logging.getLogger(__name__).info("ocr_cleanup operation=%s stage=%s elapsed_ms=%.1f",
        digest(binding.operation_id)[:16], stage, (perf_counter() - started) * 1000)


def verify_cleanup_source_file(document, source, *, accepted_source_path=None):
    # Reuse the existing bounded hash/race check; this does not prepare/extract.
    from document_preparation.accepted_source_snapshot import source_file_sha256, SourceNotReady
    path = accepted_source_path if accepted_source_path is not None else document.source_path
    if not path:
        raise ReadingContentConflict("cleanup_source_unavailable")
    try:
        actual = source_file_sha256(Path(path))
    except SourceNotReady as error:
        reason = "cleanup_source_changed" if error.reason == "source_changed" else "cleanup_source_unavailable"
        raise ReadingContentConflict(reason) from None
    if actual != source.identity.source_sha256:
        raise ReadingContentConflict("cleanup_source_changed")


def lookup_cleanup_receipt(document, *, namespace, document_id, operation_id):
    if document.document_id != document_id:
        raise ReadingContentConflict("cleanup_identity_changed")
    record = state_for(document).get(RECEIPTS, {}).get(digest([namespace, operation_id]))
    if record is None:
        return None
    result = CleanupCommitResult.model_validate_json(json.dumps(record["result"]))
    if (result.binding.namespace != namespace or result.binding.document_id != document_id
            or result.binding.operation_id != operation_id):
        raise ReadingContentConflict("cleanup_receipt_invalid")
    return result.model_copy(update={"replayed": True})


def stage_cleanup_commit(current, command, *, namespace, structure_version):
    """No I/O. Caller holds its storage lock through the sole final write."""
    snapshot, staged = command.snapshot, command.staged
    binding = staged.binding
    if namespace != binding.namespace or current.document_id != binding.document_id:
        raise ReadingContentConflict("cleanup_identity_changed")
    key = digest([namespace, binding.operation_id])
    command_hash = digest(command.model_dump(mode="json"))
    prior = lookup_cleanup_receipt(current, namespace=namespace, document_id=binding.document_id,
                                   operation_id=binding.operation_id)
    if prior:
        record = state_for(current)[RECEIPTS][key]
        if record["command"] != command_hash:
            raise ReadingContentConflict("cleanup_operation_conflict")
        if (structure_version != snapshot.structure_version
                or state_for(current).get("revision", 0) != prior.outcome.reading_revision
                or canonical_text(current) != snapshot.source.canonical_text
                or not extraction_matches(current.parse_provenance, snapshot.source.identity, canonical_text(current))):
            raise ReadingContentConflict("cleanup_operation_superseded")
        return current, prior
    if (not binding.matches(snapshot) or structure_version != snapshot.structure_version
            or current.to_dict() != snapshot.document().to_dict()):
        raise ReadingContentConflict("cleanup_confirmation_stale")
    if (binding.prompt_version != REMAINING_CLEANUP_PROMPT_VERSION
            or binding.policy_version != REMAINING_CLEANUP_POLICY_VERSION):
        raise ReadingContentConflict("cleanup_policy_changed")
    view = cleanup_reading_view(current)
    if (view.parts != snapshot.parts or view.prior_deletions != snapshot.prior_deletions
            or any(not any(p.char_start <= span.char_start and span.char_end <= p.char_end
                           for p in snapshot.protected_ranges) for span in view.protected_ranges)):
        raise ReadingContentConflict("cleanup_reading_mapping_changed")
    raw_evidence = current.parse_provenance.get("actual_ocr_evidence")
    evidence = ActualOcrEvidence.model_validate_json(json.dumps(raw_evidence)) if raw_evidence is not None else None
    # Page labels are retained in provenance, but are not executable coordinates.
    pages = [(p.page_index, p.start, p.end) for p in snapshot.pages]
    actual_pages = [(p["page_index"], p["char_start"], p["char_end"])
                    for p in current.parse_provenance.get("source_page_spans", [])]
    if evidence != snapshot.evidence or pages != actual_pages:
        raise ReadingContentConflict("cleanup_source_evidence_changed")
    resolved = validate_cleanup_input(staged.proposal, snapshot=snapshot,
                                       expected_fingerprint=binding.input_fingerprint)
    if resolved.deletions != staged.deletions or resolved.rejections != staged.rejections:
        raise ReadingContentConflict("cleanup_stage_changed")
    updated, raw_outcome = stage_validated_cleanup(current, current, source=snapshot.source,
        result=resolved, state=state_for(current), operation_id=binding.operation_id,
        proposed_count=resolved.proposed_count, change_kind="unchanged")
    outcome = CleanupOutcome.model_validate(raw_outcome)
    artifacts = updated.document_task_artifacts
    if outcome.reading_content_changed and artifacts:
        metadata = advance_layout_revision(artifacts.metadata, snapshot.reading_revision, outcome.reading_revision)
        updated = replace(updated, document_task_artifacts=replace(artifacts, metadata=metadata))
    metadata = updated.document_task_artifacts.metadata if updated.document_task_artifacts else {}
    layout = metadata.get("task_layout")
    # A carried revision is not full cache validation. The use case still checks
    # source/options/resolver freshness through the existing readiness owner.
    layout_required = (not isinstance(layout, dict) or layout.get("reading_revision") != outcome.reading_revision
                       or not any(s.task_units for c in updated.chapters for s in c.sections))
    result = CleanupCommitResult(binding=binding, outcome=outcome, replayed=False,
        reading_layout_required=layout_required,
        source_structure_fingerprint=snapshot.source_structure_fingerprint())
    receipts = updated.parse_provenance["reading_content"].setdefault(RECEIPTS, {})
    if len(receipts) >= 1000:
        raise ReadingContentConflict("cleanup_history_limit_reached")
    receipts[key] = {"command": command_hash, "result": result.model_dump(mode="json")}
    return updated, result
