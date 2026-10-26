"""Reversible canonical overlays staged before the enhancement transaction writes."""

from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import logging
from time import perf_counter

from doc_loaders.extraction_identity import ExtractionResult, read_extraction_record
from document_structure.ocr_cleanup_contract import ActualOcrEvidence, BoundCleanupProposal, CleanupOutcome
from document_structure.ocr_cleanup_validation import CleanupValidationError, validate_cleanup_proposal
from document_structure.reading_content import ReadingContentConflict, digest, state_for
from document_structure.source_mapped_text import PageRange, SourceMappedText, SourceRange
from document_structure.structured_document import StructuredDocument
from document_structure.cleanup_input import CleanupInput
from shared.task_artifacts import DocumentTaskArtifacts


@dataclass(frozen=True)
class CleanupApplication:
    plan: BoundCleanupProposal
    evidence: ActualOcrEvidence | None = None
    readable_view: SourceMappedText | None = None
    pages: tuple[PageRange, ...] = ()
    cleanup_input: CleanupInput | None = None
    baseline_digest: str | None = None
    candidate_digest: str | None = None

    def command(self, candidate):
        command = [self.plan.model_dump(mode="json"), candidate.to_dict(),
                       self.evidence.model_dump(mode="json") if self.evidence else None,
                       [asdict(page) for page in self.pages]]
        if self.cleanup_input is not None:
            command.extend([self.cleanup_input.fingerprint(), self.baseline_digest, self.candidate_digest])
        return digest(command)


@dataclass(frozen=True)
class CleanupFileCommitResult:
    document: StructuredDocument
    change_kind: str
    cleanup: dict


def log_cleanup(application, stage, *, started, outcome=None):
    logging.getLogger(__name__).info(
        "ocr_cleanup operation=%s stage=%s elapsed_ms=%.1f status=%s applied=%s skipped=%s",
        digest(application.plan.binding.operation_id)[:16], stage,
        (perf_counter() - started) * 1000, (outcome or {}).get("status"),
        (outcome or {}).get("applied_count"), (outcome or {}).get("skipped_count"))


def canonical_text(document):
    return document.raw_text or "".join(s.content for c in document.chapters for s in c.sections)


def replay_cleanup(document, application, candidate):
    receipt = state_for(document).get("cleanup_receipts", {}).get(application.plan.binding.operation_id)
    if receipt is not None and receipt["command"] != application.command(candidate):
        raise ReadingContentConflict("cleanup_operation_conflict")
    return receipt


def unit_ranges(section):
    """Map exact ordered canonical units; never guess normalized or reordered text."""
    cursor = 0
    result = {}
    for unit in section.task_units:
        if not unit.content:
            continue
        start = section.content.find(unit.content, cursor)
        if start < 0 or section.content[cursor:start].strip():
            raise ReadingContentConflict("cleanup_unit_mapping_unavailable")
        result[unit.unit_id] = SourceRange(section.char_start + start,
                                          section.char_start + start + len(unit.content))
        cursor = start + len(unit.content)
    if section.task_units and section.content[cursor:].strip():
        raise ReadingContentConflict("cleanup_unit_mapping_unavailable")
    return result


def block_range(unit, block, unit_span):
    metadata = block.metadata or {}
    start, end = metadata.get("quote_span_start"), metadata.get("quote_span_end")
    if (type(start) is not int or type(end) is not int
            or not 0 <= start < end <= len(unit.content)):
        start = unit.content.find(block.content)
        if start < 0 or unit.content.find(block.content, start + 1) >= 0:
            raise ReadingContentConflict("cleanup_block_mapping_unavailable")
        end = start + len(block.content)
    return SourceRange(unit_span.start + start, unit_span.start + end)


def manual_source_edits(document):
    from document_structure.reading_content import original_blocks
    state = state_for(document)
    result = deepcopy(state.get("source_edits", []))
    edits = state.get("blocks", {})
    if not edits:
        return result
    ranges = {s.section_id: unit_ranges(s) for c in document.chapters for s in c.sections}
    seen = set()
    text = canonical_text(document)
    for _, section, unit, block in original_blocks(document):
        if block.block_id not in edits:
            continue
        span = block_range(unit, block, ranges[section.section_id][unit.unit_id])
        edit = edits[block.block_id]
        if any(prior["char_start"] < span.end and span.start < prior["char_end"]
               and not (span.start <= prior["char_start"] and prior["char_end"] <= span.end)
               for prior in result):
            raise ReadingContentConflict("cleanup_manual_edit_mapping_unavailable")
        result = [prior for prior in result if not (
            prior["char_start"] < span.end and span.start < prior["char_end"])]
        result.append({"char_start": span.start, "char_end": span.end,
                       "quote": text[span.start:span.end],
                       "replacement": "" if edit.get("deleted") else edit["content"]})
        seen.add(block.block_id)
    if seen != set(edits):
        raise ReadingContentConflict("cleanup_manual_edit_mapping_unavailable")
    return result


def reading_slice(text, start, end, state):
    """Project a canonical interval; manual replacements take priority over cleanup."""
    edits = state.get("source_edits", [])
    deletions = state.get("cleanup_spans", [])
    operations = list(edits) + [span for span in deletions if not any(
        span["char_start"] < edit["char_end"] and edit["char_start"] < span["char_end"] for edit in edits)]
    cursor, parts = start, []
    for span in sorted(operations, key=lambda item: item["char_start"]):
        left, right = span["char_start"], span["char_end"]
        if right <= start or left >= end:
            continue
        if text[left:right] != span["quote"]:
            raise ReadingContentConflict("cleanup_source_span_changed")
        parts.append(text[cursor:max(cursor, left)])
        # A replacement spanning new blocks appears once, at its canonical start.
        if start <= left < end:
            parts.append(span.get("replacement", ""))
        cursor = max(cursor, min(end, right))
    parts.append(text[cursor:end])
    return "".join(parts)


def invalidate_generated(document):
    artifacts = document.document_task_artifacts or DocumentTaskArtifacts()
    metadata = {key: value for key, value in artifacts.metadata.items() if key not in {
        "reading_interaction_artifacts", "ancestor_summary_artifacts", "public_summary"}}
    return replace(document, document_task_artifacts=replace(artifacts, metadata=metadata, chapter_artifacts={}),
        chapters=[replace(c, task_artifacts=None, sections=[replace(s, task_artifacts=None,
            task_units=[replace(u, task_artifacts=None) for u in s.task_units]) for s in c.sections])
            for c in document.chapters])


def resolved_cleanup_regions(document, source, spans):
    resolved = []
    text, text_hash = source.canonical_text, source.text_sha256
    review = document.parse_provenance.get("source_review", {})
    if review.get("source_sha256") != source.identity.source_sha256:
        return resolved
    for region in review.get("regions", []):
        mapping = region.get("canonical_span", {})
        if not isinstance(mapping, dict):
            continue
        left, right = mapping.get("char_start"), mapping.get("char_end")
        if (mapping.get("text_sha256") != text_hash or type(left) is not int or type(right) is not int
                or not 0 <= left < right <= len(text) or not text[left:right].strip()
                or region.get("raw_candidate") != text[left:right]):
            continue
        cursor = left
        for span in sorted(spans, key=lambda value: value["char_start"]):
            if span["char_start"] <= cursor < span["char_end"]:
                cursor = span["char_end"]
        if cursor >= right:
            resolved.append(region["id"])
    return resolved


def stage_cleanup(current, candidate, application, *, namespace, structure_version,
                  fingerprint, change_kind):
    """Storage calls under its lock, before writes. LLM/provider work is forbidden."""
    started = perf_counter()
    binding = application.plan.binding
    old_state = state_for(current)
    expected = binding.model_copy(update={
        "namespace": namespace, "document_id": current.document_id,
        "structure_version": structure_version, "structure_fingerprint": fingerprint,
        "expected_reading_revision": old_state.get("revision", 0),
    })
    if binding != expected:
        raise ReadingContentConflict("cleanup_confirmation_stale")
    text = canonical_text(current)
    identity, _ = read_extraction_record(current.parse_provenance)
    source = ExtractionResult(identity, text)
    if not binding.matches_source(source):
        raise ReadingContentConflict("cleanup_source_changed")
    snapshot = application.cleanup_input
    if snapshot is not None and (
            application.baseline_digest != digest(current.to_dict())
            or application.candidate_digest != digest(candidate.to_dict())
            or snapshot.source != source or snapshot.namespace != namespace
            or snapshot.document_id != current.document_id
            or snapshot.structure_version != structure_version
            or snapshot.reading_revision != old_state.get("revision", 0)):
        raise ReadingContentConflict("cleanup_confirmation_stale")
    edits = manual_source_edits(current)
    state = deepcopy(old_state)
    if change_kind == "structure_changed":
        # Block IDs belong to the old hierarchy. Carry user edits by source position.
        state["blocks"] = {}
        state["source_edits"] = edits
        state["undo"] = None
        if edits:
            state["cleanup_source_sha256"] = source.text_sha256
    protected = [SourceRange(edit["char_start"], edit["char_end"]) for edit in edits]
    protected.extend(SourceRange(span["char_start"], span["char_end"])
                     for span in state.get("cleanup_spans", []))
    result = None
    try:
        if snapshot is not None:
            from document_structure.ocr_cleanup_validation import validate_cleanup_input
            result = validate_cleanup_input(application.plan.proposal, snapshot=snapshot,
                                             expected_fingerprint=snapshot.fingerprint())
        else:
            result = validate_cleanup_proposal(application.plan, source=source,
                expected_binding=expected, candidate=candidate, page_ranges=application.pages,
                user_edit_ranges=tuple(protected), readable_view=application.readable_view,
                evidence=application.evidence)
    except CleanupValidationError:
        # Optional validation failure retains the usable candidate and old overlay.
        pass
    count = len(application.plan.proposal.items)
    staged, outcome = stage_validated_cleanup(
        current, candidate, source=source, result=result, state=state,
        operation_id=binding.operation_id, proposed_count=count, change_kind=change_kind)
    receipts = staged.parse_provenance["reading_content"].setdefault("cleanup_receipts", {})
    if len(receipts) >= 1000:
        raise ReadingContentConflict("cleanup_history_limit_reached")
    receipts[binding.operation_id] = {"command": application.command(candidate),
                                    "change_kind": str(change_kind), "outcome": outcome}
    log_cleanup(application, "staged", started=started, outcome=outcome)
    return staged, outcome


def stage_validated_cleanup(current, candidate, *, source, result, state,
                            operation_id, proposed_count, change_kind):
    """Shared pure overlay application. Caller validates and owns receipt/commit."""
    state = deepcopy(state)
    text = source.canonical_text
    count = proposed_count
    accepted = list(result.deletions) if result else []
    previous = {key: deepcopy(state.get(key, default)) for key, default in (
        ("blocks", {}), ("regions", {}), ("source_edits", []), ("cleanup_spans", []),
        ("cleanup_record_applications", {}), ("region_applications", {}))}
    before_cleanup = deepcopy(state)
    if accepted:
        state["cleanup_source_sha256"] = source.text_sha256
        # Canonical coordinates survive storage-assigned hierarchy IDs.
        records = [{key: value for key, value in asdict(item).items()
                    if key not in {"chapter_id", "section_id"}} for item in accepted]
        state.setdefault("cleanup_spans", []).extend(records)
        state["undo"] = previous
        state["revision"] = state_for(current).get("revision", 0) + 1
        resolved = resolved_cleanup_regions(current, source, state["cleanup_spans"])
        for region_id in resolved:
            state.setdefault("regions", {}).setdefault(region_id, "cleanup_resolved")
        state.setdefault("audit", []).append({"action": "ocr_cleanup", "operation_id": operation_id,
            "revision": state["revision"], "at": datetime.now(timezone.utc).isoformat(),
            "resolved_regions": resolved, "deletions": records,
            "rejections": [asdict(item) for item in result.rejections]})
        from document_structure.reading_content import project_document
        projection = candidate if change_kind == "structure_changed" else current
        try:
            if not reading_slice(text, 0, len(text), state).strip():
                raise ReadingContentConflict("cleanup_would_empty_document")
            bodies = [s for c in candidate.chapters for s in c.sections
                      if (s.section_role or c.chapter_role) == "main_body" and s.content.strip()]
            if bodies and not any(reading_slice(text, s.char_start, s.char_end, state).strip() for s in bodies):
                raise ReadingContentConflict("cleanup_would_empty_body")
            project_document(replace(projection, parse_provenance={
                **projection.parse_provenance, "reading_content": state}))
        except ReadingContentConflict:
            state, accepted = before_cleanup, []
            logging.getLogger(__name__).info("ocr_cleanup stage=application status=skipped reason=projection_invalid")
    status = ("completed" if len(accepted) == count else "partial") if accepted else (
        "no_changes" if not count else "skipped")
    reason = {"completed": None, "partial": "some_items_rejected", "no_changes": None,
              "skipped": "no_safe_items"}[status]
    outcome = CleanupOutcome(status=status, proposed_count=count, applied_count=len(accepted),
        skipped_count=count - len(accepted), reason=reason, reading_content_changed=bool(accepted),
        reading_revision=state.get("revision", 0))
    provenance = {**candidate.parse_provenance, "reading_content": state}
    for key in ("source_review", "source_page_spans", "actual_ocr_evidence"):
        if key in current.parse_provenance:
            provenance[key] = deepcopy(current.parse_provenance[key])
    staged = replace(candidate, parse_provenance=provenance)
    if change_kind != "structure_changed":
        staged = replace(staged, document_task_artifacts=current.document_task_artifacts)
    if accepted:
        from document_structure.reading_content_artifacts import clear_source_dependents
        from document_structure.reading_refresh import changed_blocks
        affected = changed_blocks(current, staged)
        for block_id in affected:
            state.setdefault("block_content_epochs", {})[block_id] = operation_id
        staged = clear_source_dependents(staged, affected)
        staged = invalidate_generated(staged)
    # Verify effective projection now; the caller can still roll back without writes.
    reading_slice(text, 0, len(text), state)
    return staged, outcome.model_dump(mode="json")
