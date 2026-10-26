"""Exact region drafts and owned repairs; the caller owns the transaction."""

from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
import hashlib

from document_structure.ocr_cleanup_overlay import canonical_text
from document_structure.reading_content import (
    ReadingContentConflict, original_blocks, project_document,
    source_token_for,
    state_for,
)
from document_structure.source_review_blocks import region_source_intersections
from document_structure.reading_repair_guards import manual_block_protection, source_ranges_overlap, same_source_span


def _application_proof(mapping, state, region_id, text):
    application = state.get("region_applications", {}).get(region_id)
    owned = [edit for edit in state.get("source_edits", []) if edit.get("source_region_id") == region_id]
    if application is None:
        return None, [], "region_application_changed" if owned else None
    span = mapping.get("canonical_span") or {}
    item = {"char_start": span.get("char_start"), "char_end": span.get("char_end")}
    if type(item["char_start"]) is not int or type(item["char_end"]) is not int:
        return application, [], "region_application_changed"
    item["quote"] = text[item["char_start"]:item["char_end"]]
    expected = {**item, "replacement": application.get("text"), "source_region_id": region_id}
    if (not same_source_span(application, item) or not isinstance(application.get("text"), str)
            or application.get("block_spans") != mapping.get("block_spans")
            or application.get("text_sha256") != span.get("text_sha256") or owned != [expected]):
        return application, [], "region_application_changed"
    return application, owned, None


def _blocked_reason(mapping, state, region_id, text, text_hash):
    if state.get("regions", {}).get(region_id):
        return "region_not_active"
    if mapping["mapping_status"] != "complete":
        return mapping["mapping_reason"]
    links = mapping.get("block_spans", [])
    if not links:
        return "block_mapping_unavailable"
    seen = set()
    previous_end = -1
    for link in links:
        key = link["block_id"]
        if (not all(link[name] for name in ("chapter_id", "section_id", "task_unit_id", "block_id"))
                or key in seen or link["source_start"] < previous_end):
            return "ambiguous_block_coverage"
        seen.add(key)
        previous_end = link["source_end"]
    # A source repair is all-or-none, including when only one cross-block target
    # was changed in the body. Retained audit protects edits/deletions after Undo.
    protection = manual_block_protection(state, seen)
    if protection:
        return "block_manually_" + protection
    if any(key in state.get("blocks", {}) or key in state.get("review_kept_blocks", {}) for key in seen):
        return "reading_range_changed"
    _, owned, reason = _application_proof(mapping, state, region_id, text)
    if reason:
        return reason
    operations = list(state.get("source_edits", [])) + list(state.get("cleanup_spans", []))
    if operations and state.get("cleanup_source_sha256") != text_hash:
        return "reading_source_changed"
    span = mapping["canonical_span"]
    for operation in operations:
        start, end = operation.get("char_start"), operation.get("char_end")
        if (type(start) is not int or type(end) is not int or not 0 <= start < end <= len(text)
                or operation.get("quote") != text[start:end]):
            return "reading_source_changed"
        if source_ranges_overlap(operation, span) and operation not in owned:
            return "reading_range_changed"
    return None


def region_repair_targets(document):
    """Return source-bound advisory eligibility for a canonical repository snapshot.

    Commit callers resolve targets again under the repository lock; a prior result
    is not a write authorization. Only a proven own overlay can be replaced.
    """
    mappings = region_source_intersections(document)
    text = canonical_text(document)
    text_hash = hashlib.sha256(text.encode()).hexdigest()
    state = state_for(document)
    regions = {}
    for region_id, mapping in mappings.items():
        reason = _blocked_reason(mapping, state, region_id, text, text_hash)
        regions[region_id] = {
            **mapping, "canonical_span": mapping.get("canonical_span"),
            "block_spans": mapping.get("block_spans", []),
            "can_apply": reason is None, "can_delete": reason is None,
            "blocked_reason": reason,
        }
    version = document.parse_provenance.get("storage_structure_version")
    return {"document_id": document.document_id, "source_token": source_token_for(document),
            "structure_version": str(version) if version is not None else None,
            "revision": state.get("revision", 0), "canonical_sha256": text_hash,
            "offset_basis": "canonical_unicode_code_points", "regions": regions}


def region_repair_descriptions(document):
    """Minimal public projections: never expose executable coordinates/ownership."""
    targets = region_repair_targets(document)["regions"]
    state = state_for(document)
    originals = {r["id"]: r.get("raw_candidate", "")
                 for r in document.parse_provenance.get("source_review", {}).get("regions", [])}
    result = {}
    for region_id, target in targets.items():
        draft = state.get("region_drafts", {}).get(region_id, {})
        applied = state.get("region_applications", {}).get(region_id)
        result[region_id] = {"text": draft.get("text", originals[region_id]),
            "edited_at": draft.get("at"), "applied_text": applied.get("text") if applied else None,
            "applied_at": applied.get("at") if applied else None,
            "can_edit": target["can_apply"], "can_apply": target["can_apply"],
            "can_delete": target["can_delete"], "blocked_reason": target["blocked_reason"]}
    return result


def change_region(document, state, *, action, region_id, text, discard_interactions):
    targets = region_repair_targets(document)
    target = targets["regions"].get(region_id)
    if target is None:
        raise ReadingContentConflict("review_region_missing")
    if not target["can_apply"]:
        raise ReadingContentConflict(target["blocked_reason"])
    now = datetime.now(timezone.utc).isoformat()
    if action == "region_edit":
        state.setdefault("region_drafts", {})[region_id] = {"text": text, "at": now,
            "source_token": targets["source_token"]}
        return [], []
    span = target["canonical_span"]
    canonical = canonical_text(document)
    quote = canonical[span["char_start"]:span["char_end"]]
    draft = state.get("region_drafts", {}).get(region_id)
    if draft and draft.get("source_token") != targets["source_token"]:
        raise ReadingContentConflict("region_draft_source_changed")
    replacement = "" if action == "region_delete" else (text if text is not None else draft["text"] if draft else quote)
    _, owned, reason = _application_proof(target, state, region_id, canonical)
    if reason:
        raise ReadingContentConflict(reason)
    state["source_edits"] = [edit for edit in state.get("source_edits", []) if edit not in owned]
    item = {"char_start": span["char_start"], "char_end": span["char_end"], "quote": quote}
    state["source_edits"].append({**item, "replacement": replacement, "source_region_id": region_id})
    state["cleanup_source_sha256"] = targets["canonical_sha256"]
    state.setdefault("region_applications", {})[region_id] = {**item, "text": replacement,
        "block_spans": deepcopy(target["block_spans"]), "text_sha256": span["text_sha256"], "at": now}
    if text is not None:
        state.setdefault("region_drafts", {})[region_id] = {"text": text, "at": now,
            "source_token": targets["source_token"]}
    updated = replace(document, parse_provenance={**document.parse_provenance, "reading_content": state})
    visible = {block.block_id for _, _, _, block in original_blocks(project_document(updated))}
    affected = target["block_ids"]
    # Visibility deletion must also invalidate Note targets, even for cross-block
    # replacements whose surviving text appears only in the first block.
    cleared = [key for key in affected if key not in visible or discard_interactions is True]
    return affected, cleared
