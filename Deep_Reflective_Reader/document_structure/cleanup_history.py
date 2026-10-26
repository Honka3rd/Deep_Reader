"""Exact cleanup-record projection and mutations; repository owns the transaction."""

from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
import hashlib

from document_structure.reading_content import (
    ReadingContentConflict, digest, original_blocks, project_document, state_for,
)
from document_structure.ocr_cleanup_overlay import canonical_text, block_range, unit_ranges
from document_structure.reading_repair_guards import manual_block_protection, source_ranges_overlap, same_source_span


def record_id(batch, item):
    return digest([batch.get("operation_id"), batch["revision"], item])


def block_index(document):
    text = canonical_text(document)
    ranges = {}
    blocks = []
    for chapter, section, unit, block in original_blocks(document):
        if section.section_id not in ranges:
            ranges[section.section_id] = unit_ranges(section)
        span = block_range(unit, block, ranges[section.section_id][unit.unit_id])
        blocks.append((span, {"chapter_id": chapter.chapter_id, "section_id": section.section_id,
                             "task_unit_id": unit.unit_id, "block_id": block.block_id}))
    return text, sorted(blocks, key=lambda row: row[0].start)


def _application(state, identity, item):
    application = state.get("cleanup_record_applications", {}).get(identity)
    if application is not None:
        return application
    # Old restores did not store ownership. Only the exact last write plus its
    # Undo snapshot can prove which overlay it added; draft text is not evidence.
    latest = next((e for e in reversed(state.get("audit", []))
                   if e.get("action") not in {"cleanup_edit", "cleanup_delete", "region_edit"}), {})
    undo = state.get("undo") or {}
    previous_spans = list(undo.get("cleanup_spans", []))
    if (latest.get("action") != "cleanup_restore"
            or latest.get("cleanup_record_id") != identity or item not in previous_spans):
        return None
    previous_spans.remove(item)
    if previous_spans != state.get("cleanup_spans", []):
        return None
    additions = list(state.get("source_edits", []))
    for edit in undo.get("source_edits", []):
        if edit not in additions:
            return None
        additions.remove(edit)
    if len(additions) > 1:
        return None
    edit = additions[0] if additions else None
    if edit is not None and (edit.get("cleanup_record_id") is not None or any(
            edit.get(key) != item[key] for key in ("char_start", "char_end", "quote"))):
        return None
    return {**{key: item[key] for key in ("char_start", "char_end", "quote")},
            "text": edit.get("replacement") if edit else item["quote"], "legacy": True}


def _application_state(state, identity, item):
    application = _application(state, identity, item)
    overlaps = [edit for edit in state.get("source_edits", [])
                if source_ranges_overlap(edit, item)]
    if application is None:
        return None, [], "block_edited" if overlaps else None
    if (not same_source_span(application, item)
            or not isinstance(application.get("text"), str) or not application["text"].strip()):
        return application, [], "application_changed"
    owned = [edit for edit in overlaps
             if same_source_span(edit, item)
             and edit.get("replacement") == application["text"]
             and (edit.get("cleanup_record_id") == identity
                  or (application.get("legacy") and edit.get("cleanup_record_id") is None))]
    expected = int(application["text"] != item["quote"])
    if len(owned) != expected or len(overlaps) != expected:
        return application, [], "application_changed"
    return application, owned, None


def describe_record(document, batch, item, index, visible):
    state = state_for(document)
    identity = record_id(batch, item)
    saved = state.get("cleanup_record_edits", {}).get(identity, {})
    active = item in state.get("cleanup_spans", [])
    links, reason = [], None
    text, blocks = index
    start, end = item.get("char_start"), item.get("char_end")
    if (state.get("cleanup_source_sha256") != hashlib.sha256(text.encode()).hexdigest()
            or type(start) is not int or type(end) is not int
            or not 0 <= start < end <= len(text) or text[start:end] != item["quote"]):
        reason = "source_changed"
    else:
        cursor = start
        for span, target in blocks:
            left, right = max(start, span.start), min(end, span.end)
            if left >= right:
                continue
            if left < cursor or text[cursor:left].strip():
                reason = "mapping_unavailable"
            cursor = max(cursor, right)
            links.append({**target, "char_from": left - span.start, "char_to": right - span.start,
                          "can_navigate": target["block_id"] in visible})
        if not links or text[cursor:end].strip():
            reason = "mapping_unavailable"
    ids = {link["block_id"] for link in links}
    pinned = state.get("cleanup_record_links", {}).get(identity)
    if pinned is not None and pinned != [{k: v for k, v in link.items() if k != "can_navigate"} for link in links]:
        reason = "mapping_changed"
        links = []
    if reason is None:
        protection = manual_block_protection(state, ids, after_revision=batch["revision"])
        if protection == "deleted" and any(state.get("blocks", {}).get(key, {}).get("deleted") for key in ids):
            reason = "block_deleted"
        elif protection is not None:
            reason = "block_edited"
        elif any(key in state.get("blocks", {}) for key in ids):
            reason = "block_edited"
        else:
            application, _, reason = _application_state(state, identity, item)
            if reason is None and not active and application is None:
                reason = "restore_ownership_unavailable"
            if reason is None and active and application is not None:
                reason = "application_changed"
    return {"record_id": identity, "quote": item["quote"],
            "text": saved.get("text", item["quote"]), "edited_at": saved.get("at"),
            "reason": item["reason"] if item["reason"] in {"ocr_garbage", "recognition_fragment"}
                      else "unreadable_fragment",
            "page_indices": sorted({p["page_index"] for p in item.get("pages", [])}),
            "links": links, "status": "cleaned" if active else "restored",
            "can_restore": reason is None, "restore_reason": reason, "can_delete": True}


def bind_cleanup_record_links(document):
    """Stage final block identities on explicit writes; reads never backfill."""
    state = state_for(document)
    batches = [b for b in state.get("audit", []) if b.get("action") == "ocr_cleanup"]
    if not batches:
        return document
    try:
        index = block_index(document)
    except (ReadingContentConflict, KeyError, ValueError):
        return document
    links = deepcopy(state.get("cleanup_record_links", {}))
    for batch in batches:
        for item in batch.get("deletions", []):
            identity = record_id(batch, item)
            if identity not in links:
                row = describe_record(document, batch, item, index, set())
                if row["can_restore"]:
                    links[identity] = [{k: v for k, v in link.items() if k != "can_navigate"} for link in row["links"]]
    if links == state.get("cleanup_record_links", {}):
        return document
    return replace(document, parse_provenance={**document.parse_provenance,
        "reading_content": {**state, "cleanup_record_links": links}})


def history_for(document, offset=0):
    state = state_for(document)
    audit = state.get("audit", [])
    batches = [entry for entry in reversed(audit) if entry.get("action") == "ocr_cleanup"]
    latest = next((entry for entry in reversed(audit) if entry.get("action") not in {"cleanup_edit", "cleanup_delete", "region_edit"}), None)
    try:
        index = block_index(document)
        visible = {b.block_id for _, _, _, b in original_blocks(project_document(document))}
    except (ReadingContentConflict, KeyError, ValueError):
        index, visible = (canonical_text(document), []), set()
    items = []
    for batch in batches[offset:offset + 10]:
        records = [describe_record(document, batch, item, index, visible) for item in batch.get("deletions", [])]
        items.append({"revision": batch["revision"], "at": batch["at"],
            "applied_count": len(records), "skipped_count": len(batch.get("rejections", [])),
            "remaining_count": sum(row["status"] == "cleaned" for row in records),
            "undoable": bool(state.get("undo")) and batch is latest, "deletions": records})
    return {"total": len(batches), "offset": offset, "limit": 10, "items": items}


def record_change(document, event):
    """Return only the changed history row and its batch counters after commit/replay."""
    state = state_for(document)
    if event.get("action") not in {"cleanup_edit", "cleanup_delete"}:
        return None
    batch = next((b for b in state.get("audit", []) if b.get("action") == "ocr_cleanup"
                  and b["revision"] == event.get("cleanup_batch_revision")), None)
    if batch is None:
        return None
    identity = event["cleanup_record_id"]
    row = None
    if event["action"] == "cleanup_edit":
        item = next((i for i in batch["deletions"] if record_id(batch, i) == identity), None)
        if item is None:
            return None
        try:
            index = block_index(document)
            visible = {b.block_id for _, _, _, b in original_blocks(project_document(document))}
        except (ReadingContentConflict, KeyError, ValueError):
            index, visible = (canonical_text(document), []), set()
        row = describe_record(document, batch, item, index, visible)
    return {"batch_revision": batch["revision"], "record_id": identity,
            "deleted": event["action"] == "cleanup_delete", "record": row,
            "applied_count": len(batch["deletions"]),
            "remaining_count": sum(i in state.get("cleanup_spans", []) for i in batch["deletions"])}


def change_record(document, state, *, action, identity, text, previous):
    found = [(batch, item) for batch in state.get("audit", []) if batch.get("action") == "ocr_cleanup"
             for item in batch.get("deletions", []) if record_id(batch, item) == identity]
    if len(found) != 1:
        raise ReadingContentConflict("cleanup_record_missing")
    batch, item = found[0]
    if action == "cleanup_delete":
        batch["deletions"].remove(item)
        for key in ("cleanup_record_edits", "cleanup_record_links", "cleanup_record_applications"):
            state.get(key, {}).pop(identity, None)
        # History deletion is permanent; content overlays and text-only Undo remain.
        (state.get("undo") or {}).get("cleanup_record_applications", {}).pop(identity, None)
        return [], batch["revision"]
    now = datetime.now(timezone.utc).isoformat()
    if action == "cleanup_edit":
        drafts = state.setdefault("cleanup_record_edits", {})
        drafts[identity] = {**drafts.get(identity, {}), "text": text, "at": now}
        return [], batch["revision"]
    try:
        index = block_index(document)
    except (ReadingContentConflict, KeyError, ValueError):
        raise ReadingContentConflict("cleanup_mapping_unavailable") from None
    record = describe_record(document, batch, item, index, set())
    if not record["can_restore"]:
        raise ReadingContentConflict("cleanup_" + record["restore_reason"])
    replacement = record["text"] if text is None else text
    if not isinstance(replacement, str) or not replacement.strip():
        raise ValueError("reading_text_requires_1_to_10000_characters")
    application, owned, _ = _application_state(state, identity, item)
    if application is not None and application.get("legacy"):
        # Adopt the proven prior application in the Undo snapshot too, so Undo
        # retains its identity rather than stranding the old untagged overlay.
        previous["cleanup_record_applications"][identity] = {
            key: value for key, value in application.items() if key != "legacy"}
        for edit in previous["source_edits"]:
            if edit in owned:
                edit["cleanup_record_id"] = identity
    for edit in owned:
        state["source_edits"].remove(edit)
    if item in state.get("cleanup_spans", []):
        state["cleanup_spans"].remove(item)
    if replacement != item["quote"]:
        state.setdefault("source_edits", []).append({"char_start": item["char_start"],
            "char_end": item["char_end"], "quote": item["quote"], "replacement": replacement,
            "cleanup_record_id": identity})
    state.setdefault("cleanup_record_applications", {})[identity] = {
        **{key: item[key] for key in ("char_start", "char_end", "quote")}, "text": replacement}
    state.setdefault("cleanup_record_edits", {})[identity] = {"text": replacement, "at": now,
                                                               "links": deepcopy(record["links"])}
    # Reopen only regions previously auto-resolved by cleanup; explicit Ignore survives.
    from document_structure.ocr_cleanup_overlay import resolved_cleanup_regions
    from doc_loaders.extraction_identity import ExtractionResult, read_extraction_record
    extraction, _ = read_extraction_record(document.parse_provenance)
    resolved = set(resolved_cleanup_regions(document, ExtractionResult(extraction, index[0]), state["cleanup_spans"]))
    state["regions"] = {key: value for key, value in state.get("regions", {}).items()
                        if value != "cleanup_resolved" or key in resolved}
    return [link["block_id"] for link in record["links"]], batch["revision"]
