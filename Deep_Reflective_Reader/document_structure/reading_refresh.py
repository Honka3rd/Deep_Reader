"""Read-only, revision-bound block refresh for explicit reading mutations."""

from dataclasses import replace
import hashlib

from document_structure.reading_content import (
    ReadingContentConflict, block_was_manually_edited, original_blocks, project_document,
    project_task_unit, projection_source, source_token_for, state_for,
)
from document_structure.ocr_cleanup_overlay import unit_ranges
from document_structure.text_normalization import normalize_ocr_text

PAGE_SIZE = 20


def changed_blocks(before, after):
    old = {b.block_id: b.content for _, _, _, b in original_blocks(project_document(before))}
    new = {b.block_id: b.content for _, _, _, b in original_blocks(project_document(after))}
    return [b.block_id for _, _, _, b in original_blocks(before)
            if old.get(b.block_id) != new.get(b.block_id)]


def mutation_refresh(document, mutation_id, *, block_offset=0):
    state = state_for(document)
    noop = state.get("no_op_mutations", {}).get(mutation_id)
    if noop:
        token = source_token_for(document)
        return {"document_id": document.document_id,
            "structure_version": str(document.parse_provenance["storage_structure_version"])
                if document.parse_provenance.get("storage_structure_version") is not None else None,
            "revision": state.get("revision", 0), "source_token": token,
            "mutation_id": mutation_id, "mutation_revision": noop["revision"], "cleanup_record_id": None,
            "scope_current": noop["revision"] == state.get("revision", 0) and noop["source_token"] == token,
            "generated_invalidated": False, "cleared_block_id": None, "body_changed": False,
            "summaries_invalidated": False, "translation_block_ids": [], "blocks": [],
            "block_page": {"total": 0, "offset": block_offset, "limit": PAGE_SIZE}}
    matches = [event for event in state.get("audit", []) if event.get("mutation_id") == mutation_id]
    if len(matches) != 1 or not matches[0].get("refresh_source_token"):
        return None
    event = matches[0]
    token = source_token_for(document)
    current = event["revision"] == state.get("revision", 0) and event["refresh_source_token"] == token
    ids = list(dict.fromkeys(event.get("affected_block_ids", [])))
    result = {"document_id": document.document_id,
        "structure_version": str(document.parse_provenance["storage_structure_version"])
            if document.parse_provenance.get("storage_structure_version") is not None else None,
        "revision": state.get("revision", 0), "source_token": token,
        "mutation_id": mutation_id, "mutation_revision": event["revision"],
        "cleanup_record_id": event.get("cleanup_record_id"), "scope_current": current,
        "generated_invalidated": event.get("invalidate_generated", False),
        "body_changed": event.get("body_changed", bool(ids)),
        "summaries_invalidated": event.get("summaries_invalidated", event.get("invalidate_generated", False)),
        "translation_block_ids": event.get("translation_block_ids", []),
        "cleared_block_id": event.get("cleared_block_id"), "blocks": [],
        "block_page": {"total": len(ids), "offset": block_offset, "limit": PAGE_SIZE}}
    if event.get("action") in {"region_edit", "region_apply", "region_delete"}:
        result["cleared_block_ids"] = event.get("cleared_block_ids", []) if current else []
    if current and event.get("action") in {"cleanup_edit", "cleanup_delete"}:
        from document_structure.cleanup_history import record_change
        change = record_change(document, event)
        if change is not None:
            result["cleanup_change"] = change
    if not current or not ids or block_offset >= len(ids):
        return result
    rows = original_blocks(document)
    known = {b.block_id for _, _, _, b in rows}
    if len(known) != len(rows) or not set(ids) <= known:
        raise ReadingContentConflict("reading_refresh_mapping_changed")
    # One canonical-order index, followed by projection of only this page's units.
    affected = set(ids)
    selected = [row for row in rows if row[3].block_id in affected][block_offset:block_offset + PAGE_SIZE]
    state, has_spans, text = projection_source(document)
    ranges, units, orders = {}, {}, {}
    for _, _, unit, block in rows:
        order = orders.setdefault(unit.unit_id, {})
        order[block.block_id] = len(order)
    for chapter, section, unit, block in selected:
        if unit.unit_id not in units:
            if has_spans and section.section_id not in ranges:
                ranges[section.section_id] = unit_ranges(section)
            if state.get("blocks") or has_spans:
                projected = project_task_unit(unit, section, chapter, state=state, text=text,
                    unit_span=ranges.get(section.section_id, {}).get(unit.unit_id))
            else:
                projected = replace(unit, content_blocks=[b for _, _, u, b in rows if u.unit_id == unit.unit_id])
            units[unit.unit_id] = projected
        projected = units[unit.unit_id]
        visible = {b.block_id: b for b in projected.content_blocks}
        rendered = visible.get(block.block_id)
        before_id = next((b.block_id for b in projected.content_blocks
            if b.block_id not in affected and orders[unit.unit_id][b.block_id] > orders[unit.unit_id][block.block_id]), None)
        if rendered is not None:
            content = normalize_ocr_text(rendered.content)
            metadata = dict(rendered.metadata or {})
            if content != rendered.content:
                metadata["display_normalization"] = "ocr_whitespace_v1"
            # Availability is lazy; never advertise stale embedded generated artifacts.
            rendered = replace(rendered, content=content, metadata=metadata,
                artifact_ids=None, artifact_target_refs=None, interaction_availability=None).to_dict()
        result["blocks"].append({"block_id": block.block_id, "chapter_id": chapter.chapter_id,
            "section_id": section.section_id, "task_unit_id": unit.unit_id,
            "block_order": orders[unit.unit_id][block.block_id], "insert_before_block_id": before_id,
            "unit_source_hash": hashlib.sha256(projected.content.encode()).hexdigest(),
            "deleted": rendered is None, "manually_edited": block_was_manually_edited(state, block.block_id),
            "content_block": rendered})
    return result
