"""Single-block review commands; mapping and protection stay server-owned."""

from datetime import datetime, timezone

from document_structure.reading_content import ReadingContentConflict, original_blocks, project_document, state_for
from document_structure.reading_repair_guards import manual_block_protection
from document_structure.source_review_blocks import region_source_intersections


def block_repair_descriptions(document, blocks, *, mappings=None):
    state = state_for(document)
    kept = state.get("review_kept_blocks", {})
    decisions = state.get("regions", {})
    if mappings is None:
        mappings = region_source_intersections(document)
    linked = {key for region_id, mapping in mappings.items() if not decisions.get(region_id)
              for key in mapping.get("block_ids", [])}
    result = {}
    for block in blocks:
        key = block["block_id"]
        reason = None
        if key in kept:
            reason = "review_block_kept"
        elif key not in linked:
            reason = "review_block_mapping_changed"
        elif block["deleted"]:
            reason = "reading_block_deleted_use_undo"
        else:
            protection = manual_block_protection(state, [key])
            if protection:
                reason = "block_manually_" + protection
        result[key] = {"can_edit": reason is None, "can_delete": reason is None,
            "can_keep": key in linked and key not in kept and not block["deleted"],
            "blocked_reason": reason}
    return result


def validate_review_block(document, block_id, region_id, action):
    # Resolve this relation under the repository lock, not from client block IDs
    # or region-wide eligibility. One protected sibling must not block this one.
    mappings = region_source_intersections(document)
    mapping = mappings.get(region_id, {})
    if (state_for(document).get("regions", {}).get(region_id)
            or block_id not in mapping.get("block_ids", [])):
        raise ReadingContentConflict("review_block_mapping_changed")
    visible = {b.block_id for _, _, _, b in original_blocks(project_document(document))}
    blocks = [{"block_id": block_id, "deleted": block_id not in visible}]
    capability = block_repair_descriptions(document, blocks, mappings=mappings).get(block_id)
    allowed = capability and capability["can_keep" if action == "review_block_keep" else "can_edit"]
    if not allowed:
        raise ReadingContentConflict((capability or {}).get("blocked_reason") or "review_block_mapping_changed")


def keep_review_block(state, block_id):
    # Deliberately outside Undo: the body and artifacts are unchanged, and the
    # explicit confirmation permanently closes review for this canonical block.
    state.setdefault("review_kept_blocks", {})[block_id] = datetime.now(timezone.utc).isoformat()
