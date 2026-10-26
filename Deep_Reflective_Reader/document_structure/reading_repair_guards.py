"""Shared protection primitives for source regions and cleanup-history repairs."""

from document_structure.reading_content import block_was_manually_edited


def manual_block_protection(state, block_ids, *, after_revision=None):
    ids = set(block_ids)
    if any(state.get("blocks", {}).get(key, {}).get("deleted") for key in ids):
        return "deleted"
    if any(block_was_manually_edited(state, key) for key in ids):
        return "edited"
    for event in state.get("audit", []):
        if (event.get("action") in {"edit", "delete"}
                and (after_revision is None or event.get("revision", 0) > after_revision)
                and (event.get("block_id") in ids or ids.intersection(event.get("affected_block_ids", [])))):
            return "deleted" if event["action"] == "delete" else "edited"
    return None


def source_ranges_overlap(left, right):
    return left["char_start"] < right["char_end"] and right["char_start"] < left["char_end"]


def same_source_span(left, right):
    return all(left.get(key) == right.get(key) for key in ("char_start", "char_end", "quote"))
