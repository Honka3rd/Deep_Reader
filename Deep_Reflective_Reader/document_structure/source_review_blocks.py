"""Source-linked review relations and eligibility; no read-side persistence."""

from copy import deepcopy
from dataclasses import dataclass, replace
import hashlib
import json

from document_structure.ocr_cleanup_overlay import canonical_text, unit_ranges, block_range
from document_structure.reading_content import ReadingContentConflict, original_blocks

MAPPING_KEY = "source_review_mapping"
MAPPING_POLICY = "source-review-blocks-v2"


@dataclass(frozen=True)
class _BlockSpan:
    start: int
    end: int
    chapter_id: str
    section_id: str
    task_unit_id: str
    block_id: str


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _source_key(provenance):
    return _digest([provenance.get(key) for key in (
        "extraction", "pdf_extraction", "source_review", "source_page_spans", "storage_structure_version", "interaction_epoch")])


def _excluded_from_body(provenance, region):
    """Only affirmative blank-page evidence can exempt omitted text from review."""
    if region.get("raw_candidate") != "" or region.get("blank_render_confirmed") is not True:
        return False
    from doc_loaders.extraction_identity import read_extraction_record
    try:
        identity, extraction = read_extraction_record(provenance)
    except (KeyError, TypeError, ValueError, AttributeError):
        return False
    if identity.source_sha256 != provenance.get("source_review", {}).get("source_sha256") or "canonical_span" in region:
        return False
    proof = provenance.get(MAPPING_KEY, {})
    if (proof.get("policy") != MAPPING_POLICY or proof.get("source_key") != _source_key(provenance)
            or proof.get("canonical_sha256") != extraction["text_sha256"]
            or (extraction.get("text_length") is not None
                and proof.get("canonical_length") != extraction["text_length"])):
        return False
    pages = [p for p in provenance.get("source_page_spans", [])
             if p.get("page_index") == region.get("page_index")]
    if len(pages) != 1:
        return False
    start, end = pages[0].get("char_start"), pages[0].get("char_end")
    if (type(start) is not int or type(end) is not int or not 0 <= start <= end
            or type(proof.get("canonical_length")) is not int or end > proof["canonical_length"]):
        return False
    return start == end


def active_review_regions(provenance):
    """One eligibility rule for Library, reader and generation, including legacy data."""
    decisions = provenance.get("reading_content", {}).get("regions", {})
    kept = provenance.get("reading_content", {}).get("review_kept_blocks", {})
    proof = provenance.get(MAPPING_KEY, {})
    mappings = proof.get("regions", {}) if proof.get("source_key") == _source_key(provenance) else {}
    return [r for r in provenance.get("source_review", {}).get("regions", [])
            if not decisions.get(r["id"]) and not _excluded_from_body(provenance, r)
            and not (mappings.get(r["id"], {}).get("block_ids")
                     and all(key in kept for key in mappings[r["id"]]["block_ids"]))]


def needs_review_from_provenance(provenance):
    return bool(active_review_regions(provenance))


def review_provenance(document):
    """Do not use a persisted exclusion proof against a changed in-memory body."""
    provenance = document.parse_provenance
    proof = provenance.get(MAPPING_KEY)
    if proof and proof.get("canonical_sha256") != hashlib.sha256(canonical_text(document).encode()).hexdigest():
        return {key: value for key, value in provenance.items() if key != MAPPING_KEY}
    return provenance


def _region_span(document, region, text, text_hash):
    from doc_loaders.extraction_identity import read_extraction_record
    provenance = document.parse_provenance
    review = provenance.get("source_review", {})
    try:
        identity, extraction = read_extraction_record(provenance)
    except (KeyError, TypeError, ValueError, AttributeError):
        return None
    if (identity.source_sha256 != review.get("source_sha256") or extraction["text_sha256"] != text_hash
            or extraction.get("text_length", len(text)) != len(text)):
        return None
    candidate = region.get("raw_candidate")
    if not isinstance(candidate, str) or not candidate.strip():
        return None
    if "canonical_span" in region:
        mapping = region.get("canonical_span") or {}
        start, end = mapping.get("char_start"), mapping.get("char_end")
        if (mapping.get("text_sha256") == text_hash and type(start) is int and type(end) is int
                and 0 <= start < end <= len(text) and text[start:end] == candidate):
            return start, end
        return None
    if region.get("body_inclusion") == "excluded":
        return None
    # Legacy evidence may carry only a quote and page. Match exactly once inside
    # that verified source page; excluded OCR candidates do not create new blocks.
    pages = [p for p in provenance.get("source_page_spans", []) if p.get("page_index") == region.get("page_index")]
    if len(pages) != 1:
        return None
    start, end = pages[0].get("char_start"), pages[0].get("char_end")
    if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(text):
        return None
    found = text.find(candidate, start, end)
    if found < 0 or text.find(candidate, found + 1, end) >= 0:
        return None
    return found, found + len(candidate)


def _verified_block_spans(document, text):
    spans = []
    ranges = {}
    for chapter, section, unit, block in original_blocks(document):
        try:
            if text[section.char_start:section.char_end] != section.content:
                continue
            if section.section_id not in ranges:
                ranges[section.section_id] = unit_ranges(section)
            span = block_range(unit, block, ranges[section.section_id][unit.unit_id])
            metadata = block.metadata or {}
            unit_hash = hashlib.sha256(unit.content.encode()).hexdigest()
            if metadata.get("source_hash") not in (None, unit_hash):
                continue
            source = text[span.start:span.end]
            if source != block.content:
                from document_structure.text_normalization import normalize_ocr_text
                if (metadata.get("display_normalization") != "ocr_whitespace_v1"
                        or normalize_ocr_text(source) != block.content):
                    continue
            spans.append(_BlockSpan(span.start, span.end, chapter.chapter_id,
                                    section.section_id, unit.unit_id, block.block_id))
        except (ReadingContentConflict, KeyError):
            continue
    return spans


def _build_mapping(document, *, include_spans=False):
    text = canonical_text(document)
    text_hash = hashlib.sha256(text.encode()).hexdigest()
    spans = _verified_block_spans(document, text)
    result = {}
    provenance = review_provenance(document)
    for region in document.parse_provenance.get("source_review", {}).get("regions", []):
        if _excluded_from_body(provenance, region):
            result[region["id"]] = {"block_ids": [], "mapping_status": "unavailable",
                "body_status": "excluded", "mapping_reason": "never_in_body"}
            continue
        span = _region_span(document, region, text, text_hash)
        if span is None:
            result[region["id"]] = {"block_ids": [], "mapping_status": "unavailable",
                "body_status": "unknown", "mapping_reason": "source_mapping_unavailable"}
            continue
        start, end = span
        covered = sorted((block for block in spans if block.start < end and start < block.end),
                         key=lambda block: (block.start, block.end, block.block_id))
        cursor = start
        complete = True
        for block in covered:
            if text[cursor:max(cursor, block.start)].strip():
                complete = False
            cursor = max(cursor, min(end, block.end))
        complete = complete and not text[cursor:end].strip() and bool(covered)
        result[region["id"]] = {
            "block_ids": list(dict.fromkeys(block.block_id for block in covered)),
            "mapping_status": "complete" if complete else "partial",
            "body_status": "linked" if covered else "pending" if not any(
                s.task_units for c in document.chapters for s in c.sections) else "unknown",
            "mapping_reason": "verified_source_overlap" if complete else "partial_source_coverage" if covered
                else "blocks_not_materialized" if not any(s.task_units for c in document.chapters for s in c.sections)
                else "block_mapping_unavailable",
        }
        if include_spans:
            result[region["id"]].update(
                canonical_span={"char_start": start, "char_end": end, "text_sha256": text_hash},
                block_spans=[{
                    "chapter_id": block.chapter_id, "section_id": block.section_id,
                    "task_unit_id": block.task_unit_id, "block_id": block.block_id,
                    "source_start": max(start, block.start), "source_end": min(end, block.end),
                    "block_source_start": block.start, "block_source_end": block.end,
                    "char_from": max(start, block.start) - block.start,
                    "char_to": min(end, block.end) - block.start,
                } for block in covered])
    return result


def region_source_intersections(document):
    """Fresh exact source coordinates, never offsets into normalized display text.

    Internal repair planning reuses navigation's verifier without extending its
    persisted v2 cache or trusting old block-ID-only evidence for writes.
    """
    if document.parse_provenance.get("reading_projected"):
        raise ReadingContentConflict("review_mapping_requires_canonical_document")
    return _build_mapping(document, include_spans=True)


def _binding(document):
    return {"policy": MAPPING_POLICY, "document_id": document.document_id,
        "source_key": _source_key(document.parse_provenance),
        "hierarchy_key": _digest([(c.chapter_id, c.title, [(s.section_id, s.title, s.char_start, s.char_end,
            s.content, [(u.unit_id, u.title, u.container_title, u.content, [b.to_dict() for b in (u.content_blocks or [])])
                        for u in s.task_units]) for s in c.sections]) for c in document.chapters]),
        "canonical_sha256": hashlib.sha256(canonical_text(document).encode()).hexdigest(),
        "canonical_length": len(canonical_text(document))}


def region_block_mapping(document):
    if not document.parse_provenance.get("source_review", {}).get("regions"):
        return {}
    binding = _binding(document)
    cached = document.parse_provenance.get(MAPPING_KEY, {})
    if all(cached.get(key) == value for key, value in binding.items()) and isinstance(cached.get("regions"), dict):
        return deepcopy(cached["regions"])
    return _build_mapping(document)


def persist_region_mapping(document):
    """Pure staging helper, called only by explicit repository write boundaries."""
    if document.parse_provenance.get("reading_projected"):
        raise ReadingContentConflict("review_mapping_requires_canonical_document")
    from document_structure.cleanup_history import bind_cleanup_record_links
    document = bind_cleanup_record_links(document)
    if not document.parse_provenance.get("source_review", {}).get("regions"):
        provenance = {k: v for k, v in document.parse_provenance.items() if k != MAPPING_KEY}
        return replace(document, parse_provenance=provenance)
    binding = _binding(document)
    # This proof is staged from the actual canonical body, not supplied by a caller.
    staged = replace(document, parse_provenance={**document.parse_provenance, MAPPING_KEY: binding})
    record = {**binding, "regions": _build_mapping(staged)}
    if document.parse_provenance.get(MAPPING_KEY) == record:
        return document
    return replace(document, parse_provenance={**document.parse_provenance, MAPPING_KEY: record})


def region_manual_change(state, block_ids):
    """Audit is retained after Undo; automated cleanup/Ignore are not manual edits."""
    ids = set(block_ids)
    for event in reversed(state.get("audit", [])):
        if event.get("action") not in {"edit", "delete", "undo"}:
            continue
        affected = event.get("affected_block_ids", [event.get("block_id")])
        if ids.intersection(affected) and event.get("at"):
            edits = state.get("blocks", {})
            status = ("deleted" if any(edits.get(key, {}).get("deleted") for key in ids)
                      else "modified" if any(key in edits for key in ids) else "restored")
            return {"status": status, "at": event["at"]}
    return None
