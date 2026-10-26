"""Explicit layout preparation with immutable canonical units and reading overlays."""

from dataclasses import replace
import hashlib

from document_structure.ocr_cleanup_overlay import canonical_text, reading_slice, unit_ranges
from document_structure.reading_content import ReadingContentConflict, state_for
from shared.task_unit_model import TaskUnit


def resolve_reading_layout(*, document, resolver, **options):
    state = state_for(document)
    if not any(state.get(key) for key in ("blocks", "cleanup_spans", "source_edits", "standalone_cleanup_receipts")):
        return resolver(document=document, **options)
    source = canonical_text(document)
    if ((state.get("cleanup_spans") or state.get("source_edits"))
            and hashlib.sha256(source.encode("utf-8")).hexdigest() != state.get("cleanup_source_sha256")):
        raise ReadingContentConflict("cleanup_source_changed")
    result = []
    for chapter in document.chapters:
        for section in chapter.sections:
            if (state.get("cleanup_spans") or state.get("source_edits")) and (
                    type(section.char_start) is not int or type(section.char_end) is not int
                    or not 0 <= section.char_start <= section.char_end <= len(source)
                    or section.content != source[section.char_start:section.char_end]):
                raise ReadingContentConflict("cleanup_layout_mapping_unavailable")
            # Existing overlay targets must retain their backend identity, even
            # when an explicit prepare refreshes the reading-revision cache stamp.
            if section.task_units:
                if state.get("cleanup_spans") or state.get("source_edits"):
                    unit_ranges(section)
                result.extend(section.task_units)
                continue
            text, offsets = _effective_section(section, state, source)
            view = replace(section, content=text, task_units=[])
            projection = replace(document, sections=[], chapters=[replace(chapter, sections=[view])])
            units = resolver(document=projection, **options) if text.strip() else []
            cursor, cuts = 0, [0]
            for index, unit in enumerate(units):
                if not unit.content.strip() or unit.source_section_ids != [section.section_id]:
                    raise ReadingContentConflict("cleanup_layout_mapping_unavailable")
                while cursor < len(text) and text[cursor].isspace():
                    cursor += 1
                start = cursor
                # Existing resolvers normalize paragraph separators. Require
                # ordered exact non-whitespace conservation, never fuzzy matching.
                for char in unit.content:
                    if char.isspace():
                        continue
                    while cursor < len(text) and text[cursor].isspace():
                        cursor += 1
                    if cursor >= len(text) or text[cursor] != char:
                        raise ReadingContentConflict("cleanup_layout_mapping_unavailable")
                    cursor += 1
                if index:
                    if start and offsets[start] == offsets[start - 1]:
                        raise ReadingContentConflict("cleanup_layout_split_inside_edit")
                    cuts.append(offsets[start])
            if text[cursor:].strip():
                raise ReadingContentConflict("cleanup_layout_mapping_unavailable")
            if not units:
                # Persist a canonical anchor even when its effective content is
                # empty; projection and Undo must never fall back to source text.
                units = [TaskUnit(unit_id=f"task-unit-{section.section_id}", title=section.title,
                    container_title=section.container_title, content=section.content,
                    source_section_ids=[section.section_id], is_fallback_generated=False)]
            cuts.append(len(section.content))
            for unit, start, end in zip(units, cuts, cuts[1:]):
                if end <= start:
                    raise ReadingContentConflict("cleanup_layout_mapping_unavailable")
                result.append(replace(unit, content=section.content[start:end], content_blocks=[],
                                      parent_section_id=section.section_id))
    return result


def _effective_section(section, state, source):
    """Retain canonical offsets for every character supplied to the resolver."""
    edits = state.get("source_edits", [])
    spans = edits + [s for s in state.get("cleanup_spans", []) if not any(
        s["char_start"] < e["char_end"] and e["char_start"] < s["char_end"] for e in edits)]
    text, offsets, cursor = [], [], 0
    for span in sorted(spans, key=lambda s: s["char_start"]):
        start = max(0, span["char_start"] - section.char_start)
        end = min(len(section.content), span["char_end"] - section.char_start)
        if start >= end:
            continue
        if start < cursor:
            raise ReadingContentConflict("cleanup_layout_mapping_unavailable")
        text.append(section.content[cursor:start])
        offsets.extend(range(cursor, start))
        replacement = span.get("replacement", "") if span["char_start"] >= section.char_start else ""
        text.append(replacement)
        offsets.extend([start] * len(replacement))
        cursor = end
    text.append(section.content[cursor:])
    offsets.extend(range(cursor, len(section.content)))
    effective = "".join(text)
    # This verifies source quotes using the same projection as reading consumers.
    if effective != reading_slice(source, section.char_start, section.char_end, state):
        raise ReadingContentConflict("cleanup_layout_mapping_unavailable")
    return effective, offsets
