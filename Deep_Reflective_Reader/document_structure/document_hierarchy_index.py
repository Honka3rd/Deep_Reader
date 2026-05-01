from __future__ import annotations

from dataclasses import replace

from document_structure.structured_document import (
    StructuredChapter,
    StructuredDocument,
    StructuredSection,
)


def flatten_sections_from_chapters(
    document: StructuredDocument,
) -> list[StructuredSection]:
    """Flatten chapter.sections in stable reading order.

    This helper intentionally flattens only sections present under chapters.
    Standalone legacy sections (for example front matter not yet projected into
    chapters) are excluded by design.
    """
    flattened: list[StructuredSection] = []
    for chapter in document.chapters:
        flattened.extend(list(chapter.sections))
    return flattened


def build_section_index_from_chapters(
    document: StructuredDocument,
) -> dict[str, StructuredSection]:
    """Build section_id -> section index from hierarchy; reject duplicates."""
    index: dict[str, StructuredSection] = {}
    for section in flatten_sections_from_chapters(document):
        if section.section_id in index:
            raise ValueError(
                "duplicate section_id in chapters hierarchy: "
                f"{section.section_id}"
            )
        index[section.section_id] = section
    return index


def get_effective_sections(
    document: StructuredDocument,
) -> list[StructuredSection]:
    """Read sections through primary hierarchy first, then legacy fallback."""
    if document.chapters:
        return flatten_sections_from_chapters(document)
    return list(document.sections)


def validate_chapter_hierarchy_consistency(
    document: StructuredDocument,
) -> list[str]:
    """Return consistency warnings between chapters (primary) and sections (legacy)."""
    warnings: list[str] = []

    chapter_ids: set[str] = set()
    for chapter in document.chapters:
        if chapter.chapter_id in chapter_ids:
            warnings.append(f"duplicate_chapter_id:{chapter.chapter_id}")
        chapter_ids.add(chapter.chapter_id)

    hierarchy_section_ids: set[str] = set()
    hierarchy_task_unit_ids: set[str] = set()
    for chapter in document.chapters:
        for section in chapter.sections:
            if section.section_id in hierarchy_section_ids:
                warnings.append(f"duplicate_hierarchy_section_id:{section.section_id}")
            hierarchy_section_ids.add(section.section_id)

            if section.parent_chapter_id != chapter.chapter_id:
                warnings.append(
                    "section_parent_mismatch:"
                    f"{section.section_id}:parent={section.parent_chapter_id}"
                    f":chapter={chapter.chapter_id}"
                )

            for task_unit in section.task_units:
                if task_unit.parent_section_id is not None and (
                    task_unit.parent_section_id != section.section_id
                ):
                    warnings.append(
                        "task_unit_parent_section_mismatch:"
                        f"{task_unit.unit_id}:parent={task_unit.parent_section_id}"
                        f":section={section.section_id}"
                    )
                if task_unit.unit_id in hierarchy_task_unit_ids:
                    warnings.append(f"duplicate_task_unit_id:{task_unit.unit_id}")
                hierarchy_task_unit_ids.add(task_unit.unit_id)

    flat_index: dict[str, StructuredSection] = {}
    flat_duplicates: set[str] = set()
    for section in document.sections:
        if section.section_id in flat_index:
            flat_duplicates.add(section.section_id)
        else:
            flat_index[section.section_id] = section
    for section_id in sorted(flat_duplicates):
        warnings.append(f"duplicate_flat_section_id:{section_id}")

    if document.sections and hierarchy_section_ids:
        flat_ids = set(flat_index.keys())
        missing_in_flat = hierarchy_section_ids - flat_ids
        missing_in_hierarchy = flat_ids - hierarchy_section_ids
        for section_id in sorted(missing_in_flat):
            warnings.append(f"hierarchy_section_missing_in_flat:{section_id}")
        for section_id in sorted(missing_in_hierarchy):
            warnings.append(f"flat_section_missing_in_hierarchy:{section_id}")

        comparable_ids = hierarchy_section_ids & flat_ids
        for section_id in sorted(comparable_ids):
            nested = _find_hierarchy_section_by_id(document, section_id)
            if nested is None:
                continue
            flat = flat_index[section_id]
            if nested.title != flat.title:
                warnings.append(f"section_field_mismatch:{section_id}:title")
            if nested.level != flat.level:
                warnings.append(f"section_field_mismatch:{section_id}:level")
            nested_role = None if nested.section_role is None else nested.section_role.value
            flat_role = None if flat.section_role is None else flat.section_role.value
            if nested_role != flat_role:
                warnings.append(f"section_field_mismatch:{section_id}:section_role")
            if nested.char_start != flat.char_start:
                warnings.append(f"section_field_mismatch:{section_id}:char_start")
            if nested.char_end != flat.char_end:
                warnings.append(f"section_field_mismatch:{section_id}:char_end")

    return warnings


def assert_chapter_hierarchy_consistency(document: StructuredDocument) -> None:
    """Raise ValueError when hierarchy and legacy views drift."""
    warnings = validate_chapter_hierarchy_consistency(document)
    if warnings:
        raise ValueError("chapter_hierarchy_inconsistent: " + " | ".join(warnings))


def with_legacy_sections_synced_from_chapters(
    document: StructuredDocument,
) -> StructuredDocument:
    """Rebuild document.sections from chapters when hierarchy exists."""
    if not document.chapters:
        return document
    flattened = flatten_sections_from_chapters(document)
    return replace(document, sections=flattened)


def _find_hierarchy_section_by_id(
    document: StructuredDocument,
    section_id: str,
) -> StructuredSection | None:
    for chapter in document.chapters:
        for section in chapter.sections:
            if section.section_id == section_id:
                return section
    return None
