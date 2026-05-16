from __future__ import annotations

from dataclasses import replace

from document_structure.structured_hierarchy_builder import (
    build_document_hierarchy_from_sections,
)
from document_structure.structured_document import (
    StructuredChapter,
    StructuredDocument,
    StructuredSection,
)


SEVERE_HIERARCHY_WARNING_PREFIXES: tuple[str, ...] = (
    "duplicate_chapter_id:",
    "duplicate_hierarchy_section_id:",
    "duplicate_task_unit_id:",
    "section_parent_mismatch:",
    "task_unit_parent_section_mismatch:",
    "section_field_mismatch:",
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
    """Read sections from hierarchy only (chapters[].sections[])."""
    return flatten_sections_from_chapters(document)


def find_section_by_id_effective(
    document: StructuredDocument,
    section_id: str,
    *,
    allow_legacy_fallback: bool = False,
) -> StructuredSection | None:
    """Find section by id with hierarchy as primary source.

    Legacy fallback is compatibility-only and is allowed only for sections-only
    documents (chapters are empty).
    """
    normalized_section_id = section_id.strip()
    if not normalized_section_id:
        return None

    if document.chapters:
        hierarchy_matches = [
            section
            for section in flatten_sections_from_chapters(document)
            if section.section_id == normalized_section_id
        ]
        if len(hierarchy_matches) > 1:
            raise ValueError(
                "duplicate_hierarchy_section_id:"
                f"{normalized_section_id}"
            )
        if len(hierarchy_matches) == 1:
            return hierarchy_matches[0]
        # Even if allow_legacy_fallback is requested, do not fall back to root
        # legacy sections when hierarchy chapters exist.
        return None

    if not allow_legacy_fallback:
        return None

    legacy_matches = [
        section for section in document.sections if section.section_id == normalized_section_id
    ]
    if len(legacy_matches) > 1:
        raise ValueError(
            "duplicate_flat_section_id:"
            f"{normalized_section_id}"
        )
    if len(legacy_matches) == 1:
        return legacy_matches[0]
    return None


def find_sections_by_title_effective(
    document: StructuredDocument,
    title: str,
    *,
    allow_legacy_fallback: bool = False,
) -> list[StructuredSection]:
    """Find sections by exact normalized title with hierarchy-first preference.

    Legacy fallback is compatibility-only and is allowed only for sections-only
    documents (chapters are empty).
    """
    normalized_title = title.strip()
    if not normalized_title:
        return []

    if document.chapters:
        hierarchy_matches = [
            section
            for section in flatten_sections_from_chapters(document)
            if ((section.title or "").strip()) == normalized_title
        ]
        if hierarchy_matches:
            return hierarchy_matches
        # Keep hierarchy-first strictness: when chapters exist, do not consult
        # root legacy sections as a fallback title source.
        return []

    if not allow_legacy_fallback:
        return []

    return [
        section
        for section in document.sections
        if ((section.title or "").strip()) == normalized_title
    ]


def find_section_by_chapter_title_effective(
    document: StructuredDocument,
    chapter_title: str,
    *,
    allow_legacy_fallback: bool = False,
) -> StructuredSection | None:
    """Resolve chapter title into section with chapter-aware hierarchy-first semantics.

    Legacy fallback is compatibility-only and is allowed only for sections-only
    documents (chapters are empty).
    """
    normalized_chapter_title = chapter_title.strip()
    if not normalized_chapter_title:
        return None

    if document.chapters:
        matched_chapters = [
            chapter
            for chapter in document.chapters
            if ((chapter.title or "").strip()) == normalized_chapter_title
        ]
        if len(matched_chapters) > 1:
            raise ValueError(
                "ambiguous chapter title: "
                f"'{normalized_chapter_title}' matched {len(matched_chapters)} chapters"
            )
        if len(matched_chapters) == 1:
            chapter = matched_chapters[0]
            if not chapter.sections:
                raise ValueError(
                    "chapter has no sections: "
                    f"title='{normalized_chapter_title}' chapter_id='{chapter.chapter_id}'"
                )
            chapter_body_sections = [
                section
                for section in chapter.sections
                if (section.section_kind or "").strip() == "chapter_body"
            ]
            if chapter_body_sections:
                return chapter_body_sections[0]
            return chapter.sections[0]
        # Do not fall back to root legacy sections when hierarchy chapters
        # already exist in the document.
        return None

    if not allow_legacy_fallback:
        return None

    legacy_title_matches = [
        section
        for section in document.sections
        if ((section.title or "").strip()) == normalized_chapter_title
    ]
    if len(legacy_title_matches) > 1:
        raise ValueError(
            "ambiguous chapter title: "
            f"'{normalized_chapter_title}' matched {len(legacy_title_matches)} legacy sections"
        )
    if len(legacy_title_matches) == 1:
        return legacy_title_matches[0]
    return None


def find_chapter_by_id_effective(
    document: StructuredDocument,
    chapter_id: str,
) -> StructuredChapter | None:
    """Find chapter by stable chapter_id with duplicate defense."""
    normalized_chapter_id = chapter_id.strip()
    if not normalized_chapter_id:
        return None
    matches = [
        chapter for chapter in document.chapters
        if chapter.chapter_id == normalized_chapter_id
    ]
    if len(matches) > 1:
        raise ValueError(
            "duplicate_chapter_id:"
            f"{normalized_chapter_id}"
        )
    if len(matches) == 1:
        return matches[0]
    return None


def find_chapters_by_title_effective(
    document: StructuredDocument,
    chapter_title: str,
) -> list[StructuredChapter]:
    """Find chapters by exact normalized title."""
    normalized_title = chapter_title.strip()
    if not normalized_title:
        return []
    return [
        chapter
        for chapter in document.chapters
        if ((chapter.title or "").strip()) == normalized_title
    ]


def validate_chapter_hierarchy_consistency(
    document: StructuredDocument,
) -> list[str]:
    """Return consistency warnings for hierarchy-first document state."""
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

    if document.sections and document.chapters:
        warnings.append("legacy_sections_present")

    return warnings


def assert_chapter_hierarchy_consistency(document: StructuredDocument) -> None:
    """Raise ValueError when hierarchy and legacy views drift."""
    warnings = validate_chapter_hierarchy_consistency(document)
    if warnings:
        raise ValueError("chapter_hierarchy_inconsistent: " + " | ".join(warnings))


def with_legacy_sections_synced_from_chapters(
    document: StructuredDocument,
) -> StructuredDocument:
    """Deprecated transitional helper; prefer hierarchy-only persistence."""
    if not document.chapters:
        return document
    flattened = flatten_sections_from_chapters(document)
    return replace(document, sections=flattened)


def with_sections_replaced_in_hierarchy(
    document: StructuredDocument,
    sections_by_id: dict[str, StructuredSection],
) -> StructuredDocument:
    """Replace matching nested chapter sections by id, preserving chapter ordering."""
    if not document.chapters:
        return document

    updated_chapters: list[StructuredChapter] = []
    for chapter in document.chapters:
        updated_sections = [
            sections_by_id.get(section.section_id, section)
            for section in chapter.sections
        ]
        updated_chapters.append(
            replace(
                chapter,
                sections=updated_sections,
            )
        )
    return replace(document, chapters=updated_chapters)


def with_sections_synced_across_hierarchy_and_legacy(
    document: StructuredDocument,
    updated_sections: list[StructuredSection],
) -> StructuredDocument:
    """Deprecated transitional helper; prefer hierarchy-only write paths."""
    updated_sections_by_id = {
        section.section_id: section
        for section in updated_sections
    }

    if not document.chapters:
        return replace(document, sections=list(updated_sections))

    hierarchy_updated_document = with_sections_replaced_in_hierarchy(
        document,
        updated_sections_by_id,
    )
    flattened_chapter_sections = flatten_sections_from_chapters(hierarchy_updated_document)
    flattened_chapter_ids = {
        section.section_id for section in flattened_chapter_sections
    }

    preserved_non_chapter_sections = [
        updated_sections_by_id.get(section.section_id, section)
        for section in document.sections
        if section.section_id not in flattened_chapter_ids
    ]
    combined_sections = preserved_non_chapter_sections + flattened_chapter_sections
    return replace(hierarchy_updated_document, sections=combined_sections)


def is_severe_hierarchy_warning(warning: str) -> bool:
    """Return whether one hierarchy consistency warning should block save/read."""
    return warning.startswith(SEVERE_HIERARCHY_WARNING_PREFIXES)


def _find_hierarchy_section_by_id(
    document: StructuredDocument,
    section_id: str,
) -> StructuredSection | None:
    for chapter in document.chapters:
        for section in chapter.sections:
            if section.section_id == section_id:
                return section
    return None


def migrate_legacy_sections_to_chapters(
    document: StructuredDocument,
) -> StructuredDocument:
    """Upgrade sections-only legacy document into hierarchy-first structure."""
    if document.chapters:
        return replace(document, sections=[])
    if not document.sections:
        return document
    migrated = build_document_hierarchy_from_sections(document)
    return replace(migrated, sections=[])
