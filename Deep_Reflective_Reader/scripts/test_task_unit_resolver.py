import json

from document_structure.structured_document import (
    StructuredChapter,
    StructuredDocument,
    StructuredSection,
)
from section_tasks.task_unit_resolver import TaskUnitResolver


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def build_hierarchy_first_document() -> StructuredDocument:
    """Build hierarchy-first document with stale legacy section fallback payload."""
    hierarchy_content = "Hierarchy section content. " * 8
    stale_legacy_content = "Legacy stale section content. " * 8
    return StructuredDocument(
        document_id="doc-hierarchy-first",
        title="Hierarchy First Demo",
        source_path=None,
        language="en",
        raw_text=hierarchy_content,
        sections=[
            StructuredSection(
                section_id="section-stale",
                section_index=0,
                title="Legacy Stale",
                level=1,
                content=stale_legacy_content,
                char_start=0,
                char_end=len(stale_legacy_content),
                container_title=None,
            )
        ],
        chapters=[
            StructuredChapter(
                chapter_id="chapter-0",
                title="Chapter One",
                level=1,
                chapter_role="main_body",
                sections=[
                    StructuredSection(
                        section_id="section-h1",
                        section_index=1,
                        title="Chapter One",
                        level=1,
                        content=hierarchy_content,
                        char_start=0,
                        char_end=len(hierarchy_content),
                        container_title=None,
                        parent_chapter_id="chapter-0",
                        section_kind="chapter_body",
                        is_implicit_section=True,
                    )
                ],
            )
        ],
    )


def build_legacy_sections_only_document() -> StructuredDocument:
    """Build legacy sections-only document for backward-compatible fallback."""
    legacy_content = "Legacy-only section content. " * 8
    sections = [
        StructuredSection(
            section_id="section-legacy",
            section_index=0,
            title="Legacy Section",
            level=1,
            content=legacy_content,
            char_start=0,
            char_end=len(legacy_content),
            container_title=None,
        )
    ]
    return StructuredDocument(
        document_id="doc-legacy-only",
        title="Legacy Only Demo",
        source_path=None,
        language="en",
        raw_text="\n".join(section.content for section in sections),
        sections=sections,
        chapters=[],
    )


def main() -> None:
    resolver = TaskUnitResolver(
        task_unit_min_chars=40,
        task_unit_max_chars=400,
    )

    hierarchy_doc = build_hierarchy_first_document()
    hierarchy_units = resolver.resolve(hierarchy_doc)
    _assert(len(hierarchy_units) == 1, "hierarchy-first doc should resolve one task unit")
    _assert(
        hierarchy_units[0].source_section_ids == ["section-h1"],
        "resolver should use hierarchy section when chapters exist",
    )

    legacy_doc = build_legacy_sections_only_document()
    legacy_units = resolver.resolve(legacy_doc)
    _assert(len(legacy_units) == 1, "legacy sections-only doc should resolve one task unit")
    _assert(
        legacy_units[0].source_section_ids == ["section-legacy"],
        "legacy sections-only fallback should remain available",
    )

    payload = {
        "hierarchy_first_units": [unit.to_dict() for unit in hierarchy_units],
        "legacy_sections_only_units": [unit.to_dict() for unit in legacy_units],
        "status": "ok",
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
