#!/usr/bin/env python3
"""Hierarchy-first lookup smoke tests for SectionTaskContextBuilder."""

from __future__ import annotations

import json

from document_structure.structured_document import (
    StructuredChapter,
    StructuredDocument,
    StructuredSection,
)
from section_tasks.section_task_context_builder import SectionTaskContextBuilder


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _build_hierarchy_and_legacy_drift_document() -> StructuredDocument:
    hierarchy_section = StructuredSection(
        section_id="section-1",
        section_index=1,
        title="Hierarchy Title",
        level=1,
        content="hierarchy content",
        char_start=0,
        char_end=17,
        parent_chapter_id="chapter-0",
        section_kind="chapter_body",
        is_implicit_section=True,
    )
    stale_legacy_section = StructuredSection(
        section_id="section-1",
        section_index=0,
        title="Legacy Stale Title",
        level=1,
        content="legacy stale content",
        char_start=0,
        char_end=20,
    )
    return StructuredDocument(
        document_id="doc-hierarchy-drift",
        title="Hierarchy Drift",
        source_path=None,
        language="en",
        raw_text="hierarchy content",
        sections=[stale_legacy_section],
        chapters=[
            StructuredChapter(
                chapter_id="chapter-0",
                title="Chapter One",
                level=1,
                chapter_role="main_body",
                sections=[hierarchy_section],
            )
        ],
    )


def _build_legacy_only_document() -> StructuredDocument:
    legacy_section = StructuredSection(
        section_id="section-legacy",
        section_index=0,
        title="Legacy Section",
        level=1,
        content="legacy only content",
        char_start=0,
        char_end=19,
    )
    return StructuredDocument(
        document_id="doc-legacy-only",
        title="Legacy Only",
        source_path=None,
        language="en",
        raw_text="legacy only content",
        sections=[legacy_section],
        chapters=[],
    )


def main() -> None:
    builder = SectionTaskContextBuilder()

    hierarchy_doc = _build_hierarchy_and_legacy_drift_document()
    hierarchy_context = builder.build_from_document(
        document=hierarchy_doc,
        section_id="section-1",
    )
    _assert(
        hierarchy_context.section_title == "Hierarchy Title",
        "hierarchy-first lookup should prefer chapter section over legacy root section",
    )
    _assert(
        hierarchy_context.section_content == "hierarchy content",
        "hierarchy-first lookup should use hierarchy section content",
    )

    legacy_doc = _build_legacy_only_document()
    try:
        builder.build_from_document(
            document=legacy_doc,
            section_id="section-legacy",
        )
        raise AssertionError(
            "legacy sections-only document should fail when legacy fallback is disabled"
        )
    except ValueError as error:
        _assert(
            "not found in document 'doc-legacy-only'" in str(error),
            "legacy-only lookup should fail with section-not-found error",
        )

    print(
        json.dumps(
            {
                "status": "ok",
                "tests": [
                    "hierarchy_first_preferred_when_drift_exists",
                    "legacy_sections_only_fallback_disabled",
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
