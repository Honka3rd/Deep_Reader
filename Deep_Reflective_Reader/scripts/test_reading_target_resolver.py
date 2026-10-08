#!/usr/bin/env python3
"""Regression tests for hierarchy-only reading target resolution."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.section_task_coordinator import SectionTaskCoordinator  # noqa: E402
from document_preparation.preparation_mode import PreparationMode  # noqa: E402
from document_structure.structured_document import (  # noqa: E402
    StructuredChapter,
    StructuredDocument,
    StructuredSection,
)
from section_tasks.reading_target_resolver import ReadingTargetResolver  # noqa: E402
from shared.task_unit_model import TaskUnit  # noqa: E402


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _task_unit(
    unit_id: str,
    content: str,
    *,
    parent_section_id: str | None,
) -> TaskUnit:
    return TaskUnit(
        unit_id=unit_id,
        title=f"Task {unit_id}",
        container_title=None,
        content=content,
        source_section_ids=[] if parent_section_id is None else [parent_section_id],
        is_fallback_generated=False,
        parent_section_id=parent_section_id,
    )


def _section(
    section_id: str,
    content: str,
    *,
    parent_chapter_id: str,
    task_units: list[TaskUnit] | None = None,
) -> StructuredSection:
    return StructuredSection(
        section_id=section_id,
        section_index=0,
        title=f"Section {section_id}",
        level=2,
        content=content,
        char_start=0,
        char_end=len(content),
        parent_chapter_id=parent_chapter_id,
        task_units=[] if task_units is None else task_units,
    )


def _document() -> StructuredDocument:
    section_1 = _section(
        "section-1",
        "Section one content.",
        parent_chapter_id="chapter-1",
        task_units=[
            _task_unit(
                "unit-1",
                "Unit one content.",
                parent_section_id="section-1",
            )
        ],
    )
    section_2 = _section(
        "section-2",
        "Section two content.",
        parent_chapter_id="chapter-1",
        task_units=[
            _task_unit(
                "unit-2",
                "Unit two content.",
                parent_section_id="section-2",
            )
        ],
    )
    legacy_only_section = _section(
        "legacy-section",
        "Legacy-only section.",
        parent_chapter_id="legacy-chapter",
        task_units=[
            _task_unit(
                "legacy-unit",
                "Legacy-only unit.",
                parent_section_id="legacy-section",
            )
        ],
    )
    return StructuredDocument(
        document_id="doc-1",
        title="Resolver Fixture",
        source_path=None,
        language="en",
        raw_text="Full document content.",
        sections=[legacy_only_section],
        chapters=[
            StructuredChapter(
                chapter_id="chapter-1",
                title="Chapter One",
                level=1,
                chapter_role=None,
                sections=[section_1, section_2],
            )
        ],
    )


def _expect_value_error(fn, expected: str) -> None:
    try:
        fn()
    except ValueError as error:
        _assert(expected in str(error), f"expected '{expected}' in '{error}'")
        return
    raise AssertionError(f"expected ValueError containing '{expected}'")


def test_resolves_document_chapter_section_and_task_unit_targets() -> None:
    resolver = ReadingTargetResolver()
    document = _document()

    resolved_document = resolver.resolve(document=document, target_level="book")
    _assert(resolved_document.target_level == "document", "book should normalize to document")
    _assert(resolved_document.target_id == "doc-1", "document target should use document id")

    resolved_chapter = resolver.resolve(
        document=document,
        target_level="chapter",
        chapter_id="chapter-1",
    )
    _assert(resolved_chapter.target_id == "chapter-1", "chapter should resolve by id")
    _assert("Section one content." in resolved_chapter.content, "chapter should join section content")

    resolved_section = resolver.resolve(
        document=document,
        target_level="section",
        chapter_id="chapter-1",
        section_id="section-1",
    )
    _assert(resolved_section.target_id == "section-1", "section should resolve by id")
    _assert(resolved_section.chapter_id == "chapter-1", "section should retain parent chapter")

    resolved_task_unit = resolver.resolve(
        document=document,
        target_level="task_unit",
        chapter_id="chapter-1",
        section_id="section-1",
        task_unit_id="unit-1",
    )
    _assert(resolved_task_unit.target_id == "unit-1", "task unit should resolve by id")
    _assert(resolved_task_unit.section_id == "section-1", "task unit should retain section id")


def test_rejects_title_primary_and_legacy_fallback_paths() -> None:
    resolver = ReadingTargetResolver()
    document = _document()

    _expect_value_error(
        lambda: resolver.resolve(
            document=document,
            target_level="chapter",
            chapter_id="Chapter One",
        ),
        "chapter_id 'Chapter One' not found",
    )
    _expect_value_error(
        lambda: resolver.resolve(
            document=document,
            target_level="section",
            section_id="legacy-section",
        ),
        "section_id 'legacy-section' not found",
    )
    _expect_value_error(
        lambda: resolver.resolve(
            document=document,
            target_level="task_unit",
            task_unit_id="legacy-unit",
        ),
        "task_unit_id 'legacy-unit' not found",
    )
    legacy_only_document = replace(document, chapters=[])
    _expect_value_error(
        lambda: resolver.resolve(
            document=legacy_only_document,
            target_level="document",
        ),
        "legacy sections/structure_nodes fallback is not supported",
    )


def test_rejects_parent_id_mismatches_and_duplicates() -> None:
    resolver = ReadingTargetResolver()
    document = _document()

    _expect_value_error(
        lambda: resolver.resolve(
            document=document,
            target_level="section",
            chapter_id="wrong-chapter",
            section_id="section-1",
        ),
        "section parent chapter mismatch",
    )
    _expect_value_error(
        lambda: resolver.resolve(
            document=document,
            target_level="task_unit",
            chapter_id="chapter-1",
            section_id="section-2",
            task_unit_id="unit-1",
        ),
        "task_unit parent section mismatch",
    )

    first_chapter = document.chapters[0]
    duplicate_section = replace(
        first_chapter.sections[1],
        task_units=[
            _task_unit(
                "unit-1",
                "Duplicated unit id.",
                parent_section_id="section-2",
            )
        ],
    )
    duplicate_document = replace(
        document,
        chapters=[
            replace(
                first_chapter,
                sections=[first_chapter.sections[0], duplicate_section],
            )
        ],
    )
    _expect_value_error(
        lambda: resolver.resolve(
            document=duplicate_document,
            target_level="task_unit",
            task_unit_id="unit-1",
        ),
        "duplicate_task_unit_id:unit-1",
    )


@dataclass(frozen=True)
class _FakeAssets:
    errors: list[str]


@dataclass(frozen=True)
class _FakePreparationResult:
    structured_document: StructuredDocument | None
    assets: _FakeAssets


class _FakePreparationPipeline:
    def __init__(self, document: StructuredDocument) -> None:
        self.document = document
        self.calls: list[tuple[str, PreparationMode]] = []

    def prepare_and_load(self, *, doc_name: str, mode: PreparationMode) -> _FakePreparationResult:
        self.calls.append((doc_name, mode))
        return _FakePreparationResult(
            structured_document=self.document,
            assets=_FakeAssets(errors=[]),
        )


def test_coordinator_orchestrates_reading_target_resolution_once() -> None:
    document = _document()
    pipeline = _FakePreparationPipeline(document)
    coordinator = SectionTaskCoordinator.__new__(SectionTaskCoordinator)
    coordinator.document_preparation_pipeline = pipeline
    coordinator.reading_target_resolver = ReadingTargetResolver()

    resolved = coordinator.resolve_reading_target(
        doc_name=" fixture.pdf ",
        target_level="task_unit",
        chapter_id="chapter-1",
        section_id="section-1",
        task_unit_id="unit-1",
    )

    _assert(resolved.target_id == "unit-1", "coordinator should return resolved target")
    _assert(
        pipeline.calls == [("fixture.pdf", PreparationMode.BASE)],
        "coordinator should load the hierarchy once in base mode",
    )


if __name__ == "__main__":
    test_resolves_document_chapter_section_and_task_unit_targets()
    test_rejects_title_primary_and_legacy_fallback_paths()
    test_rejects_parent_id_mismatches_and_duplicates()
    test_coordinator_orchestrates_reading_target_resolution_once()
    print("reading target resolver tests passed")
