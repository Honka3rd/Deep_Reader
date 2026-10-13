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
from section_tasks.reading_target_resolver import (  # noqa: E402
    CONTENT_BLOCK_VALIDATION_READ_MODEL,
    ReadingTargetResolver,
)
from shared.task_unit_model import TaskUnit, TaskUnitContentBlock  # noqa: E402


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _task_unit(
    unit_id: str,
    content: str,
    *,
    parent_section_id: str | None,
    content_blocks: list[TaskUnitContentBlock] | None = None,
) -> TaskUnit:
    return TaskUnit(
        unit_id=unit_id,
        title=f"Task {unit_id}",
        container_title=None,
        content=content,
        source_section_ids=[] if parent_section_id is None else [parent_section_id],
        is_fallback_generated=False,
        parent_section_id=parent_section_id,
        content_blocks=[] if content_blocks is None else content_blocks,
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
    _assert("Unit one content." in resolved_document.content, "document should include hierarchy task units")
    _assert("Unit two content." in resolved_document.content, "document should aggregate across sections")
    _assert("Legacy-only" not in resolved_document.content, "document must not use legacy root sections")

    resolved_chapter = resolver.resolve(
        document=document,
        target_level="chapter",
        chapter_id="chapter-1",
    )
    _assert(resolved_chapter.target_id == "chapter-1", "chapter should resolve by id")
    _assert("Unit one content." in resolved_chapter.content, "chapter should join task-unit content")

    resolved_section = resolver.resolve(
        document=document,
        target_level="section",
        chapter_id="chapter-1",
        section_id="section-1",
    )
    _assert(resolved_section.target_id == "section-1", "section should resolve by id")
    _assert(resolved_section.chapter_id == "chapter-1", "section should retain parent chapter")
    _assert("Unit one content." in resolved_section.content, "section should prefer task-unit source content")

    resolved_task_unit = resolver.resolve(
        document=document,
        target_level="task_unit",
        chapter_id="chapter-1",
        section_id="section-1",
        task_unit_id="unit-1",
    )
    _assert(resolved_task_unit.target_id == "unit-1", "task unit should resolve by id")
    _assert(resolved_task_unit.section_id == "section-1", "task unit should retain section id")


def test_document_target_uses_hierarchy_content_when_raw_text_is_empty() -> None:
    resolver = ReadingTargetResolver()
    document = replace(_document(), raw_text="")

    resolved_document = resolver.resolve(document=document, target_level="document")

    _assert(
        len(resolved_document.content.strip()) >= 20,
        "document target should assemble readable hierarchy content without raw_text",
    )
    _assert("Unit one content." in resolved_document.content, "first task unit should be present")
    _assert("Unit two content." in resolved_document.content, "second task unit should be present")
    _assert(
        "Legacy-only section." not in resolved_document.content,
        "legacy root sections must not backfill document target content",
    )


def test_content_block_validation_read_model_accepts_segmented_render_blocks() -> None:
    resolver = ReadingTargetResolver()
    explicit_block = TaskUnitContentBlock(
        block_id="unit-rich:stored:0",
        content="Stored content block.",
        block_type="stored_paragraph",
    )
    explicit_task_unit = _task_unit(
        "unit-rich",
        "Paragraph one.\n\nParagraph two.",
        parent_section_id="section-1",
        content_blocks=[explicit_block],
    )
    explicit_blocks = resolver.content_blocks_for_target_validation(explicit_task_unit)

    _assert(
        CONTENT_BLOCK_VALIDATION_READ_MODEL == "segmented_task_unit_content_blocks",
        "resolver should declare the segmented task-unit block read model",
    )
    _assert(
        len(explicit_blocks) == 3,
        "segmented render blocks plus non-colliding explicit blocks should be accepted",
    )
    _assert(
        [block.block_id for block in explicit_blocks]
        == ["unit-rich:content:0", "unit-rich:content:1", "unit-rich:stored:0"],
        "resolver validation should preserve visible segmented ids and custom stored ids",
    )
    _assert(
        explicit_blocks[0].content == "Paragraph one."
        and explicit_blocks[1].content == "Paragraph two."
        and explicit_blocks[2].content == "Stored content block.",
        "resolver validation should use segmented text for visible ids and stored text for custom ids",
    )

    compatibility_task_unit = _task_unit(
        "unit-compat",
        "Paragraph one.\n\nParagraph two.",
        parent_section_id="section-1",
    )
    compatibility_blocks = resolver.content_blocks_for_target_validation(
        compatibility_task_unit
    )

    _assert(
        len(compatibility_blocks) == 2,
        "validation read model should compute segmented blocks for UI-visible ids",
    )
    _assert(
        [block.block_id for block in compatibility_blocks]
        == ["unit-compat:content:0", "unit-compat:content:1"],
        "missing explicit blocks should use deterministic segmented ids",
    )
    _assert(
        compatibility_blocks[0].content == "Paragraph one."
        and compatibility_blocks[1].content == "Paragraph two.",
        "segmented blocks should preserve per-block content",
    )


def test_segmented_render_block_ids_resolve_as_content_block_targets() -> None:
    resolver = ReadingTargetResolver()
    segmented_task_unit = _task_unit(
        "unit-segmented",
        "First render paragraph.\n\nSecond render paragraph.",
        parent_section_id="section-segmented",
    )
    section = _section(
        "section-segmented",
        "Section segmented content.",
        parent_chapter_id="chapter-segmented",
        task_units=[segmented_task_unit],
    )
    document = StructuredDocument(
        document_id="doc-segmented",
        title="Segmented Resolver Fixture",
        source_path=None,
        language="en",
        raw_text="Segmented document content.",
        chapters=[
            StructuredChapter(
                chapter_id="chapter-segmented",
                title="Chapter Segmented",
                level=1,
                chapter_role=None,
                sections=[section],
            )
        ],
    )

    resolved = resolver.resolve(
        document=document,
        target_level="content_block",
        chapter_id="chapter-segmented",
        section_id="section-segmented",
        task_unit_id="unit-segmented",
        content_block_id="unit-segmented:content:0",
    )
    _assert(
        resolved.content == "First render paragraph.",
        "first segmented block should target the first visible render segment",
    )

    resolved_second = resolver.resolve(
        document=document,
        target_level="content_block",
        chapter_id="chapter-segmented",
        section_id="section-segmented",
        task_unit_id="unit-segmented",
        content_block_id="unit-segmented:content:1",
    )
    _assert(
        resolved_second.content == "Second render paragraph.",
        "second segmented block should target the second visible render segment",
    )

    _expect_value_error(
        lambda: resolver.resolve(
            document=document,
            target_level="content_block",
            chapter_id="chapter-segmented",
            section_id="section-segmented",
            task_unit_id="unit-segmented",
            content_block_id="unit-segmented:content:2",
        ),
        "content_block_id 'unit-segmented:content:2' not found",
    )


def test_resolves_content_block_targets_through_task_unit_ownership() -> None:
    resolver = ReadingTargetResolver()
    document = _document()

    resolved = resolver.resolve(
        document=document,
        target_level="content_block",
        chapter_id="chapter-1",
        section_id="section-1",
        task_unit_id="unit-1",
        content_block_id="unit-1:content:0",
    )

    _assert(resolved.target_level == "content_block", "target level should resolve")
    _assert(
        resolved.target_id == "unit-1:content:0",
        "content block target_id should be the block id",
    )
    _assert(
        resolved.chapter_id == "chapter-1"
        and resolved.section_id == "section-1"
        and resolved.task_unit_id == "unit-1"
        and resolved.content_block_id == "unit-1:content:0",
        "content block target should preserve parent ownership metadata",
    )
    _assert(
        resolved.task_unit is not None and resolved.task_unit.unit_id == "unit-1",
        "content block target should retain parent task unit",
    )
    _assert(
        resolved.content_block is not None
        and resolved.content_block.block_id == "unit-1:content:0",
        "content block target should retain selected block",
    )
    _assert(
        resolved.content == "Unit one content.",
        "content block target should use the selected block text as source content",
    )


def test_rejects_content_block_missing_duplicate_parent_mismatch_and_stale_evidence() -> None:
    resolver = ReadingTargetResolver()
    block_a = TaskUnitContentBlock(
        block_id="rich-unit:block:0",
        content="Block A",
        metadata={"source_hash": "hash-a", "source_structure_version": 3},
    )
    duplicate_a = TaskUnitContentBlock(
        block_id="rich-unit:block:0",
        content="Duplicate Block A",
        metadata={"source_hash": "hash-a", "source_structure_version": 3},
    )
    rich_task_unit = _task_unit(
        "rich-unit",
        "Block A\n\nDuplicate Block A",
        parent_section_id="section-rich",
        content_blocks=[block_a],
    )
    section = _section(
        "section-rich",
        "Rich section content.",
        parent_chapter_id="chapter-rich",
        task_units=[rich_task_unit],
    )
    document = StructuredDocument(
        document_id="doc-rich",
        title="Rich Resolver Fixture",
        source_path=None,
        language="en",
        raw_text="Rich document content.",
        chapters=[
            StructuredChapter(
                chapter_id="chapter-rich",
                title="Chapter Rich",
                level=1,
                chapter_role=None,
                sections=[section],
            )
        ],
    )

    resolved = resolver.resolve(
        document=document,
        target_level="content_block",
        chapter_id="chapter-rich",
        section_id="section-rich",
        task_unit_id="rich-unit",
        content_block_id="rich-unit:block:0",
        source_hash="hash-a",
        source_structure_version=3,
    )
    _assert(
        resolved.content_block_id == "rich-unit:block:0",
        "matching source evidence should resolve",
    )
    _assert(
        resolved.content == "Block A",
        "resolved content should come from the selected explicit content block",
    )

    _expect_value_error(
        lambda: resolver.resolve(
            document=document,
            target_level="content_block",
            chapter_id="chapter-rich",
            section_id="section-rich",
            task_unit_id="rich-unit",
        ),
        "content_block target requires content_block_id",
    )
    _expect_value_error(
        lambda: resolver.resolve(
            document=document,
            target_level="task_unit",
            chapter_id="chapter-rich",
            section_id="section-rich",
            task_unit_id="rich-unit",
            content_block_id="rich-unit:block:0",
        ),
        "non-content_block targets must not include content_block_id",
    )
    _expect_value_error(
        lambda: resolver.resolve(
            document=document,
            target_level="content_block",
            chapter_id="chapter-rich",
            section_id="section-rich",
            task_unit_id="rich-unit",
            content_block_id="missing-block",
        ),
        "content_block_id 'missing-block' not found",
    )
    _expect_value_error(
        lambda: resolver.resolve(
            document=document,
            target_level="content_block",
            chapter_id="chapter-rich",
            section_id="wrong-section",
            task_unit_id="rich-unit",
            content_block_id="rich-unit:block:0",
        ),
        "content_block parent section mismatch",
    )
    _expect_value_error(
        lambda: resolver.resolve(
            document=document,
            target_level="content_block",
            chapter_id="wrong-chapter",
            section_id="section-rich",
            task_unit_id="rich-unit",
            content_block_id="rich-unit:block:0",
        ),
        "content_block parent chapter mismatch",
    )
    _expect_value_error(
        lambda: resolver.resolve(
            document=document,
            target_level="content_block",
            chapter_id="chapter-rich",
            section_id="section-rich",
            task_unit_id="rich-unit",
            content_block_id="rich-unit:block:0",
            source_hash="stale-hash",
        ),
        "stale_content_block_target: source_hash mismatch",
    )
    _expect_value_error(
        lambda: resolver.resolve(
            document=document,
            target_level="content_block",
            chapter_id="chapter-rich",
            section_id="section-rich",
            task_unit_id="rich-unit",
            content_block_id="rich-unit:block:0",
            source_structure_version=2,
        ),
        "stale_content_block_target: source_structure_version mismatch",
    )

    duplicate_document = replace(
        document,
        chapters=[
            replace(
                document.chapters[0],
                sections=[
                    replace(
                        section,
                        task_units=[
                            replace(
                                rich_task_unit,
                                content_blocks=[block_a, duplicate_a],
                            )
                        ],
                    )
                ],
            )
        ],
    )
    _expect_value_error(
        lambda: resolver.resolve(
            document=duplicate_document,
            target_level="content_block",
            task_unit_id="rich-unit",
            content_block_id="rich-unit:block:0",
        ),
        "duplicate_content_block_id:rich-unit:block:0",
    )


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
    test_document_target_uses_hierarchy_content_when_raw_text_is_empty()
    test_content_block_validation_read_model_accepts_segmented_render_blocks()
    test_segmented_render_block_ids_resolve_as_content_block_targets()
    test_resolves_content_block_targets_through_task_unit_ownership()
    test_rejects_content_block_missing_duplicate_parent_mismatch_and_stale_evidence()
    test_rejects_title_primary_and_legacy_fallback_paths()
    test_rejects_parent_id_mismatches_and_duplicates()
    test_coordinator_orchestrates_reading_target_resolution_once()
    print("reading target resolver tests passed")
