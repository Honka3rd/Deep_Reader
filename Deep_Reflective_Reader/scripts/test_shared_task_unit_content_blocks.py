#!/usr/bin/env python3
"""Shared task-unit content block foundation tests."""

from __future__ import annotations

import json

from shared.task_unit_model import (
    ArtifactTargetLevel,
    ArtifactTargetRef,
    TaskUnit,
    TaskUnitContentBlock,
    build_default_content_block_id,
)


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def test_string_content_adapts_to_single_block() -> None:
    unit = TaskUnit(
        unit_id="task-unit-42",
        title="Unit 42",
        container_title="Chapter 1",
        content="A sample content paragraph.",
        source_section_ids=["section-1"],
        is_fallback_generated=False,
    )
    blocks = unit.to_content_blocks()
    _assert(len(blocks) == 1, "non-empty string content should adapt to a single block")
    _assert(
        blocks[0].block_id == "task-unit-42:content:0",
        f"unexpected block id={blocks[0].block_id}",
    )
    _assert(
        blocks[0].content == unit.content,
        "adapted block content should preserve original task unit content",
    )
    _assert(
        len(unit.content_blocks) == 1,
        "TaskUnit should stabilize additive content_blocks during initialization",
    )


def test_empty_string_content_returns_empty_block_list() -> None:
    unit = TaskUnit(
        unit_id="task-unit-empty",
        title=None,
        container_title=None,
        content="",
        source_section_ids=["section-empty"],
        is_fallback_generated=False,
    )
    blocks = unit.to_content_blocks()
    _assert(blocks == [], "empty content should return an empty block list")
    _assert(
        unit.content_blocks == [],
        "empty content should keep stabilized content_blocks as empty list",
    )


def test_block_id_builder_is_deterministic() -> None:
    block_id_0 = build_default_content_block_id(task_unit_id="u-7", block_index=0)
    block_id_0_repeat = build_default_content_block_id(task_unit_id="u-7", block_index=0)
    block_id_3 = build_default_content_block_id(task_unit_id="u-7", block_index=3)
    _assert(block_id_0 == "u-7:content:0", f"unexpected block_id_0={block_id_0}")
    _assert(block_id_0_repeat == block_id_0, "block id builder should be deterministic")
    _assert(block_id_3 == "u-7:content:3", f"unexpected block_id_3={block_id_3}")


def test_content_block_round_trip_serialization() -> None:
    block = TaskUnitContentBlock(
        block_id="task-unit-99:content:0",
        content="paragraph",
        block_type="paragraph",
        artifact_ids=["artifact-a", "artifact-b"],
        metadata={"span_start": 0, "span_end": 9},
    )
    payload = block.to_dict()
    restored = TaskUnitContentBlock.from_dict(payload)
    _assert(restored == block, "content block should support to_dict/from_dict round-trip")


def test_content_block_artifact_target_refs_round_trip_serialization() -> None:
    content_block_target = ArtifactTargetRef(
        target_level=ArtifactTargetLevel.CONTENT_BLOCK,
        document_id="doc-1",
        chapter_id="chapter-1",
        section_id="section-1",
        task_unit_id="unit-1",
        content_block_id="unit-1:content:0",
        metadata={"quote_span_start": 4, "quote_span_end": 11},
    )
    task_unit_target = ArtifactTargetRef(
        target_level=ArtifactTargetLevel.TASK_UNIT,
        document_id="doc-1",
        chapter_id="chapter-1",
        section_id="section-1",
        task_unit_id="unit-1",
        content_block_id=None,
        metadata={"scope": "unit"},
    )
    block = TaskUnitContentBlock(
        block_id="unit-1:content:0",
        content="sample paragraph",
        artifact_ids=["artifact-1"],
        artifact_target_refs=[content_block_target, task_unit_target],
    )
    payload = block.to_dict()
    restored = TaskUnitContentBlock.from_dict(payload)
    _assert(
        restored.artifact_target_refs is not None
        and len(restored.artifact_target_refs) == 2,
        "artifact target refs should round-trip with two entries",
    )
    _assert(
        restored.artifact_target_refs[0].target_level == ArtifactTargetLevel.CONTENT_BLOCK,
        "first target level should preserve content_block level",
    )
    _assert(
        restored.artifact_target_refs[1].target_level == ArtifactTargetLevel.TASK_UNIT,
        "second target level should preserve task_unit level",
    )
    _assert(
        restored.artifact_ids == ["artifact-1"],
        "artifact_ids compatibility behavior should remain unchanged",
    )


def test_content_block_without_artifact_target_refs_remains_backward_compatible() -> None:
    legacy_payload = {
        "block_id": "legacy-unit:content:0",
        "content": "legacy",
        "block_type": None,
        "artifact_ids": ["artifact-legacy"],
        "metadata": {"hint": "legacy"},
    }
    restored = TaskUnitContentBlock.from_dict(legacy_payload)
    _assert(
        restored.artifact_target_refs is None,
        "legacy payload without artifact_target_refs should remain supported",
    )
    _assert(
        restored.artifact_ids == ["artifact-legacy"],
        "legacy artifact_ids should remain intact",
    )


def test_artifact_target_ref_invalid_target_level_fails_fast() -> None:
    invalid_payload = {
        "target_level": "invalid-level",
        "task_unit_id": "unit-x",
    }
    try:
        ArtifactTargetRef.from_dict(invalid_payload)
        raise AssertionError("invalid target_level should fail fast")
    except ValueError:
        pass


def test_task_unit_old_payload_without_content_blocks_still_works() -> None:
    legacy_payload = {
        "unit_id": "legacy-unit-1",
        "title": "Legacy Unit",
        "container_title": "Legacy Chapter",
        "content": "Legacy content text",
        "source_section_ids": ["legacy-section"],
        "is_fallback_generated": False,
        "parent_section_id": "legacy-section",
        "task_artifacts": None,
    }
    unit = TaskUnit.from_dict(legacy_payload)
    _assert(unit.content == "Legacy content text", "legacy payload content should remain available")
    _assert(
        len(unit.content_blocks) == 1,
        "legacy payload without content_blocks should auto-stabilize to one block",
    )
    _assert(
        unit.content_blocks[0].block_id == "legacy-unit-1:content:0",
        "legacy payload auto-stabilized block id should stay deterministic",
    )


def test_task_unit_content_blocks_round_trip_with_include_flag() -> None:
    unit = TaskUnit(
        unit_id="rich-unit-1",
        title="Rich Unit",
        container_title="Rich Chapter",
        content="Rich content text",
        source_section_ids=["rich-section"],
        is_fallback_generated=False,
    )
    payload_with_blocks = unit.to_dict(include_content_blocks=True)
    _assert(
        "content_blocks" in payload_with_blocks,
        "include_content_blocks=True should serialize additive content_blocks",
    )
    restored = TaskUnit.from_dict(payload_with_blocks)
    restored_blocks = restored.to_content_blocks()
    _assert(len(restored_blocks) == 1, "restored unit should retain one stabilized content block")
    _assert(
        restored_blocks[0].block_id == "rich-unit-1:content:0",
        "restored stabilized block id should remain deterministic",
    )
    _assert(
        restored.content == "Rich content text",
        "compatibility content string should remain supported after round-trip",
    )


def main() -> None:
    test_string_content_adapts_to_single_block()
    test_empty_string_content_returns_empty_block_list()
    test_block_id_builder_is_deterministic()
    test_content_block_round_trip_serialization()
    test_content_block_artifact_target_refs_round_trip_serialization()
    test_content_block_without_artifact_target_refs_remains_backward_compatible()
    test_artifact_target_ref_invalid_target_level_fails_fast()
    test_task_unit_old_payload_without_content_blocks_still_works()
    test_task_unit_content_blocks_round_trip_with_include_flag()
    print(
        json.dumps(
            {
                "status": "ok",
                "tests": [
                    "string_content_adapts_to_single_block",
                    "empty_string_content_returns_empty_block_list",
                    "block_id_builder_is_deterministic",
                    "content_block_round_trip_serialization",
                    "content_block_artifact_target_refs_round_trip_serialization",
                    "content_block_without_artifact_target_refs_remains_backward_compatible",
                    "artifact_target_ref_invalid_target_level_fails_fast",
                    "task_unit_old_payload_without_content_blocks_still_works",
                    "task_unit_content_blocks_round_trip_with_include_flag",
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
