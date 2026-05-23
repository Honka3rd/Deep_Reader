#!/usr/bin/env python3
"""Shared task-unit content block foundation tests."""

from __future__ import annotations

import hashlib
import json

from shared.task_unit_model import (
    ArtifactTargetLevel,
    ArtifactTargetRef,
    SEGMENTATION_METADATA_SCHEMA_VERSION,
    TaskUnit,
    TaskUnitContentBlock,
    build_default_content_block_id,
    segment_task_unit_content,
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


def test_segmentation_paragraph_first_with_deterministic_spans_and_hash() -> None:
    content = "Paragraph one.\n\nParagraph two."
    unit = TaskUnit(
        unit_id="seg-unit-1",
        title="Segmented",
        container_title="Chapter",
        content=content,
        source_section_ids=["section-1"],
        is_fallback_generated=False,
    )
    segmented_blocks = unit.segment_content_blocks()
    _assert(len(segmented_blocks) == 2, "paragraph-first segmentation should produce 2 blocks")

    source_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
    first = segmented_blocks[0]
    second = segmented_blocks[1]

    _assert(first.block_id == "seg-unit-1:content:0", "first segmented block id mismatch")
    _assert(second.block_id == "seg-unit-1:content:1", "second segmented block id mismatch")
    _assert(first.content == "Paragraph one.", "first paragraph content mismatch")
    _assert(second.content == "Paragraph two.", "second paragraph content mismatch")
    _assert(first.block_type == "paragraph", "first paragraph block type mismatch")
    _assert(second.block_type == "paragraph", "second paragraph block type mismatch")

    _assert(first.metadata is not None, "segmented block metadata should not be None")
    _assert(second.metadata is not None, "segmented block metadata should not be None")

    _assert(first.metadata.get("source_hash") == source_hash, "source_hash mismatch on first block")
    _assert(second.metadata.get("source_hash") == source_hash, "source_hash mismatch on second block")
    _assert(
        first.metadata.get("schema_version") == SEGMENTATION_METADATA_SCHEMA_VERSION,
        "schema_version mismatch on first block",
    )
    _assert(
        second.metadata.get("schema_version") == SEGMENTATION_METADATA_SCHEMA_VERSION,
        "schema_version mismatch on second block",
    )
    _assert(first.metadata.get("quote_span_start") == 0, "first span start mismatch")
    _assert(first.metadata.get("quote_span_end") == 14, "first span end mismatch")
    _assert(second.metadata.get("quote_span_start") == 16, "second span start mismatch")
    _assert(second.metadata.get("quote_span_end") == len(content), "second span end mismatch")

    _assert(
        content[first.metadata["quote_span_start"]:first.metadata["quote_span_end"]] == first.content,
        "first quote span should map back to block content",
    )
    _assert(
        content[second.metadata["quote_span_start"]:second.metadata["quote_span_end"]] == second.content,
        "second quote span should map back to block content",
    )


def test_segmentation_list_items_when_paragraph_is_deterministic_list() -> None:
    content = "- item one\n- item two\n\nTail paragraph."
    unit = TaskUnit(
        unit_id="seg-unit-2",
        title="List",
        container_title="Chapter",
        content=content,
        source_section_ids=["section-2"],
        is_fallback_generated=False,
    )
    segmented_blocks = segment_task_unit_content(unit)
    _assert(len(segmented_blocks) == 3, "list-aware segmentation should produce 3 blocks")
    _assert(segmented_blocks[0].block_type == "list_item", "first list item block_type mismatch")
    _assert(segmented_blocks[1].block_type == "list_item", "second list item block_type mismatch")
    _assert(segmented_blocks[2].block_type == "paragraph", "tail paragraph block_type mismatch")
    _assert(segmented_blocks[0].content == "- item one", "first list item content mismatch")
    _assert(segmented_blocks[1].content == "- item two", "second list item content mismatch")
    _assert(segmented_blocks[2].content == "Tail paragraph.", "tail paragraph content mismatch")


def test_segmentation_fallback_single_block_and_default_adapter_unchanged() -> None:
    content = "Single paragraph without deterministic split markers."
    unit = TaskUnit(
        unit_id="seg-unit-3",
        title="Single",
        container_title="Chapter",
        content=content,
        source_section_ids=["section-3"],
        is_fallback_generated=False,
    )

    default_blocks = unit.to_content_blocks()
    segmented_blocks = unit.segment_content_blocks()

    _assert(len(default_blocks) == 1, "default adapter should remain single block")
    _assert(default_blocks[0].metadata is None, "default adapter metadata should remain unchanged")
    _assert(len(segmented_blocks) == 1, "segmentation fallback should still return one block")
    _assert(segmented_blocks[0].block_type in {"paragraph", "full_content"}, "unexpected fallback block_type")
    _assert(segmented_blocks[0].content == content, "segmentation fallback should preserve content")


def test_segmentation_repeated_calls_are_idempotent() -> None:
    unit = TaskUnit(
        unit_id="seg-unit-4",
        title="Repeat",
        container_title="Chapter",
        content="Para A.\n\nPara B.",
        source_section_ids=["section-4"],
        is_fallback_generated=False,
    )
    first = [block.to_dict() for block in unit.segment_content_blocks()]
    second = [block.to_dict() for block in unit.segment_content_blocks()]
    _assert(first == second, "segmentation should be idempotent for repeated calls")


def test_segmentation_empty_content_behavior_unchanged() -> None:
    unit = TaskUnit(
        unit_id="seg-unit-5",
        title=None,
        container_title=None,
        content="",
        source_section_ids=["section-5"],
        is_fallback_generated=False,
    )
    _assert(unit.segment_content_blocks() == [], "empty content should segment to empty block list")


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
    test_segmentation_paragraph_first_with_deterministic_spans_and_hash()
    test_segmentation_list_items_when_paragraph_is_deterministic_list()
    test_segmentation_fallback_single_block_and_default_adapter_unchanged()
    test_segmentation_repeated_calls_are_idempotent()
    test_segmentation_empty_content_behavior_unchanged()
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
                    "segmentation_paragraph_first_with_deterministic_spans_and_hash",
                    "segmentation_list_items_when_paragraph_is_deterministic_list",
                    "segmentation_fallback_single_block_and_default_adapter_unchanged",
                    "segmentation_repeated_calls_are_idempotent",
                    "segmentation_empty_content_behavior_unchanged",
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
