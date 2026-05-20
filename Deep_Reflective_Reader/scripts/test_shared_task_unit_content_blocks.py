#!/usr/bin/env python3
"""Shared task-unit content block foundation tests."""

from __future__ import annotations

import json

from shared.task_unit_model import (
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


def main() -> None:
    test_string_content_adapts_to_single_block()
    test_empty_string_content_returns_empty_block_list()
    test_block_id_builder_is_deterministic()
    test_content_block_round_trip_serialization()
    print(
        json.dumps(
            {
                "status": "ok",
                "tests": [
                    "string_content_adapts_to_single_block",
                    "empty_string_content_returns_empty_block_list",
                    "block_id_builder_is_deterministic",
                    "content_block_round_trip_serialization",
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
