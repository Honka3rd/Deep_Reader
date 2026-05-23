from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import re
from typing import Any

from shared.artifact_target_model import ArtifactTargetLevel, ArtifactTargetRef
from shared.task_artifacts import TaskArtifacts


SEGMENTATION_METADATA_SCHEMA_VERSION = "content_block_segmentation_v1"
_LIST_ITEM_LINE_RE = re.compile(r"^\s*(?:[-*+]|(?:\d+|[a-zA-Z])[.)])\s+\S")


def _compute_source_hash(content: str) -> str:
    """Compute deterministic source hash from original task-unit content."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _trimmed_nonempty_span(
    content: str,
    span_start: int,
    span_end: int,
) -> tuple[int, int] | None:
    """Return trimmed non-empty span inside [span_start, span_end)."""
    start = span_start
    end = span_end
    while start < end and content[start].isspace():
        start += 1
    while end > start and content[end - 1].isspace():
        end -= 1
    if start >= end:
        return None
    return (start, end)


def _collect_paragraph_spans(content: str) -> list[tuple[int, int]]:
    """Collect deterministic paragraph spans separated by blank-line boundaries."""
    spans: list[tuple[int, int]] = []
    cursor = 0
    for match in re.finditer(r"\n\s*\n+", content):
        trimmed_span = _trimmed_nonempty_span(content, cursor, match.start())
        if trimmed_span is not None:
            spans.append(trimmed_span)
        cursor = match.end()
    trimmed_span = _trimmed_nonempty_span(content, cursor, len(content))
    if trimmed_span is not None:
        spans.append(trimmed_span)
    return spans


def _try_split_list_items(
    content: str,
    paragraph_start: int,
    paragraph_end: int,
) -> list[tuple[int, int]] | None:
    """Split paragraph into list-item spans only when all non-empty lines are list items."""
    paragraph = content[paragraph_start:paragraph_end]
    if "\n" not in paragraph:
        return None

    item_spans: list[tuple[int, int]] = []
    line_cursor = paragraph_start
    for raw_line in paragraph.splitlines(keepends=True):
        line_start = line_cursor
        line_end = line_cursor + len(raw_line)
        line_cursor = line_end

        trimmed_span = _trimmed_nonempty_span(content, line_start, line_end)
        if trimmed_span is None:
            continue

        trimmed_text = content[trimmed_span[0]:trimmed_span[1]]
        if _LIST_ITEM_LINE_RE.match(trimmed_text) is None:
            return None
        item_spans.append(trimmed_span)

    if len(item_spans) < 2:
        return None
    return item_spans


def _segment_content_spans(
    content: str,
) -> list[tuple[int, int, str]]:
    """Return deterministic segmented spans as (start, end, block_type)."""
    paragraph_spans = _collect_paragraph_spans(content)
    if not paragraph_spans:
        if content == "":
            return []
        return [(0, len(content), "full_content")]

    segmented_spans: list[tuple[int, int, str]] = []
    for paragraph_start, paragraph_end in paragraph_spans:
        list_item_spans = _try_split_list_items(content, paragraph_start, paragraph_end)
        if list_item_spans is not None:
            segmented_spans.extend((start, end, "list_item") for start, end in list_item_spans)
            continue
        segmented_spans.append((paragraph_start, paragraph_end, "paragraph"))

    if not segmented_spans:
        return [(0, len(content), "full_content")]
    return segmented_spans


def _build_segmented_content_blocks(
    task_unit_id: str,
    content: str,
) -> list[TaskUnitContentBlock]:
    """Build deterministic segmented content blocks from raw task-unit content."""
    segmented_spans = _segment_content_spans(content)
    if not segmented_spans:
        return []

    source_hash = _compute_source_hash(content)
    content_blocks: list[TaskUnitContentBlock] = []
    for block_index, (span_start, span_end, block_type) in enumerate(segmented_spans):
        block_id = build_default_content_block_id(task_unit_id=task_unit_id, block_index=block_index)
        content_blocks.append(
            TaskUnitContentBlock(
                block_id=block_id,
                content=content[span_start:span_end],
                block_type=block_type,
                metadata={
                    "source_hash": source_hash,
                    "content_block_id": block_id,
                    "quote_span_start": span_start,
                    "quote_span_end": span_end,
                    "schema_version": SEGMENTATION_METADATA_SCHEMA_VERSION,
                },
            )
        )
    return content_blocks


@dataclass(frozen=True)
class TaskUnitContentBlock:
    """Shared-layer render/interaction content block under a task unit."""

    block_id: str
    content: str
    block_type: str | None = None
    artifact_ids: list[str] | None = None
    artifact_target_refs: list[ArtifactTargetRef] | None = None
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize content block into JSON-friendly dictionary."""
        return {
            "block_id": self.block_id,
            "content": self.content,
            "block_type": self.block_type,
            "artifact_ids": None if self.artifact_ids is None else list(self.artifact_ids),
            "artifact_target_refs": (
                None
                if self.artifact_target_refs is None
                else [target_ref.to_dict() for target_ref in self.artifact_target_refs]
            ),
            "metadata": None if self.metadata is None else dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskUnitContentBlock":
        """Deserialize content block from dictionary payload."""
        artifact_ids_payload = data.get("artifact_ids")
        artifact_target_refs_payload = data.get("artifact_target_refs")
        metadata_payload = data.get("metadata")
        parsed_target_refs: list[ArtifactTargetRef] | None = None
        if isinstance(artifact_target_refs_payload, list):
            parsed_target_refs = []
            for target_payload in artifact_target_refs_payload:
                if isinstance(target_payload, ArtifactTargetRef):
                    parsed_target_refs.append(target_payload)
                    continue
                if isinstance(target_payload, dict):
                    parsed_target_refs.append(ArtifactTargetRef.from_dict(target_payload))
                    continue
                raise TypeError(
                    "TaskUnitContentBlock.from_dict expected artifact_target_refs list entries as dict payloads"
                )
        return cls(
            block_id=str(data["block_id"]),
            content=str(data.get("content", "")),
            block_type=None if data.get("block_type") is None else str(data.get("block_type")),
            artifact_ids=(
                None
                if artifact_ids_payload is None
                else [str(value) for value in artifact_ids_payload]
            ),
            artifact_target_refs=parsed_target_refs,
            metadata=(
                None
                if metadata_payload is None
                else {str(key): value for key, value in metadata_payload.items()}
            ),
        )


def build_default_content_block_id(task_unit_id: str, block_index: int = 0) -> str:
    """Build deterministic task-unit content block id for adapter-generated blocks."""
    return f"{task_unit_id}:content:{block_index}"


@dataclass(frozen=True)
class TaskUnit:
    """Resolved task-time unit for section-based summary/quiz execution."""

    unit_id: str
    title: str | None
    container_title: str | None
    content: str
    source_section_ids: list[str]
    is_fallback_generated: bool
    parent_section_id: str | None = None
    task_artifacts: TaskArtifacts | None = None
    content_blocks: list[TaskUnitContentBlock] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Stabilize additive rich-content representation for compatibility payloads."""
        normalized_blocks: list[TaskUnitContentBlock] = []
        for block in self.content_blocks:
            if isinstance(block, TaskUnitContentBlock):
                normalized_blocks.append(block)
                continue
            if isinstance(block, dict):
                normalized_blocks.append(TaskUnitContentBlock.from_dict(block))
                continue
            raise TypeError(
                "TaskUnit.content_blocks entries must be TaskUnitContentBlock or dict payloads"
            )

        if not normalized_blocks and self.content != "":
            normalized_blocks = [
                TaskUnitContentBlock(
                    block_id=build_default_content_block_id(
                        task_unit_id=self.unit_id,
                        block_index=0,
                    ),
                    content=self.content,
                )
            ]
        object.__setattr__(self, "content_blocks", normalized_blocks)

    def to_content_blocks(self) -> list[TaskUnitContentBlock]:
        """
        Return stabilized content blocks.

        `content_blocks` is preferred when present; compatibility string content
        remains available via `content`.
        """
        if self.content_blocks:
            return list(self.content_blocks)
        if self.content == "":
            return []
        return [
            TaskUnitContentBlock(
                block_id=build_default_content_block_id(
                    task_unit_id=self.unit_id,
                    block_index=0,
                ),
                content=self.content,
            )
        ]

    def segment_content_blocks(self) -> list[TaskUnitContentBlock]:
        """Explicit opt-in deterministic segmentation from `content` to multiple blocks."""
        return _build_segmented_content_blocks(task_unit_id=self.unit_id, content=self.content)

    def to_dict(self, include_content_blocks: bool = False) -> dict[str, Any]:
        """Serialize task unit into JSON-friendly dictionary."""
        payload: dict[str, Any] = {
            "unit_id": self.unit_id,
            "title": self.title,
            "container_title": self.container_title,
            "content": self.content,
            "source_section_ids": list(self.source_section_ids),
            "is_fallback_generated": self.is_fallback_generated,
            "parent_section_id": self.parent_section_id,
            "task_artifacts": (
                None if self.task_artifacts is None else self.task_artifacts.to_dict()
            ),
        }
        if include_content_blocks:
            payload["content_blocks"] = [
                content_block.to_dict()
                for content_block in self.content_blocks
            ]
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskUnit":
        """Deserialize task unit from dictionary payload."""
        source_ids_payload = data.get("source_section_ids", [])
        source_ids = [str(value) for value in source_ids_payload]
        content_blocks_payload = data.get("content_blocks")
        parsed_content_blocks: list[TaskUnitContentBlock] = []
        if isinstance(content_blocks_payload, list):
            for block_payload in content_blocks_payload:
                if isinstance(block_payload, TaskUnitContentBlock):
                    parsed_content_blocks.append(block_payload)
                    continue
                if isinstance(block_payload, dict):
                    parsed_content_blocks.append(
                        TaskUnitContentBlock.from_dict(block_payload)
                    )
                    continue
                raise TypeError(
                    "TaskUnit.from_dict expected content_blocks list entries as dict payloads"
                )
        return cls(
            unit_id=str(data["unit_id"]),
            title=(
                None if data.get("title") is None else str(data.get("title"))
            ),
            container_title=(
                None
                if data.get("container_title") is None
                else str(data.get("container_title"))
            ),
            content=str(data["content"]),
            source_section_ids=source_ids,
            is_fallback_generated=bool(data.get("is_fallback_generated", False)),
            parent_section_id=(
                None
                if data.get("parent_section_id") is None
                else str(data.get("parent_section_id"))
            ),
            task_artifacts=TaskArtifacts.from_dict(data.get("task_artifacts")),
            content_blocks=parsed_content_blocks,
        )


def segment_task_unit_content(task_unit: TaskUnit) -> list[TaskUnitContentBlock]:
    """Shared-layer helper for explicit deterministic content segmentation."""
    return task_unit.segment_content_blocks()
