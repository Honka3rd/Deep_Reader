import json
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class TaskUnitBoundaryMatchMode(StrEnum):
    """Boundary matching strategy for local split-plan application."""

    EXACT = "exact"
    CONTAINS = "contains"
    REGEX = "regex"


class TaskUnitSplitParserMode(StrEnum):
    """Parser provenance mode for task-unit split planning."""

    HEURISTIC = "heuristic"
    LLM_ENHANCED = "llm_enhanced"

    @classmethod
    def resolve(
        cls,
        value: "TaskUnitSplitParserMode | str | None",
    ) -> "TaskUnitSplitParserMode":
        """Resolve raw parser-mode input into canonical enum with safe default."""
        if isinstance(value, cls):
            return value
        if value is None:
            return cls.LLM_ENHANCED

        normalized = str(value).strip().lower().replace("-", "_")
        if not normalized:
            return cls.LLM_ENHANCED
        if normalized in {"llm_enhanced", "llm", "enhanced"}:
            return cls.LLM_ENHANCED
        if normalized in {"heuristic", "common", "default"}:
            return cls.HEURISTIC
        return cls.LLM_ENHANCED


@dataclass(frozen=True)
class TaskUnitSplitBoundaryInstruction:
    """One split-boundary instruction for section-internal split planning."""

    title: str | None
    start_anchor_text: str
    anchor_match_mode: TaskUnitBoundaryMatchMode = TaskUnitBoundaryMatchMode.EXACT
    anchor_occurrence: int = 1

    def to_dict(self) -> dict[str, Any]:
        """Serialize instruction into JSON-friendly dictionary."""
        return {
            "title": self.title,
            "start_anchor_text": self.start_anchor_text,
            "anchor_match_mode": self.anchor_match_mode.value,
            "anchor_occurrence": self.anchor_occurrence,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskUnitSplitBoundaryInstruction":
        """Deserialize one split-boundary instruction from dictionary."""
        raw_mode = str(
            data.get(
                "anchor_match_mode",
                TaskUnitBoundaryMatchMode.EXACT.value,
            )
        )
        try:
            mode = TaskUnitBoundaryMatchMode(raw_mode)
        except ValueError:
            mode = TaskUnitBoundaryMatchMode.EXACT

        occurrence = int(data.get("anchor_occurrence", 1))
        if occurrence <= 0:
            occurrence = 1

        return cls(
            title=None if data.get("title") is None else str(data.get("title")),
            start_anchor_text=str(data.get("start_anchor_text", "")),
            anchor_match_mode=mode,
            anchor_occurrence=occurrence,
        )


@dataclass(frozen=True)
class TaskUnitSplitPlan:
    """LLM-generated split plan skeleton consumed by local deterministic apply step."""

    parser_mode: TaskUnitSplitParserMode = TaskUnitSplitParserMode.LLM_ENHANCED
    instructions: list[TaskUnitSplitBoundaryInstruction] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize split plan into dictionary payload."""
        return {
            "parser_mode": self.parser_mode.value,
            "instructions": [instruction.to_dict() for instruction in self.instructions],
            "metadata": dict(self.metadata),
        }

    def to_json(self, *, ensure_ascii: bool = False, indent: int | None = 2) -> str:
        """Serialize split plan into JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=ensure_ascii, indent=indent)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskUnitSplitPlan":
        """Deserialize split plan from dictionary payload."""
        instruction_payloads = data.get("instructions", [])
        instructions: list[TaskUnitSplitBoundaryInstruction] = []
        for item in instruction_payloads:
            if not isinstance(item, dict):
                continue
            instruction = TaskUnitSplitBoundaryInstruction.from_dict(item)
            if instruction.start_anchor_text.strip():
                instructions.append(instruction)

        raw_metadata = data.get("metadata", {})
        metadata = (
            {str(key): str(value) for key, value in raw_metadata.items()}
            if isinstance(raw_metadata, dict)
            else {}
        )
        parser_mode = TaskUnitSplitParserMode.resolve(
            data.get("parser_mode", TaskUnitSplitParserMode.LLM_ENHANCED.value)
        )
        return cls(
            parser_mode=parser_mode,
            instructions=instructions,
            metadata=metadata,
        )

    @classmethod
    def from_json(cls, payload: str) -> "TaskUnitSplitPlan":
        """Deserialize split plan from JSON payload."""
        data = json.loads(payload)
        if not isinstance(data, dict):
            raise ValueError("TaskUnitSplitPlan JSON payload must be an object")
        return cls.from_dict(data)
