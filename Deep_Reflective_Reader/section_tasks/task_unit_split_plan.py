from dataclasses import dataclass, field
from enum import StrEnum


class TaskUnitBoundaryMatchMode(StrEnum):
    """Boundary matching strategy for local split-plan application."""

    EXACT = "exact"
    CONTAINS = "contains"
    REGEX = "regex"


@dataclass(frozen=True)
class TaskUnitSplitBoundaryInstruction:
    """One split-boundary instruction for section-internal split planning."""

    title: str | None
    start_anchor_text: str
    anchor_match_mode: TaskUnitBoundaryMatchMode = TaskUnitBoundaryMatchMode.EXACT
    anchor_occurrence: int = 1


@dataclass(frozen=True)
class TaskUnitSplitPlan:
    """LLM-generated split plan skeleton consumed by local deterministic apply step."""

    instructions: list[TaskUnitSplitBoundaryInstruction] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)
