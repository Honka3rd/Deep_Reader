import json
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from document_structure.section_role import SectionRole


class AnchorMatchMode(StrEnum):
    """Anchor matching strategy for local boundary resolution."""

    EXACT = "exact"
    CONTAINS = "contains"
    REGEX = "regex"


class SectionParserMode(StrEnum):
    """Parser provenance mode for split-plan generation."""

    COMMON = "common"
    LLM_ENHANCED = "llm_enhanced"

    @classmethod
    def resolve(cls, value: "SectionParserMode | str | None") -> "SectionParserMode":
        """Resolve raw parser-mode input into canonical enum."""
        if isinstance(value, cls):
            return value
        if value is None:
            return cls.LLM_ENHANCED

        normalized = str(value).strip().lower().replace("-", "_")
        if not normalized:
            return cls.LLM_ENHANCED
        normalized = normalized.replace("_", "")

        if normalized in ("llmenhanced", "llm", "enhanced"):
            return cls.LLM_ENHANCED
        if normalized in ("common", "default", "heuristic"):
            return cls.COMMON
        return cls.LLM_ENHANCED


@dataclass(frozen=True)
class SplitBoundaryInstruction:
    """One LLM-proposed boundary instruction resolved locally on raw text."""

    title: str | None
    level: int
    section_role: SectionRole | None
    container_title: str | None
    start_anchor_text: str
    anchor_match_mode: AnchorMatchMode = AnchorMatchMode.EXACT
    anchor_occurrence: int = 1

    def to_dict(self) -> dict[str, Any]:
        """Serialize instruction into JSON-friendly dictionary."""
        return {
            "title": self.title,
            "level": self.level,
            "section_role": (
                None if self.section_role is None else self.section_role.value
            ),
            "container_title": self.container_title,
            "start_anchor_text": self.start_anchor_text,
            "anchor_match_mode": self.anchor_match_mode.value,
            "anchor_occurrence": self.anchor_occurrence,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SplitBoundaryInstruction":
        """Deserialize one boundary instruction from dictionary."""
        raw_mode = str(data.get("anchor_match_mode", AnchorMatchMode.EXACT.value))
        try:
            mode = AnchorMatchMode(raw_mode)
        except ValueError:
            mode = AnchorMatchMode.EXACT

        occurrence = int(data.get("anchor_occurrence", 1))
        if occurrence <= 0:
            occurrence = 1

        return cls(
            title=None if data.get("title") is None else str(data.get("title")),
            level=max(1, int(data.get("level", 1))),
            section_role=SectionRole.resolve(data.get("section_role")),
            container_title=(
                None
                if data.get("container_title") is None
                else str(data.get("container_title"))
            ),
            start_anchor_text=str(data.get("start_anchor_text", "")),
            anchor_match_mode=mode,
            anchor_occurrence=occurrence,
        )


@dataclass(frozen=True)
class SectionSplitPlan:
    """LLM-produced split plan consumed by local deterministic slicing logic."""

    parser_mode: SectionParserMode
    instructions: list[SplitBoundaryInstruction] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def sections(self) -> list[SplitBoundaryInstruction]:
        """Region-aware alias name for prompt/API schema."""
        return self.instructions

    def to_dict(self) -> dict[str, Any]:
        """Serialize split plan into dictionary payload."""
        section_payloads = [item.to_dict() for item in self.instructions]
        return {
            "parser_mode": self.parser_mode.value,
            "sections": section_payloads,
            "instructions": section_payloads,
            "metadata": dict(self.metadata),
        }

    def to_json(self, *, ensure_ascii: bool = False, indent: int | None = 2) -> str:
        """Serialize split plan into JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=ensure_ascii, indent=indent)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SectionSplitPlan":
        """Deserialize split plan from dictionary payload."""
        instruction_payloads = data.get("sections")
        if instruction_payloads is None:
            instruction_payloads = data.get("instructions", [])
        instructions: list[SplitBoundaryInstruction] = []
        for item in instruction_payloads:
            if not isinstance(item, dict):
                continue
            instruction = SplitBoundaryInstruction.from_dict(item)
            if instruction.start_anchor_text.strip():
                instructions.append(instruction)

        raw_metadata = data.get("metadata", {})
        metadata = (
            {str(key): str(value) for key, value in raw_metadata.items()}
            if isinstance(raw_metadata, dict)
            else {}
        )
        parser_mode = SectionParserMode.resolve(
            data.get("parser_mode", SectionParserMode.LLM_ENHANCED.value)
        )
        return cls(
            parser_mode=parser_mode,
            instructions=instructions,
            metadata=metadata,
        )

    @classmethod
    def from_json(cls, payload: str) -> "SectionSplitPlan":
        """Deserialize split plan from JSON payload."""
        data = json.loads(payload)
        if not isinstance(data, dict):
            raise ValueError("SectionSplitPlan JSON payload must be an object")
        return cls.from_dict(data)
