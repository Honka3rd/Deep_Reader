from enum import StrEnum


class TaskUnitSplitMode(StrEnum):
    """Selectable split strategy for one-section internal task-unit splitting."""

    SEMANTIC_SAFE = "semantic_safe"
    PROGRESSIVE = "progressive"
    LLM_ENHANCED = "llm_enhanced"

    @classmethod
    def supported_values(cls) -> tuple[str, ...]:
        """Return canonical public values accepted by REST/API contracts."""
        return tuple(member.value for member in cls)

    @classmethod
    def resolve(cls, value: "TaskUnitSplitMode | str | None") -> "TaskUnitSplitMode":
        """Resolve raw split-mode input into canonical enum with safe default."""
        if isinstance(value, cls):
            return value
        if value is None:
            return cls.SEMANTIC_SAFE

        normalized = str(value).strip().lower().replace("-", "_")
        if not normalized:
            return cls.SEMANTIC_SAFE
        if normalized in {"semantic_safe", "semantic", "safe"}:
            return cls.SEMANTIC_SAFE
        if normalized in {"progressive", "window"}:
            return cls.PROGRESSIVE
        if normalized in {"llm_enhanced", "enhanced", "llm"}:
            return cls.LLM_ENHANCED
        return cls.SEMANTIC_SAFE

    @classmethod
    def resolve_strict(cls, value: "TaskUnitSplitMode | str") -> "TaskUnitSplitMode":
        """Resolve split mode with strict validation for API request-time input."""
        if isinstance(value, cls):
            return value

        normalized = str(value).strip().lower().replace("-", "_")
        if normalized in cls.supported_values():
            return cls(normalized)
        raise ValueError(
            "unknown task_unit_split_mode. supported values: "
            + ", ".join(cls.supported_values())
        )
