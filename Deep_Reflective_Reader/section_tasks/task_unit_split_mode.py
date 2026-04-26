from enum import StrEnum


class TaskUnitSplitMode(StrEnum):
    """Selectable split strategy for one-section internal task-unit splitting."""

    SEMANTIC_SAFE = "semantic_safe"
    PROGRESSIVE = "progressive"
    LLM_ENHANCED = "llm_enhanced"

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
