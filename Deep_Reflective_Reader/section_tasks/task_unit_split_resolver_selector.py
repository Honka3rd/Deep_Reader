from section_tasks.abstract_task_unit_split_resolver import AbstractTaskUnitSplitResolver
from section_tasks.heuristic_task_unit_split_resolver import HeuristicTaskUnitSplitResolver
from section_tasks.llm_task_unit_split_resolver import LLMTaskUnitSplitResolver
from section_tasks.task_unit_split_mode import TaskUnitSplitMode


class TaskUnitSplitResolverSelector:
    """Select split resolver implementation by task-unit split mode."""

    def __init__(
        self,
        semantic_safe_resolver: HeuristicTaskUnitSplitResolver | None = None,
        progressive_resolver: HeuristicTaskUnitSplitResolver | None = None,
        llm_resolver: LLMTaskUnitSplitResolver | None = None,
    ):
        self.semantic_safe_resolver = semantic_safe_resolver or HeuristicTaskUnitSplitResolver(
            split_mode=TaskUnitSplitMode.SEMANTIC_SAFE
        )
        self.progressive_resolver = progressive_resolver or HeuristicTaskUnitSplitResolver(
            split_mode=TaskUnitSplitMode.PROGRESSIVE
        )
        self.llm_resolver = llm_resolver or LLMTaskUnitSplitResolver(
            heuristic_fallback_resolver=self.semantic_safe_resolver
        )

    def get_resolver(
        self,
        mode: TaskUnitSplitMode | str = TaskUnitSplitMode.SEMANTIC_SAFE,
    ) -> AbstractTaskUnitSplitResolver:
        """Return resolver implementation for selected split mode."""
        resolved_mode = TaskUnitSplitMode.resolve(mode)
        if resolved_mode == TaskUnitSplitMode.PROGRESSIVE:
            return self.progressive_resolver
        if resolved_mode == TaskUnitSplitMode.LLM_ENHANCED:
            return self.llm_resolver
        return self.semantic_safe_resolver
