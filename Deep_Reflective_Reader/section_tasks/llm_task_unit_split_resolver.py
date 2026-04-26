from document_structure.structured_document import StructuredSection
from llm.llm_provider import LLMProvider
from section_tasks.abstract_task_unit_split_resolver import AbstractTaskUnitSplitResolver
from section_tasks.heuristic_task_unit_split_resolver import HeuristicTaskUnitSplitResolver
from section_tasks.task_unit import TaskUnit
from section_tasks.task_unit_split_mode import TaskUnitSplitMode
from section_tasks.task_unit_split_plan import TaskUnitSplitPlan


class LLMTaskUnitSplitResolver(AbstractTaskUnitSplitResolver):
    """High-cost split resolver skeleton: LLM builds plan, local code applies it."""

    def __init__(
        self,
        llm_provider: LLMProvider | None = None,
        heuristic_fallback_resolver: HeuristicTaskUnitSplitResolver | None = None,
    ):
        self.llm_provider = llm_provider
        self.heuristic_fallback_resolver = (
            heuristic_fallback_resolver
            or HeuristicTaskUnitSplitResolver(split_mode=TaskUnitSplitMode.SEMANTIC_SAFE)
        )

    def split_section(
        self,
        *,
        section: StructuredSection,
        section_index: int,
        task_unit_min_chars: int,
        task_unit_max_chars: int,
    ) -> list[TaskUnit]:
        """Run skeleton LLM flow; fallback to heuristic resolver safely."""
        try:
            split_plan = self.build_split_plan(
                section=section,
                task_unit_min_chars=task_unit_min_chars,
                task_unit_max_chars=task_unit_max_chars,
            )
            units = self.apply_split_plan(
                section=section,
                section_index=section_index,
                split_plan=split_plan,
                task_unit_min_chars=task_unit_min_chars,
                task_unit_max_chars=task_unit_max_chars,
            )
            if units:
                return units
        except Exception as error:
            print(
                "LLMTaskUnitSplitResolver#fallback_heuristic:",
                f"reason={error}",
            )

        # Skeleton stage: keep production path stable via heuristic fallback.
        return self.heuristic_fallback_resolver.split_section(
            section=section,
            section_index=section_index,
            task_unit_min_chars=task_unit_min_chars,
            task_unit_max_chars=task_unit_max_chars,
        )

    def build_split_plan(
        self,
        *,
        section: StructuredSection,
        task_unit_min_chars: int,
        task_unit_max_chars: int,
    ) -> TaskUnitSplitPlan:
        """Build split plan from LLM response (skeleton returns empty plan)."""
        _ = section
        _ = task_unit_min_chars
        _ = task_unit_max_chars
        if self.llm_provider is None:
            return TaskUnitSplitPlan(metadata={"fallback_reason": "llm_provider_unavailable"})

        # Future extension:
        # 1) prompt model for boundary instructions
        # 2) parse JSON plan
        # 3) validate instruction quality
        return TaskUnitSplitPlan(metadata={"fallback_reason": "llm_split_plan_not_implemented"})

    def apply_split_plan(
        self,
        *,
        section: StructuredSection,
        section_index: int,
        split_plan: TaskUnitSplitPlan,
        task_unit_min_chars: int,
        task_unit_max_chars: int,
    ) -> list[TaskUnit]:
        """Apply split plan locally (skeleton defers to heuristic fallback)."""
        _ = split_plan
        return self.heuristic_fallback_resolver.split_section(
            section=section,
            section_index=section_index,
            task_unit_min_chars=task_unit_min_chars,
            task_unit_max_chars=task_unit_max_chars,
        )
