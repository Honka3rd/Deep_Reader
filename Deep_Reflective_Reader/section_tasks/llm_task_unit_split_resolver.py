import json
import re
from dataclasses import dataclass

from document_structure.structured_document import StructuredSection
from llm.llm_provider import LLMProvider
from section_tasks.abstract_task_unit_split_resolver import AbstractTaskUnitSplitResolver
from section_tasks.heuristic_task_unit_split_resolver import HeuristicTaskUnitSplitResolver
from section_tasks.task_unit import TaskUnit
from section_tasks.task_unit_split_mode import TaskUnitSplitMode
from section_tasks.task_unit_split_plan import (
    TaskUnitBoundaryMatchMode,
    TaskUnitSplitBoundaryInstruction,
    TaskUnitSplitParserMode,
    TaskUnitSplitPlan,
)


@dataclass(frozen=True)
class _TaskUnitLineInfo:
    """One line snapshot with absolute offsets for anchor resolution."""

    stripped: str
    char_start: int
    char_end: int


class LLMTaskUnitSplitResolver(AbstractTaskUnitSplitResolver):
    """High-cost split resolver skeleton: LLM builds plan, local code applies it."""

    _JSON_BLOCK_PATTERN = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
    _WHITESPACE_COLLAPSE_PATTERN = re.compile(r"\s+")
    _MIN_ACCEPTABLE_OUTPUT_UNITS = 2
    _MAX_PLAN_INSTRUCTIONS = 32
    _MAX_PROMPT_SECTION_CHARS = 16_000

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
        """Run LLM split plan + local apply flow; fallback to heuristic resolver safely."""
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
                print(
                    "LLMTaskUnitSplitResolver#split_section:",
                    "path=llm_plan_apply",
                    f"section_id={section.section_id}",
                    f"unit_count={len(units)}",
                )
                return units
            raise ValueError("llm_apply_returned_empty_units")
        except Exception as error:
            print(
                "LLMTaskUnitSplitResolver#fallback_heuristic:",
                f"reason={error}",
                f"section_id={section.section_id}",
            )

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
        """Build split plan from LLM response with strict JSON-only contract."""
        section_text = (section.content or "").strip()
        if not section_text:
            raise ValueError("empty_section_content")

        if self.llm_provider is None:
            raise ValueError("llm_provider_unavailable")

        prompt = self._build_split_plan_prompt(
            section=section,
            task_unit_min_chars=task_unit_min_chars,
            task_unit_max_chars=task_unit_max_chars,
        )
        llm_response = self.llm_provider.complete_text(prompt)
        split_plan = self._parse_split_plan_response(llm_response)
        if not split_plan.instructions:
            raise ValueError("empty_or_invalid_llm_plan")

        if len(split_plan.instructions) > self._MAX_PLAN_INSTRUCTIONS:
            raise ValueError(
                f"too_many_plan_instructions:{len(split_plan.instructions)}"
            )
        return split_plan

    def apply_split_plan(
        self,
        *,
        section: StructuredSection,
        section_index: int,
        split_plan: TaskUnitSplitPlan,
        task_unit_min_chars: int,
        task_unit_max_chars: int,
    ) -> list[TaskUnit]:
        """Apply split plan locally with deterministic anchors and safety checks."""
        _ = task_unit_min_chars
        section_text = (section.content or "").strip()
        if not section_text:
            raise ValueError("empty_section_content")
        if not split_plan.instructions:
            raise ValueError("empty_split_instructions")

        line_infos = self._build_line_infos(section_text)
        if not line_infos:
            raise ValueError("empty_line_infos")

        boundary_items: list[tuple[int, str | None]] = []
        for instruction in split_plan.instructions:
            anchor_start = self._resolve_anchor_start(
                line_infos=line_infos,
                instruction=instruction,
            )
            if anchor_start is None:
                continue
            boundary_items.append((anchor_start, self._normalize_optional_text(instruction.title)))

        if not boundary_items:
            raise ValueError("no_resolved_boundaries")

        # Deduplicate boundaries while preserving earliest title.
        deduped_boundary_by_start: dict[int, str | None] = {}
        for start, title in sorted(boundary_items, key=lambda item: item[0]):
            if start <= 0 or start >= len(section_text):
                continue
            if start not in deduped_boundary_by_start:
                deduped_boundary_by_start[start] = title
        ordered_starts = sorted(deduped_boundary_by_start.keys())
        if not ordered_starts:
            raise ValueError("resolved_boundaries_too_few")
        if not self._is_strictly_increasing(ordered_starts):
            raise ValueError("boundaries_not_strictly_increasing")

        chunks: list[tuple[str | None, str]] = []
        chunk_starts: list[int] = [0] + ordered_starts
        chunk_ends: list[int] = ordered_starts + [len(section_text)]
        for idx, (char_start, char_end) in enumerate(zip(chunk_starts, chunk_ends)):
            if char_end <= char_start:
                continue
            chunk_text = section_text[char_start:char_end].strip()
            if not chunk_text:
                continue

            if idx == 0:
                proposed_title = self._normalize_optional_text(section.title)
            else:
                proposed_title = deduped_boundary_by_start.get(char_start)
            chunks.append((proposed_title, chunk_text))

        if len(chunks) < self._MIN_ACCEPTABLE_OUTPUT_UNITS:
            raise ValueError("resolved_chunks_too_few")
        if not self._looks_reasonable_units_output(
            section_text=section_text,
            chunks=chunks,
            task_unit_max_chars=task_unit_max_chars,
        ):
            raise ValueError("abnormal_units_output")

        units: list[TaskUnit] = []
        normalized_section_title = self._normalize_optional_text(section.title)
        normalized_container_title = self._normalize_optional_text(section.container_title)
        for unit_index, (title, chunk_text) in enumerate(chunks):
            resolved_title = title
            if not resolved_title and normalized_section_title:
                resolved_title = f"{normalized_section_title} (Part {unit_index + 1})"
            if not resolved_title:
                resolved_title = f"Task Unit {unit_index + 1}"

            units.append(
                TaskUnit(
                    unit_id=f"task-unit-{section_index}-{unit_index}",
                    title=resolved_title,
                    container_title=normalized_container_title,
                    content=chunk_text,
                    source_section_ids=[section.section_id],
                    is_fallback_generated=True,
                )
            )
        return units

    def _build_split_plan_prompt(
        self,
        *,
        section: StructuredSection,
        task_unit_min_chars: int,
        task_unit_max_chars: int,
    ) -> str:
        """Build strict JSON-only prompt for single-section split planning."""
        section_text = (section.content or "").strip()
        preview_text = section_text[:self._MAX_PROMPT_SECTION_CHARS]
        section_title = self._normalize_optional_text(section.title) or "Untitled Section"
        return (
            "You are a single-section task-unit split planner.\n"
            "Return ONLY a JSON object with this exact schema:\n"
            "{\n"
            '  "parser_mode": "llm_enhanced",\n'
            '  "instructions": [\n'
            "    {\n"
            '      "title": string|null,\n'
            '      "start_anchor_text": string,\n'
            '      "anchor_match_mode": "exact"|"contains"|"regex",\n'
            '      "anchor_occurrence": integer\n'
            "    }\n"
            "  ]\n"
            "}\n\n"
            "Rules:\n"
            "1) Split ONLY inside this one section.\n"
            "2) Do NOT output rewritten content.\n"
            "3) Do NOT output char offsets.\n"
            "4) Use stable anchor texts that appear in the section.\n"
            "5) Keep instruction count compact and practical.\n\n"
            f"section_title: {section_title}\n"
            f"target_min_chars_per_unit: {max(1, int(task_unit_min_chars))}\n"
            f"target_max_chars_per_unit: {max(1, int(task_unit_max_chars))}\n"
            "section_text_preview:\n"
            f"{preview_text}"
        )

    def _parse_split_plan_response(self, response_text: str) -> TaskUnitSplitPlan:
        """Parse LLM response into task-unit split plan with tolerant JSON extraction."""
        payload = self._extract_json_payload(response_text)
        try:
            decoded = json.loads(payload)
        except json.JSONDecodeError:
            raise ValueError("llm_plan_json_decode_error")
        if not isinstance(decoded, dict):
            raise ValueError("llm_plan_payload_non_object")
        try:
            split_plan = TaskUnitSplitPlan.from_dict(decoded)
        except Exception as error:
            raise ValueError(f"llm_plan_schema_error:{error}") from error

        if split_plan.parser_mode != TaskUnitSplitParserMode.LLM_ENHANCED:
            split_plan = TaskUnitSplitPlan(
                parser_mode=TaskUnitSplitParserMode.LLM_ENHANCED,
                instructions=split_plan.instructions,
                metadata=split_plan.metadata,
            )
        return split_plan

    def _extract_json_payload(self, response_text: str) -> str:
        """Extract JSON payload from raw response, tolerant to fenced output."""
        stripped = (response_text or "").strip()
        if not stripped:
            raise ValueError("llm_plan_empty_response")

        block_match = self._JSON_BLOCK_PATTERN.search(stripped)
        if block_match:
            return block_match.group(1).strip()
        return stripped

    def _resolve_anchor_start(
        self,
        *,
        line_infos: list[_TaskUnitLineInfo],
        instruction: TaskUnitSplitBoundaryInstruction,
    ) -> int | None:
        """Resolve one instruction to absolute char offset via local line scan."""
        anchor = instruction.start_anchor_text.strip()
        if not anchor:
            return None

        matched_offsets: list[int] = []
        for info in line_infos:
            if not info.stripped:
                continue
            if self._anchor_matches(
                line_text=info.stripped,
                anchor_text=anchor,
                match_mode=instruction.anchor_match_mode,
            ):
                matched_offsets.append(info.char_start)

        if not matched_offsets:
            return None
        occurrence = max(1, int(instruction.anchor_occurrence))
        occurrence_index = min(len(matched_offsets), occurrence) - 1
        return matched_offsets[occurrence_index]

    def _anchor_matches(
        self,
        *,
        line_text: str,
        anchor_text: str,
        match_mode: TaskUnitBoundaryMatchMode,
    ) -> bool:
        """Check one line against anchor matching strategy."""
        if match_mode == TaskUnitBoundaryMatchMode.REGEX:
            try:
                return re.search(anchor_text, line_text) is not None
            except re.error:
                return False

        normalized_line = self._normalize_line(line_text)
        normalized_anchor = self._normalize_line(anchor_text)
        if not normalized_anchor:
            return False

        if match_mode == TaskUnitBoundaryMatchMode.CONTAINS:
            return normalized_anchor in normalized_line
        return normalized_line == normalized_anchor

    def _build_line_infos(self, text: str) -> list[_TaskUnitLineInfo]:
        """Build line infos with absolute offsets for anchor resolution."""
        line_infos: list[_TaskUnitLineInfo] = []
        cursor = 0
        for line in text.splitlines(keepends=True):
            char_start = cursor
            char_end = cursor + len(line)
            cursor = char_end
            line_infos.append(
                _TaskUnitLineInfo(
                    stripped=line.strip(),
                    char_start=char_start,
                    char_end=char_end,
                )
            )

        if not line_infos:
            line_infos.append(
                _TaskUnitLineInfo(
                    stripped=text.strip(),
                    char_start=0,
                    char_end=len(text),
                )
            )
        return line_infos

    @staticmethod
    def _is_strictly_increasing(boundaries: list[int]) -> bool:
        """Validate boundaries are strictly increasing."""
        if not boundaries:
            return False
        previous = boundaries[0]
        for current in boundaries[1:]:
            if current <= previous:
                return False
            previous = current
        return True

    @staticmethod
    def _looks_reasonable_units_output(
        *,
        section_text: str,
        chunks: list[tuple[str | None, str]],
        task_unit_max_chars: int,
    ) -> bool:
        """Validate final chunks before accepting LLM split output."""
        if len(chunks) < LLMTaskUnitSplitResolver._MIN_ACCEPTABLE_OUTPUT_UNITS:
            return False
        total_chunk_chars = sum(len(chunk_text) for _, chunk_text in chunks)
        if total_chunk_chars < max(1, int(len(section_text) * 0.7)):
            return False
        oversized_chunks = [
            chunk_text
            for _, chunk_text in chunks
            if len(chunk_text) > max(task_unit_max_chars * 2, 2400)
        ]
        if oversized_chunks:
            return False
        return True

    @classmethod
    def _normalize_line(cls, value: str) -> str:
        """Normalize line text for exact/contains anchor matching."""
        return cls._WHITESPACE_COLLAPSE_PATTERN.sub(" ", value.strip().lower())

    @staticmethod
    def _normalize_optional_text(value: str | None) -> str | None:
        """Normalize optional text to either None or non-empty string."""
        normalized = (value or "").strip()
        if not normalized:
            return None
        return normalized
