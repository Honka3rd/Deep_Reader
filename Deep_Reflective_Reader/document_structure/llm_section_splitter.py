import json
import re

from document_structure.abstract_section_splitter import AbstractSectionSplitter
from document_structure.section_split_plan import (
    AnchorMatchMode,
    SectionParserMode,
    SectionSplitPlan,
    SplitBoundaryInstruction,
)
from document_structure.section_splitter import CommonSectionSplitter
from document_structure.section_splitter_dto import LineInfo
from document_structure.structured_document import StructuredSection
from language.language_code import LanguageCode
from llm.llm_provider import LLMProvider


class LLMSectionSplitter(AbstractSectionSplitter):
    """Optional high-cost splitter: LLM plans boundaries, local code applies them."""

    _JSON_BLOCK_PATTERN = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
    _WHITESPACE_COLLAPSE_PATTERN = re.compile(r"\s+")

    def __init__(
        self,
        llm_provider: LLMProvider | None = None,
        common_splitter: CommonSectionSplitter | None = None,
    ):
        self.llm_provider = llm_provider
        self.common_splitter = common_splitter or CommonSectionSplitter()

    def split(
        self,
        raw_text: str,
        language: LanguageCode = LanguageCode.UNKNOWN,
    ) -> list[StructuredSection]:
        """Split by LLM-generated plan, with safe fallback to common splitter."""
        try:
            plan = self.build_split_plan(raw_text=raw_text, language=language)
            return self.apply_split_plan(
                raw_text=raw_text,
                language=language,
                split_plan=plan,
            )
        except Exception as error:
            print(
                "LLMSectionSplitter#split_fallback_common:",
                f"reason={error}",
            )
            return self.common_splitter.split(raw_text=raw_text, language=language)

    def build_split_plan(
        self,
        *,
        raw_text: str,
        language: LanguageCode,
    ) -> SectionSplitPlan:
        """Build one split plan from LLM response. Plan can be empty safely."""
        if language == LanguageCode.UNKNOWN:
            raise ValueError("bad_request: unsupported document structure language 'unknown'")
        if not raw_text.strip():
            return SectionSplitPlan(
                parser_mode=SectionParserMode.LLM_ENHANCED,
                instructions=[],
                metadata={"fallback_reason": "empty_raw_text"},
            )
        if self.llm_provider is None:
            return SectionSplitPlan(
                parser_mode=SectionParserMode.LLM_ENHANCED,
                instructions=[],
                metadata={"fallback_reason": "llm_provider_unavailable"},
            )

        prompt = self._build_split_plan_prompt(raw_text=raw_text, language=language)
        llm_response = self.llm_provider.complete_text(prompt)
        plan = self._parse_split_plan_response(llm_response)
        if plan.instructions:
            return plan
        return SectionSplitPlan(
            parser_mode=SectionParserMode.LLM_ENHANCED,
            instructions=[],
            metadata={"fallback_reason": "empty_or_invalid_llm_plan"},
        )

    def apply_split_plan(
        self,
        *,
        raw_text: str,
        language: LanguageCode,
        split_plan: SectionSplitPlan,
    ) -> list[StructuredSection]:
        """Apply LLM split plan locally; fallback to common splitter when unresolved."""
        def fallback_common(reason: str) -> list[StructuredSection]:
            print(
                "LLMSectionSplitter#apply_fallback_common:",
                f"reason={reason}",
            )
            return self.common_splitter.split(raw_text=raw_text, language=language)

        if language == LanguageCode.UNKNOWN:
            raise ValueError("bad_request: unsupported document structure language 'unknown'")
        if not raw_text:
            return [self.common_splitter._single_fallback_section(raw_text)]
        if not split_plan.instructions:
            return fallback_common("empty_split_instructions")

        line_infos = self.common_splitter._build_line_infos(raw_text)
        if not line_infos:
            return fallback_common("empty_line_infos")

        boundary_items: list[tuple[int, str, str | None]] = []
        for instruction in split_plan.instructions:
            anchor_start = self._resolve_anchor_start(
                line_infos=line_infos,
                instruction=instruction,
            )
            if anchor_start is None:
                continue
            resolved_title = (instruction.title or "").strip() or instruction.start_anchor_text.strip()
            boundary_items.append(
                (
                    anchor_start,
                    resolved_title,
                    self._normalize_optional_text(instruction.container_title),
                )
            )

        if not boundary_items:
            return fallback_common("no_resolved_boundaries")

        ordered_boundaries: list[tuple[int, str, str | None]] = []
        seen_starts: set[int] = set()
        for start, title, container_title in sorted(boundary_items, key=lambda item: item[0]):
            if start < 0 or start >= len(raw_text):
                continue
            if start in seen_starts:
                continue
            seen_starts.add(start)
            ordered_boundaries.append((start, title, container_title))

        if len(ordered_boundaries) < 2:
            return fallback_common("resolved_boundaries_too_few")
        if not self._is_strictly_increasing_boundaries(ordered_boundaries):
            return fallback_common("boundaries_not_strictly_increasing")

        sections = self._build_sections_from_boundaries(
            raw_text=raw_text,
            boundaries=ordered_boundaries,
        )
        if not self._is_sections_output_reasonable(raw_text=raw_text, sections=sections):
            return fallback_common("abnormal_section_output")
        return sections

    def _build_split_plan_prompt(self, *, raw_text: str, language: LanguageCode) -> str:
        """Build strict JSON-only prompt for boundary planning."""
        preview = raw_text[:16_000]
        return (
            "You are a document structure planner. "
            "Return ONLY JSON object with this schema:\n"
            "{\n"
            '  "parser_mode": "llm_enhanced",\n'
            '  "instructions": [\n'
            "    {\n"
            '      "title": string|null,\n'
            '      "level": integer,\n'
            '      "container_title": string|null,\n'
            '      "start_anchor_text": string,\n'
            '      "anchor_match_mode": "exact"|"contains"|"regex",\n'
            '      "anchor_occurrence": integer\n'
            "    }\n"
            "  ]\n"
            "}\n\n"
            "Rules:\n"
            "1) Do not rewrite content.\n"
            "2) Do not return char offsets.\n"
            "3) Choose stable anchors likely to exist verbatim in raw text.\n"
            "4) Keep instructions concise and high-confidence.\n\n"
            f"Document language code: {language.value}\n"
            "Raw text preview:\n"
            f"{preview}"
        )

    def _parse_split_plan_response(self, response_text: str) -> SectionSplitPlan:
        """Parse LLM response to SectionSplitPlan with tolerant JSON extraction."""
        payload = self._extract_json_payload(response_text)
        try:
            decoded = json.loads(payload)
        except json.JSONDecodeError:
            return SectionSplitPlan(
                parser_mode=SectionParserMode.LLM_ENHANCED,
                instructions=[],
                metadata={"parse_error": "json_decode_error"},
            )
        if not isinstance(decoded, dict):
            return SectionSplitPlan(
                parser_mode=SectionParserMode.LLM_ENHANCED,
                instructions=[],
                metadata={"parse_error": "non_object_payload"},
            )
        try:
            return SectionSplitPlan.from_dict(decoded)
        except Exception as error:
            return SectionSplitPlan(
                parser_mode=SectionParserMode.LLM_ENHANCED,
                instructions=[],
                metadata={"parse_error": str(error)},
            )

    def _extract_json_payload(self, response_text: str) -> str:
        """Extract JSON payload from raw response, tolerant to fenced output."""
        stripped = response_text.strip()
        block_match = self._JSON_BLOCK_PATTERN.search(stripped)
        if block_match:
            return block_match.group(1).strip()
        return stripped

    def _resolve_anchor_start(
        self,
        *,
        line_infos: list[LineInfo],
        instruction: SplitBoundaryInstruction,
    ) -> int | None:
        """Resolve one instruction to absolute start offset using local line scan."""
        anchor = instruction.start_anchor_text.strip()
        if not anchor:
            return None

        matched_offsets: list[int] = []
        for info in line_infos:
            stripped = info.stripped
            if not stripped:
                continue
            if self._anchor_matches(
                line_text=stripped,
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
        match_mode: AnchorMatchMode,
    ) -> bool:
        """Check one line against anchor matching strategy."""
        if match_mode == AnchorMatchMode.REGEX:
            try:
                return re.search(anchor_text, line_text) is not None
            except re.error:
                return False

        normalized_line = self._normalize_line(line_text)
        normalized_anchor = self._normalize_line(anchor_text)
        if not normalized_anchor:
            return False

        if match_mode == AnchorMatchMode.CONTAINS:
            return normalized_anchor in normalized_line

        return normalized_line == normalized_anchor

    def _build_sections_from_boundaries(
        self,
        *,
        raw_text: str,
        boundaries: list[tuple[int, str, str | None]],
    ) -> list[StructuredSection]:
        """Build contiguous sections from resolved boundary starts."""
        if not boundaries:
            return [self.common_splitter._single_fallback_section(raw_text)]

        sections: list[StructuredSection] = []
        current_container_title: str | None = None
        first_start = boundaries[0][0]
        if first_start > 0:
            sections.append(
                self.common_splitter._build_section(
                    section_index=0,
                    title=None,
                    raw_text=raw_text,
                    char_start=0,
                    char_end=first_start,
                )
            )

        for idx, (char_start, title, container_title) in enumerate(boundaries):
            if container_title is not None:
                current_container_title = container_title
            if idx + 1 < len(boundaries):
                char_end = boundaries[idx + 1][0]
            else:
                char_end = len(raw_text)
            if char_end <= char_start:
                continue

            sections.append(
                self.common_splitter._build_section(
                    section_index=len(sections),
                    title=title,
                    raw_text=raw_text,
                    char_start=char_start,
                    char_end=char_end,
                    container_title=current_container_title,
                )
            )

        if not sections:
            return [self.common_splitter._single_fallback_section(raw_text)]
        return sections

    @staticmethod
    def _is_strictly_increasing_boundaries(
        boundaries: list[tuple[int, str, str | None]],
    ) -> bool:
        """Validate boundaries are strictly increasing by start offset."""
        if not boundaries:
            return False
        previous = boundaries[0][0]
        for current, _, _ in boundaries[1:]:
            if current <= previous:
                return False
            previous = current
        return True

    @staticmethod
    def _is_sections_output_reasonable(
        *,
        raw_text: str,
        sections: list[StructuredSection],
    ) -> bool:
        """Validate section output shape before accepting enhanced parse result."""
        if len(sections) < 2:
            return False

        text_length = len(raw_text)
        last_end = 0
        non_empty_sections = 0
        for section in sections:
            if section.char_start < 0 or section.char_end > text_length:
                return False
            if section.char_end <= section.char_start:
                return False
            if section.char_start < last_end:
                return False
            last_end = section.char_end
            if section.content.strip():
                non_empty_sections += 1

        if non_empty_sections < 2:
            return False
        if sections[-1].char_end != text_length:
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
