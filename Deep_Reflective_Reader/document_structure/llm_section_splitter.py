import json
import re

from config.app_DI_config import (
    LLMSectionPreviewPolicyConfig,
    PromptTextNormalizationConfig,
)
from document_structure.abstract_section_splitter import AbstractSectionSplitter
from document_structure.section_split_plan import (
    AnchorMatchMode,
    SectionParserMode,
    SectionSplitPlan,
    SplitBoundaryInstruction,
)
from document_structure.section_role import SectionRole
from document_structure.section_splitter import CommonSectionSplitter
from document_structure.section_splitter_dto import LineInfo
from document_structure.structured_document import StructuredSection
from document_structure.text_normalization import normalize_ocr_text
from language.language_code import LanguageCode
from llm.llm_provider import LLMProvider

_JSON_BLOCK_PATTERN = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


class LLMSectionSplitter(AbstractSectionSplitter):
    """Optional high-cost splitter: LLM plans boundaries, local code applies them."""

    def __init__(
        self,
        llm_provider: LLMProvider | None = None,
        common_splitter: CommonSectionSplitter | None = None,
        prompt_text_normalization: PromptTextNormalizationConfig | None = None,
        preview_policy: LLMSectionPreviewPolicyConfig | None = None,
    ):
        self.llm_provider = llm_provider
        self.common_splitter = common_splitter or CommonSectionSplitter()
        self.prompt_text_normalization = (
            prompt_text_normalization
            if prompt_text_normalization is not None
            else PromptTextNormalizationConfig(
                default_excerpt_chars=16_000,
                min_excerpt_chars=2_000,
                max_excerpt_chars_hard_cap=400_000,
                chars_per_token=4,
            )
        )
        self.preview_policy = (
            preview_policy
            if preview_policy is not None
            else LLMSectionPreviewPolicyConfig(
                prompt_overhead_tokens=1_200,
                excerpt_token_utilization_ratio=0.65,
                preview_head_ratio=0.65,
                preview_omitted_separator="\n\n[... middle omitted ...]\n\n",
            )
        )

    def split(
        self,
        raw_text: str,
        language: LanguageCode = LanguageCode.UNKNOWN,
    ) -> list[StructuredSection]:
        """Split by LLM-generated plan, with safe fallback to common splitter."""
        sections, _provenance = self.split_with_provenance(
            raw_text=raw_text,
            language=language,
        )
        return sections

    def split_with_provenance(
        self,
        raw_text: str,
        language: LanguageCode = LanguageCode.UNKNOWN,
    ) -> tuple[list[StructuredSection], dict[str, object]]:
        """Split by LLM plan and return advisory parser provenance."""
        try:
            plan = self.build_split_plan(raw_text=raw_text, language=language)
            return self.apply_split_plan_with_provenance(
                raw_text=raw_text,
                language=language,
                split_plan=plan,
            )
        except Exception as error:
            print(
                "LLMSectionSplitter#split_fallback_common:",
                f"reason={error}",
            )
            return self._fallback_common_with_provenance(
                raw_text=raw_text,
                language=language,
                reason=f"exception:{type(error).__name__}",
            )

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
        if plan.sections:
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
        sections, _provenance = self.apply_split_plan_with_provenance(
            raw_text=raw_text,
            language=language,
            split_plan=split_plan,
        )
        return sections

    def apply_split_plan_with_provenance(
        self,
        *,
        raw_text: str,
        language: LanguageCode,
        split_plan: SectionSplitPlan,
    ) -> tuple[list[StructuredSection], dict[str, object]]:
        """Apply LLM split plan and return advisory fallback provenance."""
        if language == LanguageCode.UNKNOWN:
            raise ValueError("bad_request: unsupported document structure language 'unknown'")
        if not raw_text:
            return self._fallback_common_with_provenance(
                raw_text=raw_text,
                language=language,
                reason="empty_raw_text",
            )
        if not split_plan.sections:
            fallback_reason = (
                str(split_plan.metadata.get("fallback_reason"))
                if isinstance(split_plan.metadata, dict)
                and split_plan.metadata.get("fallback_reason") is not None
                else "empty_split_instructions"
            )
            return self._fallback_common_with_provenance(
                raw_text=raw_text,
                language=language,
                reason=fallback_reason,
            )

        line_infos = self.common_splitter._build_line_infos(raw_text)
        if not line_infos:
            return self._fallback_common_with_provenance(
                raw_text=raw_text,
                language=language,
                reason="empty_line_infos",
            )
        common_sections = self._build_common_baseline_sections(
            raw_text=raw_text,
            language=language,
        )

        boundary_items: list[tuple[int, str, int, SectionRole | None, str | None]] = []
        for instruction in split_plan.sections:
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
                    max(1, int(instruction.level)),
                    instruction.section_role,
                    self._normalize_optional_text(instruction.container_title),
                )
            )

        if not boundary_items:
            return self._fallback_common_with_provenance(
                raw_text=raw_text,
                language=language,
                reason="no_resolved_boundaries",
            )

        ordered_boundaries: list[tuple[int, str, int, SectionRole | None, str | None]] = []
        seen_starts: set[int] = set()
        for start, title, level, section_role, container_title in sorted(
            boundary_items,
            key=lambda item: item[0],
        ):
            if start < 0 or start >= len(raw_text):
                continue
            if start in seen_starts:
                continue
            seen_starts.add(start)
            ordered_boundaries.append((start, title, level, section_role, container_title))

        if len(ordered_boundaries) < 2:
            ordered_boundaries = self._augment_boundaries_with_common_split(
                raw_text=raw_text,
                language=language,
                resolved_boundaries=ordered_boundaries,
                common_sections=common_sections,
            )
        if len(ordered_boundaries) < 2:
            return self._fallback_common_with_provenance(
                raw_text=raw_text,
                language=language,
                reason="resolved_boundaries_too_few_after_augmentation",
            )
        if not self._is_strictly_increasing_boundaries(ordered_boundaries):
            return self._fallback_common_with_provenance(
                raw_text=raw_text,
                language=language,
                reason="boundaries_not_strictly_increasing",
            )

        ordered_boundaries = self._merge_boundaries_with_common_baseline(
            resolved_boundaries=ordered_boundaries,
            common_sections=common_sections,
        )

        sections = self._build_sections_from_boundaries(
            raw_text=raw_text,
            boundaries=ordered_boundaries,
        )
        if not self._is_sections_output_reasonable(raw_text=raw_text, sections=sections):
            return self._fallback_common_with_provenance(
                raw_text=raw_text,
                language=language,
                reason="abnormal_section_output",
            )
        return sections, self._build_parse_provenance(
            requested_parser_mode=SectionParserMode.LLM_ENHANCED.value,
            effective_parser_mode=SectionParserMode.LLM_ENHANCED.value,
            fallback_used=False,
            fallback_reason=None,
        )

    def _fallback_common_with_provenance(
        self,
        *,
        raw_text: str,
        language: LanguageCode,
        reason: str,
    ) -> tuple[list[StructuredSection], dict[str, object]]:
        print(
            "LLMSectionSplitter#apply_fallback_common:",
            f"reason={reason}",
        )
        sections = self.common_splitter.split(raw_text=raw_text, language=language)
        return sections, self._build_parse_provenance(
            requested_parser_mode=SectionParserMode.LLM_ENHANCED.value,
            effective_parser_mode=SectionParserMode.COMMON.value,
            fallback_used=True,
            fallback_reason=reason,
        )

    @staticmethod
    def _build_parse_provenance(
        *,
        requested_parser_mode: str,
        effective_parser_mode: str,
        fallback_used: bool,
        fallback_reason: str | None,
    ) -> dict[str, object]:
        return {
            "requested_parser_mode": requested_parser_mode,
            "effective_parser_mode": effective_parser_mode,
            "fallback_used": fallback_used,
            "fallback_reason": fallback_reason,
            "source": "llm_section_splitter",
        }

    def _build_split_plan_prompt(self, *, raw_text: str, language: LanguageCode) -> str:
        """Build strict JSON-only prompt for region-aware structure planning."""
        preview_budget_chars = self._compute_preview_char_budget(raw_text_length=len(raw_text))
        preview = self._build_preview_text(raw_text=raw_text, preview_budget_chars=preview_budget_chars)
        return (
            "You are a high-cost fallback document structure restorer. "
            "Return ONLY JSON object with this schema:\n"
            "{\n"
            '  "parser_mode": "llm_enhanced",\n'
            '  "sections": [\n'
            "    {\n"
            '      "title": string|null,\n'
            '      "level": integer,\n'
            '      "section_role": "toc"|"front_matter"|"main_body"|"appendix"|"back_matter",\n'
            '      "container_title": string|null,\n'
            '      "start_anchor_text": string,\n'
            '      "anchor_match_mode": "exact"|"contains"|"regex",\n'
            '      "anchor_occurrence": integer\n'
            "    }\n"
            "  ]\n"
            "}\n\n"
            "Rules:\n"
            "1) Do not rewrite content and do not output section content text blocks.\n"
            "2) Do not return char offsets. Use executable start anchors only.\n"
            "3) Detect document regions first: toc, front_matter, main_body, appendix, back_matter.\n"
            "4) TOC headings must not be misclassified as main_body.\n"
            "5) front_matter/back_matter/appendix must stay out of main_body role.\n"
            "6) Keep section count conservative; avoid over-segmentation.\n"
            "7) Use stable anchors likely to appear verbatim in raw text.\n"
            "8) If heading text appears in both TOC and body, use anchor_occurrence to target the intended one.\n"
            "9) Output valid JSON only, no markdown or extra commentary.\n\n"
            f"Document language code: {language.value}\n"
            "Raw text preview (head+tail when truncated):\n"
            f"{preview}"
        )

    def _compute_preview_char_budget(self, *, raw_text_length: int) -> int:
        """Compute preview-char budget via provider-level prompt text normalization."""
        if raw_text_length <= 0:
            return 0
        sample_text = "x" * raw_text_length
        excerpt = self._normalize_preview_excerpt(sample_text)
        return max(1, len(excerpt))

    def _normalize_preview_excerpt(self, raw_text: str) -> str:
        if not raw_text:
            return ""
        if self.llm_provider is None:
            fallback_chars = min(
                len(raw_text),
                self.prompt_text_normalization.default_excerpt_chars,
            )
            return raw_text[:fallback_chars]
        return self.llm_provider.normalize_input_text_for_prompt(
            text=raw_text,
            prompt_overhead_tokens=self.preview_policy.prompt_overhead_tokens,
            token_utilization_ratio=self.preview_policy.excerpt_token_utilization_ratio,
            default_excerpt_chars=self.prompt_text_normalization.default_excerpt_chars,
            min_excerpt_chars=self.prompt_text_normalization.min_excerpt_chars,
            max_excerpt_chars_hard_cap=self.prompt_text_normalization.max_excerpt_chars_hard_cap,
            chars_per_token=self.prompt_text_normalization.chars_per_token,
        )

    def _build_preview_text(
        self,
        *,
        raw_text: str,
        preview_budget_chars: int,
    ) -> str:
        """Build preview using head+tail slices to improve rear-region visibility."""
        if preview_budget_chars <= 0:
            return ""
        if len(raw_text) <= preview_budget_chars:
            return raw_text

        separator = self.preview_policy.preview_omitted_separator
        if preview_budget_chars <= len(separator) + 20:
            return raw_text[:preview_budget_chars]

        content_budget = preview_budget_chars - len(separator)
        head_chars = max(20, int(content_budget * self.preview_policy.preview_head_ratio))
        tail_chars = max(20, content_budget - head_chars)
        if head_chars + tail_chars > content_budget:
            tail_chars = max(20, content_budget - head_chars)
        return f"{raw_text[:head_chars]}{separator}{raw_text[-tail_chars:]}"

    def _augment_boundaries_with_common_split(
        self,
        *,
        raw_text: str,
        language: LanguageCode,
        resolved_boundaries: list[tuple[int, str, int, SectionRole | None, str | None]],
        common_sections: list[StructuredSection] | None = None,
    ) -> list[tuple[int, str, int, SectionRole | None, str | None]]:
        """Augment partial LLM boundaries with conservative common-split boundaries."""
        common_sections = (
            common_sections
            if common_sections is not None
            else self._build_common_baseline_sections(raw_text=raw_text, language=language)
        )

        merged_by_start: dict[int, tuple[int, str, int, SectionRole | None, str | None]] = {
            item[0]: item for item in resolved_boundaries
        }
        for section in common_sections:
            title = (section.title or "").strip()
            if not title:
                continue
            start = int(section.char_start)
            if start < 0 or start >= len(raw_text):
                continue
            if start in merged_by_start:
                continue
            merged_by_start[start] = (
                start,
                title,
                max(1, int(section.level)),
                SectionRole.resolve(section.section_role),
                self._normalize_optional_text(section.container_title),
            )

        augmented = sorted(merged_by_start.values(), key=lambda item: item[0])
        print(
            "LLMSectionSplitter#boundary_augmentation:",
            f"resolved={len(resolved_boundaries)} augmented={len(augmented)}",
        )
        return augmented

    def _merge_boundaries_with_common_baseline(
        self,
        *,
        resolved_boundaries: list[tuple[int, str, int, SectionRole | None, str | None]],
        common_sections: list[StructuredSection],
    ) -> list[tuple[int, str, int, SectionRole | None, str | None]]:
        """Use common split as boundary baseline while preserving same-anchor LLM metadata."""
        common_boundaries = self._build_common_boundary_items(common_sections)
        if len(common_boundaries) < 2:
            return resolved_boundaries

        resolved_by_start = {item[0]: item for item in resolved_boundaries}
        merged: list[tuple[int, str, int, SectionRole | None, str | None]] = []
        overlay_count = 0
        for common_item in common_boundaries:
            resolved_item = resolved_by_start.get(common_item[0])
            if resolved_item is None:
                merged.append(common_item)
                continue
            merged.append(resolved_item)
            overlay_count += 1

        common_starts = {item[0] for item in common_boundaries}
        llm_special_only = [
            item
            for item in resolved_boundaries
            if item[0] not in common_starts
            and SectionRole.resolve(item[3]) in {
                SectionRole.TOC,
                SectionRole.FRONT_MATTER,
                SectionRole.APPENDIX,
                SectionRole.BACK_MATTER,
            }
        ]
        if llm_special_only:
            merged.extend(llm_special_only)
            merged.sort(key=lambda item: item[0])

        if len(merged) > len(resolved_boundaries) or overlay_count < len(resolved_boundaries):
            print(
                "LLMSectionSplitter#hybrid_common_baseline:",
                f"resolved={len(resolved_boundaries)} common={len(common_boundaries)} "
                f"merged={len(merged)} overlays={overlay_count}",
            )
        return merged

    def _build_common_baseline_sections(
        self,
        *,
        raw_text: str,
        language: LanguageCode,
    ) -> list[StructuredSection]:
        try:
            return self.common_splitter.split(raw_text=raw_text, language=language)
        except Exception as error:
            print(
                "LLMSectionSplitter#common_baseline_skipped:",
                f"reason=common_split_failed error={error}",
            )
            return []

    def _build_common_boundary_items(
        self,
        common_sections: list[StructuredSection],
    ) -> list[tuple[int, str, int, SectionRole | None, str | None]]:
        boundaries: list[tuple[int, str, int, SectionRole | None, str | None]] = []
        seen_starts: set[int] = set()
        for section in common_sections:
            title = (section.title or "").strip()
            if not title:
                continue
            start = int(section.char_start)
            if start < 0 or start in seen_starts:
                continue
            seen_starts.add(start)
            boundaries.append(
                (
                    start,
                    title,
                    max(1, int(section.level)),
                    SectionRole.resolve(section.section_role),
                    self._normalize_optional_text(section.container_title),
                )
            )
        return sorted(boundaries, key=lambda item: item[0])

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
        block_match = _JSON_BLOCK_PATTERN.search(stripped)
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
        boundaries: list[tuple[int, str, int, SectionRole | None, str | None]],
    ) -> list[StructuredSection]:
        """Build contiguous sections from resolved boundary starts."""
        if not boundaries:
            return [self.common_splitter._single_fallback_section(raw_text)]

        sections: list[StructuredSection] = []
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

        for idx, (char_start, title, level, section_role, container_title) in enumerate(boundaries):
            if idx + 1 < len(boundaries):
                char_end = boundaries[idx + 1][0]
            else:
                char_end = len(raw_text)
            if char_end <= char_start:
                continue

            resolved_role = section_role or SectionRole.MAIN_BODY
            sections.append(
                self.common_splitter._build_section(
                    section_index=len(sections),
                    title=title,
                    raw_text=raw_text,
                    char_start=char_start,
                    char_end=char_end,
                    container_title=container_title,
                    section_role=resolved_role,
                    level=max(1, int(level)),
                )
            )

        if not sections:
            return [self.common_splitter._single_fallback_section(raw_text)]
        return sections

    @staticmethod
    def _is_strictly_increasing_boundaries(
        boundaries: list[tuple[int, str, int, SectionRole | None, str | None]],
    ) -> bool:
        """Validate boundaries are strictly increasing by start offset."""
        if not boundaries:
            return False
        previous = boundaries[0][0]
        for current, _, _, _, _ in boundaries[1:]:
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
        if not LLMSectionSplitter._has_reasonable_main_body_density(
            raw_text=raw_text,
            sections=sections,
        ):
            return False
        return True

    @staticmethod
    def _has_reasonable_main_body_density(
        *,
        raw_text: str,
        sections: list[StructuredSection],
    ) -> bool:
        """Reject LLM plans that resolved duplicated headings into TOC-only spans."""
        if len(raw_text.strip()) < 5_000:
            return True

        main_body_sections = [
            section
            for section in sections
            if SectionRole.resolve(section.section_role) == SectionRole.MAIN_BODY
        ]
        if len(main_body_sections) < 5:
            return True

        tiny_sections = [
            section
            for section in main_body_sections
            if len(section.content.strip()) < 120
        ]
        if len(tiny_sections) / len(main_body_sections) > 0.65:
            return False

        substantive_sections = [
            section
            for section in main_body_sections
            if LLMSectionSplitter._has_substantive_body_text(section)
        ]
        return len(substantive_sections) / len(main_body_sections) >= 0.35

    @staticmethod
    def _has_substantive_body_text(section: StructuredSection) -> bool:
        """Return whether a main-body section contains more than heading/list text."""
        lines = [line.strip() for line in section.content.splitlines() if line.strip()]
        if not lines:
            return False

        excluded = {
            LLMSectionSplitter._normalize_line(value)
            for value in (section.title, section.container_title)
            if value
        }
        body_lines = [
            line
            for line in lines
            if LLMSectionSplitter._normalize_line(line) not in excluded
        ]
        if not body_lines:
            return False

        body_text = " ".join(body_lines)
        if len(body_text) < 80:
            return False
        return len(re.findall(r"\w+", body_text, flags=re.UNICODE)) >= 10

    @staticmethod
    def _normalize_line(value: str) -> str:
        """Normalize line text for exact/contains anchor matching."""
        return normalize_ocr_text(value).lower()

    @staticmethod
    def _normalize_optional_text(value: str | None) -> str | None:
        """Normalize optional text to either None or non-empty string."""
        normalized = normalize_ocr_text(value or "")
        if not normalized:
            return None
        return normalized
