from __future__ import annotations

import re
from typing import Any

from document_structure.document_structure_language_registry import (
    DocumentStructureLanguageRegistry,
)
from language.language_code import LanguageCode, LanguageCodeResolver
from llm.llm_provider import LLMProvider


class DocumentProfileEvidenceBuilder:
    """Build compact classification evidence; not parser boundaries or split rules."""

    def __init__(
        self,
        *,
        structure_language_registry: DocumentStructureLanguageRegistry,
    ):
        self.structure_language_registry = structure_language_registry

    def build(
        self,
        *,
        text: str,
        document_language: LanguageCode | str,
        profile_prompt_policy: Any,
        llm_provider: LLMProvider,
        prompt_text_normalization: Any,
    ) -> dict[str, Any]:
        normalized_text = text or ""
        resolved_language = self._resolve_language(document_language)

        head_chars = max(200, int(profile_prompt_policy.evidence_head_chars))
        tail_chars = max(200, int(profile_prompt_policy.evidence_tail_chars))
        middle_chars = max(200, int(profile_prompt_policy.evidence_middle_chars))
        line_limit = max(1, int(profile_prompt_policy.evidence_heading_lines_limit))

        excerpt_for_classification = llm_provider.normalize_input_text_for_prompt(
            text=normalized_text,
            prompt_overhead_tokens=profile_prompt_policy.structure_prompt_overhead_tokens,
            token_utilization_ratio=profile_prompt_policy.structure_excerpt_token_utilization_ratio,
            default_excerpt_chars=prompt_text_normalization.default_excerpt_chars,
            min_excerpt_chars=prompt_text_normalization.min_excerpt_chars,
            max_excerpt_chars_hard_cap=prompt_text_normalization.max_excerpt_chars_hard_cap,
            chars_per_token=prompt_text_normalization.chars_per_token,
        )
        head_excerpt = normalized_text[:head_chars]
        tail_excerpt = (
            normalized_text[-tail_chars:]
            if len(normalized_text) > tail_chars
            else normalized_text
        )
        middle_excerpt = ""
        if len(normalized_text) > middle_chars:
            mid_start = max(0, (len(normalized_text) // 2) - (middle_chars // 2))
            middle_excerpt = normalized_text[mid_start : mid_start + middle_chars]

        candidate_lines = self._extract_candidate_lines(
            text=normalized_text,
            language=resolved_language,
            limit=line_limit,
        )

        return {
            "document_language": resolved_language.value,
            "raw_text_length": len(normalized_text),
            "excerpt_for_classification": excerpt_for_classification,
            "head_excerpt": head_excerpt,
            "tail_excerpt": tail_excerpt,
            "middle_excerpt": middle_excerpt,
            # Sampled evidence only for profile classification, not parser boundaries.
            "candidate_lines": candidate_lines,
        }

    def _extract_candidate_lines(
        self,
        *,
        text: str,
        language: LanguageCode,
        limit: int,
    ) -> list[str]:
        patterns = self.structure_language_registry.get_profile_evidence_patterns(language)
        keywords = tuple(
            keyword.lower().strip()
            for keyword in self.structure_language_registry.get_profile_evidence_keywords(language)
            if keyword.strip()
        )

        candidates: list[str] = []
        seen: set[str] = set()

        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if len(stripped) > 90:
                continue

            lowered = stripped.lower()
            keyword_hit = self._matches_keyword_evidence_line(
                stripped=stripped,
                lowered=lowered,
                keywords=keywords,
            )
            pattern_hit = any(item.pattern.search(stripped) for item in patterns)
            if not (keyword_hit or pattern_hit):
                continue

            dedupe_key = stripped.lower()
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            candidates.append(stripped)
            if len(candidates) >= limit:
                break

        return candidates

    def _matches_keyword_evidence_line(
        self,
        *,
        stripped: str,
        lowered: str,
        keywords: tuple[str, ...],
    ) -> bool:
        # candidate_lines are sampled evidence for LLM profile classification only.
        # They are not authoritative headings, parser boundaries, or split rules.
        dedup_keywords = tuple(dict.fromkeys(keyword for keyword in keywords if keyword))
        for keyword in dedup_keywords:
            if self._is_latin_keyword(keyword):
                if self._matches_latin_keyword_line(
                    stripped=stripped,
                    lowered=lowered,
                    keyword=keyword.lower(),
                ):
                    return True
            elif self._matches_cjk_keyword_line(
                stripped=stripped,
                keyword=keyword,
            ):
                return True
        return False

    def _matches_latin_keyword_line(
        self,
        *,
        stripped: str,
        lowered: str,
        keyword: str,
    ) -> bool:
        if lowered == keyword:
            return True
        separators = (" ", ":", "：", "-", "–", "—", ".", "\t")
        if any(lowered.startswith(f"{keyword}{separator}") for separator in separators):
            return True
        if self._has_numbered_prefix_before_keyword(
            stripped=stripped,
            keyword=keyword,
            case_insensitive=True,
        ):
            return True
        return False

    def _matches_cjk_keyword_line(
        self,
        *,
        stripped: str,
        keyword: str,
    ) -> bool:
        if stripped == keyword:
            return True
        separators = (":", "：", " ", "\t", "-", "–", "—", "（", "(", "【", "[", ".", "。")
        if any(stripped.startswith(f"{keyword}{separator}") for separator in separators):
            return True
        if self._has_numbered_prefix_before_keyword(
            stripped=stripped,
            keyword=keyword,
            case_insensitive=False,
        ):
            return True

        # Very short CJK keywords must be very conservative to avoid sequence-like false positives.
        if len(keyword) <= 1:
            return False
        return False

    def _has_numbered_prefix_before_keyword(
        self,
        *,
        stripped: str,
        keyword: str,
        case_insensitive: bool,
    ) -> bool:
        escaped_keyword = re.escape(keyword)
        prefix_pattern = (
            r"^[\s　]*[（(]?(?:\d+|[ivxlcdmIVXLCDM]+|[一二三四五六七八九十百千万〇零两兩]+)[）)]?"
            r"[、.．:：\-–—)\]]\s*"
            + escaped_keyword
            + r"(?:[\s:：\-–—\t（(【\[].*)?$"
        )
        flags = re.IGNORECASE if case_insensitive else 0
        return re.match(prefix_pattern, stripped, flags) is not None

    @staticmethod
    def _is_latin_keyword(keyword: str) -> bool:
        return any("a" <= char.lower() <= "z" for char in keyword)

    @staticmethod
    def _resolve_language(language: LanguageCode | str) -> LanguageCode:
        if isinstance(language, LanguageCode):
            return language
        return LanguageCodeResolver.resolve(language)
