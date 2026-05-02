import json
import re
from typing import Any

from llm.llm_provider import LLMProvider
from config.app_DI_config import (
    ProfilePromptPolicyConfig,
    PromptTextNormalizationConfig,
)
from profile.document_profile import DocumentProfile, DocumentStructureProfile


_FENCED_JSON_PATTERN = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_HEADING_LIKE_LINE_PATTERNS = [
    re.compile(r"^\s*chapter\s+[ivxlcdm0-9]+\b", re.IGNORECASE),
    re.compile(r"^\s*part\s+[ivxlcdm0-9]+\b", re.IGNORECASE),
    re.compile(r"^\s*第[一二三四五六七八九十百千万〇零两\d]+[章节回部卷篇]\b"),
    re.compile(r"^\s*[（(]?[一二三四五六七八九十\d]+[）)]?[、.．]\s*"),
    re.compile(r"^\s*(preface|foreword|contents|appendix|afterword|epilogue)\b", re.IGNORECASE),
    re.compile(r"^\s*(序|前言|目录|目錄|附录|附錄|后记|後記|跋)\s*$"),
]
_QUALITY_LABEL_ALLOWLIST = {
    "good",
    "usable_with_warnings",
    "needs_enhanced_parse",
    "needs_manual_review",
    "unknown",
}


class DocumentProfileBuilder:
    """Generate topic/summary profile for a document."""
    llm_provider: LLMProvider
    prompt_text_normalization: PromptTextNormalizationConfig
    profile_prompt_policy: ProfilePromptPolicyConfig

    def __init__(
        self,
        llm_provider: LLMProvider,
        prompt_text_normalization: PromptTextNormalizationConfig,
        profile_prompt_policy: ProfilePromptPolicyConfig,
    ):
        """Initialize object state and injected dependencies.

Args:
    llm_provider: Llm provider.
    prompt_text_normalization: Generic prompt-text normalization policy.
    profile_prompt_policy: Profile-specific prompt policy.
"""
        self.llm_provider = llm_provider
        self.prompt_text_normalization = prompt_text_normalization
        self.profile_prompt_policy = profile_prompt_policy

    def build(
        self,
        text: str,
        document_language: str,
    ) -> DocumentProfile:
        """Build document profile from raw text and detected language.

Args:
    text: Input text content.
    document_language: Primary document language code (e.g. en/zh).

Returns:
    Profile containing topic, summary, and document language."""
        structured_profile = self._build_structured_profile_payload(
            text=text,
            document_language=document_language,
        )
        if structured_profile is not None:
            return structured_profile

        topic: str = self._detect_topic(text, document_language) or "other"
        summary: str = self._generate_summary(text, document_language, topic) or ""

        return DocumentProfile(
            topic=topic,
            summary=summary,
            document_language=document_language,
        )

    def _build_structured_profile_payload(
        self,
        *,
        text: str,
        document_language: str,
    ) -> DocumentProfile | None:
        try:
            evidence = self._build_profile_evidence(
                text=text,
                document_language=document_language,
            )
            prompt = self._build_structured_profile_prompt(
                evidence=evidence,
            )
            raw_response = self.llm_provider.complete_text(prompt)
            payload = self._parse_json_object_payload(raw_response)
            if payload is None:
                return None
            return self._build_document_profile_from_payload(
                payload=payload,
                fallback_document_language=document_language,
            )
        except Exception as error:
            print(
                "DocumentProfileBuilder#structured_profile_fallback:",
                f"reason={error}",
            )
            return None

    def _detect_topic(self, text: str, document_language: str) -> str:
        """Internal helper for detect topic.

Args:
    text: Input text content.
    document_language: Primary document language code (e.g. en/zh).

Returns:
    Short English topic label inferred from document content."""
        excerpt = self._build_prompt_excerpt(
            text=text,
            prompt_overhead_tokens=self.profile_prompt_policy.topic_prompt_overhead_tokens,
            token_utilization_ratio=self.profile_prompt_policy.topic_excerpt_token_utilization_ratio,
        )
        prompt = f"""
You are a document classifier.

Determine the primary topic/category of the following document.

Possible examples include:
- literary fiction
- science fiction
- biography
- history
- philosophy
- financial report
- news
- technical documentation
- essay
- academic paper
- other

Return ONLY a short topic label in English.

Document language: {document_language}

Document excerpt:
{excerpt}
"""
        return self.llm_provider.complete_text(prompt).strip().lower()

    def _generate_summary(
        self,
        text: str,
        document_language: str,
        topic: str,
    ) -> str:
        """Internal helper for generate summary.

Args:
    text: Input text content.
    document_language: Primary document language code (e.g. en/zh).
    topic: Topic.

Returns:
    Prompt-conditioning summary text for downstream QA."""
        excerpt = self._build_prompt_excerpt(
            text=text,
            prompt_overhead_tokens=self.profile_prompt_policy.summary_prompt_overhead_tokens,
            token_utilization_ratio=self.profile_prompt_policy.summary_excerpt_token_utilization_ratio,
        )
        prompt = f"""
You are a document profiling assistant.

The document topic is: {topic}
The document language is: {document_language}

Generate a concise background summary for prompt conditioning.

Rules:
1. The summary must be useful for later question answering.
2. Keep it factual and high-level.
3. Do not include unnecessary details.
4. Prefer stable background information over narrow local details.
5. Keep it within 120-180 English words.
6. If the document is literary, emphasize main characters, relationships, and disambiguation notes.
7. If the document is a biography, emphasize subject identity, timeline, and major roles.
8. If the document is a financial report, emphasize company/entity, reporting period, business scope, and major financial themes.
9. Return only the summary text.

Document excerpt:
{excerpt}
"""
        return self.llm_provider.complete_text(prompt).strip()

    def _build_prompt_excerpt(
        self,
        *,
        text: str,
        prompt_overhead_tokens: int,
        token_utilization_ratio: float,
    ) -> str:
        """Build prompt excerpt with provider-level capability normalization."""
        return self.llm_provider.normalize_input_text_for_prompt(
            text=text,
            prompt_overhead_tokens=prompt_overhead_tokens,
            token_utilization_ratio=token_utilization_ratio,
            default_excerpt_chars=self.prompt_text_normalization.default_excerpt_chars,
            min_excerpt_chars=self.prompt_text_normalization.min_excerpt_chars,
            max_excerpt_chars_hard_cap=self.prompt_text_normalization.max_excerpt_chars_hard_cap,
            chars_per_token=self.prompt_text_normalization.chars_per_token,
        )

    def _build_profile_evidence(
        self,
        *,
        text: str,
        document_language: str,
    ) -> dict[str, Any]:
        normalized_text = text or ""
        head_chars = max(200, int(self.profile_prompt_policy.evidence_head_chars))
        tail_chars = max(200, int(self.profile_prompt_policy.evidence_tail_chars))
        middle_chars = max(200, int(self.profile_prompt_policy.evidence_middle_chars))
        heading_lines_limit = max(10, int(self.profile_prompt_policy.evidence_heading_lines_limit))

        excerpt_for_structure = self._build_prompt_excerpt(
            text=normalized_text,
            prompt_overhead_tokens=self.profile_prompt_policy.structure_prompt_overhead_tokens,
            token_utilization_ratio=self.profile_prompt_policy.structure_excerpt_token_utilization_ratio,
        )

        head_excerpt = normalized_text[:head_chars]
        tail_excerpt = normalized_text[-tail_chars:] if len(normalized_text) > tail_chars else normalized_text
        middle_excerpt = ""
        if len(normalized_text) > middle_chars:
            mid_start = max(0, (len(normalized_text) // 2) - (middle_chars // 2))
            middle_excerpt = normalized_text[mid_start : mid_start + middle_chars]

        heading_like_lines = self._extract_heading_like_lines(
            text=normalized_text,
            limit=heading_lines_limit,
        )

        return {
            "document_language": document_language,
            "raw_text_length": len(normalized_text),
            "excerpt_for_structure": excerpt_for_structure,
            "head_excerpt": head_excerpt,
            "tail_excerpt": tail_excerpt,
            "middle_excerpt": middle_excerpt,
            "candidate_heading_lines": heading_like_lines,
        }

    def _extract_heading_like_lines(
        self,
        *,
        text: str,
        limit: int,
    ) -> list[str]:
        lines = text.splitlines()
        candidates: list[str] = []
        seen: set[str] = set()
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            if len(stripped) > 90:
                continue

            is_upper_heading = (
                len(stripped) <= 80
                and stripped.upper() == stripped
                and any(char.isalpha() for char in stripped)
            )
            matches_pattern = any(pattern.search(stripped) for pattern in _HEADING_LIKE_LINE_PATTERNS)
            if not (is_upper_heading or matches_pattern):
                continue

            normalized_key = stripped.lower()
            if normalized_key in seen:
                continue
            seen.add(normalized_key)
            candidates.append(stripped)
            if len(candidates) >= limit:
                break
        return candidates

    def _build_structured_profile_prompt(
        self,
        *,
        evidence: dict[str, Any],
    ) -> str:
        evidence_json = json.dumps(evidence, ensure_ascii=False, indent=2)
        return (
            "You are a document profile and structure-hint extractor.\n"
            "Return ONLY one valid JSON object.\n"
            "Do not output markdown. Do not output prose explanation.\n"
            "Do not invent headings or examples that are not present in the evidence package.\n"
            "If confidence is below 0.5, prefer unknown/false decisions.\n"
            "suggested_regex is optional and may be null; do not assume it is authoritative.\n"
            "examples must be sourced from evidence package text.\n\n"
            "Required JSON schema:\n"
            "{\n"
            '  "topic": string,\n'
            '  "summary": string,\n'
            '  "document_language": string,\n'
            '  "structure_profile": {\n'
            '    "profile_version": "structure_profile_v1",\n'
            '    "generated_by": "llm_profile_builder",\n'
            '    "document_structure_type": string|null,\n'
            '    "confidence": number|null,\n'
            '    "heading_patterns": {\n'
            '      "chapter": {"exists": bool, "pattern_type": string|null, "description": string|null, "examples": string[], "confidence": number|null, "suggested_regex": string|null}|null,\n'
            '      "section": {"exists": bool, "pattern_type": string|null, "description": string|null, "examples": string[], "confidence": number|null, "suggested_regex": string|null}|null,\n'
            '      "front_matter": {"exists": bool, "pattern_type": string|null, "description": string|null, "examples": string[], "confidence": number|null, "suggested_regex": string|null}|null,\n'
            '      "appendix": {"exists": bool, "pattern_type": string|null, "description": string|null, "examples": string[], "confidence": number|null, "suggested_regex": string|null}|null,\n'
            '      "back_matter": {"exists": bool, "pattern_type": string|null, "description": string|null, "examples": string[], "confidence": number|null, "suggested_regex": string|null}|null\n'
            "    },\n"
            '    "special_regions": {\n'
            '      "toc": {"exists": bool, "count_estimate": number|null, "examples": string[], "confidence": number|null, "notes": string|null}|null,\n'
            '      "front_matter": {"exists": bool, "count_estimate": number|null, "examples": string[], "confidence": number|null, "notes": string|null}|null,\n'
            '      "appendix": {"exists": bool, "count_estimate": number|null, "examples": string[], "confidence": number|null, "notes": string|null}|null,\n'
            '      "back_matter": {"exists": bool, "count_estimate": number|null, "examples": string[], "confidence": number|null, "notes": string|null}|null\n'
            "    },\n"
            '    "quality_hints": {\n'
            '      "quality_label": "good"|"usable_with_warnings"|"needs_enhanced_parse"|"needs_manual_review"|"unknown"|null,\n'
            '      "likely_single_blob": bool,\n'
            '      "likely_over_fragmented": bool,\n'
            '      "likely_chapter_only": bool,\n'
            '      "likely_chapter_section": bool,\n'
            '      "likely_essay": bool,\n'
            '      "likely_ocr_noisy": bool,\n'
            '      "confidence": number|null\n'
            "    },\n"
            '    "recommended_strategy": {\n'
            '      "structured_parser_mode": string|null,\n'
            '      "task_unit_split_mode": string|null,\n'
            '      "semantic_top_k_candidates": number|null,\n'
            '      "needs_enhanced_parse": bool,\n'
            '      "needs_manual_review": bool,\n'
            '      "reason": string|null\n'
            "    }|null,\n"
            '    "risks": string[],\n'
            '    "evidence": string[]\n'
            "  }\n"
            "}\n\n"
            "Evidence package JSON:\n"
            f"{evidence_json}\n"
        )

    def _parse_json_object_payload(self, response_text: str) -> dict[str, Any] | None:
        stripped = (response_text or "").strip()
        if not stripped:
            return None
        block_match = _FENCED_JSON_PATTERN.search(stripped)
        if block_match:
            stripped = block_match.group(1).strip()
        try:
            decoded = json.loads(stripped)
        except json.JSONDecodeError:
            return None
        if not isinstance(decoded, dict):
            return None
        return decoded

    def _build_document_profile_from_payload(
        self,
        *,
        payload: dict[str, Any],
        fallback_document_language: str,
    ) -> DocumentProfile | None:
        topic = str(payload.get("topic") or "").strip()
        summary = str(payload.get("summary") or "").strip()
        document_language = str(
            payload.get("document_language") or fallback_document_language
        ).strip()
        if not topic or not summary or not document_language:
            return None

        structure_profile_payload = payload.get("structure_profile")
        if not isinstance(structure_profile_payload, dict):
            return None
        if not self._is_structure_profile_payload_valid(structure_profile_payload):
            return None
        structure_profile = self._safe_parse_structure_profile(
            structure_profile_payload
        )
        if structure_profile is None:
            return None

        return DocumentProfile(
            topic=topic,
            summary=summary,
            document_language=document_language,
            structure_profile=structure_profile,
        )

    def _safe_parse_structure_profile(
        self,
        payload: dict[str, Any],
    ) -> DocumentStructureProfile | None:
        try:
            normalized_payload = dict(payload)
            quality_hints_payload = normalized_payload.get("quality_hints")
            if isinstance(quality_hints_payload, dict):
                quality_label = quality_hints_payload.get("quality_label")
                if (
                    isinstance(quality_label, str)
                    and quality_label.strip()
                    and quality_label.strip() not in _QUALITY_LABEL_ALLOWLIST
                ):
                    quality_hints_payload["quality_label"] = "unknown"
            return DocumentStructureProfile.from_dict(normalized_payload)
        except Exception as error:
            print(
                "DocumentProfileBuilder#structure_profile_parse_fallback:",
                f"reason={error}",
            )
            return None

    def _is_structure_profile_payload_valid(
        self,
        payload: dict[str, Any],
    ) -> bool:
        confidence = payload.get("confidence")
        if confidence is not None:
            try:
                confidence_value = float(confidence)
            except (TypeError, ValueError):
                return False
            if confidence_value < 0.0 or confidence_value > 1.0:
                return False

        heading_patterns = payload.get("heading_patterns")
        if heading_patterns is not None:
            if not isinstance(heading_patterns, dict):
                return False
            for region_key in (
                "chapter",
                "section",
                "front_matter",
                "appendix",
                "back_matter",
            ):
                region_value = heading_patterns.get(region_key)
                if region_value is None:
                    continue
                if not isinstance(region_value, dict):
                    return False
                examples = region_value.get("examples")
                if examples is not None and not isinstance(examples, list):
                    return False

        special_regions = payload.get("special_regions")
        if special_regions is not None:
            if not isinstance(special_regions, dict):
                return False
            for region_key in ("toc", "front_matter", "appendix", "back_matter"):
                region_value = special_regions.get(region_key)
                if region_value is None:
                    continue
                if not isinstance(region_value, dict):
                    return False
                examples = region_value.get("examples")
                if examples is not None and not isinstance(examples, list):
                    return False

        risks = payload.get("risks")
        if risks is not None and not isinstance(risks, list):
            return False
        evidence = payload.get("evidence")
        if evidence is not None and not isinstance(evidence, list):
            return False
        return True
