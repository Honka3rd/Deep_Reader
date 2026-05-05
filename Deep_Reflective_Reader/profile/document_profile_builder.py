from __future__ import annotations

import json
import re
from typing import Any

from document_structure.document_structure_language_registry import (
    DocumentStructureLanguageRegistry,
)
from llm.llm_provider import LLMProvider
from language.language_code import LanguageCode, LanguageCodeResolver
from profile.document_profile import (
    DiscourseMode,
    DocumentProfile,
    DocumentStructureShape,
    HeadingStyle,
    LikelihoodLevel,
    ParserRelevantMetadata,
    TextForm,
)
from profile.parser_metadata_extractor import ParserMetadataExtractor
from profile.document_profile_evidence_builder import DocumentProfileEvidenceBuilder


_FENCED_JSON_PATTERN = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_TEXT_FORM_ALLOWLIST = {item.value for item in TextForm}
_DISCOURSE_MODE_ALLOWLIST = {item.value for item in DiscourseMode}
_DOCUMENT_STRUCTURE_SHAPE_ALLOWLIST = {item.value for item in DocumentStructureShape}
_LIKELY_HEADING_STYLE_ALLOWLIST = {item.value for item in HeadingStyle}
_RISK_ALLOWLIST = {item.value for item in LikelihoodLevel}


class DocumentProfileBuilder:
    """Generate document profile with deterministic parser metadata + light LLM classification."""

    llm_provider: LLMProvider
    prompt_text_normalization: Any
    profile_prompt_policy: Any
    metadata_extractor: ParserMetadataExtractor
    evidence_builder: DocumentProfileEvidenceBuilder

    def __init__(
        self,
        llm_provider: LLMProvider,
        prompt_text_normalization: Any,
        profile_prompt_policy: Any,
        evidence_builder: DocumentProfileEvidenceBuilder | None = None,
    ):
        self.llm_provider = llm_provider
        self.prompt_text_normalization = prompt_text_normalization
        self.profile_prompt_policy = profile_prompt_policy
        self.metadata_extractor = ParserMetadataExtractor()
        self.evidence_builder = evidence_builder or DocumentProfileEvidenceBuilder(
            structure_language_registry=DocumentStructureLanguageRegistry(),
        )

    def build(
        self,
        text: str,
        document_language: str,
    ) -> DocumentProfile:
        resolved_language = LanguageCodeResolver.resolve(document_language)
        deterministic_metadata = self.metadata_extractor.extract(
            text=text or "",
            document_language=resolved_language,
        )
        classification_profile = self._build_profile_classification_payload(
            text=text,
            document_language=resolved_language,
            deterministic_metadata=deterministic_metadata,
        )
        if classification_profile is not None:
            return classification_profile

        topic = self._detect_topic(text, resolved_language.value) or "other"
        summary = self._generate_summary(text, resolved_language.value, topic) or ""
        return DocumentProfile(
            topic=topic,
            summary=summary,
            document_language=resolved_language,
            parser_metadata=deterministic_metadata,
            structure_profile=None,
        )

    def _build_profile_classification_payload(
        self,
        *,
        text: str,
        document_language: LanguageCode,
        deterministic_metadata: ParserRelevantMetadata,
    ) -> DocumentProfile | None:
        try:
            print(
                "DocumentProfileBuilder#parser_metadata_classification_start",
                f"document_language={document_language.value}",
                f"text_len={len(text or '')}",
            )
            evidence = self._build_profile_evidence(
                text=text,
                document_language=document_language.value,
            )
            prompt = self._build_profile_classification_prompt(
                evidence=evidence,
                deterministic_metadata=deterministic_metadata,
            )
            raw_response = self.llm_provider.complete_text(prompt)
            payload = self._parse_json_object_payload(raw_response)
            if payload is None:
                print(
                    "DocumentProfileBuilder#parser_metadata_classification_fallback",
                    "reason=parse_json_object_failed",
                    f"response_len={len(raw_response or '')}",
                )
                return None
            profile = self._build_document_profile_from_classification_payload(
                payload=payload,
                fallback_document_language=document_language,
                deterministic_metadata=deterministic_metadata,
            )
            if profile is None:
                print(
                    "DocumentProfileBuilder#parser_metadata_classification_fallback",
                    "reason=payload_validation_failed",
                )
                return None
            print("DocumentProfileBuilder#parser_metadata_classification_success")
            return profile
        except Exception as error:
            print(
                "DocumentProfileBuilder#parser_metadata_classification_fallback",
                f"reason={error}",
            )
            return None

    def _build_document_profile_from_classification_payload(
        self,
        *,
        payload: dict[str, Any],
        fallback_document_language: LanguageCode,
        deterministic_metadata: ParserRelevantMetadata,
    ) -> DocumentProfile | None:
        topic = str(payload.get("topic") or "").strip()
        summary = str(payload.get("summary") or "").strip()
        document_language = LanguageCodeResolver.resolve(
            payload.get("document_language") or fallback_document_language
        )
        if not topic or not summary or document_language == LanguageCode.UNKNOWN:
            print(
                "DocumentProfileBuilder#parser_metadata_payload_invalid",
                "reason=missing_topic_or_summary_or_language",
            )
            return None
        if not self._is_classification_payload_valid(payload):
            return None

        merged_metadata = self.metadata_extractor.merge_classification(
            base_metadata=deterministic_metadata,
            text_form=self._normalize_enum(
                payload.get("text_form"),
                TextForm,
            ),
            discourse_mode=self._normalize_enum(
                payload.get("discourse_mode"),
                DiscourseMode,
            ),
            document_structure_shape=self._normalize_enum(
                payload.get("document_structure_shape"),
                DocumentStructureShape,
            ),
            likely_heading_style=self._normalize_enum(
                payload.get("likely_heading_style"),
                HeadingStyle,
            ),
            title_uniqueness_risk=self._normalize_enum(
                payload.get("title_uniqueness_risk"),
                LikelihoodLevel,
            ),
            confidence=self._safe_confidence(payload.get("confidence")),
            notes=self._safe_notes(payload.get("notes")),
        )
        return DocumentProfile(
            topic=topic,
            summary=summary,
            document_language=document_language,
            parser_metadata=merged_metadata,
            structure_profile=None,
        )

    def _is_classification_payload_valid(self, payload: dict[str, Any]) -> bool:
        field_allowlist_pairs = (
            ("text_form", _TEXT_FORM_ALLOWLIST),
            ("discourse_mode", _DISCOURSE_MODE_ALLOWLIST),
            ("document_structure_shape", _DOCUMENT_STRUCTURE_SHAPE_ALLOWLIST),
            ("likely_heading_style", _LIKELY_HEADING_STYLE_ALLOWLIST),
            ("title_uniqueness_risk", _RISK_ALLOWLIST),
        )
        for field_name, allowlist in field_allowlist_pairs:
            if field_name not in payload:
                continue
            normalized = self._normalize_allowlist(payload.get(field_name), allowlist)
            if normalized is None:
                print(
                    "DocumentProfileBuilder#parser_metadata_payload_invalid",
                    f"reason={field_name}_invalid",
                )
                return False
        if "confidence" in payload and self._safe_confidence(payload.get("confidence")) is None:
            print(
                "DocumentProfileBuilder#parser_metadata_payload_invalid",
                "reason=confidence_invalid",
            )
            return False
        if "notes" in payload and not isinstance(payload.get("notes"), list):
            print(
                "DocumentProfileBuilder#parser_metadata_payload_invalid",
                "reason=notes_not_list",
            )
            return False
        return True

    def _normalize_allowlist(self, value: Any, allowlist: set[str]) -> str | None:
        if value is None:
            return None
        normalized = str(value).strip()
        if not normalized:
            return None
        if normalized in allowlist:
            return normalized
        if normalized.lower() == "unknown":
            return "unknown"
        return None

    def _normalize_enum(
        self,
        value: Any,
        enum_type: type[Any],
    ) -> Any | None:
        normalized = self._normalize_allowlist(value, {item.value for item in enum_type})
        if normalized is None:
            return None
        try:
            return enum_type(normalized)
        except ValueError:
            try:
                return enum_type("unknown")
            except ValueError:
                return None

    def _safe_confidence(self, value: Any) -> float | None:
        if value is None:
            return None
        try:
            confidence = float(value)
        except (TypeError, ValueError):
            return None
        if confidence < 0.0 or confidence > 1.0:
            return None
        return confidence

    def _safe_notes(self, value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        notes: list[str] = []
        for item in value:
            if not isinstance(item, str):
                continue
            normalized = item.strip()
            if normalized:
                notes.append(normalized)
        return notes

    def _build_profile_classification_prompt(
        self,
        *,
        evidence: dict[str, Any],
        deterministic_metadata: ParserRelevantMetadata,
    ) -> str:
        evidence_json = json.dumps(evidence, ensure_ascii=False, indent=2)
        metadata_json = json.dumps(
            deterministic_metadata.to_dict(),
            ensure_ascii=False,
            indent=2,
        )
        return (
            "You are a document profile classifier.\n"
            "Return ONLY one valid JSON object. No markdown. No explanations.\n"
            "Do not invent headings. Do not output regex. Do not output exact split locations.\n"
            "candidate_lines are sampled evidence for classification only; they are not authoritative headings or parser boundaries.\n"
            "If uncertain, use 'unknown'.\n\n"
            "Return schema:\n"
            "{\n"
            '  "topic": string,\n'
            '  "summary": string,\n'
            '  "document_language": string,\n'
            '  "text_form": "novel"|"essay"|"academic_paper"|"technical_document"|"financial_report"|"legal_document"|"news_article"|"manual"|"dialogue_script"|"poetry"|"mixed"|"unknown",\n'
            '  "discourse_mode": "narrative"|"expository"|"argumentative"|"instructional"|"dialogue"|"reference"|"mixed"|"unknown",\n'
            '  "document_structure_shape": "chapter_only"|"part_chapter"|"chapter_section"|"essay_sections"|"flat_long_text"|"over_fragmented"|"mixed"|"unknown",\n'
            '  "likely_heading_style": "chinese_chapter_numbers"|"english_chapter_words"|"roman_numeral_parts"|"numbered_decimal_sections"|"numbered_chinese_points"|"plain_title_headings"|"none"|"mixed"|"unknown",\n'
            '  "title_uniqueness_risk": "none"|"low"|"medium"|"high"|"unknown",\n'
            '  "confidence": number|null,\n'
            '  "notes": string[]\n'
            "}\n\n"
            "Deterministic parser metadata (already computed, treat as trustworthy baseline):\n"
            f"{metadata_json}\n\n"
            "Evidence package JSON:\n"
            f"{evidence_json}\n"
        )

    def _detect_topic(self, text: str, document_language: str) -> str:
        excerpt = self._build_prompt_excerpt(
            text=text,
            prompt_overhead_tokens=self.profile_prompt_policy.topic_prompt_overhead_tokens,
            token_utilization_ratio=self.profile_prompt_policy.topic_excerpt_token_utilization_ratio,
        )
        prompt = f"""
You are a document classifier.

Determine the primary topic/category of the following document.
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
Return only the summary text.

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
        return self.evidence_builder.build(
            text=text,
            document_language=document_language,
            profile_prompt_policy=self.profile_prompt_policy,
            llm_provider=self.llm_provider,
            prompt_text_normalization=self.prompt_text_normalization,
        )

    def _parse_json_object_payload(self, response_text: str) -> dict[str, Any] | None:
        stripped = (response_text or "").strip()
        if not stripped:
            print(
                "DocumentProfileBuilder#classification_parse_failed",
                "reason=empty_response",
            )
            return None
        block_match = _FENCED_JSON_PATTERN.search(stripped)
        if block_match:
            stripped = block_match.group(1).strip()
        try:
            decoded = json.loads(stripped)
        except json.JSONDecodeError as error:
            error_context = self._build_json_error_context(
                text=stripped,
                error_position=error.pos,
                window_chars=180,
            )
            print(
                "DocumentProfileBuilder#classification_parse_failed",
                "reason=json_decode_error",
                f"error={error}",
                f"response_len={len(stripped)}",
                f"json_error_context={error_context}",
                "sanitized_preview=" + self._log_preview_text(
                    stripped,
                    max_chars=1800,
                ),
            )
            return None
        if not isinstance(decoded, dict):
            print(
                "DocumentProfileBuilder#classification_parse_failed",
                "reason=non_object_payload",
            )
            return None
        return decoded

    def _build_json_error_context(
        self,
        *,
        text: str,
        error_position: int,
        window_chars: int,
    ) -> str:
        if not text:
            return ""
        safe_position = max(0, min(error_position, len(text)))
        start = max(0, safe_position - window_chars)
        end = min(len(text), safe_position + window_chars)
        context = text[start:end]
        context_sanitized = self._sanitize_control_chars(context)
        return (
            f"pos={safe_position} start={start} end={end} "
            f"context={context_sanitized}"
        )

    def _log_preview_text(self, text: str, *, max_chars: int) -> str:
        preview = text[:max_chars]
        if len(text) > max_chars:
            preview = f"{preview}...<truncated total_len={len(text)}>"
        return self._sanitize_control_chars(preview)

    def _sanitize_control_chars(self, text: str) -> str:
        return (
            text.replace("\\", "\\\\")
            .replace("\r", "\\r")
            .replace("\n", "\\n")
            .replace("\t", "\\t")
        )
