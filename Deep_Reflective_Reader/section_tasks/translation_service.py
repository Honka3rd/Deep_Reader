"""Stateless, provider-injected translation without hierarchy or storage access."""

import json
import logging
from time import monotonic

from llm.llm_provider import LLMProvider
from section_tasks.interaction_output_language import validate_output_language, OutputLanguageMismatch
from section_tasks.translation_contracts import (
    TranslationError, TranslationResult, normalize_translation_language, ensure_distinct_translation_languages,
)
from section_tasks.translation_prompt import build_translation_prompt, TRANSLATION_PROMPT_VERSION


logger = logging.getLogger(__name__)
_OVERHEAD_TOKENS = 256


class TranslationService:
    def __init__(self, llm_provider: LLMProvider):
        self._provider = llm_provider

    def translate(self, *, text: str, source_language: str, target_language: str) -> TranslationResult:
        started = monotonic()
        outcome = "failed"
        try:
            if not isinstance(text, str) or not text.strip():
                raise TranslationError("translation_invalid_text")
            source = normalize_translation_language(source_language, field="source_language")
            target = normalize_translation_language(target_language, field="target_language")
            ensure_distinct_translation_languages(source, target)
            prompt = build_translation_prompt(text=text, source_language=source, target_language=target)
            output_override = self._check_capacity(prompt)
            try:
                response = (self._provider.complete_text_with_output_budget(prompt, output_override)
                            if output_override is not None else self._provider.complete_text(prompt))
            except Exception:
                raise TranslationError("translation_provider_failed") from None
            status, translated = _parse_response(response)
            try:
                validate_output_language(translated, target.lower().split("-")[0])
            except OutputLanguageMismatch:
                raise TranslationError("translation_invalid_output") from None
            outcome = status
            return TranslationResult(source, target, status, translated)
        except TranslationError as error:
            outcome = error.code
            raise
        finally:
            logger.info("translation_finished prompt_version=%s outcome=%s duration_ms=%.1f",
                        TRANSLATION_PROMPT_VERSION, outcome, (monotonic() - started) * 1000)

    def _count(self, text: str) -> int:
        count = self._provider.count_input_tokens(text)
        if count is None:
            return len(text.encode("utf-8"))
        if type(count) is not int or count <= 0:
            raise ValueError("invalid token count")
        return count

    def _check_capacity(self, prompt: str) -> int | None:
        try:
            capabilities = self._provider.get_model_capabilities()
            output_limit = capabilities.max_output_tokens
            supports_override = callable(getattr(type(self._provider), "complete_text_with_output_budget", None))
            if not supports_override:
                default_output = getattr(self._provider, "effective_output_budget", output_limit)
                if type(default_output) is not int or default_output != output_limit:
                    raise ValueError("provider cannot dispatch the model output limit")
            input_tokens = self._count(prompt) + _OVERHEAD_TOKENS
            input_limit = capabilities.input_token_limit(output_limit)
        except Exception:
            raise TranslationError("translation_capacity_unavailable") from None
        logger.info("translation_capacity input_tokens=%d input_limit=%d output_limit=%d",
                    input_tokens, input_limit, output_limit)
        if input_tokens > input_limit:
            raise TranslationError("translation_capacity_exceeded")
        # Match the preflight reserve at dispatch without changing the shared provider.
        return output_limit if supports_override else None


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate key")
        result[key] = value
    return result


def _parse_response(response: str) -> tuple[str, str | None]:
    try:
        if not isinstance(response, str):
            raise ValueError("non-text response")
        payload = json.loads(response, object_pairs_hook=_unique_object)
        if not isinstance(payload, dict) or set(payload) != {"status", "translated_text"}:
            raise ValueError("invalid fields")
        status, text = payload["status"], payload["translated_text"]
        if status == "untranslatable" and text is None:
            return status, text
        if status not in ("completed", "partial") or not isinstance(text, str) or not text.strip():
            raise ValueError("invalid translation")
        return status, text
    except (ValueError, TypeError, RecursionError):
        raise TranslationError("translation_invalid_output") from None
