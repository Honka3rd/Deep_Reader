"""Generation-only language policy; never parser authority or source rewriting."""

import re
from collections.abc import Mapping

from language.language_code import LanguageCode, LanguageCodeResolver
from section_tasks.reading_interaction_service_contracts import (
    ReadingInteractionRequest, ReadingInteractionArtifact,
)


LANGUAGE_LABELS_BY_CODE = {
    "ar": "Arabic", "de": "German", "en": "English", "es": "Spanish",
    "fr": "French", "hi": "Hindi", "id": "Indonesian", "it": "Italian",
    "ja": "Japanese", "ko": "Korean", "nl": "Dutch", "pl": "Polish",
    "pt": "Portuguese", "ru": "Russian", "th": "Thai", "tr": "Turkish",
    "uk": "Ukrainian", "vi": "Vietnamese", "zh": "Chinese",
    "zh-hans": "Simplified Chinese", "zh-hant": "Traditional Chinese",
}
OUTPUT_LANGUAGE_POLICY_VERSION = "profile-first-language-v1"
_QUOTATIONS = re.compile(r'"[^"\n]*"|\u201c[^\u201d\n]*\u201d|\u2018[^\u2019\n]*\u2019|\u300c[^\u300d\n]*\u300d|\u300e[^\u300f\n]*\u300f')


class OutputLanguageMismatch(ValueError):
    """Stable failure code without disclosing generated or source text."""

    def __init__(self):
        super().__init__("output_language_mismatch")


def resolve_language(context: Mapping[str, object], document_language: str | None = None) -> str | None:
    # context.document_language is selected profile-first by the coordinator.
    for raw in (context.get("document_language"), document_language):
        if not isinstance(raw, str):
            continue
        normalized = raw.strip().lower().replace("_", "-")
        if normalized in {"zh-hans", "zh-hant"}:
            return normalized
        code = LanguageCodeResolver.resolve(raw)
        if code != LanguageCode.UNKNOWN:
            return code.value
    return None


def request_language(request: ReadingInteractionRequest) -> str | None:
    return resolve_language(request.context_metadata, request.target.document_language)


def session_language(session: ReadingInteractionArtifact) -> str | None:
    context = session.metadata.get("context")
    return resolve_language(context if isinstance(context, dict) else {})


def language_instruction(language: str | None, *, evaluation: bool = False) -> str:
    if language is None:
        source = "the session question" if evaluation else "the Primary Source"
        instruction = f"Output language: Use the language of {source}; do not default to English.\n"
    else:
        instruction = (
            "Output language: Write every user-facing JSON string in "
            f"{LANGUAGE_LABELS_BY_CODE[language]} ({language}).\n"
        )
    return instruction + (
        "This applies to every thesis, reasoning, explanation, question, choice, "
        "answer and feedback field. Keep JSON keys, type codes, booleans and numbers "
        "unchanged. Brief original quotations, proper names and technical terms may "
        "retain their original language. Ancestor summaries, secondary context, "
        "examples and the user's answer do not override this output-language policy.\n"
    )


def validate_output_language(payload: object, language: str | None) -> None:
    """Reject clear zh/en script mismatches, abstaining on short/mixed text.

    This is deliberately not a universal language detector. Unsupported language
    pairs and uncertain evidence stay prompt-constrained, not falsely rejected.
    Only normalized generated payloads belong here, never the user's answer.
    """
    if language not in {"zh", "zh-hans", "zh-hant", "en"}:
        return
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key not in {"type", "id"}:
                validate_output_language(value, language)
    elif isinstance(payload, list):
        for value in payload:
            validate_output_language(value, language)
    elif isinstance(payload, str):
        prose = _QUOTATIONS.sub("", payload)
        han = len(re.findall(r"[\u3400-\u9fff]", prose))
        latin = len(re.findall(r"[A-Za-z]", prose))
        words = re.findall(r"[A-Za-z]+", prose)
        if language.startswith("zh") and han == 0 and latin >= 40 and len(words) >= 8:
            raise OutputLanguageMismatch()
        if language == "en" and han >= 20 and latin < 4:
            raise OutputLanguageMismatch()
