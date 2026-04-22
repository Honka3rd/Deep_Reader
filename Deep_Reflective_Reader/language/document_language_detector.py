import json
import os

from language.language_code import LanguageCode
from language.language_profile_registry import LanguageProfileRegistry
from llm.llm_provider import LLMProvider
from config.faiss_storage_config import FaissStorageConfig

class DocumentLanguageDetector:
    """Detect and cache primary language for a document."""
    llm_provider: LLMProvider

    def __init__(self, llm_provider: LLMProvider):
        """Initialize object state and injected dependencies.

Args:
    llm_provider: Llm provider.
"""
        self.llm_provider = llm_provider

    def detect(self, text: str, config: FaissStorageConfig | None = None) -> str:
        """Detect primary language for the given document text.

Args:
    text: Input text content.
    config: FaissStorageConfig describing filesystem artifact paths.

Returns:
    Normalized language code (for example: ``en``, ``zh``, ``ja``)."""
        language = self._load_from_profile(config)
        if language:
            print(f"DocumentLanguageDetector#detect: loaded from profile -> {language}")
            return language

        language = self._load_from_records(config)
        if language:
            print(f"DocumentLanguageDetector#detect: loaded from records -> {language}")
            return language

        language = self._detect_with_llm(text)
        print(f"DocumentLanguageDetector#detect: detected by llm -> {language}")
        return language

    @staticmethod
    def _load_from_profile(config: FaissStorageConfig | None) -> str | None:
        """Internal helper for load from profile.

Args:
    config: FaissStorageConfig describing filesystem artifact paths.

Returns:
    Language code from profile when available; otherwise ``None``."""
        if config is None:
            return None

        path = config.get_raw_profile_path()
        if not os.path.exists(path):
            return None

        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)

            language = payload.get("document_language")
            normalized = DocumentLanguageDetector._normalize_language(language)
            if normalized is not None:
                return normalized
        except Exception as e:
            print(f"DocumentLanguageDetector#_load_from_profile failed: {e}")

        return None

    @staticmethod
    def _load_from_records(config: FaissStorageConfig | None) -> str | None:
        """Internal helper for load from records.

Args:
    config: FaissStorageConfig describing filesystem artifact paths.

Returns:
    Language code from records when available; otherwise ``None``."""
        if config is None:
            return None

        path = config.get_raw_records_path()
        if not os.path.exists(path):
            return None

        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)

            language = payload.get("document_language")
            normalized = DocumentLanguageDetector._normalize_language(language)
            if normalized is not None:
                return normalized
        except Exception as e:
            print(f"DocumentLanguageDetector#_load_from_records failed: {e}")

        return None

    @staticmethod
    def _normalize_language(value: str | None) -> str | None:
        """Normalize persisted/detected language value to supported code string."""
        normalized = LanguageProfileRegistry.normalize_detector_output(value)
        if normalized == LanguageCode.UNKNOWN:
            return None
        return normalized.value

    def _detect_with_llm(self, text: str) -> str:
        """Internal helper for detect with llm.

Args:
    text: Input text content.

Returns:
    Language code inferred from LLM output after normalization."""
        supported_codes = ", ".join(
            LanguageProfileRegistry.get_supported_language_codes()
        )
        prompt = f"""
Detect the primary language of the following document.
Return only one short language code from this list: {supported_codes}.

Document:
{text[:4000]}
"""
        result = self.llm_provider.complete_text(prompt).strip()
        normalized = LanguageProfileRegistry.normalize_detector_output(result)
        if normalized == LanguageCode.UNKNOWN:
            print(
                "Warn:DocumentLanguageDetector#_detect_with_llm: "
                f"unknown detector output={result!r}, fallback_to_en"
            )
            return LanguageCode.EN.value
        return normalized.value
