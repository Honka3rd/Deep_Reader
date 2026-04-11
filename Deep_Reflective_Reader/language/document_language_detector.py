import json
import os

from llm_provider import LLMProvider
from storage_config import StorageConfig

class DocumentLanguageDetector:
    llm_provider: LLMProvider

    def __init__(self, llm_provider: LLMProvider):
        self.llm_provider = llm_provider

    def detect(self, text: str, config: StorageConfig | None = None) -> str:
        """
        Detect document language with persistence-aware fallback:

        1. Try document_profile.json
        2. Try records.json
        3. Fallback to LLM detection
        """
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
    def _load_from_profile(config: StorageConfig | None) -> str | None:
        if config is None:
            return None

        path = config.get_raw_profile_path()
        if not os.path.exists(path):
            return None

        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)

            language = payload.get("document_language")
            if isinstance(language, str) and language.strip():
                return language.strip().lower()
        except Exception as e:
            print(f"DocumentLanguageDetector#_load_from_profile failed: {e}")

        return None

    @staticmethod
    def _load_from_records(config: StorageConfig | None) -> str | None:
        if config is None:
            return None

        path = config.get_raw_records_path()
        if not os.path.exists(path):
            return None

        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)

            language = payload.get("document_language")
            if isinstance(language, str) and language.strip():
                return language.strip().lower()
        except Exception as e:
            print(f"DocumentLanguageDetector#_load_from_records failed: {e}")

        return None

    def _detect_with_llm(self, text: str) -> str:
        prompt = f"""
Detect the primary language of the following document.
Return only a short language code such as: en, zh, ja, fr, de.

Document:
{text[:4000]}
"""
        result = self.llm_provider.complete_text(prompt).strip().lower()

        # 小防禦，避免回 "english" / "en." 這種
        normalized = result.replace(".", "").split()[0]

        mapping = {
            "english": "en",
            "chinese": "zh",
            "japanese": "ja",
            "french": "fr",
            "german": "de",
        }

        return mapping.get(normalized, normalized)