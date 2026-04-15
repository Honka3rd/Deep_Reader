from enum import StrEnum
import re


class LanguageCode(StrEnum):
    """Canonical language codes used by QA pipeline."""
    EN = "en"
    ZH = "zh"
    JA = "ja"
    FR = "fr"
    DE = "de"
    ES = "es"
    PT = "pt"
    IT = "it"
    RU = "ru"
    KO = "ko"
    AR = "ar"
    HI = "hi"
    TR = "tr"
    NL = "nl"
    PL = "pl"
    UK = "uk"
    ID = "id"
    VI = "vi"
    TH = "th"
    UNKNOWN = "unknown"


class LanguageCodeResolver:
    """Normalize raw language strings to ``LanguageCode``."""

    _ALIASES: dict[str, LanguageCode] = {
        "english": LanguageCode.EN,
        "en-us": LanguageCode.EN,
        "en-gb": LanguageCode.EN,
        "chinese": LanguageCode.ZH,
        "zh-cn": LanguageCode.ZH,
        "zh-tw": LanguageCode.ZH,
        "zh-hans": LanguageCode.ZH,
        "zh-hant": LanguageCode.ZH,
        "japanese": LanguageCode.JA,
        "french": LanguageCode.FR,
        "german": LanguageCode.DE,
        "spanish": LanguageCode.ES,
        "portuguese": LanguageCode.PT,
        "italian": LanguageCode.IT,
        "russian": LanguageCode.RU,
        "korean": LanguageCode.KO,
        "arabic": LanguageCode.AR,
        "hindi": LanguageCode.HI,
        "turkish": LanguageCode.TR,
        "dutch": LanguageCode.NL,
        "polish": LanguageCode.PL,
        "ukrainian": LanguageCode.UK,
        "indonesian": LanguageCode.ID,
        "vietnamese": LanguageCode.VI,
        "thai": LanguageCode.TH,
    }

    @classmethod
    def resolve(cls, value: str | None) -> LanguageCode:
        """Resolve raw language string to canonical enum (or ``UNKNOWN``)."""
        if value is None:
            return LanguageCode.UNKNOWN

        normalized = value.strip().lower()
        if not normalized:
            return LanguageCode.UNKNOWN

        normalized = normalized.replace("_", "-")
        normalized = re.sub(r"[^a-z-]", "", normalized)

        if normalized in cls._ALIASES:
            return cls._ALIASES[normalized]

        if normalized in LanguageCode._value2member_map_:
            return LanguageCode(normalized)

        short = normalized.split("-")[0]
        if short in LanguageCode._value2member_map_:
            return LanguageCode(short)

        return LanguageCode.UNKNOWN

    @classmethod
    def infer_from_text(cls, text: str | None) -> LanguageCode:
        """Infer language code from script/keyword hints without LLM."""
        if text is None:
            return LanguageCode.UNKNOWN

        content = text.strip()
        if not content:
            return LanguageCode.UNKNOWN

        lowered = content.lower()

        if re.search(r"[\u4e00-\u9fff]", content):
            return LanguageCode.ZH
        if re.search(r"[\u3040-\u30ff]", content):
            return LanguageCode.JA
        if re.search(r"[\uac00-\ud7af]", content):
            return LanguageCode.KO
        if re.search(r"[\u0600-\u06ff]", content):
            return LanguageCode.AR
        if re.search(r"[\u0900-\u097f]", content):
            return LanguageCode.HI
        if re.search(r"[\u0e00-\u0e7f]", content):
            return LanguageCode.TH
        if re.search(r"[а-яёіїєґ]", lowered):
            if re.search(r"[іїєґ]", lowered):
                return LanguageCode.UK
            return LanguageCode.RU

        if re.search(r"[a-z]", lowered):
            # Heuristic keyword hints for Latin-script languages.
            if any(token in lowered for token in (" cuáles", " qué ", " tema", " relaciones", " razones")):
                return LanguageCode.ES
            if any(token in lowered for token in (" quais", " resumo", " temas", " relações")):
                return LanguageCode.PT
            if any(token in lowered for token in (" quels", " résumé", " thèmes", " relations")):
                return LanguageCode.FR
            if any(token in lowered for token in (" welche", " zusammenfassung", " themen", " beziehungen")):
                return LanguageCode.DE
            if any(token in lowered for token in (" quali", " riassunto", " temi", " relazioni")):
                return LanguageCode.IT
            if any(token in lowered for token in (" welke", " samenvatting", " relaties", " thema")):
                return LanguageCode.NL
            if any(token in lowered for token in (" które", " podsumowanie", " relacje", " tematy")):
                return LanguageCode.PL
            if any(token in lowered for token in (" yang ", " ringkasan", " hubungan", " tema")):
                return LanguageCode.ID
            if any(token in lowered for token in (" những", " tóm tắt", " mối quan hệ", " chủ đề")):
                return LanguageCode.VI
            if any(token in lowered for token in (" which", " what are", " overall", " list")):
                return LanguageCode.EN
            return LanguageCode.EN

        return LanguageCode.UNKNOWN
