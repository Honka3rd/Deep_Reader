"""Tesseract OCR language selection policy for PDF raw loading."""

from language.language_code import LanguageCode, LanguageCodeResolver


DEFAULT_TESSERACT_OCR_LANGUAGE = "eng+chi_sim+chi_tra"

_TESSERACT_LANGUAGE_BY_DOCUMENT_LANGUAGE: dict[LanguageCode, str] = {
    LanguageCode.EN: "eng",
    LanguageCode.ZH: "chi_sim+chi_tra+eng",
    LanguageCode.UNKNOWN: DEFAULT_TESSERACT_OCR_LANGUAGE,
}


def get_tesseract_language_for_document_language(language: str | LanguageCode | None) -> str:
    """Map project language code values to Tesseract language pack names."""
    if isinstance(language, LanguageCode):
        resolved_language = language
    else:
        resolved_language = LanguageCodeResolver.resolve(language)
    return _TESSERACT_LANGUAGE_BY_DOCUMENT_LANGUAGE.get(
        resolved_language,
        DEFAULT_TESSERACT_OCR_LANGUAGE,
    )


def normalize_tesseract_language_config(value: str | None) -> str:
    """Normalize explicit OCR language config with a safe multilingual fallback."""
    if value is None:
        return DEFAULT_TESSERACT_OCR_LANGUAGE
    normalized = value.strip()
    return normalized or DEFAULT_TESSERACT_OCR_LANGUAGE
