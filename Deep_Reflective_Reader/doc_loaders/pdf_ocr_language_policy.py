"""Engine adapter over the system registry, shared by runtime and image build."""

from language.language_code import LanguageCode, LanguageCodeResolver
from language.language_profile_registry import LanguageProfileRegistry


# These are engine identifiers, not an independent supported-language registry.
_MODELS_BY_LANGUAGE: dict[str, tuple[str, ...]] = {
    "en": ("eng",), "zh": ("chi_sim", "chi_tra"), "ja": ("jpn",),
    "ko": ("kor",), "es": ("spa",), "pt": ("por",), "fr": ("fra",),
    "de": ("deu",), "it": ("ita",), "ru": ("rus",), "ar": ("ara",),
    "hi": ("hin",), "tr": ("tur",), "nl": ("nld",), "pl": ("pol",),
    "uk": ("ukr",), "id": ("ind",), "vi": ("vie",), "th": ("tha",),
}

# Existing Chinese strategy requires these exact model builds.
VERTICAL_MODEL_HASHES = {
    "chi_tra_vert": "1df02a4b210e5c217b783819538b63e9dfe6904e2b5e53b62664f1b9f7a989d0",
    "chi_sim_vert": "20590de84725bab69cde93bd6e8ed360a13cc5421a7e7364ddeb93e9af53d6da",
}


def registered_ocr_models() -> tuple[str, ...]:
    codes = LanguageProfileRegistry.get_supported_language_codes()
    missing = set(codes) - _MODELS_BY_LANGUAGE.keys()
    if missing:
        raise ValueError("ocr_registry_mapping_missing:" + ",".join(sorted(missing)))
    return tuple(dict.fromkeys(model for code in codes for model in _MODELS_BY_LANGUAGE[code]))


def required_ocr_models() -> tuple[str, ...]:
    """Installed capabilities include strategy dependencies, regardless of hints."""
    horizontal = registered_ocr_models()
    vertical = tuple(VERTICAL_MODEL_HASHES) if "chi_sim" in horizontal else ()
    return (*horizontal, *vertical, "osd")


DEFAULT_TESSERACT_OCR_LANGUAGE = "+".join(registered_ocr_models())


def get_tesseract_language_for_document_language(language: str | LanguageCode | None) -> str:
    """Map project language code values to Tesseract language pack names."""
    if isinstance(language, LanguageCode):
        resolved_language = language
    else:
        resolved_language = LanguageCodeResolver.resolve(language)
    if resolved_language == LanguageCode.UNKNOWN:
        return "+".join(registered_ocr_models())
    registered_ocr_models()  # Validate registry coverage before narrowing selection.
    if resolved_language.value not in LanguageProfileRegistry.get_supported_language_codes():
        raise ValueError("ocr_language_not_registered")
    models = _MODELS_BY_LANGUAGE[resolved_language.value]
    # Preserve the established bilingual Chinese selection contract.
    return "+".join((*models, "eng") if resolved_language == LanguageCode.ZH else models)


def normalize_tesseract_language_config(value: str | None) -> str:
    """Expand registry codes or validated legacy model hints; never silently drop one."""
    models = registered_ocr_models()
    normalized = (value or "").strip().lower()
    if normalized in {"", "registry"}:
        return "+".join(models)
    allowed = set(required_ocr_models()) - {"osd"}
    selected: list[str] = []
    codes = LanguageProfileRegistry.get_supported_language_codes()
    for token in normalized.split("+"):
        token = token.strip()
        if token in allowed:
            selected.append(token)
        elif token in codes:
            selected.extend(_MODELS_BY_LANGUAGE[token])
        else:
            raise ValueError("ocr_language_config_invalid: use registry, registered codes or OCR model names")
    return "+".join(dict.fromkeys(selected))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--debian-packages", action="store_true", required=True)
    parser.parse_args()
    print(" ".join("tesseract-ocr-" + model.replace("_", "-")
                   for model in (*registered_ocr_models(), "osd")))
