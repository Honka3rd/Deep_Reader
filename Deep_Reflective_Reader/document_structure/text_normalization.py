"""Deterministic normalization views for noisy OCR text.

The source text remains untouched. These helpers are for matching and
structure detection only, so source offsets and provenance stay traceable.
"""

from __future__ import annotations

import unicodedata


def normalize_ocr_whitespace(value: str) -> str:
    """Remove OCR-inserted intra-word spaces while preserving punctuation.

    Spaces between Latin/number tokens are retained. Whitespace surrounded by
    CJK characters or punctuation is removed, and paragraph breaks remain
    paragraph breaks.
    """
    if not value:
        return ""

    normalized = value.replace("\u3000", " ")
    output: list[str] = []
    index = 0
    length = len(normalized)
    while index < length:
        if not normalized[index].isspace():
            output.append(normalized[index])
            index += 1
            continue

        start = index
        newline_count = 0
        while index < length and normalized[index].isspace():
            if normalized[index] in "\r\n":
                newline_count += 1
            index += 1

        previous = _previous_non_whitespace(normalized, start)
        following = _next_non_whitespace(normalized, index)
        if newline_count >= 2:
            output.append("\n\n")
        elif _should_remove_between(previous, following):
            continue
        else:
            output.append(" ")

    return "".join(output).strip()


def normalize_ocr_text(value: str) -> str:
    """Return a Unicode-normalized OCR matching view."""
    return normalize_ocr_whitespace(unicodedata.normalize("NFKC", value or ""))


def _previous_non_whitespace(value: str, index: int) -> str | None:
    index -= 1
    while index >= 0 and value[index].isspace():
        index -= 1
    return value[index] if index >= 0 else None


def _next_non_whitespace(value: str, index: int) -> str | None:
    while index < len(value) and value[index].isspace():
        index += 1
    return value[index] if index < len(value) else None


def _should_remove_between(previous: str | None, following: str | None) -> bool:
    if previous is None or following is None:
        return False
    if _is_cjk(previous) or _is_cjk(following):
        return True
    if _is_punctuation(previous) and (_is_cjk(following) or _is_punctuation(following)):
        return True
    if _is_punctuation(following) and (_is_cjk(previous) or _is_punctuation(previous)):
        return True
    return False


def _is_cjk(value: str) -> bool:
    codepoint = ord(value)
    return (
        0x3400 <= codepoint <= 0x4DBF
        or 0x4E00 <= codepoint <= 0x9FFF
        or 0xF900 <= codepoint <= 0xFAFF
    )


def _is_punctuation(value: str) -> bool:
    return unicodedata.category(value).startswith("P")
