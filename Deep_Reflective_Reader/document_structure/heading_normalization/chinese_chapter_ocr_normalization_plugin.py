import re

from language.language_code import LanguageCode


class ChineseChapterOcrNormalizationPlugin:
    """Normalize OCR-confusable dash glyph in Chinese chapter headings."""

    name = "chinese_chapter_ocr_normalization"

    _CHINESE_CHAPTER_OCR_DASH_PATTERN = re.compile(
        r"^\s*(?P<prefix>第[0-9零〇一二两兩三四五六七八九十百千]+)\s*-\s*章(?P<suffix>\s*(?:[:：.\-–—].*)?)\s*$"
    )

    def normalize(self, heading: str, language: LanguageCode) -> str:
        """Convert OCR dash-like chapter glyph to canonical Chinese numeral one."""
        if language != LanguageCode.ZH:
            return heading

        matched = self._CHINESE_CHAPTER_OCR_DASH_PATTERN.match(heading.strip())
        if matched is None:
            return heading

        prefix = matched.group("prefix")
        if not self._looks_like_missing_one(prefix):
            return heading

        suffix = (matched.group("suffix") or "").strip()
        normalized = f"{prefix}一章"
        if suffix:
            normalized = f"{normalized}{suffix}"
        return normalized

    @staticmethod
    def _looks_like_missing_one(prefix: str) -> bool:
        """Only normalize likely '...十-章' / '...0-章' OCR patterns."""
        number_text = prefix[1:] if prefix.startswith("第") else prefix
        if not number_text:
            return False
        return number_text.endswith(("十", "拾", "0"))
