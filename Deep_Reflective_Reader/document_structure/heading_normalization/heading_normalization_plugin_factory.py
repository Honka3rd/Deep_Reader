from document_structure.heading_normalization.chinese_chapter_ocr_normalization_plugin import (
    ChineseChapterOcrNormalizationPlugin,
)
from document_structure.heading_normalization.common_heading_typography_normalization_plugin import (
    CommonHeadingTypographyNormalizationPlugin,
)
from document_structure.heading_normalization.heading_normalization_plugin import (
    HeadingNormalizationPlugin,
)
from language.language_code import LanguageCode


class HeadingNormalizationPluginFactory:
    """Factory that resolves heading-normalization plugin list by language."""

    def __init__(
        self,
        common_plugin: CommonHeadingTypographyNormalizationPlugin | None = None,
        chinese_chapter_ocr_plugin: ChineseChapterOcrNormalizationPlugin | None = None,
    ):
        self.common_plugin = (
            common_plugin or CommonHeadingTypographyNormalizationPlugin()
        )
        self.chinese_chapter_ocr_plugin = (
            chinese_chapter_ocr_plugin or ChineseChapterOcrNormalizationPlugin()
        )

    def get_plugins(self, language: LanguageCode) -> tuple[HeadingNormalizationPlugin, ...]:
        """Return plugin pipeline for target language."""
        if language == LanguageCode.ZH:
            return (self.common_plugin, self.chinese_chapter_ocr_plugin)
        return (self.common_plugin,)
