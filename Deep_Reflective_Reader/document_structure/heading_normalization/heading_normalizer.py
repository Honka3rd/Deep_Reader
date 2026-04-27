from document_structure.heading_normalization.heading_normalization_executor import (
    HeadingNormalizationExecutor,
)
from document_structure.heading_normalization.heading_normalization_plugin_factory import (
    HeadingNormalizationPluginFactory,
)
from language.language_code import LanguageCode


class HeadingNormalizer:
    """Language-aware heading normalizer backed by plugin factory + executor."""

    def __init__(
        self,
        plugin_factory: HeadingNormalizationPluginFactory | None = None,
        executor: HeadingNormalizationExecutor | None = None,
    ):
        self.plugin_factory = plugin_factory or HeadingNormalizationPluginFactory()
        self.executor = executor or HeadingNormalizationExecutor()

    def normalize(
        self,
        *,
        heading: str,
        language: LanguageCode,
    ) -> str:
        """Normalize heading text with language-specific plugin chain."""
        plugins = self.plugin_factory.get_plugins(language)
        return self.executor.normalize(
            heading=heading,
            language=language,
            plugins=plugins,
        )
