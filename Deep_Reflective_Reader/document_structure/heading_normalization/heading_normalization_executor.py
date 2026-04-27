from document_structure.heading_normalization.heading_normalization_plugin import (
    HeadingNormalizationPlugin,
)
from language.language_code import LanguageCode


class HeadingNormalizationExecutor:
    """Execute heading-normalization plugins sequentially with safety guards."""

    def normalize(
        self,
        *,
        heading: str,
        language: LanguageCode,
        plugins: tuple[HeadingNormalizationPlugin, ...],
    ) -> str:
        """Run plugin pipeline and return final normalized heading."""
        normalized = heading
        for plugin in plugins:
            try:
                candidate = plugin.normalize(normalized, language)
                if isinstance(candidate, str):
                    normalized = candidate
            except Exception as error:
                plugin_name = getattr(plugin, "name", plugin.__class__.__name__)
                print(
                    "HeadingNormalizationExecutor#plugin_failed:",
                    f"plugin={plugin_name}",
                    f"language={language.value}",
                    f"error={error}",
                )
        return normalized
