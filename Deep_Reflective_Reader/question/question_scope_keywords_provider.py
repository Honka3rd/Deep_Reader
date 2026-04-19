from language.language_code import LanguageCode
from language.language_profile_registry import LanguageProfileRegistry


class QuestionScopeKeywordsProvider:
    """Provide global-scope intent keywords by language from registry."""

    @staticmethod
    def get_keywords(language: LanguageCode) -> tuple[str, ...]:
        """Return global-intent keywords for a specific language code."""
        return LanguageProfileRegistry.get_scope_keywords(language)

    @staticmethod
    def get_all_keywords() -> tuple[str, ...]:
        """Return deduplicated keyword set across all supported languages."""
        return LanguageProfileRegistry.get_all_scope_keywords()
