"""Application resource ownership and lazy provider/profile public adapters."""
from dataclasses import dataclass

from learning import LearningService
from reading_preferences.models import identifier
from reading_preferences.profile_source import ExistingProfileLanguageReader
from reading_preferences.service import ReadingPreferencesService
from section_tasks.learning_hints_service import LearningHintsService
from section_tasks.learning_translation_service import LearningTranslationService


class LearningProvider:
    """Resolve the configured singleton only on explicit generation preflight.

    No fallback to a retrying completion API: an injected provider must expose
    the single-attempt method. Capacity failures remain domain-owned errors.
    """
    def __init__(self, factory):
        self.factory = factory

    def _provider(self):
        provider = self.factory()
        if not callable(getattr(provider, "complete_text_once_with_output_budget", None)):
            raise RuntimeError("learning_provider_unavailable")
        return provider

    def get_model_capabilities(self):
        return self._provider().get_model_capabilities()

    def count_input_tokens(self, text):
        return self._provider().count_input_tokens(text)

    @property
    def tokenizer_identity(self):
        return self._provider().tokenizer_identity

    def complete_text_with_output_budget(self, prompt, output_tokens):
        return self._provider().complete_text_once_with_output_budget(prompt, output_tokens)


def existing_profile_reader(scope, store, config_factory):
    def load(request_scope, document_id):
        identifier(document_id)
        if request_scope != scope:
            raise ValueError("preference_scope_mismatch")
        config = config_factory(document_id, read_only=True)
        if config.doc_name != document_id:
            raise ValueError("profile_document_identity_mismatch")
        loader = getattr(store, "load_source", None)
        if not callable(loader):
            raise OSError("profile_source_reader_unavailable")
        return loader(config)
    return ExistingProfileLanguageReader(load)


@dataclass(frozen=True)
class LearningRuntime:
    preferences: ReadingPreferencesService
    service: LearningService


def open_learning_runtime(*, storage, documents, profile_store, config_factory,
                          provider_factory, policy):
    if documents is not storage.documents:
        raise ValueError("learning_document_repository_mismatch")
    profiles = existing_profile_reader(storage.preferences.scope, profile_store, config_factory)
    preferences = ReadingPreferencesService(storage.preferences, profiles)
    provider = LearningProvider(provider_factory)
    service = LearningService(storage.learning, preferences,
        LearningTranslationService(provider, provider_name="configured"),
        LearningHintsService(provider, provider_name="configured"),
        lease_ms=policy.lease_ms, max_workers=policy.max_workers)
    try:
        yield LearningRuntime(preferences, service)
    finally:
        # Application shutdown drains accepted jobs. Browser requests do not own it.
        service.shutdown(wait=True)
