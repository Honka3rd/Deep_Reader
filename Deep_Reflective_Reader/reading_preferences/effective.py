"""Pure, immutable effective settings projection. No storage, provider or repair."""

from dataclasses import dataclass

from language.language_proficiency_registry import (
    LanguageProficiencyRegistry, ProficiencyScheme, ProficiencySelectionError,
    ResolvedReadingProficiency,
)
from reading_preferences.models import (
    BookPreferences, LanguageLearningSettings, ModeDefinition, PreferenceError,
    WorkspacePreferences, language, list_modes,
)
from reading_preferences.profile_source import SourceLanguage


@dataclass(frozen=True, slots=True)
class PreferenceIssue:
    code: str
    field: str
    message: str


# Stable reason codes; transport/localization remain the API/UI owners' concern.
_ISSUES = {
    "learning_mode_not_active": ("active_mode", "Enable language learning for this book before generating learning material."),
    "profile_missing": ("source_language", "Choose the book's original language; no saved profile is available."),
    "profile_invalid": ("source_language", "Choose the book's original language or repair the saved profile."),
    "profile_unavailable": ("source_language", "The saved profile could not be read. Retry or explicitly choose the original language."),
    "source_language_required": ("source_language", "Choose the book's original language."),
    "source_language_invalid": ("source_language", "Choose an original language supported by the language registry."),
    "native_language_required": ("native_language", "Set your native language in workspace preferences."),
    "native_language_invalid": ("native_language", "Choose a native language supported by the language registry."),
    "same_language": ("source_language", "Language learning requires a book written in a language different from your native language."),
    "proficiency_required": ("proficiency", "Choose your reading level for this book's original language."),
    "proficiency_language_mismatch": ("proficiency", "Choose a reading level for the current original language; the saved level belongs to another language."),
    "proficiency_invalid": ("proficiency", "Choose a current reading level and scheme for this language."),
    "settings_unsupported": ("settings", "This saved mode settings version is not supported."),
    "mode_unknown": ("active_mode", "This saved reading mode is not supported. Basic source text remains readable."),
    "book_deleted": ("document_id", "These preferences belong to a deleted book identity. Reload the library."),
}


def issue(code: str) -> PreferenceIssue:
    field, message = _ISSUES[code]
    return PreferenceIssue(code, field, message)


class PreferenceValidationError(PreferenceError):
    def __init__(self, reasons: tuple[PreferenceIssue, ...]):
        self.reasons = reasons
        super().__init__(reasons[0].code)


@dataclass(frozen=True, slots=True)
class ModeEligibility:
    mode_id: str
    reasons: tuple[PreferenceIssue, ...] = ()

    @property
    def eligible(self) -> bool:
        return not self.reasons


@dataclass(frozen=True, slots=True)
class EffectiveBookPreferences:
    workspace: WorkspacePreferences
    book: BookPreferences
    source: SourceLanguage
    catalog: tuple[ModeDefinition, ...]
    modes: tuple[ModeEligibility, ...]
    proficiency_scheme: ProficiencyScheme | None
    proficiency: ResolvedReadingProficiency | None
    active: ModeEligibility

    @property
    def effective_mode(self) -> str | None:
        # None means unavailable, not a persisted or implicit normal-mode switch.
        return self.active.mode_id if self.active.eligible else None


def project_preferences(workspace: WorkspacePreferences, book: BookPreferences,
                        source: SourceLanguage) -> EffectiveBookPreferences:
    reasons = []
    native = workspace.native_language
    if native is None:
        reasons.append(issue("native_language_required"))
    else:
        try:
            language(native)
        except PreferenceError:
            reasons.append(issue("native_language_invalid"))
    if source.problem:
        reasons.append(issue(source.problem))
    if source.code is not None and source.code == native:
        reasons.append(issue("same_language"))
    scheme = (LanguageProficiencyRegistry.get_scheme(source.code)
              if source.code is not None else None)
    setting = next((entry for entry in book.settings.modes
                    if entry.mode_id == "language_learning"), None)
    proficiency = None
    if setting is None:
        reasons.append(issue("proficiency_required"))
    elif type(setting) is not LanguageLearningSettings or setting.version != "1":
        reasons.append(issue("settings_unsupported"))
    elif source.code is not None and setting.language_code != source.code:
        reasons.append(issue("proficiency_language_mismatch"))
    elif source.code is not None:
        try:
            proficiency = LanguageProficiencyRegistry.resolve_selection(
                language_code=source.code, scheme_id=setting.scheme_id,
                scheme_version=setting.scheme_version, level_id=setting.level_id)
        except ProficiencySelectionError:
            reasons.append(issue("proficiency_invalid"))
    deleted = (issue("book_deleted"),) if book.deleted else ()
    catalog = list_modes()
    policies = {"normal": (), "language_learning": tuple(reasons)}
    modes = tuple(ModeEligibility(mode.mode_id, deleted + policies.get(
        mode.mode_id, (issue("mode_unknown"),))) for mode in catalog)
    active = next((mode for mode in modes if mode.mode_id == book.settings.active_mode),
                  ModeEligibility(book.settings.active_mode, deleted + (issue("mode_unknown"),)))
    return EffectiveBookPreferences(workspace, book, source, catalog, modes, scheme, proficiency, active)
