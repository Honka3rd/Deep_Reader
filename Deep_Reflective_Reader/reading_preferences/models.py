"""Immutable preferences and mode contracts, with no user/profile/storage lookup."""

from dataclasses import dataclass
from typing import ClassVar

from language.language_profile_registry import LanguageProfileRegistry
from language.language_proficiency_registry import LanguageProficiencyRegistry


class PreferenceError(ValueError):
    pass


class PreferenceConflict(PreferenceError):
    pass


class PreferenceStorageError(RuntimeError):
    pass


class PreferenceCommitUnknown(PreferenceStorageError):
    """Storage commit may have succeeded; read/reconcile before any retry."""


def identifier(value: str) -> str:
    # Opaque, exact server-resolved identity; no lossy path/name normalization.
    if (not isinstance(value, str) or not value or value != value.strip()
            or not value.isprintable() or value in (".", "..")
            or "/" in value or "\\" in value):
        raise PreferenceError("preference_identity_invalid")
    return value


def revision(value: int) -> int:
    if type(value) is not int or value < 0:
        raise PreferenceError("preference_revision_invalid")
    return value


def language(value: str | None) -> None:
    if value is not None and (not isinstance(value, str) or value == "unknown"
                             or value not in LanguageProfileRegistry.get_supported_language_codes()):
        raise PreferenceError("preference_language_invalid")


@dataclass(frozen=True, slots=True)
class WorkspaceScope:
    """Constructed by server composition from its storage namespace, not a DTO.

    A workspace is single-person/shared installation scope, not a fake user.
    Future user ownership requires a new scope contract and explicit migration.
    """
    namespace: str
    kind: ClassVar[str] = "workspace"

    def __post_init__(self):
        identifier(self.namespace)


@dataclass(frozen=True, slots=True)
class ModeDefinition:
    mode_id: str
    label: str
    description: str
    settings_version: str


MODES = (
    ModeDefinition("normal", "Normal reading", "Read with the usual reading interactions.", "1"),
    ModeDefinition("language_learning", "Language learning", "Read sentence translations and vocabulary guidance.", "1"),
)


def list_modes() -> tuple[ModeDefinition, ...]:
    """Pure catalog, not book eligibility or an executable UI description."""
    return MODES


@dataclass(frozen=True, slots=True)
class LanguageLearningSettings:
    # Language binds the proficiency choice; it is not source/profile authority.
    language_code: str
    scheme_id: str
    scheme_version: str
    level_id: str
    version: str = "1"
    mode_id: ClassVar[str] = "language_learning"


@dataclass(frozen=True, slots=True)
class PreservedModeSettings:
    """Read-only representation of unknown stored modes/settings versions.

    Canonical JSON is immutable; writes may only retain the identical stored
    value. This is not a public arbitrary-JSON settings submission contract.
    """
    mode_id: str
    version: str
    payload_json: str


ModeSettings = LanguageLearningSettings | PreservedModeSettings


@dataclass(frozen=True, slots=True)
class BookSettings:
    active_mode: str = "normal"
    source_language_override: str | None = None
    modes: tuple[ModeSettings, ...] = ()


@dataclass(frozen=True, slots=True)
class WorkspacePreferences:
    scope: WorkspaceScope
    revision: int = 0
    native_language: str | None = None


@dataclass(frozen=True, slots=True)
class BookPreferences:
    scope: WorkspaceScope
    document_id: str
    revision: int = 0
    settings: BookSettings = BookSettings()
    deleted: bool = False


def validate_book_write(settings: BookSettings, previous: BookSettings,
                        native_language: str | None, resolved_source_language: str | None) -> None:
    """Lower-level write invariants only. LLRN-09 resolves actual profile/source.

    The internal caller must supply a currently resolved source and later enforce
    its document revision. File preferences cannot transact with document files.
    """
    if type(settings) is not BookSettings or type(settings.modes) is not tuple:
        raise PreferenceError("preference_settings_invalid")
    if settings.active_mode not in tuple(mode.mode_id for mode in MODES):
        raise PreferenceError("preference_mode_unknown")
    language(settings.source_language_override)
    seen = set()
    for entry in settings.modes:
        if type(entry) not in (LanguageLearningSettings, PreservedModeSettings):
            raise PreferenceError("preference_settings_invalid")
        identifier(entry.mode_id)
        if entry.mode_id in seen:
            raise PreferenceError("preference_settings_duplicate")
        seen.add(entry.mode_id)
        if entry == next((old for old in previous.modes if old.mode_id == entry.mode_id), None):
            # Inactive settings remain readable even after a scheme version change.
            if settings.active_mode != entry.mode_id:
                continue
        if type(entry) is not LanguageLearningSettings or entry.version != "1":
            raise PreferenceError("preference_settings_unsupported")
        LanguageProficiencyRegistry.resolve_selection(
            language_code=entry.language_code, scheme_id=entry.scheme_id,
            scheme_version=entry.scheme_version, level_id=entry.level_id)
    # Switching mode must not silently discard another mode's saved settings.
    if not {entry.mode_id for entry in previous.modes}.issubset(seen):
        raise PreferenceError("preference_settings_retention_required")
    if settings.active_mode == "language_learning":
        learning = next((e for e in settings.modes if e.mode_id == "language_learning"), None)
        if type(learning) is not LanguageLearningSettings:
            raise PreferenceError("preference_learning_settings_required")
        language(native_language)
        language(resolved_source_language)
        if native_language is None or resolved_source_language is None:
            raise PreferenceError("preference_languages_required")
        if native_language == resolved_source_language:
            raise PreferenceError("preference_same_language")
        if (learning.language_code != resolved_source_language
                or settings.source_language_override not in (None, resolved_source_language)):
            raise PreferenceError("preference_source_mismatch")
