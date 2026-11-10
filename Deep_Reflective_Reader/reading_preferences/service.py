"""Validated settings service; persistence CAS remains repository-owned.

Profile language is advisory source selection, not generation authorization.
Lifecycle/orchestration must revalidate profile/source at admission and commit.
No read/save calls generation, preparation, profile mutation or cache cleanup.
"""

from dataclasses import replace

from reading_preferences.effective import (
    EffectiveBookPreferences, PreferenceValidationError, issue, project_preferences,
)
from reading_preferences.models import (
    BookPreferences, BookSettings, PreferenceConflict, PreferenceError, WorkspacePreferences,
    identifier, language, list_modes, revision,
)
from reading_preferences.profile_source import ExistingProfileLanguageReader, SourceLanguage
from reading_preferences.repository import ReadingPreferencesRepository
from reading_preferences.serialization import decode_settings, encode_settings


class ReadingPreferencesService:
    def __init__(self, repository: ReadingPreferencesRepository,
                 profiles: ExistingProfileLanguageReader):
        self._repository = repository
        self._profiles = profiles

    @staticmethod
    def catalog():
        return list_modes()

    def read_workspace(self) -> WorkspacePreferences:
        return self._repository.read_workspace()

    def save_workspace(self, native_language: str | None, *, expected_revision: int,
                       mutation_id: str) -> WorkspacePreferences:
        return self._repository.save_workspace(native_language, expected_revision=expected_revision,
                                               mutation_id=mutation_id)

    def _source(self, document_id: str, settings: BookSettings) -> SourceLanguage:
        override = settings.source_language_override
        if override is not None:
            try:
                language(override)
            except PreferenceError:
                return SourceLanguage(None, "override", "source_language_invalid")
            return SourceLanguage(override, "override")
        return self._profiles.read(self._repository.scope, document_id)

    @property
    def scope(self):
        return self._repository.scope

    def resolve_snapshot(self, workspace: WorkspacePreferences, book: BookPreferences) -> EffectiveBookPreferences:
        """Resolve an owner-locked snapshot without reacquiring preference locks.

        Only the injected existing-profile reader may perform read-only IO here.
        It must not acquire document/preference locks or open a write transaction.
        """
        if workspace.scope != self.scope or book.scope != self.scope:
            raise PreferenceError("preference_scope_mismatch")
        return project_preferences(workspace, book, self._source(book.document_id, book.settings))

    def read_book(self, document_id: str) -> EffectiveBookPreferences:
        identifier(document_id)
        workspace = self._repository.read_workspace()
        book = self._repository.read_book(document_id)
        source = self._source(document_id, book.settings)
        # Read-only optimistic snapshot: no locks/files/default writes, no retry
        # loop under contention. Both revisions are monotonic, including ABA.
        if (workspace != self._repository.read_workspace()
                or book != self._repository.read_book(document_id)):
            raise PreferenceConflict("preference_snapshot_changed")
        return project_preferences(workspace, book, source)

    def save_book(self, document_id: str, settings: BookSettings, *, expected_revision: int,
                  expected_workspace_revision: int, mutation_id: str) -> EffectiveBookPreferences:
        identifier(document_id)
        revision(expected_revision)
        revision(expected_workspace_revision)
        identifier(mutation_id)
        if type(settings) is not BookSettings or type(settings.modes) is not tuple:
            raise PreferenceError("preference_settings_invalid")
        if settings.active_mode not in tuple(mode.mode_id for mode in list_modes()):
            raise PreferenceError("preference_mode_unknown")
        language(settings.source_language_override)
        decode_settings(encode_settings(settings))
        workspace = self._repository.read_workspace()
        if workspace.revision != expected_workspace_revision:
            raise PreferenceConflict("preference_revision_conflict")
        source = self._source(document_id, settings)
        if settings.active_mode == "language_learning" and source.problem:
            raise PreferenceValidationError((issue(source.problem),))
        projected = project_preferences(workspace, BookPreferences(
            self._repository.scope, document_id, settings=settings), source)
        if not projected.active.eligible:
            raise PreferenceValidationError(projected.active.reasons)
        # Mode/settings/native validation and both CAS checks are repeated inside
        # the repository write boundary, including durable idempotent replay.
        # Normal reading does not depend on profile availability or learning fields.
        book = self._repository.save_book(
            document_id, settings, expected_revision=expected_revision,
            expected_workspace_revision=expected_workspace_revision,
            resolved_source_language=source.code if settings.active_mode == "language_learning" else None,
            mutation_id=mutation_id)
        # No I/O after commit: a failed reread must not disguise a successful save.
        return replace(projected, book=book)
