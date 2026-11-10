"""Storage-neutral preference port. Scope is bound by server composition.

All mutations are explicit CAS commands. The most recent command per affected
record is replayable by mutation_id; older/unknown outcomes require a read and
conflict handling. No blind replay after intervening revisions. Rename/delete
apply to preferences only; the app must coordinate document lifecycle separately.
"""

from typing import Protocol

from reading_preferences.models import BookPreferences, BookSettings, WorkspacePreferences, WorkspaceScope


class ReadingPreferencesRepository(Protocol):
    @property
    def scope(self) -> WorkspaceScope: ...

    def read_workspace(self) -> WorkspacePreferences: ...

    def read_book(self, document_id: str) -> BookPreferences: ...

    def save_workspace(self, native_language: str | None, *, expected_revision: int,
                       mutation_id: str) -> WorkspacePreferences: ...

    def save_book(self, document_id: str, settings: BookSettings, *, expected_revision: int,
                  expected_workspace_revision: int, resolved_source_language: str | None,
                  mutation_id: str) -> BookPreferences: ...

    def delete_book(self, document_id: str, *, expected_revision: int,
                    mutation_id: str) -> BookPreferences: ...

    def rename_book(self, document_id: str, new_document_id: str, *, expected_revision: int,
                    expected_destination_revision: int, mutation_id: str) -> BookPreferences: ...
