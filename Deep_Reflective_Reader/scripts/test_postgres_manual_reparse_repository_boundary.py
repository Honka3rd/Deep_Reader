#!/usr/bin/env python3
"""Regression coverage for PostgreSQL manual reparse save boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from db.postgres_structured_document_artifact_repository import (
    PostgresStructuredDocumentArtifactRepository,
)
from document_structure.structured_document import (
    StructuredChapter,
    StructuredDocument,
    StructuredSection,
)


@dataclass(frozen=True)
class _Target:
    namespace: str
    document_name: str


class _FakePostgresStore:
    def __init__(self) -> None:
        self.save_calls: list[tuple[StructuredDocument, _Target, bool]] = []

    def target_for_doc_name(self, doc_name: str) -> _Target:
        return _Target(namespace="default", document_name=doc_name)

    def save(
        self,
        *,
        document: StructuredDocument,
        target: _Target,
        replace_existing_hierarchy: bool,
    ) -> None:
        self.save_calls.append((document, target, replace_existing_hierarchy))

    def load(self, target: _Target) -> StructuredDocument:
        raise AssertionError(f"unexpected load call: {target}")

    def list_documents(self, query=None, limit=50):  # noqa: ANN001
        raise AssertionError((query, limit))


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _document(doc_name: str, chapter_id: str) -> StructuredDocument:
    section = StructuredSection(
        section_id=f"{chapter_id}-section",
        section_index=0,
        title="Manual Section",
        level=1,
        content="Manual body",
        char_start=0,
        char_end=11,
        parent_chapter_id=chapter_id,
    )
    chapter = StructuredChapter(
        chapter_id=chapter_id,
        title="Manual Chapter",
        level=1,
        chapter_role=None,
        sections=[section],
    )
    return StructuredDocument(
        document_id=doc_name,
        title=doc_name,
        source_path=doc_name,
        language="en",
        raw_text="Manual body",
        chapters=[chapter],
        parse_provenance={
            "requested_parser_mode": "manual_structure",
            "effective_parser_mode": "manual_structure_projection",
        },
    )


def test_postgres_repository_keeps_save_boundaries_distinct() -> None:
    store = _FakePostgresStore()
    repository = PostgresStructuredDocumentArtifactRepository(store)  # type: ignore[arg-type]
    document = _document("Book.pdf", "manual-chapter-0")

    repository.save_document(document, doc_name="Book.pdf")
    repository.save_reparsed_document(document, doc_name="Book.pdf")

    _assert(len(store.save_calls) == 2, "both saves should reach the store")
    _assert(
        store.save_calls[0][2] is False,
        "generic PostgreSQL save must preserve current hierarchy/version",
    )
    _assert(
        store.save_calls[1][2] is True,
        "manual reparse save must request hierarchy replacement",
    )
    _assert(
        [call[1].document_name for call in store.save_calls]
        == ["Book.pdf", "Book.pdf"],
        "repository should resolve both saves through doc_name target",
    )


def main() -> None:
    test_postgres_repository_keeps_save_boundaries_distinct()
    print("postgres manual reparse repository boundary regression passed")


if __name__ == "__main__":
    main()
