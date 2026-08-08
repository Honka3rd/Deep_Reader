#!/usr/bin/env python3
"""Regression test for structured storage URI handling in prepare_and_load."""

from __future__ import annotations

from typing import Any

from document_preparation.document_preparation_pipeline import DocumentPreparationPipeline
from document_preparation.prepared_document_assets import PreparedDocumentAssets
from document_preparation.preparation_mode import PreparationMode
from document_structure.structured_document import (
    StructuredChapter,
    StructuredDocument,
    StructuredSection,
)


class _FakeStructuredDocumentStore:
    def __init__(self, document: StructuredDocument):
        self.document = document
        self.exists_targets: list[str] = []
        self.load_targets: list[str] = []

    def exists(self, target: str) -> bool:
        self.exists_targets.append(target)
        return target == "postgres://structured/default/Madame Bovary"

    def load(self, target: str) -> StructuredDocument:
        self.load_targets.append(target)
        if target != "postgres://structured/default/Madame Bovary":
            raise AssertionError(f"unexpected load target: {target}")
        return self.document


def _fake_document() -> StructuredDocument:
    section = StructuredSection(
        section_id="section-0",
        section_index=0,
        title="Chapter One",
        level=1,
        content="Madame Bovary regression section.",
        char_start=0,
        char_end=34,
    )
    chapter = StructuredChapter(
        chapter_id="chapter-0",
        title="Chapter One",
        level=1,
        chapter_role=None,
        sections=[section],
    )
    return StructuredDocument(
        document_id="Madame Bovary",
        title="Madame Bovary",
        source_path=None,
        language="en",
        raw_text=section.content,
        chapters=[chapter],
    )


def main() -> None:
    document = _fake_document()
    store = _FakeStructuredDocumentStore(document)
    pipeline = DocumentPreparationPipeline(
        loader_factory=None,  # type: ignore[arg-type]
        language_detector=None,  # type: ignore[arg-type]
        structured_document_builder=None,  # type: ignore[arg-type]
        structured_document_store=store,  # type: ignore[arg-type]
        node_provider=None,  # type: ignore[arg-type]
        faiss_index_builder=None,  # type: ignore[arg-type]
        faiss_index_store=None,  # type: ignore[arg-type]
        fingerprint_handler=None,  # type: ignore[arg-type]
        profile_builder=None,  # type: ignore[arg-type]
        profile_store=None,  # type: ignore[arg-type]
        bundle_provider=None,  # type: ignore[arg-type]
    )

    assets = PreparedDocumentAssets(
        doc_name="Madame Bovary",
        raw_text=document.raw_text,
        language="en",
        structured_document_ready=True,
        faiss_ready=False,
        profile_ready=True,
        bundle_ready=False,
        structured_document_path="postgres://structured/default/Madame Bovary",
        faiss_namespace=None,
        errors=[],
    )

    def fake_prepare(*args: Any, **kwargs: Any) -> PreparedDocumentAssets:
        return assets

    pipeline.prepare = fake_prepare  # type: ignore[method-assign]

    result = pipeline.prepare_and_load(
        doc_name="Madame Bovary",
        mode=PreparationMode.BASE,
        force_rebuild=False,
    )

    assert result.structured_document is document
    assert result.assets.errors == []
    assert store.exists_targets == ["postgres://structured/default/Madame Bovary"]
    assert store.load_targets == ["postgres://structured/default/Madame Bovary"]
    print("postgres structured URI prepare_and_load regression passed")


if __name__ == "__main__":
    main()
