#!/usr/bin/env python3
"""Regression tests for document-backed reading interaction artifact storage."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from document_structure.structured_document import (  # noqa: E402
    StructuredChapter,
    StructuredDocument,
    StructuredSection,
)
from document_structure.structured_document_artifact_repository import (  # noqa: E402
    StructuredDocumentArtifactRepository,
)
from section_tasks.reading_interaction_artifact_store import (  # noqa: E402
    DocumentReadingInteractionArtifactStore,
)
from section_tasks.reading_interaction_service_contracts import (  # noqa: E402
    ReadingInteractionArtifact,
    ReadingInteractionRequest,
)
from section_tasks.reading_target_resolver import ReadingTargetResolver  # noqa: E402
from shared.task_unit_model import TaskUnit  # noqa: E402


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _document() -> StructuredDocument:
    task_unit = TaskUnit(
        unit_id="unit-1",
        title="Unit One",
        container_title="Section One",
        content="A task unit about assumptions and conclusions.",
        source_section_ids=["section-1"],
        is_fallback_generated=False,
        parent_section_id="section-1",
    )
    section = StructuredSection(
        section_id="section-1",
        section_index=0,
        title="Section One",
        level=2,
        content="A section about assumptions and conclusions.",
        char_start=0,
        char_end=44,
        parent_chapter_id="chapter-1",
        task_units=[task_unit],
    )
    chapter = StructuredChapter(
        chapter_id="chapter-1",
        title="Chapter One",
        level=1,
        chapter_role=None,
        sections=[section],
    )
    return StructuredDocument(
        document_id="doc-1",
        title="Document One",
        source_path=None,
        language="en",
        raw_text="A document about assumptions and conclusions.",
        chapters=[chapter],
    )


def test_document_backed_analysis_store_round_trips_current_artifact() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        repository = StructuredDocumentArtifactRepository(base_dir=temp_dir)
        repository.save_document(_document(), doc_name="book")
        target = ReadingTargetResolver().resolve(
            document=repository.load_document("book"),
            target_level="section",
            section_id="section-1",
        )
        request = ReadingInteractionRequest(
            target=target,
            interaction_type="analysis",
            context_metadata={"context_mode": "full_target", "doc_name": "book"},
        )
        store = DocumentReadingInteractionArtifactStore(repository)

        missing = store.get_analysis_artifact(request)
        artifact = ReadingInteractionArtifact.from_target(
            target=target,
            interaction_type="analysis",
            status="completed",
            payload={
                "summary": "Summary",
                "reasoning": "Reasoning",
                "explanation": "Explanation",
            },
            metadata={"context": {"doc_name": "book"}},
        )
        saved = store.save_analysis_artifact(artifact)
        restored = store.get_analysis_artifact(request)
        persisted_document = repository.load_document("book")

        _assert(missing is None, "missing artifact should read as none")
        _assert(saved.metadata["artifact_id"], "save should stamp artifact id metadata")
        _assert(restored is not None, "saved artifact should be readable")
        _assert(restored.payload == artifact.payload, "payload should round-trip")
        _assert(
            restored.metadata["artifact_id"] == saved.metadata["artifact_id"],
            "artifact id should round-trip",
        )
        _assert(
            "reading_interaction_artifacts"
            in persisted_document.document_task_artifacts.metadata,
            "artifact should persist in document-level metadata",
        )


if __name__ == "__main__":
    test_document_backed_analysis_store_round_trips_current_artifact()
    print("reading interaction artifact store tests passed")
