#!/usr/bin/env python3
"""Regression tests for hard-reparse cleanup of reading interaction artifacts."""

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
from shared.task_artifacts import (  # noqa: E402
    DocumentTaskArtifacts,
    QuizArtifact,
    SummaryArtifact,
    TaskArtifacts,
)
from shared.task_unit_model import TaskUnit  # noqa: E402


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _stale_artifacts() -> TaskArtifacts:
    return TaskArtifacts(
        summary=SummaryArtifact(
            content="stale analysis payload",
            metadata={
                "interaction_type": "analysis",
                "artifact_reference": {"referenced_artifact_ids": ["stale-unit"]},
            },
        ),
        quiz=QuizArtifact(
            items=[
                {
                    "type": "short_answer",
                    "question": "Stale question?",
                    "answer": "Stale answer",
                }
            ],
            metadata={
                "interaction_type": "quiz",
                "artifact_reference": {"referenced_artifact_ids": ["stale-quiz"]},
            },
        ),
    )


def _document_with_stale_interaction_artifacts(
    *,
    doc_id: str = "hard-reparse-doc",
    chapter_id: str = "chapter-old",
    section_id: str = "section-old",
    task_unit_id: str = "unit-old",
) -> StructuredDocument:
    task_unit = TaskUnit(
        unit_id=task_unit_id,
        title="Old Unit",
        container_title=None,
        content="Old task unit content.",
        source_section_ids=[section_id],
        is_fallback_generated=False,
        parent_section_id=section_id,
        task_artifacts=_stale_artifacts(),
    )
    section = StructuredSection(
        section_id=section_id,
        section_index=0,
        title="Old Section",
        level=1,
        content="Old section content.",
        char_start=0,
        char_end=20,
        parent_chapter_id=chapter_id,
        task_units=[task_unit],
        task_artifacts=_stale_artifacts(),
    )
    chapter = StructuredChapter(
        chapter_id=chapter_id,
        title="Old Chapter",
        level=1,
        chapter_role=None,
        sections=[section],
        task_artifacts=_stale_artifacts(),
    )
    return StructuredDocument(
        document_id=doc_id,
        title="Hard Reparse Doc",
        source_path=None,
        language="en",
        raw_text="Old section content.",
        chapters=[chapter],
        document_task_artifacts=DocumentTaskArtifacts(
            chapter_artifacts={
                chapter_id: _stale_artifacts(),
            },
            metadata={
                "critical_thinking_sessions": [
                    {
                        "status": "question_generated",
                        "target_level": "section",
                        "target_id": section_id,
                    }
                ],
                "artifact_reference": {
                    "referenced_artifact_ids": ["stale-analysis", "stale-quiz"],
                },
            },
        ),
    )


def _replacement_document_with_accidentally_copied_artifacts() -> StructuredDocument:
    return _document_with_stale_interaction_artifacts(
        chapter_id="chapter-new",
        section_id="section-new",
        task_unit_id="unit-new",
    )


def test_file_repository_save_reparsed_document_clears_interaction_artifacts() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        repository = StructuredDocumentArtifactRepository(base_dir=temp_dir)
        repository.save_document(
            _document_with_stale_interaction_artifacts(),
            doc_name="hard-reparse-doc",
        )

        repository.save_reparsed_document(
            _replacement_document_with_accidentally_copied_artifacts(),
            doc_name="hard-reparse-doc",
        )

        reloaded = repository.load_document("hard-reparse-doc")
        chapter = reloaded.chapters[0]
        section = chapter.sections[0]
        task_unit = section.task_units[0]

        _assert(
            reloaded.document_task_artifacts is None,
            "hard reparse should clear document-level interaction artifacts",
        )
        _assert(
            chapter.task_artifacts is None,
            "hard reparse should clear chapter-level interaction artifacts",
        )
        _assert(
            section.task_artifacts is None,
            "hard reparse should clear section-level interaction artifacts",
        )
        _assert(
            task_unit.task_artifacts is None,
            "hard reparse should clear task-unit interaction artifacts",
        )
        _assert(
            reloaded.sections == [],
            "hard reparse save should remain hierarchy-only",
        )


def test_cleanup_helper_removes_referenced_artifact_metadata() -> None:
    cleaned = StructuredDocumentArtifactRepository.cleanup_reparsed_document_derived_artifacts(
        _replacement_document_with_accidentally_copied_artifacts()
    )

    _assert(
        cleaned.document_task_artifacts is None,
        "cleanup should remove document artifact references",
    )
    _assert(
        cleaned.chapters[0].sections[0].task_units[0].task_artifacts is None,
        "cleanup should remove task-unit artifact references",
    )


if __name__ == "__main__":
    test_file_repository_save_reparsed_document_clears_interaction_artifacts()
    test_cleanup_helper_removes_referenced_artifact_metadata()
    print("hard reparse reading interaction cleanup tests passed")
