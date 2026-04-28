#!/usr/bin/env python3
"""Smoke tests for task-artifact models and repository skeleton."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from document_structure.structured_document import StructuredDocument, StructuredSection
from document_structure.structured_document_artifact_repository import (
    StructuredDocumentArtifactRepository,
)
from document_structure.section_role import SectionRole
from section_tasks.task_unit import TaskUnit
from shared.task_artifacts import (
    DocumentTaskArtifacts,
    QuizArtifact,
    SummaryArtifact,
    TaskArtifacts,
)


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def test_legacy_json_compatibility() -> None:
    """Load legacy structured JSON payload without task_artifacts fields."""
    legacy_payload = json.dumps(
        {
            "document_id": "legacy-doc",
            "title": "Legacy",
            "source_path": None,
            "language": "en",
            "raw_text": "Chapter 1\nlegacy content",
            "sections": [
                {
                    "section_id": "section-0",
                    "section_index": 0,
                    "title": "Chapter 1",
                    "level": 1,
                    "content": "Chapter 1\nlegacy content",
                    "char_start": 0,
                    "char_end": 24,
                    "container_title": None,
                    "section_role": "main_body",
                }
            ],
            "structure_error_code": None,
            "structure_error_message": None,
        },
        ensure_ascii=False,
    )
    loaded = StructuredDocument.from_json(legacy_payload)
    _assert(loaded.sections[0].task_artifacts is None, "legacy section should load without artifacts")
    _assert(
        loaded.document_task_artifacts is None,
        "legacy document should load without document_task_artifacts",
    )


def test_artifact_round_trip() -> None:
    """Serialize/deserialize section artifact payload and verify consistency."""
    artifact = TaskArtifacts(
        summary=SummaryArtifact(
            content="summary text",
            language="zh",
            generated_at="2026-04-28T10:00:00Z",
            prompt_version="summary.v1",
            task_unit_split_mode="semantic_safe",
            semantic_top_k_candidates=3,
        ),
        quiz=QuizArtifact(
            items=[
                {
                    "question_id": "q1",
                    "question_text": "Who is the protagonist?",
                    "answer_text": "Xu Sanguan",
                }
            ],
            language="zh",
            generated_at="2026-04-28T10:01:00Z",
            prompt_version="quiz.v1",
            quiz_schema_version="quiz.schema.v1",
            task_unit_split_mode="semantic_safe",
            semantic_top_k_candidates=3,
        ),
    )
    section = StructuredSection(
        section_id="section-0",
        section_index=0,
        title="第一章",
        level=1,
        content="第一章\n正文",
        char_start=0,
        char_end=6,
        section_role=SectionRole.MAIN_BODY,
        task_artifacts=artifact,
    )
    payload = section.to_dict()
    restored = StructuredSection.from_dict(payload)
    _assert(restored.task_artifacts is not None, "restored section should keep task artifacts")
    _assert(
        restored.task_artifacts.summary is not None
        and restored.task_artifacts.summary.content == "summary text",
        "summary content should round-trip",
    )
    _assert(
        restored.task_artifacts.quiz is not None
        and len(restored.task_artifacts.quiz.items) == 1,
        "quiz items should round-trip",
    )


def test_repository_atomic_save_smoke() -> None:
    """Copy one small structured JSON, update section artifact, save atomically, reload."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_root = Path(temp_dir)
        source_path = temp_root / "source.structured.json"
        copied_path = temp_root / "artifact-doc.structured.json"

        base_document = StructuredDocument(
            document_id="artifact-doc",
            title="Artifact Doc",
            source_path=None,
            language="en",
            raw_text="Chapter 1\ncontent",
            sections=[
                StructuredSection(
                    section_id="section-0",
                    section_index=0,
                    title="Chapter 1",
                    level=1,
                    content="Chapter 1\ncontent",
                    char_start=0,
                    char_end=17,
                    section_role=SectionRole.MAIN_BODY,
                )
            ],
        )
        source_path.write_text(base_document.to_json(), encoding="utf-8")
        shutil.copy(source_path, copied_path)

        repository = StructuredDocumentArtifactRepository(base_dir=temp_dir)
        artifacts = TaskArtifacts(
            summary=SummaryArtifact(
                content="atomic summary",
                language="en",
                prompt_version="summary.v1",
            )
        )
        updated_document = repository.update_section_artifacts(
            doc_name="artifact-doc",
            section_id="section-0",
            artifacts=artifacts,
        )
        _assert(
            updated_document.sections[0].task_artifacts is not None,
            "updated document should contain section artifacts",
        )

        reloaded = repository.load_document("artifact-doc")
        _assert(
            reloaded.sections[0].task_artifacts is not None,
            "reloaded document should persist section artifacts",
        )
        _assert(
            reloaded.sections[0].task_artifacts.summary is not None
            and reloaded.sections[0].task_artifacts.summary.content == "atomic summary",
            "persisted artifact content mismatch",
        )


def test_task_unit_artifact_smoke() -> None:
    """Construct TaskUnit with task_artifacts and round-trip through dict payload."""
    task_unit = TaskUnit(
        unit_id="task-unit-0",
        title="Task Unit 1",
        container_title=None,
        content="unit content",
        source_section_ids=["section-0"],
        is_fallback_generated=False,
        task_artifacts=TaskArtifacts(
            summary=SummaryArtifact(content="task unit summary"),
        ),
    )
    restored = TaskUnit.from_dict(task_unit.to_dict())
    _assert(restored.task_artifacts is not None, "task unit artifacts should round-trip")
    _assert(
        restored.task_artifacts.summary is not None
        and restored.task_artifacts.summary.content == "task unit summary",
        "task unit summary artifact mismatch",
    )


def test_document_level_artifacts_round_trip() -> None:
    """Validate document-level artifact container serialization compatibility."""
    document = StructuredDocument(
        document_id="doc-with-artifacts",
        title="Doc",
        source_path=None,
        language="en",
        raw_text="raw",
        sections=[],
        document_task_artifacts=DocumentTaskArtifacts(
            chapter_artifacts={
                "chapter-1": TaskArtifacts(
                    summary=SummaryArtifact(content="chapter summary"),
                )
            },
            metadata={"layout_version": "v1"},
        ),
    )
    restored = StructuredDocument.from_dict(document.to_dict())
    _assert(
        restored.document_task_artifacts is not None,
        "document_task_artifacts should round-trip",
    )
    _assert(
        "chapter-1" in restored.document_task_artifacts.chapter_artifacts,
        "chapter artifact key should persist",
    )


def main() -> None:
    test_legacy_json_compatibility()
    test_artifact_round_trip()
    test_repository_atomic_save_smoke()
    test_task_unit_artifact_smoke()
    test_document_level_artifacts_round_trip()

    print(
        json.dumps(
            {
                "status": "ok",
                "tests": [
                    "legacy_json_compatibility",
                    "artifact_round_trip",
                    "repository_atomic_save_smoke",
                    "task_unit_artifact_smoke",
                    "document_level_artifacts_round_trip",
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
