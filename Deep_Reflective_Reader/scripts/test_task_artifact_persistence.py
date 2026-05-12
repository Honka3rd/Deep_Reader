#!/usr/bin/env python3
"""Smoke tests for task-artifact models and repository skeleton."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from document_structure.document_hierarchy_index import get_effective_sections
from document_structure.structured_document import StructuredDocument, StructuredSection
from document_structure.structured_document_artifact_repository import (
    StructuredDocumentArtifactRepository,
)
from document_structure.section_role import SectionRole
from shared.task_artifacts import (
    DocumentTaskArtifacts,
    QuizArtifact,
    SummaryArtifact,
    TaskArtifacts,
)
from shared.task_unit_model import TaskUnit


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def test_legacy_json_compatibility() -> None:
    """Load legacy structured JSON payload without task_artifacts/task_units fields."""
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
    _assert(
        loaded.sections[0].task_units == [],
        "legacy section should load with empty task_units",
    )
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


def test_section_task_units_round_trip() -> None:
    """Round-trip section task_units payload including per-unit task artifacts."""
    section = StructuredSection(
        section_id="section-0",
        section_index=0,
        title="第一章",
        level=1,
        content="第一章\n正文",
        char_start=0,
        char_end=6,
        section_role=SectionRole.MAIN_BODY,
        task_units=[
            TaskUnit(
                unit_id="task-unit-0",
                title="第一章 (Part 1)",
                container_title=None,
                content="正文第一段",
                source_section_ids=["section-0"],
                is_fallback_generated=True,
                task_artifacts=TaskArtifacts(
                    summary=SummaryArtifact(content="unit summary"),
                ),
            ),
            TaskUnit(
                unit_id="task-unit-1",
                title="第一章 (Part 2)",
                container_title=None,
                content="正文第二段",
                source_section_ids=["section-0"],
                is_fallback_generated=False,
                task_artifacts=None,
            ),
        ],
    )

    restored = StructuredSection.from_dict(section.to_dict())
    _assert(len(restored.task_units) == 2, "task_units count should round-trip")
    _assert(restored.task_units[0].unit_id == "task-unit-0", "first task unit id mismatch")
    _assert(restored.task_units[1].content == "正文第二段", "second task unit content mismatch")
    _assert(
        restored.task_units[0].task_artifacts is not None
        and restored.task_units[0].task_artifacts.summary is not None
        and restored.task_units[0].task_artifacts.summary.content == "unit summary",
        "task unit artifact should round-trip",
    )


def test_repository_atomic_save_smoke() -> None:
    """Copy one small structured JSON, update section artifact, save atomically, reload."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_root = Path(temp_dir)
        target_path = temp_root / "artifact-doc.structured.json"

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
        target_path.write_text(
            base_document.to_json(include_legacy_sections=True),
            encoding="utf-8",
        )

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
        updated_sections = get_effective_sections(updated_document)
        _assert(
            updated_sections[0].task_artifacts is not None,
            "updated document should contain section artifacts",
        )

        reloaded = repository.load_document("artifact-doc")
        reloaded_sections = get_effective_sections(reloaded)
        _assert(
            reloaded_sections[0].task_artifacts is not None,
            "reloaded document should persist section artifacts",
        )
        _assert(
            reloaded_sections[0].task_artifacts.summary is not None
            and reloaded_sections[0].task_artifacts.summary.content == "atomic summary",
            "persisted artifact content mismatch",
        )


def test_repository_update_section_task_units() -> None:
    """Persist section task_units list and verify reloaded structured document."""
    with tempfile.TemporaryDirectory() as temp_dir:
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
        target_path = Path(temp_dir) / "artifact-doc.structured.json"
        target_path.write_text(
            base_document.to_json(include_legacy_sections=True),
            encoding="utf-8",
        )

        repository = StructuredDocumentArtifactRepository(base_dir=temp_dir)
        persisted_units = [
            TaskUnit(
                unit_id="task-unit-0",
                title="Chapter 1 (Part 1)",
                container_title=None,
                content="chunk 1",
                source_section_ids=["section-0"],
                is_fallback_generated=True,
            ),
            TaskUnit(
                unit_id="task-unit-1",
                title="Chapter 1 (Part 2)",
                container_title=None,
                content="chunk 2",
                source_section_ids=["section-0"],
                is_fallback_generated=False,
            ),
        ]
        repository.update_section_task_units(
            doc_name="artifact-doc",
            section_id="section-0",
            task_units=persisted_units,
        )

        reloaded = repository.load_document("artifact-doc")
        reloaded_sections = get_effective_sections(reloaded)
        _assert(len(reloaded_sections[0].task_units) == 2, "section task_units should persist")
        _assert(
            reloaded_sections[0].task_units[0].unit_id == "task-unit-0",
            "persisted task unit id mismatch",
        )


def test_repository_update_task_unit_artifacts() -> None:
    """Update one task-unit artifact and ensure other units stay unchanged."""
    with tempfile.TemporaryDirectory() as temp_dir:
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
                    task_units=[
                        TaskUnit(
                            unit_id="task-unit-0",
                            title="u0",
                            container_title=None,
                            content="chunk 0",
                            source_section_ids=["section-0"],
                            is_fallback_generated=True,
                        ),
                        TaskUnit(
                            unit_id="task-unit-1",
                            title="u1",
                            container_title=None,
                            content="chunk 1",
                            source_section_ids=["section-0"],
                            is_fallback_generated=False,
                        ),
                    ],
                )
            ],
        )
        target_path = Path(temp_dir) / "artifact-doc.structured.json"
        target_path.write_text(
            base_document.to_json(include_legacy_sections=True),
            encoding="utf-8",
        )
        repository = StructuredDocumentArtifactRepository(base_dir=temp_dir)

        repository.update_task_unit_artifacts(
            doc_name="artifact-doc",
            task_unit_id="task-unit-1",
            artifacts=TaskArtifacts(
                summary=SummaryArtifact(content="unit-1 summary"),
            ),
        )

        reloaded = repository.load_document("artifact-doc")
        reloaded_sections = get_effective_sections(reloaded)
        unit0 = reloaded_sections[0].task_units[0]
        unit1 = reloaded_sections[0].task_units[1]
        _assert(unit0.task_artifacts is None, "other task unit should remain unchanged")
        _assert(
            unit1.task_artifacts is not None
            and unit1.task_artifacts.summary is not None
            and unit1.task_artifacts.summary.content == "unit-1 summary",
            "target task unit artifact should persist",
        )


def test_repository_update_task_unit_artifacts_unknown_id() -> None:
    """Unknown task_unit_id should raise ValueError."""
    with tempfile.TemporaryDirectory() as temp_dir:
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
                    task_units=[
                        TaskUnit(
                            unit_id="task-unit-0",
                            title="u0",
                            container_title=None,
                            content="chunk 0",
                            source_section_ids=["section-0"],
                            is_fallback_generated=True,
                        ),
                    ],
                )
            ],
        )
        target_path = Path(temp_dir) / "artifact-doc.structured.json"
        target_path.write_text(
            base_document.to_json(include_legacy_sections=True),
            encoding="utf-8",
        )
        repository = StructuredDocumentArtifactRepository(base_dir=temp_dir)
        try:
            repository.update_task_unit_artifacts(
                doc_name="artifact-doc",
                task_unit_id="unknown-unit",
                artifacts=TaskArtifacts(summary=SummaryArtifact(content="x")),
            )
        except ValueError:
            return
        raise AssertionError("unknown task_unit_id should raise ValueError")


def test_repository_update_task_unit_artifacts_duplicate_id() -> None:
    """Duplicate task_unit_id in one document must be rejected defensively."""
    with tempfile.TemporaryDirectory() as temp_dir:
        base_document = StructuredDocument(
            document_id="artifact-doc",
            title="Artifact Doc",
            source_path=None,
            language="en",
            raw_text="content",
            sections=[
                StructuredSection(
                    section_id="section-0",
                    section_index=0,
                    title="S0",
                    level=1,
                    content="chunk A",
                    char_start=0,
                    char_end=7,
                    section_role=SectionRole.MAIN_BODY,
                    task_units=[
                        TaskUnit(
                            unit_id="task-unit-0",
                            title="u0",
                            container_title=None,
                            content="chunk 0",
                            source_section_ids=["section-0"],
                            is_fallback_generated=False,
                        ),
                    ],
                ),
                StructuredSection(
                    section_id="section-1",
                    section_index=1,
                    title="S1",
                    level=1,
                    content="chunk B",
                    char_start=8,
                    char_end=15,
                    section_role=SectionRole.MAIN_BODY,
                    task_units=[
                        TaskUnit(
                            unit_id="task-unit-0",
                            title="u1",
                            container_title=None,
                            content="chunk 1",
                            source_section_ids=["section-1"],
                            is_fallback_generated=False,
                        ),
                    ],
                ),
            ],
        )
        target_path = Path(temp_dir) / "artifact-doc.structured.json"
        target_path.write_text(
            base_document.to_json(include_legacy_sections=True),
            encoding="utf-8",
        )
        repository = StructuredDocumentArtifactRepository(base_dir=temp_dir)

        try:
            repository.update_task_unit_artifacts(
                doc_name="artifact-doc",
                task_unit_id="task-unit-0",
                artifacts=TaskArtifacts(summary=SummaryArtifact(content="dup update")),
            )
        except ValueError as error:
            message = str(error)
            _assert("duplicate task_unit_id" in message, "error should mention duplicate id")
            _assert("match_count=2" in message, "error should include duplicate match count")
            reloaded = repository.load_document("artifact-doc")
            reloaded_sections = get_effective_sections(reloaded)
            _assert(
                all(
                    task_unit.task_artifacts is None
                    for section in reloaded_sections
                    for task_unit in section.task_units
                ),
                "duplicate-id defensive failure should not update any task unit",
            )
            return
        raise AssertionError("duplicate task_unit_id should raise ValueError")


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
    test_section_task_units_round_trip()
    test_repository_atomic_save_smoke()
    test_repository_update_section_task_units()
    test_repository_update_task_unit_artifacts()
    test_repository_update_task_unit_artifacts_unknown_id()
    test_repository_update_task_unit_artifacts_duplicate_id()
    test_task_unit_artifact_smoke()
    test_document_level_artifacts_round_trip()

    print(
        json.dumps(
            {
                "status": "ok",
                "tests": [
                    "legacy_json_compatibility",
                    "artifact_round_trip",
                    "section_task_units_round_trip",
                    "repository_atomic_save_smoke",
                    "repository_update_section_task_units",
                    "repository_update_task_unit_artifacts",
                    "repository_update_task_unit_artifacts_unknown_id",
                    "repository_update_task_unit_artifacts_duplicate_id",
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
