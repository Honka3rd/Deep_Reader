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
from section_tasks.ancestor_summary_context import (  # noqa: E402
    AncestorSummaryArtifact,
    AncestorSummaryRequirement,
    AncestorSummaryTarget,
)
from section_tasks.reading_interaction_artifact_store import (  # noqa: E402
    DocumentAncestorSummaryArtifactStore,
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


def test_document_backed_quiz_store_round_trips_current_artifact() -> None:
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
            interaction_type="quiz",
            context_metadata={"context_mode": "full_target", "doc_name": "book"},
        )
        store = DocumentReadingInteractionArtifactStore(repository)

        missing = store.get_quiz_artifact(request)
        artifact = ReadingInteractionArtifact.from_target(
            target=target,
            interaction_type="quiz",
            status="completed",
            payload={
                "items": [
                    {
                        "item_id": "q1",
                        "type": "short_answer",
                        "question": "What does the unit discuss?",
                        "answer": "Assumptions and conclusions.",
                    }
                ]
            },
            metadata={"context": {"doc_name": "book"}},
        )
        saved = store.save_quiz_artifact(artifact)
        restored = store.get_quiz_artifact(request)

        _assert(missing is None, "missing quiz should read as none")
        _assert(saved.metadata["artifact_id"], "quiz save should stamp artifact id")
        _assert(restored is not None, "saved quiz should be readable")
        _assert(restored.payload == artifact.payload, "quiz payload should round-trip")
        _assert(
            restored.metadata["artifact_id"] == saved.metadata["artifact_id"],
            "quiz artifact id should round-trip",
        )


def test_document_backed_ancestor_summary_store_round_trips_summary_artifact() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        repository = StructuredDocumentArtifactRepository(base_dir=temp_dir)
        repository.save_document(_document(), doc_name="book")
        requirement = AncestorSummaryRequirement(
            doc_name="book",
            target=AncestorSummaryTarget(
                target_level="task_unit",
                target_id="unit-1",
                document_id="doc-1",
                chapter_id="chapter-1",
                section_id="section-1",
                task_unit_id="unit-1",
                title="Unit One",
            ),
            prompt_instruction_version="ancestor_summary_prompt_v1",
            source_structure_version=3,
            source_hash="hash-unit-1",
            document_language="en",
        )
        summary = AncestorSummaryArtifact(
            target=requirement.target,
            summary="The unit introduces assumptions and consequences.",
            schema_version=requirement.schema_version,
            prompt_instruction_version=requirement.prompt_instruction_version,
            source_structure_version=requirement.source_structure_version,
            source_hash=requirement.source_hash,
            document_language=requirement.document_language,
            generated_at="2026-10-12T12:00:00+00:00",
            metadata={"context_mode": "ancestor_summary"},
        )
        store = DocumentAncestorSummaryArtifactStore(repository)

        missing = store.get_ancestor_summary(requirement)
        saved = store.save_ancestor_summary(requirement, summary)
        restored = store.get_ancestor_summary(requirement)
        persisted_document = repository.load_document("book")
        metadata = persisted_document.document_task_artifacts.metadata

        _assert(missing is None, "missing ancestor summary should read as none")
        _assert(saved.artifact_id is not None, "save should stamp artifact id")
        _assert(restored is not None, "saved summary should be readable")
        _assert(restored.summary == summary.summary, "summary should round-trip")
        _assert(restored.target.task_unit_id == "unit-1", "target should round-trip")
        _assert(
            "ancestor_summary_artifacts" in metadata,
            "summary should persist under dedicated metadata key",
        )
        _assert(
            "reading_interaction_artifacts" not in metadata,
            "ancestor summary persistence should not create interaction artifacts",
        )


def test_content_block_current_artifacts_do_not_collide_with_parent_task_unit() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        repository = StructuredDocumentArtifactRepository(base_dir=temp_dir)
        repository.save_document(_document(), doc_name="book")
        document = repository.load_document("book")
        resolver = ReadingTargetResolver()
        task_unit_target = resolver.resolve(
            document=document,
            target_level="task_unit",
            task_unit_id="unit-1",
        )
        content_block_target = resolver.resolve(
            document=document,
            target_level="content_block",
            task_unit_id="unit-1",
            content_block_id="unit-1:content:0",
        )
        task_unit_analysis_request = ReadingInteractionRequest(
            target=task_unit_target,
            interaction_type="analysis",
            context_metadata={"context_mode": "full_target", "doc_name": "book"},
        )
        block_analysis_request = ReadingInteractionRequest(
            target=content_block_target,
            interaction_type="analysis",
            context_metadata={"context_mode": "full_target", "doc_name": "book"},
        )
        task_unit_quiz_request = ReadingInteractionRequest(
            target=task_unit_target,
            interaction_type="quiz",
            context_metadata={"context_mode": "full_target", "doc_name": "book"},
        )
        block_quiz_request = ReadingInteractionRequest(
            target=content_block_target,
            interaction_type="quiz",
            context_metadata={"context_mode": "full_target", "doc_name": "book"},
        )
        store = DocumentReadingInteractionArtifactStore(repository)

        saved_task_analysis = store.save_analysis_artifact(
            ReadingInteractionArtifact.from_target(
                target=task_unit_target,
                interaction_type="analysis",
                status="completed",
                payload={
                    "summary": "Task summary",
                    "reasoning": "Task reasoning",
                    "explanation": "Task explanation",
                },
                metadata={"context": {"doc_name": "book"}},
            )
        )
        saved_block_analysis = store.save_analysis_artifact(
            ReadingInteractionArtifact.from_target(
                target=content_block_target,
                interaction_type="analysis",
                status="completed",
                payload={
                    "summary": "Block summary",
                    "reasoning": "Block reasoning",
                    "explanation": "Block explanation",
                },
                metadata={"context": {"doc_name": "book"}},
            )
        )
        saved_task_quiz = store.save_quiz_artifact(
            ReadingInteractionArtifact.from_target(
                target=task_unit_target,
                interaction_type="quiz",
                status="completed",
                payload={"items": [{"type": "short_answer", "question": "Task?", "answer": "Task"}]},
                metadata={"context": {"doc_name": "book"}},
            )
        )
        saved_block_quiz = store.save_quiz_artifact(
            ReadingInteractionArtifact.from_target(
                target=content_block_target,
                interaction_type="quiz",
                status="completed",
                payload={"items": [{"type": "short_answer", "question": "Block?", "answer": "Block"}]},
                metadata={"context": {"doc_name": "book"}},
            )
        )

        restored_task_analysis = store.get_analysis_artifact(task_unit_analysis_request)
        restored_block_analysis = store.get_analysis_artifact(block_analysis_request)
        restored_task_quiz = store.get_quiz_artifact(task_unit_quiz_request)
        restored_block_quiz = store.get_quiz_artifact(block_quiz_request)

        _assert(
            saved_task_analysis.metadata["artifact_id"]
            != saved_block_analysis.metadata["artifact_id"],
            "task-unit and content-block analysis keys should not collide",
        )
        _assert(
            "::content_block::unit-1:content:0"
            in saved_block_analysis.metadata["artifact_id"],
            "content-block analysis key should include content_block_id",
        )
        _assert(
            saved_task_quiz.metadata["artifact_id"]
            != saved_block_quiz.metadata["artifact_id"],
            "task-unit and content-block quiz keys should not collide",
        )
        _assert(
            "::content_block::unit-1:content:0"
            in saved_block_quiz.metadata["artifact_id"],
            "content-block quiz key should include content_block_id",
        )
        _assert(
            restored_task_analysis is not None
            and restored_task_analysis.payload["summary"] == "Task summary",
            "task-unit analysis should read only the task-unit artifact",
        )
        _assert(
            restored_block_analysis is not None
            and restored_block_analysis.payload["summary"] == "Block summary"
            and restored_block_analysis.content_block_id == "unit-1:content:0",
            "content-block analysis should read only the matching block artifact",
        )
        _assert(
            restored_task_quiz is not None
            and restored_task_quiz.payload["items"][0]["answer"] == "Task",
            "task-unit quiz should read only the task-unit artifact",
        )
        _assert(
            restored_block_quiz is not None
            and restored_block_quiz.payload["items"][0]["answer"] == "Block"
            and restored_block_quiz.content_block_id == "unit-1:content:0",
            "content-block quiz should read only the matching block artifact",
        )


def test_document_backed_critical_thinking_sessions_preserve_session_id() -> None:
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
            interaction_type="critical_thinking_session",
            context_metadata={"context_mode": "full_target", "doc_name": "book"},
        )
        store = DocumentReadingInteractionArtifactStore(repository)

        generated = ReadingInteractionArtifact.from_target(
            target=target,
            interaction_type="critical_thinking_session",
            status="question_generated",
            payload={"question": "What assumption should be challenged?"},
            metadata={"context": {"doc_name": "book"}},
        )
        saved = store.save_critical_thinking_session(generated)
        restored = store.get_critical_thinking_session(request, saved.session_id)
        completed = ReadingInteractionArtifact(
            interaction_type="critical_thinking_session",
            status="completed",
            target_level=generated.target_level,
            target_id=generated.target_id,
            document_id=generated.document_id,
            chapter_id=generated.chapter_id,
            section_id=generated.section_id,
            task_unit_id=generated.task_unit_id,
            payload={
                "question": "What assumption should be challenged?",
                "answer": "The assumed link between claim and evidence.",
                "evaluation": {
                    "feedback": "Grounded",
                    "strengths": "Uses evidence",
                    "improvements": "Add counterargument",
                },
            },
            metadata=dict(saved.artifact.metadata),
        )
        resaved = store.save_critical_thinking_session(completed)
        restored_completed = store.get_critical_thinking_session(
            request,
            saved.session_id,
        )

        _assert(saved.session_id, "generated session should receive an id")
        _assert(restored is not None, "saved session should be readable")
        _assert(
            restored.metadata["session_id"] == saved.session_id,
            "session id should be stored in metadata",
        )
        _assert(
            resaved.session_id == saved.session_id,
            "saving an existing session should preserve identity",
        )
        _assert(
            restored_completed is not None
            and restored_completed.status == "completed",
            "completed session should overwrite the same session",
        )


def test_content_block_critical_thinking_sessions_do_not_match_parent_task_unit() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        repository = StructuredDocumentArtifactRepository(base_dir=temp_dir)
        repository.save_document(_document(), doc_name="book")
        document = repository.load_document("book")
        resolver = ReadingTargetResolver()
        task_unit_target = resolver.resolve(
            document=document,
            target_level="task_unit",
            task_unit_id="unit-1",
        )
        content_block_target = resolver.resolve(
            document=document,
            target_level="content_block",
            task_unit_id="unit-1",
            content_block_id="unit-1:content:0",
        )
        task_unit_request = ReadingInteractionRequest(
            target=task_unit_target,
            interaction_type="critical_thinking_session",
            context_metadata={"context_mode": "full_target", "doc_name": "book"},
        )
        block_request = ReadingInteractionRequest(
            target=content_block_target,
            interaction_type="critical_thinking_session",
            context_metadata={"context_mode": "full_target", "doc_name": "book"},
        )
        store = DocumentReadingInteractionArtifactStore(repository)

        saved_block_session = store.save_critical_thinking_session(
            ReadingInteractionArtifact.from_target(
                target=content_block_target,
                interaction_type="critical_thinking_session",
                status="question_generated",
                payload={"question": "Which block-level assumption matters?"},
                metadata={"context": {"doc_name": "book"}},
            )
        )
        restored_block_session = store.get_critical_thinking_session(
            block_request,
            saved_block_session.session_id,
        )
        parent_lookup = store.get_critical_thinking_session(
            task_unit_request,
            saved_block_session.session_id,
        )

        _assert(
            "::content_block::unit-1:content:0::"
            in saved_block_session.session_id,
            "content-block session id should include content_block_id",
        )
        _assert(
            restored_block_session is not None
            and restored_block_session.content_block_id == "unit-1:content:0",
            "content-block session should round-trip with block identity",
        )
        _assert(
            parent_lookup is None,
            "parent task-unit request must not match a content-block session",
        )


def test_document_backed_current_critical_thinking_session_matches_block_target() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        repository = StructuredDocumentArtifactRepository(base_dir=temp_dir)
        repository.save_document(_document(), doc_name="book")
        document = repository.load_document("book")
        resolver = ReadingTargetResolver()
        task_unit_target = resolver.resolve(
            document=document,
            target_level="task_unit",
            task_unit_id="unit-1",
        )
        content_block_target = resolver.resolve(
            document=document,
            target_level="content_block",
            task_unit_id="unit-1",
            content_block_id="unit-1:content:0",
        )
        task_unit_request = ReadingInteractionRequest(
            target=task_unit_target,
            interaction_type="critical_thinking_session",
            context_metadata={"context_mode": "full_target", "doc_name": "book"},
        )
        block_request = ReadingInteractionRequest(
            target=content_block_target,
            interaction_type="critical_thinking_session",
            context_metadata={"context_mode": "full_target", "doc_name": "book"},
        )
        store = DocumentReadingInteractionArtifactStore(repository)

        first = store.save_critical_thinking_session(
            ReadingInteractionArtifact.from_target(
                target=content_block_target,
                interaction_type="critical_thinking_session",
                status="question_generated",
                payload={"question": "First block question?"},
                metadata={"context": {"doc_name": "book"}},
            )
        )
        second = store.save_critical_thinking_session(
            ReadingInteractionArtifact.from_target(
                target=content_block_target,
                interaction_type="critical_thinking_session",
                status="question_generated",
                payload={"question": "Second block question?"},
                metadata={"context": {"doc_name": "book"}},
            )
        )

        current_block = store.get_current_critical_thinking_session(block_request)
        current_parent = store.get_current_critical_thinking_session(task_unit_request)

        _assert(first.session_id != second.session_id, "sessions should be distinct")
        _assert(
            current_block is not None
            and current_block.metadata["session_id"] == second.session_id
            and current_block.payload["question"] == "Second block question?",
            "current-session lookup should return the latest matching block session",
        )
        _assert(
            current_parent is None,
            "parent task-unit current lookup must not match a content-block session",
        )


if __name__ == "__main__":
    test_document_backed_analysis_store_round_trips_current_artifact()
    test_document_backed_quiz_store_round_trips_current_artifact()
    test_document_backed_ancestor_summary_store_round_trips_summary_artifact()
    test_content_block_current_artifacts_do_not_collide_with_parent_task_unit()
    test_document_backed_critical_thinking_sessions_preserve_session_id()
    test_content_block_critical_thinking_sessions_do_not_match_parent_task_unit()
    test_document_backed_current_critical_thinking_session_matches_block_target()
    print("reading interaction artifact store tests passed")
