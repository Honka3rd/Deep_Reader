#!/usr/bin/env python3
"""Task-unit id normalizer tests for document-scope uniqueness."""

from __future__ import annotations

import json
from pathlib import Path

from document_structure.section_role import SectionRole
from document_structure.structured_document import StructuredDocument, StructuredSection
from section_tasks.task_unit_id_normalizer import TaskUnitIdNormalizer
from shared.task_artifacts import QuizArtifact, SummaryArtifact, TaskArtifacts
from shared.task_unit_model import TaskUnit


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _build_duplicate_id_document() -> StructuredDocument:
    return StructuredDocument(
        document_id="dup-doc",
        title="Duplicate Task Unit IDs",
        source_path=None,
        language="zh",
        raw_text="S0\nS1\nS2",
        sections=[
            StructuredSection(
                section_id="section-0",
                section_index=0,
                title="S0",
                level=1,
                content="A",
                char_start=0,
                char_end=1,
                section_role=SectionRole.MAIN_BODY,
                task_units=[
                    TaskUnit(
                        unit_id="task-unit-0",
                        title="u0",
                        container_title=None,
                        content="u0-content",
                        source_section_ids=["section-0"],
                        is_fallback_generated=False,
                        task_artifacts=TaskArtifacts(
                            summary=SummaryArtifact(content="summary-u0"),
                        ),
                    )
                ],
            ),
            StructuredSection(
                section_id="section-1",
                section_index=1,
                title="S1",
                level=1,
                content="B",
                char_start=2,
                char_end=3,
                section_role=SectionRole.MAIN_BODY,
                task_units=[
                    TaskUnit(
                        unit_id="task-unit-0",
                        title="u1",
                        container_title=None,
                        content="u1-content",
                        source_section_ids=["section-1"],
                        is_fallback_generated=False,
                        task_artifacts=TaskArtifacts(
                            quiz=QuizArtifact(
                                items=[
                                    {
                                        "question_id": "q1",
                                        "question_text": "t",
                                        "answer_text": "a",
                                    }
                                ]
                            )
                        ),
                    )
                ],
            ),
            StructuredSection(
                section_id="section-2",
                section_index=2,
                title="S2",
                level=1,
                content="C",
                char_start=4,
                char_end=5,
                section_role=SectionRole.MAIN_BODY,
                task_units=[
                    TaskUnit(
                        unit_id="task-unit-local",
                        title="u2",
                        container_title=None,
                        content="u2-content",
                        source_section_ids=["section-2"],
                        is_fallback_generated=True,
                    )
                ],
            ),
        ],
    )


def test_normalizer_basic_and_artifact_preservation() -> None:
    normalizer = TaskUnitIdNormalizer()
    document = _build_duplicate_id_document()

    duplicates_before = normalizer.find_duplicate_task_unit_ids(document=document)
    _assert(
        duplicates_before == {"task-unit-0": 2},
        f"expected duplicate task-unit-0 before normalization, got={duplicates_before}",
    )

    normalized = normalizer.normalize_document_task_unit_ids(document=document)
    ids = [
        task_unit.unit_id
        for section in normalized.sections
        for task_unit in section.task_units
    ]
    _assert(ids == ["task-unit-0", "task-unit-1", "task-unit-2"], f"unexpected ids={ids}")
    _assert(
        not normalizer.has_duplicate_task_unit_ids(document=normalized),
        "normalized document should not contain duplicate task_unit ids",
    )

    # Ensure artifacts stay on original unit positions; no merging by old duplicate id.
    s0_u0 = normalized.sections[0].task_units[0]
    s1_u0 = normalized.sections[1].task_units[0]
    _assert(
        s0_u0.task_artifacts is not None
        and s0_u0.task_artifacts.summary is not None
        and s0_u0.task_artifacts.summary.content == "summary-u0",
        "section-0 artifact should stay on the same task unit position",
    )
    _assert(
        s1_u0.task_artifacts is not None
        and s1_u0.task_artifacts.quiz is not None
        and len(s1_u0.task_artifacts.quiz.items) == 1,
        "section-1 quiz artifact should stay on the same task unit position",
    )

    # Source section ids and content must stay unchanged.
    _assert(s0_u0.source_section_ids == ["section-0"], "source_section_ids should remain unchanged")
    _assert(s1_u0.source_section_ids == ["section-1"], "source_section_ids should remain unchanged")
    _assert(s0_u0.content == "u0-content", "content should remain unchanged")
    _assert(s1_u0.content == "u1-content", "content should remain unchanged")


def test_real_document_regression_snapshot() -> dict[str, int | bool]:
    """Real document snapshot check (without mutating files)."""
    path = Path("data/structured/许三观卖血记.structured.json")
    if not path.exists():
        return {
            "exists": False,
            "total_task_units": 0,
            "unique_task_unit_ids": 0,
            "normalized_unique": False,
        }

    payload = json.loads(path.read_text(encoding="utf-8"))
    document = StructuredDocument.from_dict(payload)
    normalizer = TaskUnitIdNormalizer()
    total = sum(len(section.task_units) for section in document.sections)
    unique_before = len(normalizer.collect_task_unit_id_counts(document))
    normalized = normalizer.normalize_document_task_unit_ids(document=document)
    unique_after = len(normalizer.collect_task_unit_id_counts(normalized))
    return {
        "exists": True,
        "total_task_units": total,
        "unique_task_unit_ids": unique_before,
        "normalized_unique": unique_after == total,
    }


def main() -> None:
    test_normalizer_basic_and_artifact_preservation()
    real_doc = test_real_document_regression_snapshot()
    print(
        json.dumps(
            {
                "status": "ok",
                "tests": [
                    "normalizer_basic_and_artifacts_preserved",
                    "real_document_regression_snapshot",
                ],
                "real_document": real_doc,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

