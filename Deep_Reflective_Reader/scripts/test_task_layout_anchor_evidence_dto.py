#!/usr/bin/env python3
"""Regression tests for task-layout anchor evidence DTOs."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from section_tasks.document_task_layout import (  # noqa: E402
    AnchorEvidenceDTO,
    DocumentTaskLayoutChapterDTO,
    DocumentTaskLayoutSectionDTO,
    SectionTaskMode,
)


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def test_chapter_and_section_dtos_carry_lightweight_anchor_evidence() -> None:
    section = DocumentTaskLayoutSectionDTO(
        section_id="section-1",
        title="Section One",
        container_title="Chapter One",
        section_role="main_body",
        parent_chapter_id="chapter-1",
        section_kind="manual_structure_section",
        is_implicit_section=False,
        task_mode=SectionTaskMode.DIRECT,
        task_units=[],
        anchor_evidence=AnchorEvidenceDTO(
            anchor_type="page_range",
            status="available",
            char_start=10,
            char_end=50,
            page_start_index=1,
            page_end_index=2,
            page_start_label="2",
            page_end_label="3",
        ),
    )
    chapter = DocumentTaskLayoutChapterDTO(
        chapter_id="chapter-1",
        title="Chapter One",
        level=1,
        chapter_role=None,
        sections=[section],
        anchor_evidence=AnchorEvidenceDTO(
            anchor_type="char_range",
            status="available",
            reason="page_boundaries_unavailable",
            char_start=0,
            char_end=100,
        ),
    )

    payload = chapter.to_dict()

    _assert(
        payload["anchor_evidence"]["anchor_type"] == "char_range",
        "chapter anchor evidence should serialize",
    )
    section_payload = payload["sections"][0]
    _assert(
        section_payload["anchor_evidence"]["anchor_type"] == "page_range",
        "section anchor evidence should serialize",
    )
    _assert(
        section_payload["anchor_evidence"]["page_start_label"] == "2",
        "page labels should remain lightweight prefill metadata",
    )
    serialized = str(payload)
    _assert("raw_text" not in serialized, "anchor evidence must not expose raw text")
    _assert("ocr_text" not in serialized, "anchor evidence must not expose OCR text")
    _assert("bbox" not in serialized, "anchor evidence must not expose geometry")


def test_unset_anchor_evidence_preserves_current_payload_shape() -> None:
    section = DocumentTaskLayoutSectionDTO(
        section_id="section-1",
        title="Section One",
        container_title="Chapter One",
        section_role=None,
        parent_chapter_id="chapter-1",
        section_kind=None,
        is_implicit_section=False,
        task_mode=SectionTaskMode.DIRECT,
        task_units=[],
    )

    payload = section.to_dict()

    _assert(
        "anchor_evidence" not in payload,
        "unset anchor evidence should not alter current public payload before API mapping",
    )


if __name__ == "__main__":
    test_chapter_and_section_dtos_carry_lightweight_anchor_evidence()
    test_unset_anchor_evidence_preserves_current_payload_shape()
    print("task-layout anchor evidence DTO tests passed")
