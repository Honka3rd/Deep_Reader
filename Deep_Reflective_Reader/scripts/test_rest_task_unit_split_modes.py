#!/usr/bin/env python3
"""Minimal API smoke test for request-time task-unit split mode wiring."""

from __future__ import annotations

import json
from contextlib import suppress
from typing import Any

from fastapi.testclient import TestClient

import main
from document_preparation.prepared_document_assets import PreparedDocumentAssets
from document_preparation.prepared_document_result import PreparedDocumentResult
from document_structure.structured_document import StructuredDocument, StructuredSection
from section_tasks.section_task_result import SectionTaskResult
from section_tasks.task_unit import TaskUnit


def _build_fake_document() -> StructuredDocument:
    content = (
        "Chapter One\n"
        "This is a minimal structured section for split-mode wiring smoke tests. "
        "It is long enough to be non-empty and deterministic."
    )
    return StructuredDocument(
        document_id="smoke-doc",
        title="Smoke Doc",
        source_path=None,
        language="en",
        raw_text=content,
        sections=[
            StructuredSection(
                section_id="section-0",
                section_index=0,
                title="Chapter One",
                level=1,
                content=content,
                char_start=0,
                char_end=len(content),
                container_title=None,
            )
        ],
    )


def _build_fake_prepared_result(document: StructuredDocument) -> PreparedDocumentResult:
    assets = PreparedDocumentAssets(
        doc_name=document.title,
        raw_text=document.raw_text,
        language=document.language,
        structured_document_ready=True,
        faiss_ready=False,
        profile_ready=False,
        bundle_ready=False,
        structured_document_path="/tmp/smoke-doc.structured.json",
        faiss_namespace=None,
        errors=[],
    )
    return PreparedDocumentResult(
        assets=assets,
        structured_document=document,
        bundle=None,
    )


def _print_exchange(label: str, payload: dict[str, Any], response: Any) -> None:
    print(f"\n[{label}] request:")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"[{label}] status: {response.status_code}")
    print(f"[{label}] response:")
    with suppress(Exception):
        print(json.dumps(response.json(), ensure_ascii=False, indent=2))


def main_smoke() -> None:
    client = TestClient(main.app)
    coordinator = main.section_task_coordinator
    fake_document = _build_fake_document()
    fake_result = _build_fake_prepared_result(fake_document)
    captured_resolve_calls: list[dict[str, Any]] = []

    original_prepare_and_load = coordinator.document_preparation_pipeline.prepare_and_load
    original_load_profile = coordinator._load_existing_document_profile
    original_resolve_with_options = coordinator.task_unit_resolver.resolve_with_options
    original_summary_task_unit = coordinator.chapter_summary_service.summarize_task_unit

    def fake_prepare_and_load(*args: Any, **kwargs: Any) -> PreparedDocumentResult:
        return fake_result

    def fake_load_profile(*args: Any, **kwargs: Any) -> None:
        return None

    def fake_resolve_with_options(
        *,
        document: StructuredDocument,
        split_mode: Any = None,
        semantic_top_k_candidates: int | None = None,
    ) -> list[TaskUnit]:
        captured_resolve_calls.append(
            {
                "split_mode": None if split_mode is None else str(split_mode),
                "semantic_top_k_candidates": semantic_top_k_candidates,
                "section_count": len(document.sections),
            }
        )
        section = document.sections[0]
        return [
            TaskUnit(
                unit_id="task-unit-0",
                title=section.title,
                container_title=section.container_title,
                content=section.content,
                source_section_ids=[section.section_id],
                is_fallback_generated=False,
            )
        ]

    def fake_summarize_task_unit(*args: Any, **kwargs: Any) -> SectionTaskResult[str]:
        return SectionTaskResult.ok("summary-smoke-ok")

    coordinator.document_preparation_pipeline.prepare_and_load = fake_prepare_and_load
    coordinator._load_existing_document_profile = fake_load_profile
    coordinator.task_unit_resolver.resolve_with_options = fake_resolve_with_options
    coordinator.chapter_summary_service.summarize_task_unit = fake_summarize_task_unit

    try:
        exchanges = [
            (
                "task-layout semantic_safe top_k=1",
                "/documents/task-layout",
                {
                    "doc_name": "Smoke Doc",
                    "task_unit_split_mode": "semantic_safe",
                    "semantic_top_k_candidates": 1,
                },
            ),
            (
                "task-layout semantic_safe top_k=5",
                "/documents/task-layout",
                {
                    "doc_name": "Smoke Doc",
                    "task_unit_split_mode": "semantic_safe",
                    "semantic_top_k_candidates": 5,
                },
            ),
            (
                "task-layout progressive",
                "/documents/task-layout",
                {
                    "doc_name": "Smoke Doc",
                    "task_unit_split_mode": "progressive",
                },
            ),
            (
                "task-layout llm_enhanced",
                "/documents/task-layout",
                {
                    "doc_name": "Smoke Doc",
                    "task_unit_split_mode": "llm_enhanced",
                },
            ),
            (
                "section-summary semantic_safe top_k=3",
                "/documents/section-summary",
                {
                    "doc_name": "Smoke Doc",
                    "section_id": "section-0",
                    "task_unit_split_mode": "semantic_safe",
                    "semantic_top_k_candidates": 3,
                },
            ),
        ]

        for label, path, payload in exchanges:
            response = client.post(path, json=payload)
            _print_exchange(label, payload, response)
            assert response.status_code == 200, (
                f"{label} expected status=200, got {response.status_code}"
            )

        invalid_mode_payload = {
            "doc_name": "Smoke Doc",
            "task_unit_split_mode": "bad_mode",
        }
        invalid_mode_response = client.post(
            "/documents/task-layout",
            json=invalid_mode_payload,
        )
        _print_exchange("task-layout invalid mode", invalid_mode_payload, invalid_mode_response)
        assert invalid_mode_response.status_code == 400

        invalid_top_k_payload = {
            "doc_name": "Smoke Doc",
            "task_unit_split_mode": "semantic_safe",
            "semantic_top_k_candidates": 0,
        }
        invalid_top_k_response = client.post(
            "/documents/task-layout",
            json=invalid_top_k_payload,
        )
        _print_exchange(
            "task-layout invalid semantic_top_k",
            invalid_top_k_payload,
            invalid_top_k_response,
        )
        assert invalid_top_k_response.status_code == 400

        print("\n[captured resolve_with_options calls]")
        print(json.dumps(captured_resolve_calls, ensure_ascii=False, indent=2))
    finally:
        coordinator.document_preparation_pipeline.prepare_and_load = original_prepare_and_load
        coordinator._load_existing_document_profile = original_load_profile
        coordinator.task_unit_resolver.resolve_with_options = original_resolve_with_options
        coordinator.chapter_summary_service.summarize_task_unit = original_summary_task_unit


if __name__ == "__main__":
    main_smoke()
