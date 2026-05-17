#!/usr/bin/env python3
"""REST and coordinator smoke tests for on-demand task-unit content lookup."""

from __future__ import annotations

from dataclasses import dataclass, replace

from fastapi.testclient import TestClient

import main
from app.section_task_coordinator import SectionTaskCoordinator
from document_preparation.prepared_document_assets import PreparedDocumentAssets
from document_preparation.prepared_document_result import PreparedDocumentResult
from document_preparation.preparation_mode import PreparationMode
from document_structure.enhanced_parse_trigger_evaluator import EnhancedParseTriggerEvaluator
from document_structure.section_role import SectionRole
from document_structure.structured_document import (
    StructuredChapter,
    StructuredDocument,
    StructuredSection,
)
from section_tasks.task_unit_split_mode import TaskUnitSplitMode
from shared.task_artifacts import DocumentTaskArtifacts
from shared.task_unit_model import TaskUnit


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


@dataclass(frozen=True)
class _FakeAssets:
    errors: list[str]


@dataclass(frozen=True)
class _FakePreparedResult:
    assets: _FakeAssets
    structured_document: StructuredDocument | None
    bundle: object | None = None


class _FakePipeline:
    def __init__(self, document: StructuredDocument):
        self.document = document

    def prepare_and_load(
        self,
        doc_name: str,
        force_rebuild: bool = False,
        mode: PreparationMode | str = PreparationMode.BASE,
        structured_parser_mode: str = "common",
    ) -> PreparedDocumentResult:
        _ = (force_rebuild, mode, structured_parser_mode)
        assets = PreparedDocumentAssets(
            doc_name=doc_name,
            raw_text=self.document.raw_text,
            language=self.document.language,
            structured_document_ready=True,
            faiss_ready=False,
            profile_ready=False,
            bundle_ready=False,
            structured_document_path=None,
            faiss_namespace=None,
            errors=[],
        )
        return PreparedDocumentResult(
            assets=assets,
            structured_document=self.document,
            bundle=None,
        )


class _SpyRepository:
    def __init__(self) -> None:
        self.write_calls = 0

    def save_document(self, document, doc_name):  # noqa: ANN001
        _ = (document, doc_name)
        self.write_calls += 1
        return document

    def update_task_layout(self, doc_name, task_units_by_section_id, task_layout_metadata):  # noqa: ANN001
        _ = (doc_name, task_units_by_section_id, task_layout_metadata)
        self.write_calls += 1
        raise RuntimeError("update_task_layout should not be called in this smoke test")


class _MissingProfileStore:
    @staticmethod
    def exists(config) -> bool:  # noqa: ANN001
        _ = config
        return False

    @staticmethod
    def load(config):  # noqa: ANN001
        _ = config
        raise RuntimeError("profile should not be loaded")


class _NoopSummaryService:
    pass


class _NoopQuizService:
    pass


class _FailIfResolverCalled:
    def __init__(self) -> None:
        self.split_mode = TaskUnitSplitMode.SEMANTIC_SAFE

    def resolve_with_options(self, **kwargs):  # noqa: ANN003
        _ = kwargs
        raise RuntimeError("task unit resolver should not be called when cache is valid")


def _build_section(*, section_id: str, chapter_id: str, title: str, unit_id: str, content: str) -> StructuredSection:
    return StructuredSection(
        section_id=section_id,
        section_index=0,
        title=title,
        level=1,
        content=content,
        char_start=0,
        char_end=len(content),
        section_role=SectionRole.MAIN_BODY,
        parent_chapter_id=chapter_id,
        section_kind="chapter_body",
        is_implicit_section=True,
        task_units=[
            TaskUnit(
                unit_id=unit_id,
                title=f"{title} Unit",
                container_title=title,
                content=content,
                source_section_ids=[section_id],
                is_fallback_generated=False,
                parent_section_id=section_id,
            )
        ],
    )


def _build_cache_valid_document(*, duplicate_unit_id: bool = False) -> StructuredDocument:
    section_a = _build_section(
        section_id="section-a",
        chapter_id="chapter-a",
        title="Chapter A",
        unit_id="task-unit-1",
        content="Content A",
    )
    section_b = _build_section(
        section_id="section-b",
        chapter_id="chapter-b",
        title="Chapter B",
        unit_id=("task-unit-1" if duplicate_unit_id else "task-unit-2"),
        content="Content B",
    )

    document = StructuredDocument(
        document_id="doc-content",
        title="Doc Content",
        source_path=None,
        language="en",
        raw_text="Content A\n\nContent B",
        chapters=[
            StructuredChapter(
                chapter_id="chapter-a",
                title="Chapter A",
                level=1,
                chapter_role="main_body",
                sections=[section_a],
            ),
            StructuredChapter(
                chapter_id="chapter-b",
                title="Chapter B",
                level=1,
                chapter_role="main_body",
                sections=[section_b],
            ),
        ],
        sections=[],
        structure_nodes=[],
    )

    source_hash = SectionTaskCoordinator._compute_source_hash(document)
    task_layout_metadata = {
        "task_layout": {
            "source_hash": source_hash,
            "task_unit_split_mode": TaskUnitSplitMode.SEMANTIC_SAFE.value,
            "semantic_top_k_candidates": None,
            "resolver_version": "task_unit_resolver_v2",
        }
    }
    return replace(
        document,
        document_task_artifacts=DocumentTaskArtifacts(metadata=task_layout_metadata),
    )


def _build_legacy_sections_only_document() -> StructuredDocument:
    section = _build_section(
        section_id="legacy-section",
        chapter_id="legacy-chapter",
        title="Legacy Chapter",
        unit_id="legacy-task-unit",
        content="Legacy Content",
    )
    return StructuredDocument(
        document_id="legacy-doc",
        title="Legacy Doc",
        source_path=None,
        language="en",
        raw_text="Legacy Content",
        chapters=[],
        sections=[section],
        structure_nodes=[],
    )


def _build_coordinator(document: StructuredDocument) -> tuple[SectionTaskCoordinator, _SpyRepository]:
    repository = _SpyRepository()
    coordinator = SectionTaskCoordinator(
        document_preparation_pipeline=_FakePipeline(document),
        document_artifact_repository=repository,
        document_profile_store=_MissingProfileStore(),
        chapter_summary_service=_NoopSummaryService(),
        chapter_quiz_service=_NoopQuizService(),
        task_unit_resolver=_FailIfResolverCalled(),
        enhanced_parse_trigger_evaluator=EnhancedParseTriggerEvaluator(),
    )
    return coordinator, repository


def test_task_layout_id_then_content_lookup_success() -> None:
    coordinator, repository = _build_coordinator(_build_cache_valid_document())
    client = TestClient(main.app)
    original = main.section_task_coordinator
    main.section_task_coordinator = coordinator
    try:
        layout_response = client.post(
            "/documents/task-layout",
            json={"doc_name": "Doc Content", "refresh_task_units": False},
        )
        _assert(layout_response.status_code == 200, "task-layout should succeed")
        layout_payload = layout_response.json()
        unit_id = layout_payload["chapters"][0]["sections"][0]["task_units"][0]["unit_id"]

        content_response = client.get(
            f"/documents/Doc Content/task-units/{unit_id}/content"
        )
        _assert(content_response.status_code == 200, "content lookup should succeed")
        payload = content_response.json()

        _assert(payload["task_unit_id"] == unit_id, "task_unit_id mismatch")
        _assert(payload["content"] == "Content A", "content mismatch")
        _assert(payload["section_id"] == "section-a", "section_id mismatch")
        _assert(payload["chapter_id"] == "chapter-a", "chapter_id mismatch")
        _assert(repository.write_calls == 0, "content lookup path must not write persistence")
    finally:
        main.section_task_coordinator = original


def test_task_unit_content_missing_id_returns_404() -> None:
    coordinator, repository = _build_coordinator(_build_cache_valid_document())
    client = TestClient(main.app)
    original = main.section_task_coordinator
    main.section_task_coordinator = coordinator
    try:
        response = client.get(
            "/documents/Doc Content/task-units/missing-unit/content"
        )
        _assert(response.status_code == 404, "missing unit should return 404")
        _assert("not found" in response.json().get("detail", "").lower(), "error should mention not found")
        _assert(repository.write_calls == 0, "missing lookup must not write persistence")
    finally:
        main.section_task_coordinator = original


def test_task_unit_content_duplicate_id_fails_fast() -> None:
    coordinator, repository = _build_coordinator(_build_cache_valid_document(duplicate_unit_id=True))
    client = TestClient(main.app)
    original = main.section_task_coordinator
    main.section_task_coordinator = coordinator
    try:
        response = client.get(
            "/documents/Doc Content/task-units/task-unit-1/content"
        )
        _assert(response.status_code == 400, "duplicate task_unit_id should fail with 400")
        _assert(
            "duplicate task_unit_id" in response.json().get("detail", ""),
            "duplicate id error should be explicit",
        )
        _assert(repository.write_calls == 0, "duplicate lookup must not write persistence")
    finally:
        main.section_task_coordinator = original


def test_task_unit_content_does_not_fallback_to_root_sections() -> None:
    coordinator, repository = _build_coordinator(_build_legacy_sections_only_document())
    client = TestClient(main.app)
    original = main.section_task_coordinator
    main.section_task_coordinator = coordinator
    try:
        response = client.get(
            "/documents/Legacy Doc/task-units/legacy-task-unit/content"
        )
        _assert(response.status_code == 400, "legacy sections-only runtime path should fail-fast")
        detail = response.json().get("detail", "").lower()
        _assert("requires migration" in detail, "error should mention migration requirement")
        _assert(repository.write_calls == 0, "fail-fast lookup must not write persistence")
    finally:
        main.section_task_coordinator = original


if __name__ == "__main__":
    test_task_layout_id_then_content_lookup_success()
    test_task_unit_content_missing_id_returns_404()
    test_task_unit_content_duplicate_id_fails_fast()
    test_task_unit_content_does_not_fallback_to_root_sections()
    print("ok")
