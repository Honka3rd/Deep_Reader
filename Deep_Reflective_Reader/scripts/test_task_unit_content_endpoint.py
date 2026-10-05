#!/usr/bin/env python3
"""REST and coordinator smoke tests for on-demand task-unit content lookup."""

from __future__ import annotations

from dataclasses import dataclass, replace

from fastapi.testclient import TestClient
from pydantic import ValidationError

import main
from api_schemas import (
    ArtifactTargetRefResponse,
    BatchTaskUnitContentResponse,
    TaskUnitContentResponse,
)
from app.section_task_coordinator import SectionTaskCoordinator
from doc_loaders.pdf_page_evidence import PdfPageTextBoundary
from document_preparation.document_preparation_pipeline import (
    ManualStructureSourceEvidence,
)
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
from document_structure.text_normalization import normalize_ocr_text
from section_tasks.task_unit_split_mode import TaskUnitSplitMode
from shared.task_artifacts import DocumentTaskArtifacts
from shared.task_unit_model import (
    ArtifactTargetLevel,
    ArtifactTargetRef,
    TaskUnit,
    TaskUnitContentBlock,
)


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
    def __init__(
        self,
        document: StructuredDocument,
        page_boundaries: list[PdfPageTextBoundary] | None = None,
    ):
        self.document = document
        self.page_boundaries = list(page_boundaries or [])
        self.prepare_calls = 0

    def prepare_and_load(
        self,
        doc_name: str,
        force_rebuild: bool = False,
        mode: PreparationMode | str = PreparationMode.BASE,
        structured_parser_mode: str = "common",
    ) -> PreparedDocumentResult:
        self.prepare_calls += 1
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

    def load_manual_structure_source_evidence(
        self,
        doc_name: str,
    ) -> ManualStructureSourceEvidence:
        return ManualStructureSourceEvidence(
            doc_name=doc_name,
            source_identity=doc_name,
            raw_text=self.document.raw_text,
            source_hash=SectionTaskCoordinator._compute_raw_text_hash(
                self.document.raw_text
            ),
            language=self.document.language,
            page_boundaries=list(self.page_boundaries),
            errors=[],
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
    content_block_id = f"{unit_id}:content:0"
    content_block = TaskUnitContentBlock(
        block_id=content_block_id,
        content=content,
        artifact_target_refs=[
            ArtifactTargetRef(
                target_level=ArtifactTargetLevel.CONTENT_BLOCK,
                document_id="doc-content",
                chapter_id=chapter_id,
                section_id=section_id,
                task_unit_id=unit_id,
                content_block_id=content_block_id,
                metadata={
                    "source_hash": "hash-123",
                    "content_block_id": content_block_id,
                    "quote_span_start": 0,
                    "quote_span_end": 7,
                    "schema_version": "v1",
                    "unexpected_key": "should-be-filtered",
                },
            ),
            ArtifactTargetRef(
                target_level=ArtifactTargetLevel.TASK_UNIT,
                document_id="doc-content",
                chapter_id=chapter_id,
                section_id=section_id,
                task_unit_id=unit_id,
                metadata={"schema_version": "v1"},
            ),
        ],
    )
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
                content_blocks=[content_block],
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
        parse_provenance={
            "requested_parser_mode": "llm_enhanced",
            "effective_parser_mode": "common",
            "fallback_used": True,
            "fallback_reason": "abnormal_section_output",
            "source": "llm_section_splitter",
        },
    )

    source_hash = SectionTaskCoordinator._compute_source_hash(document)
    task_layout_metadata = {
        "task_layout": {
            "source_hash": source_hash,
            "task_unit_split_mode": TaskUnitSplitMode.SEMANTIC_SAFE.value,
            "semantic_top_k_candidates": None,
            "resolver_version": SectionTaskCoordinator._TASK_LAYOUT_RESOLVER_VERSION,
        }
    }
    return replace(
        document,
        document_task_artifacts=DocumentTaskArtifacts(metadata=task_layout_metadata),
    )


def _build_segmentable_document() -> StructuredDocument:
    section = _build_section(
        section_id="section-seg",
        chapter_id="chapter-seg",
        title="Chapter Seg",
        unit_id="task-unit-seg",
        content="Paragraph one.\n\n- item one\n- item two\n\nParagraph tail.",
    )
    document = StructuredDocument(
        document_id="doc-seg",
        title="Doc Seg",
        source_path=None,
        language="en",
        raw_text="Paragraph one.\n\n- item one\n- item two\n\nParagraph tail.",
        chapters=[
            StructuredChapter(
                chapter_id="chapter-seg",
                title="Chapter Seg",
                level=1,
                chapter_role="main_body",
                sections=[section],
            )
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
            "resolver_version": SectionTaskCoordinator._TASK_LAYOUT_RESOLVER_VERSION,
        }
    }
    return replace(
        document,
        document_task_artifacts=DocumentTaskArtifacts(metadata=task_layout_metadata),
    )


def _build_multilingual_segmentable_document() -> StructuredDocument:
    content = "第一段：中文段落。\n\n第二段：日本語の段落。"
    section = _build_section(
        section_id="section-cjk",
        chapter_id="chapter-cjk",
        title="Chapter CJK",
        unit_id="task-unit-cjk",
        content=content,
    )
    document = StructuredDocument(
        document_id="doc-cjk",
        title="Doc CJK",
        source_path=None,
        language="zh",
        raw_text=content,
        chapters=[
            StructuredChapter(
                chapter_id="chapter-cjk",
                title="Chapter CJK",
                level=1,
                chapter_role="main_body",
                sections=[section],
            )
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
            "resolver_version": SectionTaskCoordinator._TASK_LAYOUT_RESOLVER_VERSION,
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


def _build_coordinator(
    document: StructuredDocument,
    page_boundaries: list[PdfPageTextBoundary] | None = None,
) -> tuple[SectionTaskCoordinator, _SpyRepository]:
    repository = _SpyRepository()
    coordinator = SectionTaskCoordinator(
        document_preparation_pipeline=_FakePipeline(
            document,
            page_boundaries=page_boundaries,
        ),
        document_artifact_repository=repository,
        document_profile_store=_MissingProfileStore(),
        chapter_summary_service=_NoopSummaryService(),
        chapter_quiz_service=_NoopQuizService(),
        task_unit_resolver=_FailIfResolverCalled(),
        enhanced_parse_trigger_evaluator=EnhancedParseTriggerEvaluator(),
    )
    return coordinator, repository


def test_task_layout_anchor_evidence_prefers_page_range_when_available() -> None:
    document = _build_segmentable_document()
    page_boundaries = [
        PdfPageTextBoundary(
            page_index=4,
            char_start=0,
            char_end=len(document.raw_text),
            text=document.raw_text,
            page_label="5",
        )
    ]
    coordinator, repository = _build_coordinator(
        document,
        page_boundaries=page_boundaries,
    )

    layout = coordinator.get_document_task_layout(
        doc_name="Doc Seg",
        refresh_task_units=False,
        include_anchor_page_evidence=True,
    )
    chapter_anchor = layout.chapters[0].anchor_evidence
    section_anchor = layout.chapters[0].sections[0].anchor_evidence
    _assert(chapter_anchor is not None, "chapter anchor evidence should exist")
    _assert(section_anchor is not None, "section anchor evidence should exist")
    _assert(
        chapter_anchor.anchor_type == "page_range",
        "chapter anchor evidence should prefer page_range when page boundaries exist",
    )
    _assert(
        section_anchor.anchor_type == "page_range",
        "section anchor evidence should prefer page_range when page boundaries exist",
    )
    _assert(
        section_anchor.page_start_index == 4
        and section_anchor.page_end_index == 4
        and section_anchor.page_start_label == "5",
        "section anchor evidence should expose lightweight page coordinates",
    )

    client = TestClient(main.app)
    original = main.section_task_coordinator
    main.section_task_coordinator = coordinator
    try:
        layout_response = client.post(
            "/documents/task-layout",
            json={
                "doc_name": "Doc Seg",
                "refresh_task_units": False,
                "include_anchor_page_evidence": True,
            },
        )
        _assert(layout_response.status_code == 200, "task-layout should succeed")
        anchor_payload = layout_response.json()["chapters"][0]["sections"][0][
            "anchor_evidence"
        ]
        _assert(
            anchor_payload["anchor_type"] == "page_range",
            "public task-layout response should expose page_range anchor evidence",
        )
        serialized_anchor = str(anchor_payload)
        _assert(
            "raw_text" not in serialized_anchor
            and "ocr_text" not in serialized_anchor
            and document.raw_text not in serialized_anchor,
            "page-backed task-layout anchor evidence must not expose page text",
        )
        _assert(
            repository.write_calls == 0,
            "page-backed task-layout anchor projection must remain read-only",
        )
    finally:
        main.section_task_coordinator = original


def test_task_layout_id_then_content_lookup_success() -> None:
    coordinator, repository = _build_coordinator(_build_cache_valid_document())
    layout = coordinator.get_document_task_layout(
        doc_name="Doc Content",
        refresh_task_units=False,
    )
    _assert(
        layout.chapters[0].anchor_evidence is not None,
        "coordinator layout should include chapter anchor evidence",
    )
    _assert(
        layout.chapters[0].anchor_evidence.anchor_type == "char_range",
        "chapter anchor evidence should fall back to char_range without page boundaries",
    )
    _assert(
        layout.chapters[0].sections[0].anchor_evidence is not None,
        "coordinator layout should include section anchor evidence",
    )
    _assert(
        layout.chapters[0].sections[0].anchor_evidence.char_start == 0,
        "section anchor evidence should preserve parsed char start",
    )
    _assert(
        "raw_text" not in str(layout.chapters[0].to_dict()),
        "task-layout anchor evidence must not expose raw text",
    )

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
        _assert(
            layout_payload["parse_provenance"]
            == {
                "requested_parser_mode": "llm_enhanced",
                "effective_parser_mode": "common",
                "fallback_used": True,
                "fallback_reason": "abnormal_section_output",
                "source": "llm_section_splitter",
            },
            "task-layout should expose lightweight parser provenance",
        )
        unit_id = layout_payload["chapters"][0]["sections"][0]["task_units"][0]["unit_id"]
        _assert(
            "content_blocks" not in layout_payload["chapters"][0]["sections"][0]["task_units"][0],
            "task-layout task_unit metadata must stay lightweight without content_blocks",
        )
        _assert(
            layout_payload["chapters"][0]["anchor_evidence"]["anchor_type"] == "char_range",
            "public task-layout response should expose chapter anchor evidence",
        )
        _assert(
            layout_payload["chapters"][0]["sections"][0]["anchor_evidence"]["char_start"] == 0,
            "public task-layout response should expose section anchor evidence",
        )
        serialized_anchor_evidence = str(
            layout_payload["chapters"][0]["sections"][0]["anchor_evidence"]
        )
        _assert(
            (
                "raw_text" not in serialized_anchor_evidence
                and "ocr_text" not in serialized_anchor_evidence
                and "bbox" not in serialized_anchor_evidence
            ),
            "public task-layout anchor evidence must stay lightweight",
        )

        content_response = client.get(
            f"/documents/Doc Content/task-units/{unit_id}/content"
        )
        _assert(content_response.status_code == 200, "content lookup should succeed")
        payload = content_response.json()

        _assert(payload["task_unit_id"] == unit_id, "task_unit_id mismatch")
        _assert(payload["content"] is None, "default response should not expose raw content")
        _assert("content_blocks" in payload, "content_blocks should exist in content endpoint payload")
        _assert(
            isinstance(payload["content_blocks"], list) and len(payload["content_blocks"]) == 1,
            "content_blocks should contain a single adapter-generated block",
        )

        content_response_segmented_false = client.get(
            f"/documents/Doc Content/task-units/{unit_id}/content",
            params={"segmented": "false"},
        )
        _assert(content_response_segmented_false.status_code == 200, "segmented=false should succeed")
        payload_segmented_false = content_response_segmented_false.json()
        _assert(
            len(payload_segmented_false["content_blocks"]) == 1,
            "segmented=false should preserve compatibility-safe single-block behavior",
        )
        _assert(
            payload["content_blocks"] == payload_segmented_false["content_blocks"],
            "omitted segmented and segmented=false should produce identical compatibility-safe blocks",
        )
        _assert(
            payload["content_blocks"][0]["block_id"] == f"{unit_id}:content:0",
            "content block id should be deterministic",
        )
        _assert(
            payload["content_blocks"][0]["content"] == "Content A",
            "default content block should preserve original text",
        )

        content_response_with_raw = client.get(
            f"/documents/Doc Content/task-units/{unit_id}/content",
            params={"include_raw_content": "true"},
        )
        _assert(content_response_with_raw.status_code == 200, "include_raw_content=true should succeed")
        payload_with_raw = content_response_with_raw.json()
        _assert(
            payload_with_raw["content"] == "Content A",
            "include_raw_content=true should expose legacy raw content for compatibility",
        )
        _assert(
            payload_with_raw["content_blocks"][0]["content"] == "Content A",
            "content_blocks should remain unchanged when raw-content compatibility is enabled",
        )
        _assert(
            set(payload["content_blocks"][0].keys())
            == {
                "block_id",
                "content",
                "block_type",
                "artifact_ids",
                "artifact_target_refs",
                "metadata",
            },
            "content block response should keep normalized API schema shape",
        )
        _assert(
            payload["content_blocks"][0]["block_type"] is None
            and payload["content_blocks"][0]["artifact_ids"] is None
            and isinstance(payload["content_blocks"][0]["artifact_target_refs"], list)
            and len(payload["content_blocks"][0]["artifact_target_refs"]) == 2
            and payload["content_blocks"][0]["metadata"] is None,
            "default optional block fields should serialize as null in current adapter path",
        )
        _assert(
            payload["content_blocks"][0]["artifact_target_refs"][0]["target_level"]
            == "content_block",
            "first artifact target ref should preserve content_block target level",
        )
        _assert(
            set(payload["content_blocks"][0]["artifact_target_refs"][0]["metadata"].keys())
            == {
                "source_hash",
                "content_block_id",
                "quote_span_start",
                "quote_span_end",
                "schema_version",
            },
            "artifact target metadata should be limited to approved glossary keys",
        )
        _assert(
            "unexpected_key"
            not in payload["content_blocks"][0]["artifact_target_refs"][0]["metadata"],
            "unexpected artifact target metadata keys should be filtered out",
        )
        _assert(
            TaskUnitContentResponse.model_validate(payload).task_unit_id == unit_id,
            "endpoint payload should validate against official TaskUnitContentResponse schema",
        )
        _assert(payload["section_id"] == "section-a", "section_id mismatch")
        _assert(payload["chapter_id"] == "chapter-a", "chapter_id mismatch")
        _assert(repository.write_calls == 0, "content lookup path must not write persistence")
    finally:
        main.section_task_coordinator = original


def test_batch_task_unit_content_lookup_preserves_order_and_read_only() -> None:
    coordinator, repository = _build_coordinator(_build_cache_valid_document())
    pipeline = coordinator.document_preparation_pipeline
    client = TestClient(main.app)
    original = main.section_task_coordinator
    main.section_task_coordinator = coordinator
    try:
        response = client.post(
            "/documents/Doc Content/task-units/content",
            json={
                "task_unit_ids": ["task-unit-2", "task-unit-1"],
                "segmented": True,
            },
        )
        _assert(response.status_code == 200, "batch content lookup should succeed")
        payload = response.json()

        _assert(payload["document_id"] == "doc-content", "batch document_id mismatch")
        _assert(
            [item["task_unit_id"] for item in payload["contents"]]
            == ["task-unit-2", "task-unit-1"],
            "batch response should preserve requested task-unit order",
        )
        _assert(
            [item["content_blocks"][0]["content"] for item in payload["contents"]]
            == ["Content B", "Content A"],
            "batch response should include requested task-unit content in order",
        )
        _assert(
            all(item["content"] is None for item in payload["contents"]),
            "batch response should suppress raw content by default",
        )
        _assert(
            BatchTaskUnitContentResponse.model_validate(payload).contents[0].task_unit_id
            == "task-unit-2",
            "batch payload should validate against official schema",
        )
        _assert(
            getattr(pipeline, "prepare_calls") == 1,
            "batch content lookup should prepare/load the document once",
        )
        _assert(repository.write_calls == 0, "batch content lookup must not write persistence")
    finally:
        main.section_task_coordinator = original


def test_batch_task_unit_content_duplicate_request_id_fails_fast() -> None:
    coordinator, repository = _build_coordinator(_build_cache_valid_document())
    client = TestClient(main.app)
    original = main.section_task_coordinator
    main.section_task_coordinator = coordinator
    try:
        response = client.post(
            "/documents/Doc Content/task-units/content",
            json={
                "task_unit_ids": ["task-unit-1", "task-unit-1"],
                "segmented": True,
            },
        )
        _assert(response.status_code == 422, "duplicate request ids should fail schema validation")
        _assert(repository.write_calls == 0, "invalid batch request must not write persistence")
    finally:
        main.section_task_coordinator = original


def test_task_unit_content_segmented_true_returns_deterministic_multi_blocks() -> None:
    coordinator, repository = _build_coordinator(_build_segmentable_document())
    client = TestClient(main.app)
    original = main.section_task_coordinator
    main.section_task_coordinator = coordinator
    expected_raw_content = "Paragraph one.\n\n- item one\n- item two\n\nParagraph tail."
    try:
        response = client.get(
            "/documents/Doc Seg/task-units/task-unit-seg/content",
            params={"segmented": "true"},
        )
        _assert(response.status_code == 200, "segmented=true lookup should succeed")
        payload = response.json()

        _assert(payload["content"] is None, "segmented=true default should not expose raw content")
        _assert(len(payload["content_blocks"]) == 4, "segmented=true should return deterministic multi-block output")

        block_0 = payload["content_blocks"][0]
        block_1 = payload["content_blocks"][1]
        block_2 = payload["content_blocks"][2]
        block_3 = payload["content_blocks"][3]

        _assert(block_0["block_id"] == "task-unit-seg:content:0", "block 0 id mismatch")
        _assert(block_1["block_id"] == "task-unit-seg:content:1", "block 1 id mismatch")
        _assert(block_2["block_id"] == "task-unit-seg:content:2", "block 2 id mismatch")
        _assert(block_3["block_id"] == "task-unit-seg:content:3", "block 3 id mismatch")

        _assert(block_0["content"] == "Paragraph one.", "block 0 content mismatch")
        _assert(block_1["content"] == "- item one", "block 1 content mismatch")
        _assert(block_2["content"] == "- item two", "block 2 content mismatch")
        _assert(block_3["content"] == "Paragraph tail.", "block 3 content mismatch")

        expected_metadata_keys = {
            "source_hash",
            "content_block_id",
            "quote_span_start",
            "quote_span_end",
            "schema_version",
        }
        for block in payload["content_blocks"]:
            _assert(block["metadata"] is not None, "segmented block metadata must exist")
            _assert(
                set(block["metadata"].keys()) == expected_metadata_keys,
                "segmented block metadata keys mismatch",
            )
            start = block["metadata"]["quote_span_start"]
            end = block["metadata"]["quote_span_end"]
            _assert(expected_raw_content[start:end] == block["content"], "quote span must map to block content")

        _assert(repository.write_calls == 0, "segmented=true lookup must not write persistence")

        response_with_raw = client.get(
            "/documents/Doc Seg/task-units/task-unit-seg/content",
            params={"segmented": "true", "include_raw_content": "true"},
        )
        _assert(response_with_raw.status_code == 200, "segmented+include_raw_content should succeed")
        payload_with_raw = response_with_raw.json()
        _assert(
            payload_with_raw["content"] == expected_raw_content,
            "raw content should be available with compatibility flag",
        )
    finally:
        main.section_task_coordinator = original


def test_task_unit_content_segmented_true_suppresses_leading_hierarchy_title() -> None:
    content = (
        "Chapter Three\n"
        "One morning old Rouault brought Charles the money for setting his leg.\n\n"
        "Charles followed his advice."
    )
    coordinator, repository = _build_coordinator(
        StructuredDocument(
            document_id="doc-heading-trim",
            title="Doc Heading Trim",
            source_path=None,
            language="en",
            raw_text=content,
            chapters=[
                StructuredChapter(
                    chapter_id="chapter-three",
                    title="Chapter Three",
                    level=2,
                    chapter_role="main_body",
                    sections=[
                        _build_section(
                            section_id="section-three",
                            chapter_id="chapter-three",
                            title="Chapter Three",
                            unit_id="task-unit-heading-trim",
                            content=content,
                        )
                    ],
                )
            ],
            sections=[],
            structure_nodes=[],
        )
    )
    client = TestClient(main.app)
    original = main.section_task_coordinator
    main.section_task_coordinator = coordinator
    try:
        response = client.get(
            "/documents/Doc Heading Trim/task-units/task-unit-heading-trim/content",
            params={"segmented": "true"},
        )
        _assert(response.status_code == 200, "heading-trim segmented lookup should succeed")
        payload = response.json()

        first_block = payload["content_blocks"][0]
        expected_first_content = (
            "One morning old Rouault brought Charles the money for setting his leg."
        )
        _assert(
            first_block["content"] == expected_first_content,
            "segmented first block should not duplicate the hierarchy title",
        )
        _assert(
            first_block["metadata"]["quote_span_start"] == len("Chapter Three\n"),
            "trimmed first block span should start after the hierarchy title line",
        )
        _assert(
            content[
                first_block["metadata"]["quote_span_start"]:
                first_block["metadata"]["quote_span_end"]
            ]
            == first_block["content"],
            "trimmed first block quote span should still map to original task-unit content",
        )
        _assert(
            payload["content_blocks"][1]["content"] == "Charles followed his advice.",
            "second block should remain unchanged",
        )
        _assert(repository.write_calls == 0, "heading-trim lookup must not write persistence")
    finally:
        main.section_task_coordinator = original


def test_task_unit_content_segmented_true_supports_multilingual_paragraph_split() -> None:
    coordinator, repository = _build_coordinator(_build_multilingual_segmentable_document())
    client = TestClient(main.app)
    original = main.section_task_coordinator
    main.section_task_coordinator = coordinator
    expected_raw_content = "第一段：中文段落。\n\n第二段：日本語の段落。"
    try:
        response = client.get(
            "/documents/Doc CJK/task-units/task-unit-cjk/content",
            params={"segmented": "true"},
        )
        _assert(response.status_code == 200, "multilingual segmented=true lookup should succeed")
        payload = response.json()

        _assert(payload["content"] is None, "default multilingual segmented response should not expose raw content")
        _assert(len(payload["content_blocks"]) == 2, "CJK blank-line paragraphs should split into two blocks")
        _assert(payload["content_blocks"][0]["content"] == "第一段:中文段落。", "Chinese paragraph mismatch")
        _assert(payload["content_blocks"][1]["content"] == "第二段:日本語の段落。", "Japanese paragraph mismatch")
        _assert(payload["content_blocks"][0]["block_id"] == "task-unit-cjk:content:0", "first CJK block id mismatch")
        _assert(payload["content_blocks"][1]["block_id"] == "task-unit-cjk:content:1", "second CJK block id mismatch")

        expected_metadata_keys = {
            "source_hash",
            "content_block_id",
            "quote_span_start",
            "quote_span_end",
            "schema_version",
            "display_normalization",
        }
        for block in payload["content_blocks"]:
            _assert(block["metadata"] is not None, "CJK segmented block metadata must exist")
            _assert(
                set(block["metadata"].keys()) == expected_metadata_keys,
                "CJK segmented metadata keys mismatch",
            )
            start = block["metadata"]["quote_span_start"]
            end = block["metadata"]["quote_span_end"]
            _assert(
                normalize_ocr_text(expected_raw_content[start:end]) == block["content"],
                "CJK quote span should map to normalized block content",
            )
            _assert(
                block["metadata"]["display_normalization"] == "ocr_whitespace_v1",
                "CJK normalized blocks should declare display normalization",
            )

        _assert(repository.write_calls == 0, "multilingual segmented lookup must not write persistence")
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


def test_artifact_target_schema_invalid_target_level_fails_fast() -> None:
    try:
        ArtifactTargetRefResponse.model_validate(
            {
                "target_level": "invalid-level",
                "task_unit_id": "task-unit-1",
            }
        )
        raise AssertionError("invalid target_level should fail schema validation")
    except ValidationError:
        pass


def test_artifact_target_schema_level_constraints_fail_fast() -> None:
    try:
        ArtifactTargetRefResponse.model_validate(
            {
                "target_level": "content_block",
                "task_unit_id": "task-unit-1",
            }
        )
        raise AssertionError("content_block target should require content_block_id")
    except ValidationError:
        pass

    try:
        ArtifactTargetRefResponse.model_validate(
            {
                "target_level": "task_unit",
                "content_block_id": "task-unit-1:content:0",
            }
        )
        raise AssertionError("task_unit target should require task_unit_id")
    except ValidationError:
        pass


if __name__ == "__main__":
    test_task_layout_anchor_evidence_prefers_page_range_when_available()
    test_task_layout_id_then_content_lookup_success()
    test_batch_task_unit_content_lookup_preserves_order_and_read_only()
    test_batch_task_unit_content_duplicate_request_id_fails_fast()
    test_task_unit_content_segmented_true_returns_deterministic_multi_blocks()
    test_task_unit_content_segmented_true_suppresses_leading_hierarchy_title()
    test_task_unit_content_segmented_true_supports_multilingual_paragraph_split()
    test_task_unit_content_missing_id_returns_404()
    test_task_unit_content_duplicate_id_fails_fast()
    test_task_unit_content_does_not_fallback_to_root_sections()
    test_artifact_target_schema_invalid_target_level_fails_fast()
    test_artifact_target_schema_level_constraints_fail_fast()
    print("ok")
