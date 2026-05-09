#!/usr/bin/env python3
"""Post-structure profile metadata enrichment tests."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

from document_preparation.document_preparation_pipeline import DocumentPreparationPipeline
from document_preparation.prepared_document_assets import PreparedDocumentAssets
from document_preparation.preparation_mode import PreparationMode
from document_structure.section_role import SectionRole
from document_structure.structured_document import (
    StructuredChapter,
    StructuredDocument,
    StructuredSection,
)
from document_structure.structured_document_store import StructuredDocumentStore
from language.language_code import LanguageCode
from profile.document_profile import (
    DocumentProfile,
    DocumentStructureShape,
    LikelihoodLevel,
    ParserRelevantMetadata,
    PostStructureMetadata,
)
from profile.document_profile_store import DocumentProfileStore
from profile.post_structure_metadata_enricher import PostStructureMetadataEnricher
from shared.task_unit_model import TaskUnit


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _unit(unit_id: str, section_id: str) -> TaskUnit:
    return TaskUnit(
        unit_id=unit_id,
        title=None,
        container_title=None,
        content=f"content-{unit_id}",
        source_section_ids=[section_id],
        is_fallback_generated=False,
        parent_section_id=section_id,
    )


def _section(
    section_id: str,
    *,
    title: str | None,
    content: str = "section-content",
    section_index: int = 0,
    section_role: SectionRole | None = SectionRole.MAIN_BODY,
    parent_chapter_id: str | None = None,
    section_kind: str | None = "chapter_body",
    is_implicit_section: bool = True,
    task_unit_count: int = 0,
) -> StructuredSection:
    return StructuredSection(
        section_id=section_id,
        section_index=section_index,
        title=title,
        level=1,
        content=content,
        char_start=0,
        char_end=len(content),
        section_role=section_role,
        parent_chapter_id=parent_chapter_id,
        section_kind=section_kind,
        is_implicit_section=is_implicit_section,
        task_units=[_unit(f"{section_id}-unit-{i}", section_id) for i in range(task_unit_count)],
    )


def _chapter(
    chapter_id: str,
    *,
    title: str | None,
    chapter_role: str | None = "main_body",
    sections: list[StructuredSection] | None = None,
) -> StructuredChapter:
    return StructuredChapter(
        chapter_id=chapter_id,
        title=title,
        level=1,
        chapter_role=chapter_role,
        sections=list(sections or []),
    )


def _document(chapters: list[StructuredChapter]) -> StructuredDocument:
    return StructuredDocument(
        document_id="doc",
        title="Doc",
        source_path=None,
        language="en",
        raw_text="raw",
        chapters=chapters,
        sections=[],
        structure_nodes=[],
    )


def _profile() -> DocumentProfile:
    return DocumentProfile(
        topic="topic",
        summary="summary",
        document_language=LanguageCode.EN,
        parser_metadata=ParserRelevantMetadata(),
    )


class _Noop:
    @staticmethod
    def detect(raw_text: str) -> str:
        _ = raw_text
        return "en"

    @staticmethod
    def get(doc_name: str):
        _ = doc_name
        return _Noop()

    @staticmethod
    def load(doc_name: str) -> str:
        _ = doc_name
        return "raw text"

    @staticmethod
    def parse(raw_text, config):
        _ = (raw_text, config)
        return None

    @staticmethod
    def build_from_parsed_document(parsed_document):
        _ = parsed_document
        return None

    @staticmethod
    def has_position_metadata(config):
        _ = config
        return False

    @staticmethod
    def clear(config):
        _ = config

    @staticmethod
    def save(*args, **kwargs):
        _ = (args, kwargs)

    @staticmethod
    def matches(raw_text, path):
        _ = (raw_text, path)
        return False

    @staticmethod
    def get_bundle_from_raw_text(doc_name, raw_text, force_rebuild=False):
        _ = (doc_name, raw_text, force_rebuild)
        return {"ok": True}


class _CaptureProfileStore:
    def __init__(self) -> None:
        self.saved_profiles: list[DocumentProfile] = []

    def exists(self, config) -> bool:
        _ = config
        return False

    def load(self, config) -> DocumentProfile:
        _ = config
        raise RuntimeError("not expected")

    def clear(self, config) -> None:
        _ = config

    def save(self, profile: DocumentProfile, config) -> None:
        _ = config
        self.saved_profiles.append(profile)


class _RaisingEnricher:
    def enrich(self, *, profile: DocumentProfile, structured_document: StructuredDocument) -> DocumentProfile:
        _ = (profile, structured_document)
        raise RuntimeError("boom")


class _PipelineForEnrichment(DocumentPreparationPipeline):
    def __init__(
        self,
        *,
        structured_document: StructuredDocument,
        profile_store: _CaptureProfileStore,
        enricher=None,
    ) -> None:
        self._structured_document = structured_document
        self._structured_path = Path("data/structured") / f"test-{uuid.uuid4()}.structured.json"
        super().__init__(
            loader_factory=_Noop(),
            language_detector=_Noop(),
            structured_document_builder=_Noop(),
            structured_document_store=StructuredDocumentStore(),
            node_provider=_Noop(),
            faiss_index_builder=_Noop(),
            faiss_index_store=_Noop(),
            fingerprint_handler=_Noop(),
            profile_builder=_Noop(),
            profile_store=profile_store,
            bundle_provider=_Noop(),
            post_structure_enricher=enricher or PostStructureMetadataEnricher(),
        )

    def _prepare_profile(
        self,
        doc_name: str,
        raw_text: str | None,
        language: str | None,
        assets: PreparedDocumentAssets,
        force_rebuild: bool,
    ) -> tuple[bool, DocumentProfile | None]:
        _ = (doc_name, raw_text, language, assets, force_rebuild)
        return True, _profile()

    def _prepare_structured_document(
        self,
        doc_name: str,
        raw_text: str | None,
        language: str | None,
        document_profile: DocumentProfile | None,
        assets: PreparedDocumentAssets,
        force_rebuild: bool,
        parser_mode,
    ) -> tuple[bool, str | None]:
        _ = (doc_name, raw_text, language, document_profile, assets, force_rebuild, parser_mode)
        self._structured_path.parent.mkdir(parents=True, exist_ok=True)
        StructuredDocumentStore.save(self._structured_document, str(self._structured_path))
        return True, str(self._structured_path)


def test_profile_round_trip_with_post_metadata() -> None:
    profile = DocumentProfile(
        topic="topic",
        summary="summary",
        document_language=LanguageCode.EN,
        post_structure_metadata=PostStructureMetadata(
            chapter_count=2,
            section_count=3,
            task_unit_stats_available=True,
            title_uniqueness_risk=LikelihoodLevel.HIGH,
            actual_structure_shape=DocumentStructureShape.CHAPTER_ONLY,
            chapter_title_coverage=1.0,
            section_title_coverage=0.66,
        ),
    )
    payload = profile.to_dict()
    restored = DocumentProfile.from_dict(payload)
    _assert(restored.post_structure_metadata is not None, "post metadata should round-trip")
    _assert(restored.post_structure_metadata.chapter_count == 2, "chapter_count should persist")
    _assert(
        restored.post_structure_metadata.actual_structure_shape == DocumentStructureShape.CHAPTER_ONLY,
        "shape should round-trip",
    )

    namespace = f"post-meta-roundtrip-{uuid.uuid4()}"
    store = DocumentProfileStore()
    from config.faiss_storage_config import FaissStorageConfig

    config = FaissStorageConfig(namespace=namespace)
    try:
        store.save(profile, config)
        loaded = store.load(config)
        _assert(loaded.post_structure_metadata is not None, "store round-trip should keep post metadata")
        _assert(loaded.post_structure_metadata.section_count == 3, "section_count should persist in store")
    finally:
        config_path = Path(config.get_raw_profile_path())
        if config_path.exists():
            config_path.unlink()


def test_basic_counts_and_duplicate_risk() -> None:
    chapter_1_sections = [
        _section("s1", title="Chapter One", parent_chapter_id="c1", task_unit_count=2),
    ]
    chapter_2_sections = [
        _section("s2", title="Chapter One", parent_chapter_id="c2", task_unit_count=2),
    ]
    document = _document(
        [
            _chapter("c1", title="Chapter One", sections=chapter_1_sections),
            _chapter("c2", title="Chapter One", sections=chapter_2_sections),
        ]
    )
    enriched = PostStructureMetadataEnricher().enrich(
        profile=_profile(),
        structured_document=document,
    )
    metadata = enriched.post_structure_metadata
    _assert(metadata is not None, "metadata should exist")
    _assert(metadata.chapter_count == 2, "chapter_count should be 2")
    _assert(metadata.section_count == 2, "section_count should be 2")
    _assert(metadata.task_unit_count == 4, "task_unit_count should be 4")
    _assert(metadata.task_unit_stats_available is True, "task unit stats should be marked available")
    _assert("Chapter One" in metadata.duplicate_chapter_titles, "duplicate chapter title should be detected")
    _assert(metadata.title_uniqueness_risk == LikelihoodLevel.HIGH, "duplicate chapter titles should be high risk")


def test_task_unit_stats_availability() -> None:
    no_units_doc = _document(
        [
            _chapter(
                "c1",
                title="Chapter One",
                sections=[_section("s1", title="Chapter One", parent_chapter_id="c1", task_unit_count=0)],
            )
        ]
    )
    no_units_meta = PostStructureMetadataEnricher().enrich(
        profile=_profile(),
        structured_document=no_units_doc,
    ).post_structure_metadata
    _assert(no_units_meta is not None, "metadata should exist")
    _assert(no_units_meta.task_unit_stats_available is False, "task unit stats should be unavailable")
    _assert(no_units_meta.avg_task_units_per_section is None, "avg task units should be None when unavailable")
    _assert(
        "task_units_not_available_at_enrichment_time" in no_units_meta.notes,
        "missing task units note should be emitted",
    )

    with_units_doc = _document(
        [
            _chapter(
                "c2",
                title="Chapter Two",
                sections=[_section("s2", title="Chapter Two", parent_chapter_id="c2", task_unit_count=3)],
            )
        ]
    )
    with_units_meta = PostStructureMetadataEnricher().enrich(
        profile=_profile(),
        structured_document=with_units_doc,
    ).post_structure_metadata
    _assert(with_units_meta is not None, "metadata should exist")
    _assert(with_units_meta.task_unit_stats_available is True, "task unit stats should be available")
    _assert(with_units_meta.avg_task_units_per_section == 3.0, "avg task units should be computed")


def test_duplicate_section_title_noise_filtering() -> None:
    implicit_mirror_doc = _document(
        [
            _chapter(
                "c1",
                title="Chapter One",
                sections=[_section("s1", title="Chapter One", parent_chapter_id="c1", is_implicit_section=True)],
            ),
            _chapter(
                "c2",
                title="Chapter One",
                sections=[_section("s2", title="Chapter One", parent_chapter_id="c2", is_implicit_section=True)],
            ),
        ]
    )
    mirror_meta = PostStructureMetadataEnricher().enrich(
        profile=_profile(),
        structured_document=implicit_mirror_doc,
    ).post_structure_metadata
    _assert(mirror_meta is not None, "metadata should exist")
    _assert("Chapter One" in mirror_meta.duplicate_chapter_titles, "chapter duplicate should remain")
    _assert(
        "Chapter One" not in mirror_meta.duplicate_section_titles,
        "implicit mirror section duplicates should be excluded",
    )

    explicit_dup_doc = _document(
        [
            _chapter(
                "c3",
                title="Chapter Three",
                sections=[
                    _section(
                        "s3",
                        title="Background",
                        parent_chapter_id="c3",
                        is_implicit_section=False,
                        section_kind="subsection",
                    ),
                    _section(
                        "s4",
                        title="Background",
                        parent_chapter_id="c3",
                        is_implicit_section=False,
                        section_kind="subsection",
                    ),
                ],
            )
        ]
    )
    explicit_meta = PostStructureMetadataEnricher().enrich(
        profile=_profile(),
        structured_document=explicit_dup_doc,
    ).post_structure_metadata
    _assert(explicit_meta is not None, "metadata should exist")
    _assert(
        "Background" in explicit_meta.duplicate_section_titles,
        "explicit section duplicate should be detected",
    )


def test_shape_detection_cases() -> None:
    madame_like_doc = _document(
        [
            _chapter(
                "c1",
                title="Chapter One",
                sections=[_section("s1", title="Chapter One", parent_chapter_id="c1", is_implicit_section=True)],
            ),
            _chapter(
                "c2",
                title="Chapter Two",
                sections=[_section("s2", title="Chapter Two", parent_chapter_id="c2", is_implicit_section=True)],
            ),
            _chapter(
                "c3",
                title="Chapter One",
                sections=[_section("s3", title="Chapter One", parent_chapter_id="c3", is_implicit_section=True)],
            ),
            _chapter(
                "c4",
                title="Chapter Two",
                sections=[_section("s4", title="Chapter Two", parent_chapter_id="c4", is_implicit_section=True)],
            ),
            _chapter(
                "c5",
                title="Chapter Three",
                sections=[_section("s5", title="Chapter Three", parent_chapter_id="c5", is_implicit_section=True)],
            ),
            _chapter(
                "c6",
                title="Chapter Three",
                sections=[_section("s6", title="Chapter Three", parent_chapter_id="c6", is_implicit_section=True)],
            ),
        ]
    )
    madame_like_meta = PostStructureMetadataEnricher().enrich(
        profile=_profile(),
        structured_document=madame_like_doc,
    ).post_structure_metadata
    _assert(madame_like_meta is not None, "madame-like metadata should exist")
    _assert(
        madame_like_meta.actual_structure_shape in {
            DocumentStructureShape.PART_CHAPTER,
            DocumentStructureShape.MIXED,
        },
        "madame-like duplicate local chapter titles should avoid chapter_only",
    )
    _assert(
        madame_like_meta.title_uniqueness_risk == LikelihoodLevel.HIGH,
        "madame-like duplicate titles should be high risk",
    )
    _assert(
        "possible_part_chapter_repeated_local_titles" in madame_like_meta.notes,
        "madame-like structure should carry repeated-local-title note",
    )

    chapter_only_doc = _document(
        [
            _chapter(
                f"c{i}",
                title=f"Chapter {i}",
                sections=[
                    _section(
                        f"s{i}",
                        title=f"Chapter {i}",
                        parent_chapter_id=f"c{i}",
                        is_implicit_section=True,
                        task_unit_count=1,
                    )
                ],
            )
            for i in range(10)
        ]
    )
    chapter_only_meta = PostStructureMetadataEnricher().enrich(
        profile=_profile(),
        structured_document=chapter_only_doc,
    ).post_structure_metadata
    _assert(chapter_only_meta is not None, "chapter_only metadata should exist")
    _assert(
        chapter_only_meta.actual_structure_shape == DocumentStructureShape.CHAPTER_ONLY,
        "chapter_only shape should be detected",
    )
    _assert(
        chapter_only_meta.title_uniqueness_risk in {LikelihoodLevel.LOW, LikelihoodLevel.NONE},
        "xu-sanguan-like unique titles should be low/none risk",
    )

    essay_doc = _document(
        [
            _chapter(
                "e1",
                title="Theme",
                sections=[
                    _section(
                        "es1",
                        title="Point One",
                        content="short content",
                        parent_chapter_id="e1",
                        is_implicit_section=False,
                    )
                ],
            ),
            _chapter(
                "e2",
                title="Method",
                sections=[
                    _section(
                        "es2",
                        title="Point Two",
                        content="short content",
                        parent_chapter_id="e2",
                        is_implicit_section=False,
                    )
                ],
            ),
        ]
    )
    essay_meta = PostStructureMetadataEnricher().enrich(
        profile=_profile(),
        structured_document=essay_doc,
    ).post_structure_metadata
    _assert(essay_meta is not None, "essay metadata should exist")
    _assert(
        essay_meta.actual_structure_shape == DocumentStructureShape.ESSAY_SECTIONS,
        "essay_sections shape should be detected",
    )

    flat_doc = _document(
        [
            _chapter(
                "f1",
                title=None,
                sections=[
                    _section(
                        "fs1",
                        title=None,
                        content="x" * 8000,
                        parent_chapter_id="f1",
                        is_implicit_section=True,
                    )
                ],
            )
        ]
    )
    flat_meta = PostStructureMetadataEnricher().enrich(
        profile=_profile(),
        structured_document=flat_doc,
    ).post_structure_metadata
    _assert(flat_meta is not None, "flat metadata should exist")
    _assert(
        flat_meta.actual_structure_shape == DocumentStructureShape.FLAT_LONG_TEXT,
        "flat long text shape should be detected",
    )


def test_pipeline_enrichment_non_blocking_and_base_mode_save() -> None:
    document = _document(
        [
            _chapter(
                "c1",
                title="Chapter One",
                sections=[_section("s1", title="Chapter One", parent_chapter_id="c1", task_unit_count=1)],
            )
        ]
    )
    failing_store = _CaptureProfileStore()
    failing_pipeline = _PipelineForEnrichment(
        structured_document=document,
        profile_store=failing_store,
        enricher=_RaisingEnricher(),
    )
    assets_fail = failing_pipeline.prepare(
        doc_name="enrich-fail-doc",
        force_rebuild=True,
        mode=PreparationMode.BASE,
    )
    _assert(
        assets_fail.structured_document_ready is True,
        "enrichment failure should not block structured readiness",
    )
    _assert(
        any("post_structure_enrichment_failed" in error for error in assets_fail.errors),
        "enrichment failure should be recorded",
    )

    ok_store = _CaptureProfileStore()
    ok_pipeline = _PipelineForEnrichment(
        structured_document=document,
        profile_store=ok_store,
    )
    assets_ok = ok_pipeline.prepare(
        doc_name="enrich-ok-doc",
        force_rebuild=True,
        mode=PreparationMode.BASE,
    )
    _assert(assets_ok.profile_ready is True, "base mode should keep profile ready")
    _assert(
        assets_ok.structured_document_ready is True,
        "base mode should keep structured ready",
    )
    _assert(ok_store.saved_profiles, "enriched profile should be saved")
    saved_profile = ok_store.saved_profiles[-1]
    _assert(
        saved_profile.post_structure_metadata is not None,
        "saved profile should include post_structure_metadata",
    )


def main() -> None:
    test_profile_round_trip_with_post_metadata()
    test_basic_counts_and_duplicate_risk()
    test_task_unit_stats_availability()
    test_duplicate_section_title_noise_filtering()
    test_shape_detection_cases()
    test_pipeline_enrichment_non_blocking_and_base_mode_save()
    print(
        json.dumps(
            {
                "status": "ok",
                "tests": [
                    "post_structure_round_trip",
                    "counts_and_duplicate_risk",
                    "task_unit_stats_availability",
                    "duplicate_section_title_noise_filtering",
                    "shape_detection",
                    "pipeline_non_blocking_and_base_mode_save",
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
