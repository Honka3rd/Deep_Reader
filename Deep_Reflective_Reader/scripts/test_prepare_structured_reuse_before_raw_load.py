#!/usr/bin/env python3
"""Regression checks for structured reuse before expensive raw PDF loading."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from document_preparation.document_preparation_pipeline import (  # noqa: E402
    DocumentPreparationPipeline,
)
from document_preparation.preparation_mode import PreparationMode  # noqa: E402
from document_structure.section_splitter_selector import SectionSplitterMode  # noqa: E402
from document_structure.structured_document import (  # noqa: E402
    StructuredChapter,
    StructuredDocument,
    StructuredSection,
)
from language.language_code import LanguageCode  # noqa: E402
from profile.document_profile import DocumentProfile  # noqa: E402


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _fake_document(doc_name: str = "國富論lite") -> StructuredDocument:
    section = StructuredSection(
        section_id="section-0",
        section_index=0,
        title="第一章",
        level=1,
        content="已持久化的 OCR 結構內容。",
        char_start=0,
        char_end=13,
    )
    chapter = StructuredChapter(
        chapter_id="chapter-0",
        title="第一章",
        level=0,
        chapter_role=None,
        sections=[section],
    )
    return StructuredDocument(
        document_id=doc_name,
        title=doc_name,
        source_path=None,
        language="zh",
        raw_text=section.content,
        chapters=[chapter],
    )


class _FailIfLoadedLoader:
    def load(self, doc_name: str) -> str:
        raise AssertionError(f"raw loader should not be called for {doc_name}")


class _TextLoader:
    def __init__(self) -> None:
        self.load_calls = 0

    def load(self, doc_name: str) -> str:
        _ = doc_name
        self.load_calls += 1
        return "第一章\nforce rebuild raw text"


class _LoaderFactory:
    def __init__(self, loader: object) -> None:
        self.loader = loader
        self.get_calls = 0

    def get(self, doc_name: str) -> object:
        _ = doc_name
        self.get_calls += 1
        return self.loader


class _StructuredStore:
    def __init__(self, *, exists_value: bool, document: StructuredDocument) -> None:
        self.exists_value = exists_value
        self.document = document
        self.exists_calls = 0
        self.load_calls = 0
        self.save_calls = 0

    def location(self, target: object) -> str:
        _ = target
        return f"postgres://structured/default/{self.document.document_id}"

    def exists(self, target: object) -> bool:
        _ = target
        self.exists_calls += 1
        return self.exists_value

    def load(self, target: object) -> StructuredDocument:
        _ = target
        self.load_calls += 1
        return self.document

    def save(self, document: StructuredDocument, target: object) -> None:
        _ = target
        self.save_calls += 1
        self.document = document
        self.exists_value = True


class _LanguageDetector:
    def detect(self, raw_text: str) -> str:
        _ = raw_text
        return "zh"


class _ProfileBuilder:
    def build(self, *, text: str, document_language: str) -> DocumentProfile:
        _ = text
        return DocumentProfile(
            topic="test",
            summary="test",
            document_language=LanguageCode(document_language),
        )


class _ProfileStore:
    def exists(self, config: object) -> bool:
        _ = config
        return False

    def load(self, config: object) -> DocumentProfile:
        raise AssertionError(f"profile load should not be called: {config}")

    def clear(self, config: object) -> None:
        _ = config

    def save(self, profile: DocumentProfile, config: object) -> None:
        _ = (profile, config)


class _StructuredBuilder:
    def build(self, **kwargs: Any) -> StructuredDocument:
        return _fake_document(str(kwargs["document_id"]))


class _NoopPostStructureEnricher:
    def enrich(
        self,
        *,
        profile: DocumentProfile,
        structured_document: StructuredDocument,
    ) -> DocumentProfile:
        _ = structured_document
        return profile


class _NoopFaissStore:
    def has_position_metadata(self, config: object) -> bool:
        _ = config
        return False

    def clear(self, config: object) -> None:
        _ = config

    def save(self, bundle: object, config: object) -> None:
        _ = (bundle, config)


class _NoopFingerprintHandler:
    def matches(self, raw_text: str, path: str) -> bool:
        _ = (raw_text, path)
        return False

    def clear(self, path: str) -> None:
        _ = path

    def save(self, raw_text: str, path: str) -> None:
        _ = (raw_text, path)


class _NoopNodeProvider:
    def parse(self, raw_text: str, config: object) -> object:
        _ = (raw_text, config)
        return object()


class _NoopFaissBuilder:
    def build_from_parsed_document(self, parsed_document: object) -> object:
        _ = parsed_document
        return object()


class _NoopBundleProvider:
    def get_bundle_from_raw_text(
        self,
        *,
        doc_name: str,
        raw_text: str,
        force_rebuild: bool,
    ) -> object:
        _ = (doc_name, raw_text, force_rebuild)
        return object()


def _build_pipeline(
    *,
    loader: object,
    store: _StructuredStore,
) -> DocumentPreparationPipeline:
    return DocumentPreparationPipeline(
        loader_factory=_LoaderFactory(loader),  # type: ignore[arg-type]
        language_detector=_LanguageDetector(),  # type: ignore[arg-type]
        structured_document_builder=_StructuredBuilder(),  # type: ignore[arg-type]
        structured_document_store=store,  # type: ignore[arg-type]
        node_provider=_NoopNodeProvider(),  # type: ignore[arg-type]
        faiss_index_builder=_NoopFaissBuilder(),  # type: ignore[arg-type]
        faiss_index_store=_NoopFaissStore(),  # type: ignore[arg-type]
        fingerprint_handler=_NoopFingerprintHandler(),  # type: ignore[arg-type]
        profile_builder=_ProfileBuilder(),  # type: ignore[arg-type]
        profile_store=_ProfileStore(),  # type: ignore[arg-type]
        bundle_provider=_NoopBundleProvider(),  # type: ignore[arg-type]
        post_structure_enricher=_NoopPostStructureEnricher(),  # type: ignore[arg-type]
    )


def test_base_prepare_reuses_existing_structured_before_raw_load() -> None:
    document = _fake_document()
    store = _StructuredStore(exists_value=True, document=document)
    pipeline = _build_pipeline(loader=_FailIfLoadedLoader(), store=store)

    assets = pipeline.prepare(
        doc_name="國富論lite",
        force_rebuild=False,
        mode=PreparationMode.BASE,
        structured_parser_mode=SectionSplitterMode.COMMON,
    )

    loader_factory = pipeline.loader_factory
    _assert(
        getattr(loader_factory, "get_calls") == 0,
        "raw loader factory should not be used",
    )
    _assert(store.exists_calls == 1, "structured store should be checked once")
    _assert(store.load_calls == 1, "existing structured document should be validated")
    _assert(store.save_calls == 0, "existing structured document should not be rewritten")
    _assert(assets.structured_document_ready is True, "structured should be ready")
    _assert(
        assets.structured_document_path == "postgres://structured/default/國富論lite",
        f"unexpected structured path: {assets.structured_document_path}",
    )
    _assert(assets.raw_text is None, "raw text should not be loaded on structured reuse")
    _assert(assets.errors == [], f"unexpected errors: {assets.errors}")


def test_force_rebuild_keeps_raw_load_and_rebuild_behavior() -> None:
    document = _fake_document()
    loader = _TextLoader()
    store = _StructuredStore(exists_value=True, document=document)
    pipeline = _build_pipeline(loader=loader, store=store)

    assets = pipeline.prepare(
        doc_name="國富論lite",
        force_rebuild=True,
        mode=PreparationMode.BASE,
        structured_parser_mode=SectionSplitterMode.COMMON,
    )

    _assert(loader.load_calls == 1, "force rebuild should load raw text")
    _assert(store.save_calls == 1, "force rebuild should rewrite structured document")
    _assert(
        assets.raw_text is not None and assets.raw_text.strip(),
        "raw text should be present",
    )
    _assert(assets.structured_document_ready is True, "force rebuild should prepare structured")


if __name__ == "__main__":
    test_base_prepare_reuses_existing_structured_before_raw_load()
    test_force_rebuild_keeps_raw_load_and_rebuild_behavior()
    print("prepare structured reuse before raw-load checks passed")
