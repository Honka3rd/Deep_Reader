#!/usr/bin/env python3
"""Pipeline-order regression tests: prepare profile before structured document."""

from __future__ import annotations

import json
from dataclasses import dataclass

from document_preparation.document_preparation_pipeline import DocumentPreparationPipeline
from document_preparation.prepared_document_assets import PreparedDocumentAssets
from document_preparation.preparation_mode import PreparationMode
from profile.document_profile import DocumentProfile


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


class _NoopLoader:
    @staticmethod
    def load(doc_name: str) -> str:
        _ = doc_name
        return "raw-text"


class _NoopLoaderFactory:
    @staticmethod
    def get(doc_name: str):
        _ = doc_name
        return _NoopLoader()


class _NoopLanguageDetector:
    @staticmethod
    def detect(raw_text: str) -> str:
        _ = raw_text
        return "zh"


class _NoopStructuredBuilder:
    @staticmethod
    def build(**kwargs):
        _ = kwargs
        return {"ok": True}


class _NoopStructuredStore:
    @staticmethod
    def save(document, target):
        _ = (document, target)

    @staticmethod
    def load(config_or_path):
        _ = config_or_path
        return {"ok": True}


class _NoopNodeProvider:
    @staticmethod
    def parse(raw_text, config):
        _ = (raw_text, config)
        return None


class _NoopFaissBuilder:
    @staticmethod
    def build_from_parsed_document(parsed_document):
        _ = parsed_document
        return None


class _NoopFaissStore:
    @staticmethod
    def has_position_metadata(config):
        _ = config
        return False

    @staticmethod
    def clear(config):
        _ = config

    @staticmethod
    def save(bundle, config):
        _ = (bundle, config)


class _NoopFingerprintHandler:
    @staticmethod
    def matches(raw_text, path):
        _ = (raw_text, path)
        return False

    @staticmethod
    def clear(path):
        _ = path

    @staticmethod
    def save(raw_text, path):
        _ = (raw_text, path)


class _NoopBundleProvider:
    @staticmethod
    def get_bundle_from_raw_text(doc_name, raw_text, force_rebuild=False):
        _ = (doc_name, raw_text, force_rebuild)
        return {"ok": True}


class _FakeProfileBuilder:
    def __init__(self, *, should_raise: bool = False):
        self.should_raise = should_raise
        self.build_calls = 0

    def build(self, text: str, document_language: str) -> DocumentProfile:
        self.build_calls += 1
        if self.should_raise:
            raise RuntimeError("profile_builder_error")
        return DocumentProfile(
            topic="essay",
            summary=f"summary for {text[:8]}",
            document_language=document_language,
        )


@dataclass
class _FakeProfileStore:
    exists_value: bool = False
    load_should_raise: bool = False
    loaded_profile: DocumentProfile | None = None
    exists_calls: int = 0
    load_calls: int = 0
    clear_calls: int = 0
    save_calls: int = 0

    def exists(self, config) -> bool:
        _ = config
        self.exists_calls += 1
        return self.exists_value

    def load(self, config) -> DocumentProfile:
        _ = config
        self.load_calls += 1
        if self.load_should_raise:
            raise RuntimeError("profile_load_error")
        return self.loaded_profile or DocumentProfile(
            topic="loaded",
            summary="loaded-summary",
            document_language="zh",
        )

    def clear(self, config) -> None:
        _ = config
        self.clear_calls += 1
        self.exists_value = False

    def save(self, profile: DocumentProfile, config) -> None:
        _ = config
        self.save_calls += 1
        self.loaded_profile = profile
        self.exists_value = True


class _OrderTrackingPipeline(DocumentPreparationPipeline):
    def __init__(self, *, profile_should_fail: bool = False):
        self.order: list[str] = []
        self.profile_should_fail = profile_should_fail
        self.profile_calls = 0
        self.structured_calls = 0
        self.faiss_calls = 0
        self.bundle_calls = 0
        self.profile_arg_seen: DocumentProfile | None = None
        super().__init__(
            loader_factory=_NoopLoaderFactory(),
            language_detector=_NoopLanguageDetector(),
            structured_document_builder=_NoopStructuredBuilder(),
            structured_document_store=_NoopStructuredStore(),
            node_provider=_NoopNodeProvider(),
            faiss_index_builder=_NoopFaissBuilder(),
            faiss_index_store=_NoopFaissStore(),
            fingerprint_handler=_NoopFingerprintHandler(),
            profile_builder=_FakeProfileBuilder(),
            profile_store=_FakeProfileStore(),
            bundle_provider=_NoopBundleProvider(),
        )

    def _load_raw_text(self, doc_name: str, assets: PreparedDocumentAssets) -> str | None:
        _ = doc_name
        self.order.append("raw")
        return "raw-text"

    def _detect_language(self, doc_name: str, raw_text: str | None, assets: PreparedDocumentAssets) -> str | None:
        _ = (doc_name, raw_text, assets)
        self.order.append("language")
        return "zh"

    def _prepare_profile(
        self,
        doc_name: str,
        raw_text: str | None,
        language: str | None,
        assets: PreparedDocumentAssets,
        force_rebuild: bool,
    ) -> tuple[bool, DocumentProfile | None]:
        _ = (doc_name, raw_text, language, force_rebuild)
        self.order.append("profile")
        self.profile_calls += 1
        if self.profile_should_fail:
            assets.errors.append("prepare_profile_failed:mock_failure")
            return False, None
        return True, DocumentProfile(topic="essay", summary="ok", document_language="zh")

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
        _ = (doc_name, raw_text, language, assets, force_rebuild, parser_mode)
        self.order.append("structured")
        self.structured_calls += 1
        self.profile_arg_seen = document_profile
        return True, "/tmp/mock.structured.json"

    def _prepare_faiss(self, doc_name: str, raw_text: str | None, assets: PreparedDocumentAssets, force_rebuild: bool) -> tuple[bool, str | None]:
        _ = (doc_name, raw_text, assets, force_rebuild)
        self.order.append("faiss")
        self.faiss_calls += 1
        return True, "mock-namespace"

    def _prepare_bundle(self, doc_name: str, raw_text: str | None, assets: PreparedDocumentAssets, force_rebuild: bool) -> bool:
        _ = (doc_name, raw_text, assets, force_rebuild)
        self.order.append("bundle")
        self.bundle_calls += 1
        return True


def _build_real_pipeline_for_profile_tests(
    *,
    profile_builder: _FakeProfileBuilder,
    profile_store: _FakeProfileStore,
) -> DocumentPreparationPipeline:
    return DocumentPreparationPipeline(
        loader_factory=_NoopLoaderFactory(),
        language_detector=_NoopLanguageDetector(),
        structured_document_builder=_NoopStructuredBuilder(),
        structured_document_store=_NoopStructuredStore(),
        node_provider=_NoopNodeProvider(),
        faiss_index_builder=_NoopFaissBuilder(),
        faiss_index_store=_NoopFaissStore(),
        fingerprint_handler=_NoopFingerprintHandler(),
        profile_builder=profile_builder,
        profile_store=profile_store,
        bundle_provider=_NoopBundleProvider(),
    )


def test_base_mode_prepares_profile_before_structured() -> None:
    pipeline = _OrderTrackingPipeline(profile_should_fail=False)
    assets = pipeline.prepare(
        doc_name="order-doc",
        force_rebuild=False,
        mode=PreparationMode.BASE,
    )
    _assert(pipeline.order == ["raw", "language", "profile", "structured"], "base mode order should be raw->language->profile->structured")
    _assert(pipeline.profile_calls == 1, "profile should be prepared once in base mode")
    _assert(pipeline.structured_calls == 1, "structured should be prepared once in base mode")
    _assert(pipeline.faiss_calls == 0 and pipeline.bundle_calls == 0, "base mode should skip faiss/bundle")
    _assert(assets.profile_ready is True, "base mode should mark profile_ready true when profile succeeds")
    _assert(assets.structured_document_ready is True, "base mode should still prepare structured document")
    _assert(pipeline.profile_arg_seen is not None, "structured step should receive prepared profile")


def test_free_qa_no_duplicate_profile_build() -> None:
    pipeline = _OrderTrackingPipeline(profile_should_fail=False)
    assets = pipeline.prepare(
        doc_name="free-doc",
        force_rebuild=False,
        mode=PreparationMode.FREE_QA,
    )
    _assert(pipeline.profile_calls == 1, "free_qa should prepare profile exactly once")
    _assert(pipeline.order == ["raw", "language", "profile", "structured", "faiss", "bundle"], "free_qa order should include profile before structured and no duplicate profile step")
    _assert(assets.profile_ready is True and assets.structured_document_ready is True, "free_qa should report both profile and structured ready")


def test_profile_failure_does_not_block_structured() -> None:
    pipeline = _OrderTrackingPipeline(profile_should_fail=True)
    assets = pipeline.prepare(
        doc_name="profile-fail-doc",
        force_rebuild=False,
        mode=PreparationMode.BASE,
    )
    _assert(assets.profile_ready is False, "profile failure should mark profile_ready false")
    _assert(assets.structured_document_ready is True, "profile failure must not block structured build")
    _assert(any("prepare_profile_failed" in error for error in assets.errors), "profile failure reason should be recorded")
    _assert(pipeline.profile_arg_seen is None, "structured should receive None profile when profile preparation failed")


def test_existing_profile_reuse_without_rebuild() -> None:
    profile_store = _FakeProfileStore(exists_value=True, load_should_raise=False)
    profile_builder = _FakeProfileBuilder(should_raise=False)
    pipeline = _build_real_pipeline_for_profile_tests(
        profile_builder=profile_builder,
        profile_store=profile_store,
    )
    assets = PreparedDocumentAssets(
        doc_name="reuse-doc",
        raw_text="raw",
        language="zh",
        structured_document_ready=False,
        faiss_ready=False,
        profile_ready=False,
        bundle_ready=False,
        structured_document_path=None,
        faiss_namespace=None,
        errors=[],
    )
    ready, profile = pipeline._prepare_profile(
        doc_name="reuse-doc",
        raw_text="raw",
        language="zh",
        assets=assets,
        force_rebuild=False,
    )
    _assert(ready is True and profile is not None, "existing profile should be loaded successfully")
    _assert(profile_store.load_calls == 1, "existing profile should be loaded")
    _assert(profile_builder.build_calls == 0, "existing profile should skip rebuild")
    _assert(profile_store.clear_calls == 0, "existing profile reuse should not clear store")


def test_force_rebuild_clears_and_rebuilds_profile() -> None:
    profile_store = _FakeProfileStore(exists_value=True, load_should_raise=False)
    profile_builder = _FakeProfileBuilder(should_raise=False)
    pipeline = _build_real_pipeline_for_profile_tests(
        profile_builder=profile_builder,
        profile_store=profile_store,
    )
    assets = PreparedDocumentAssets(
        doc_name="rebuild-doc",
        raw_text="raw",
        language="zh",
        structured_document_ready=False,
        faiss_ready=False,
        profile_ready=False,
        bundle_ready=False,
        structured_document_path=None,
        faiss_namespace=None,
        errors=[],
    )
    ready, profile = pipeline._prepare_profile(
        doc_name="rebuild-doc",
        raw_text="raw",
        language="zh",
        assets=assets,
        force_rebuild=True,
    )
    _assert(ready is True and profile is not None, "force_rebuild should rebuild profile")
    _assert(profile_store.clear_calls == 1, "force_rebuild should clear profile artifact before rebuild")
    _assert(profile_builder.build_calls == 1, "force_rebuild should invoke profile builder")
    _assert(profile_store.save_calls == 1, "force_rebuild should save rebuilt profile")


def main() -> None:
    test_base_mode_prepares_profile_before_structured()
    test_free_qa_no_duplicate_profile_build()
    test_profile_failure_does_not_block_structured()
    test_existing_profile_reuse_without_rebuild()
    test_force_rebuild_clears_and_rebuilds_profile()
    print(
        json.dumps(
            {
                "status": "ok",
                "tests": [
                    "base_order_profile_before_structured",
                    "free_qa_no_duplicate_profile_build",
                    "profile_failure_non_blocking",
                    "existing_profile_reuse",
                    "force_rebuild_profile_clear_and_build",
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
