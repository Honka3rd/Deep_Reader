#!/usr/bin/env python3
"""Regression tests for manual-structure source evidence handoff."""

from __future__ import annotations

from pathlib import Path
import hashlib
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from document_preparation.document_preparation_pipeline import (  # noqa: E402
    DocumentPreparationPipeline,
)
from doc_loaders.pdf_page_evidence import PdfPageTextBoundary  # noqa: E402


class _FakeLoader:
    def __init__(
        self,
        raw_text: str,
        *,
        page_boundaries: list[PdfPageTextBoundary] | None = None,
    ) -> None:
        self.raw_text = raw_text
        self.page_boundaries = page_boundaries
        self.load_calls: list[str] = []
        self.boundary_calls: list[str] = []

    def load(self, doc_name: str) -> str:
        self.load_calls.append(doc_name)
        return self.raw_text

    def load_page_text_boundaries(self, doc_name: str) -> list[PdfPageTextBoundary]:
        self.boundary_calls.append(doc_name)
        return list(self.page_boundaries or [])


class _FakeLoaderFactory:
    def __init__(self, loader: _FakeLoader) -> None:
        self.loader = loader
        self.get_calls: list[str] = []

    def get(self, doc_name: str) -> _FakeLoader:
        self.get_calls.append(doc_name)
        return self.loader


class _FakeLanguageDetector:
    def __init__(self, language: str = "EN") -> None:
        self.language = language
        self.detect_calls: list[str] = []

    def detect(self, text: str) -> str:
        self.detect_calls.append(text)
        return self.language


class _FailingStructuredStore:
    def save(self, *args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("manual evidence handoff must not save structured artifacts")


def _pipeline(loader: _FakeLoader, detector: _FakeLanguageDetector) -> DocumentPreparationPipeline:
    return DocumentPreparationPipeline(
        loader_factory=_FakeLoaderFactory(loader),  # type: ignore[arg-type]
        language_detector=detector,  # type: ignore[arg-type]
        structured_document_builder=None,  # type: ignore[arg-type]
        structured_document_store=_FailingStructuredStore(),  # type: ignore[arg-type]
        node_provider=None,  # type: ignore[arg-type]
        faiss_index_builder=None,  # type: ignore[arg-type]
        faiss_index_store=None,  # type: ignore[arg-type]
        fingerprint_handler=None,  # type: ignore[arg-type]
        profile_builder=None,  # type: ignore[arg-type]
        profile_store=None,  # type: ignore[arg-type]
        bundle_provider=None,  # type: ignore[arg-type]
    )


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def test_manual_structure_source_evidence_is_read_only_and_source_agnostic() -> None:
    raw_text = "Chapter One\nBody"
    loader = _FakeLoader(raw_text=raw_text)
    detector = _FakeLanguageDetector(language="EN")

    evidence = _pipeline(loader, detector).load_manual_structure_source_evidence(
        " Book.txt "
    )

    _assert(evidence.doc_name == "Book.txt", "doc_name should be normalized")
    _assert(evidence.source_identity == "Book.txt", "source identity should be stable")
    _assert(evidence.raw_text == raw_text, "raw text should be returned")
    _assert(
        evidence.source_hash == hashlib.sha256(raw_text.encode("utf-8")).hexdigest(),
        "source hash should be computed from raw text",
    )
    _assert(evidence.language == "en", "language should be normalized")
    _assert(evidence.page_boundaries == [], "page boundaries should be optional")
    _assert(evidence.errors == [], "successful handoff should not report errors")
    _assert(loader.load_calls == ["Book.txt"], "loader should use normalized doc name")


def test_manual_structure_source_evidence_rejects_empty_source() -> None:
    loader = _FakeLoader(raw_text="   ")
    detector = _FakeLanguageDetector()

    try:
        _pipeline(loader, detector).load_manual_structure_source_evidence("Book.txt")
    except ValueError as error:
        _assert("source is empty" in str(error), f"unexpected error: {error}")
        return
    raise AssertionError("empty manual source should fail")


def test_manual_structure_source_evidence_includes_valid_page_boundaries() -> None:
    raw_text = "Chapter One\nBody\n\nChapter Two\nBody"
    page_boundaries = [
        PdfPageTextBoundary(
            page_index=0,
            char_start=0,
            char_end=len("Chapter One\nBody"),
            text="Chapter One\nBody",
            page_label="1",
            source_sha256="pdf-hash",
        ),
        PdfPageTextBoundary(
            page_index=1,
            char_start=len("Chapter One\nBody\n\n"),
            char_end=len(raw_text),
            text="Chapter Two\nBody",
            page_label="2",
            source_sha256="pdf-hash",
        ),
    ]
    loader = _FakeLoader(raw_text=raw_text, page_boundaries=page_boundaries)
    detector = _FakeLanguageDetector(language="EN")

    evidence = _pipeline(loader, detector).load_manual_structure_source_evidence(
        "Book.pdf"
    )

    _assert(
        evidence.source_hash == hashlib.sha256(raw_text.encode("utf-8")).hexdigest(),
        "manual source hash should remain the raw-text hash",
    )
    _assert(
        evidence.page_boundaries == page_boundaries,
        "valid page boundaries should be preserved for page_range validation",
    )
    _assert(
        [boundary.page_label for boundary in evidence.page_boundaries] == ["1", "2"],
        "page labels should be available to UI-facing callers",
    )
    _assert(evidence.errors == [], "valid page boundaries should not add errors")
    _assert(loader.boundary_calls == ["Book.pdf"], "page boundary loader should run once")


def test_manual_structure_source_evidence_keeps_char_fallback_for_invalid_page_boundaries() -> None:
    raw_text = "Chapter One\nBody"
    loader = _FakeLoader(
        raw_text=raw_text,
        page_boundaries=[
            PdfPageTextBoundary(
                page_index=0,
                char_start=0,
                char_end=len(raw_text),
                text="different text",
                page_label="1",
            )
        ],
    )
    detector = _FakeLanguageDetector(language="EN")

    evidence = _pipeline(loader, detector).load_manual_structure_source_evidence(
        "Book.pdf"
    )

    _assert(evidence.raw_text == raw_text, "raw text should remain available")
    _assert(evidence.page_boundaries == [], "invalid page evidence should be dropped")
    _assert(
        any(
            error.startswith("manual_structure_page_boundaries_invalid:text_mismatch")
            for error in evidence.errors
        ),
        f"invalid boundary should be reported explicitly: {evidence.errors}",
    )


if __name__ == "__main__":
    test_manual_structure_source_evidence_is_read_only_and_source_agnostic()
    test_manual_structure_source_evidence_rejects_empty_source()
    test_manual_structure_source_evidence_includes_valid_page_boundaries()
    test_manual_structure_source_evidence_keeps_char_fallback_for_invalid_page_boundaries()
    print("manual structure preparation handoff tests passed")
