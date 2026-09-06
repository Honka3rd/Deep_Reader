#!/usr/bin/env python3
"""Regression checks for preparation raw-load error mapping."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from doc_loaders.document_load_errors import (  # noqa: E402
    RawTextOcrFailedError,
    RawTextRequiresOcrError,
)
from document_preparation.document_preparation_pipeline import (  # noqa: E402
    DocumentPreparationPipeline,
)
from document_preparation.prepared_document_assets import PreparedDocumentAssets  # noqa: E402


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


class _RequiresOcrLoader:
    def load(self, doc_name: str) -> str:
        raise RawTextRequiresOcrError(doc_name)


class _OcrFailedLoader:
    def load(self, doc_name: str) -> str:
        raise RawTextOcrFailedError(doc_name, detail="tesseract_not_found:tesseract")


class _LoaderFactory:
    def __init__(self, loader: object) -> None:
        self.loader = loader

    def get(self, doc_name: str) -> object:
        return self.loader


def test_requires_ocr_error_maps_to_specific_prepare_reason() -> None:
    pipeline = DocumentPreparationPipeline.__new__(DocumentPreparationPipeline)
    pipeline.loader_factory = _LoaderFactory(_RequiresOcrLoader())
    assets = PreparedDocumentAssets(
        doc_name="Scanned PDF",
        raw_text=None,
        language=None,
        structured_document_ready=False,
        faiss_ready=False,
        profile_ready=False,
        bundle_ready=False,
        structured_document_path=None,
        faiss_namespace=None,
        errors=[],
    )

    raw_text = pipeline._load_raw_text(
        doc_name="Scanned PDF",
        assets=assets,
    )

    _assert(raw_text is None, "requires-OCR load should not return raw text")
    _assert(
        assets.errors == ["load_raw_text_requires_ocr:Scanned PDF"],
        f"unexpected raw-load errors: {assets.errors}",
    )


def test_ocr_failed_error_maps_to_specific_prepare_reason() -> None:
    pipeline = DocumentPreparationPipeline.__new__(DocumentPreparationPipeline)
    pipeline.loader_factory = _LoaderFactory(_OcrFailedLoader())
    assets = PreparedDocumentAssets(
        doc_name="Scanned PDF",
        raw_text=None,
        language=None,
        structured_document_ready=False,
        faiss_ready=False,
        profile_ready=False,
        bundle_ready=False,
        structured_document_path=None,
        faiss_namespace=None,
        errors=[],
    )

    raw_text = pipeline._load_raw_text(
        doc_name="Scanned PDF",
        assets=assets,
    )

    _assert(raw_text is None, "failed OCR should not return raw text")
    _assert(
        assets.errors == [
            "load_raw_text_ocr_failed:Scanned PDF:tesseract_not_found:tesseract"
        ],
        f"unexpected OCR failure errors: {assets.errors}",
    )


if __name__ == "__main__":
    test_requires_ocr_error_maps_to_specific_prepare_reason()
    test_ocr_failed_error_maps_to_specific_prepare_reason()
    print("OK: document preparation raw-load error tests passed")
