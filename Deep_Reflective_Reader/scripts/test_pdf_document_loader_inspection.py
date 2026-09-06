#!/usr/bin/env python3
"""Regression checks for PDF text/image inspection."""

from __future__ import annotations

import sys
import tempfile
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from language.language_code import LanguageCode  # noqa: E402
from doc_loaders.pdf_document_loader import (  # noqa: E402
    PdfDocumentLoader,
    PdfInspectionMetrics,
)
from doc_loaders.document_load_errors import RawTextRequiresOcrError  # noqa: E402
from doc_loaders.pdf_ocr_language_policy import (  # noqa: E402
    DEFAULT_TESSERACT_OCR_LANGUAGE,
    get_tesseract_language_for_document_language,
)


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def test_scanned_pdf_classification_uses_text_image_and_font_signals() -> None:
    scanned = PdfInspectionMetrics(
        page_count=10,
        native_text_chars=0,
        pages_with_images=9,
        pages_with_fonts=0,
    )
    _assert(scanned.is_scanned_image_pdf, "image-heavy textless PDFs should require OCR")

    born_digital = PdfInspectionMetrics(
        page_count=10,
        native_text_chars=1000,
        pages_with_images=10,
        pages_with_fonts=10,
    )
    _assert(
        not born_digital.is_scanned_image_pdf,
        "PDFs with native text should not be classified as scanned image PDFs",
    )

    textless_vector = PdfInspectionMetrics(
        page_count=10,
        native_text_chars=0,
        pages_with_images=0,
        pages_with_fonts=0,
    )
    _assert(
        not textless_vector.is_scanned_image_pdf,
        "textless non-image PDFs should not be treated as scanned-image PDFs",
    )

    textless_font_pdf = PdfInspectionMetrics(
        page_count=10,
        native_text_chars=0,
        pages_with_images=10,
        pages_with_fonts=10,
    )
    _assert(
        not textless_font_pdf.is_scanned_image_pdf,
        "PDFs with font resources should not be classified by image presence alone",
    )


def test_native_pdf_text_is_preserved_while_collecting_metrics() -> None:
    class _Page:
        def __init__(self, text: str) -> None:
            self._text = text

        def extract_text(self) -> str:
            return self._text

        def get(self, key: str) -> dict[str, object]:
            if key == "/Resources":
                return {"/Font": {"F1": object()}}
            return {}

    class _Reader:
        pages = [_Page("First page"), _Page("Second page")]

    text, metrics = PdfDocumentLoader()._extract_text_and_metrics(_Reader())

    _assert(text == "First page\nSecond page", f"unexpected native text: {text!r}")
    _assert(metrics.native_text_chars == len("First pageSecond page"), "native text chars should be counted")
    _assert(metrics.pages_with_fonts == 2, "font resources should be counted")
    _assert(not metrics.is_scanned_image_pdf, "native text PDFs should not require OCR")


def test_real_scanned_pdf_inspection_for_dark_water_fixture() -> None:
    raw_dir = Path(__file__).resolve().parents[1] / "data" / "raw"
    fixture = raw_dir / "暗水幽灵.pdf"
    if not fixture.exists():
        return

    loader = PdfDocumentLoader(base_dir=str(raw_dir))
    metrics = loader.inspect("暗水幽灵")

    _assert(metrics.page_count == 261, f"unexpected page count: {metrics}")
    _assert(metrics.native_text_chars == 0, f"unexpected native text: {metrics}")
    _assert(metrics.pages_with_images == 261, f"unexpected image page count: {metrics}")
    _assert(metrics.pages_with_fonts == 0, f"unexpected font page count: {metrics}")
    _assert(metrics.is_scanned_image_pdf, "暗水幽灵.pdf should be detected as scanned")


def test_real_scanned_pdf_load_requires_ocr() -> None:
    raw_dir = Path(__file__).resolve().parents[1] / "data" / "raw"
    fixture = raw_dir / "暗水幽灵.pdf"
    if not fixture.exists():
        return

    loader = PdfDocumentLoader(base_dir=str(raw_dir))
    try:
        loader.load("暗水幽灵")
    except RawTextRequiresOcrError as error:
        _assert(error.doc_name == "暗水幽灵", "error should preserve doc_name")
        return
    raise AssertionError("scanned PDF load should raise RawTextRequiresOcrError")


def test_real_scanned_pdf_can_use_explicit_ocr_fallback() -> None:
    raw_dir = Path(__file__).resolve().parents[1] / "data" / "raw"
    fixture = raw_dir / "暗水幽灵.pdf"
    if not fixture.exists():
        return

    with tempfile.TemporaryDirectory() as temp_dir:
        fake_tesseract = Path(temp_dir) / "fake-tesseract"
        fake_tesseract.write_text("#!/bin/sh\nprintf 'OCR fallback text\\n'\n", encoding="utf-8")
        fake_tesseract.chmod(0o755)

        loader = PdfDocumentLoader(
            base_dir=str(raw_dir),
            ocr_enabled=True,
            tesseract_cmd=str(fake_tesseract),
            ocr_language="eng",
            ocr_page_limit=1,
            ocr_cache_dir=str(Path(temp_dir) / "ocr-cache"),
        )

        text = loader.load("暗水幽灵")
        _assert(text == "OCR fallback text", f"unexpected OCR text: {text!r}")


def test_ocr_cache_reuses_text_for_matching_provenance() -> None:
    raw_dir = Path(__file__).resolve().parents[1] / "data" / "raw"
    fixture = raw_dir / "暗水幽灵.pdf"
    if not fixture.exists():
        return

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        count_file = temp_path / "ocr-count"
        fake_tesseract = temp_path / "fake-tesseract"
        fake_tesseract.write_text(
            "#!/bin/sh\n"
            "if [ \"$1\" = \"--version\" ]; then printf 'tesseract 5.5.0\\n'; exit 0; fi\n"
            "count_file=\"$FAKE_TESSERACT_COUNT_FILE\"\n"
            "count=$(cat \"$count_file\" 2>/dev/null || printf '0')\n"
            "count=$((count + 1))\n"
            "printf '%s' \"$count\" > \"$count_file\"\n"
            "printf 'Cached OCR text\\n'\n",
            encoding="utf-8",
        )
        fake_tesseract.chmod(0o755)

        previous = os.environ.get("FAKE_TESSERACT_COUNT_FILE")
        os.environ["FAKE_TESSERACT_COUNT_FILE"] = str(count_file)
        try:
            loader = PdfDocumentLoader(
                base_dir=str(raw_dir),
                ocr_enabled=True,
                tesseract_cmd=str(fake_tesseract),
                ocr_language="eng",
                ocr_page_limit=1,
                ocr_cache_dir=str(temp_path / "ocr-cache"),
            )

            first_text = loader.load("暗水幽灵")
            boundaries = loader.load_page_text_boundaries("暗水幽灵")
            second_text = loader.load("暗水幽灵")

            _assert(first_text == "Cached OCR text", "first OCR pass should return recognized text")
            _assert(
                boundaries[0].text == "Cached OCR text",
                "page-boundary load should reuse page-aware OCR cache",
            )
            _assert(second_text == first_text, "second OCR pass should reuse cached text")
            _assert(count_file.read_text(encoding="utf-8") == "1", "OCR command should run only once")
        finally:
            if previous is None:
                os.environ.pop("FAKE_TESSERACT_COUNT_FILE", None)
            else:
                os.environ["FAKE_TESSERACT_COUNT_FILE"] = previous


def test_ocr_language_policy_uses_project_language_codes() -> None:
    _assert(
        get_tesseract_language_for_document_language(LanguageCode.EN) == "eng",
        "English should map to the English Tesseract pack",
    )
    _assert(
        get_tesseract_language_for_document_language("zh-cn") == "chi_sim+chi_tra+eng",
        "Chinese aliases should map to Chinese OCR packs with English fallback",
    )
    _assert(
        get_tesseract_language_for_document_language(None) == DEFAULT_TESSERACT_OCR_LANGUAGE,
        "unknown raw-load language should use the multilingual OCR fallback",
    )


def test_pdf_loader_defaults_to_multilingual_ocr_language() -> None:
    previous = os.environ.pop("DEEP_READER_PDF_OCR_LANGUAGE", None)
    try:
        loader = PdfDocumentLoader()
        _assert(
            loader.ocr_language == DEFAULT_TESSERACT_OCR_LANGUAGE,
            f"unexpected default OCR language: {loader.ocr_language}",
        )
    finally:
        if previous is not None:
            os.environ["DEEP_READER_PDF_OCR_LANGUAGE"] = previous


if __name__ == "__main__":
    test_scanned_pdf_classification_uses_text_image_and_font_signals()
    test_native_pdf_text_is_preserved_while_collecting_metrics()
    test_real_scanned_pdf_inspection_for_dark_water_fixture()
    test_real_scanned_pdf_load_requires_ocr()
    test_real_scanned_pdf_can_use_explicit_ocr_fallback()
    test_ocr_cache_reuses_text_for_matching_provenance()
    test_ocr_language_policy_uses_project_language_codes()
    test_pdf_loader_defaults_to_multilingual_ocr_language()
    print("OK: PDF document loader inspection tests passed")
