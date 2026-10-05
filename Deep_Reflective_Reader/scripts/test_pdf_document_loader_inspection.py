#!/usr/bin/env python3
"""Regression checks for PDF text/image inspection."""

from __future__ import annotations

import sys
import tempfile
import os
import hashlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import doc_loaders.pdf_document_loader as pdf_loader_module  # noqa: E402
from language.language_code import LanguageCode  # noqa: E402
from doc_loaders.pdf_document_loader import (  # noqa: E402
    PdfDocumentLoader,
    PdfInspectionMetrics,
)
from doc_loaders.document_loader_factory import DocumentLoaderFactory  # noqa: E402
from doc_loaders.raw_data_paths import DEFAULT_RAW_DATA_DIR  # noqa: E402
from doc_loaders.text_document_loader import TextDocumentLoader  # noqa: E402
from doc_loaders.document_load_errors import RawTextRequiresOcrError  # noqa: E402
from doc_loaders.document_load_errors import RawTextOcrLowQualityError  # noqa: E402
from doc_loaders.pdf_ocr_language_policy import (  # noqa: E402
    DEFAULT_TESSERACT_OCR_LANGUAGE,
    get_tesseract_language_for_document_language,
)


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def test_default_raw_data_dir_is_independent_of_cwd() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        shadow_raw_dir = temp_path / "data" / "raw"
        shadow_raw_dir.mkdir(parents=True)
        (shadow_raw_dir / "cwd-shadow.pdf").write_bytes(b"not the project raw fixture")

        previous_cwd = Path.cwd()
        try:
            os.chdir(temp_path)
            text_loader = TextDocumentLoader()
            pdf_loader = PdfDocumentLoader()
            factory = DocumentLoaderFactory()
            selected_loader = factory.get("cwd-shadow")
        finally:
            os.chdir(previous_cwd)

    _assert(
        text_loader.base_dir == DEFAULT_RAW_DATA_DIR,
        f"text loader should resolve project raw dir, got {text_loader.base_dir}",
    )
    _assert(
        pdf_loader.base_dir == DEFAULT_RAW_DATA_DIR,
        f"pdf loader should resolve project raw dir, got {pdf_loader.base_dir}",
    )
    _assert(
        factory.base_dir == DEFAULT_RAW_DATA_DIR,
        f"factory should resolve project raw dir, got {factory.base_dir}",
    )
    _assert(
        isinstance(selected_loader, TextDocumentLoader),
        "factory should not select PDF from cwd-relative shadow data/raw",
    )


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
    extract_calls = 0

    class _Page:
        def __init__(self, text: str) -> None:
            self._text = text

        def extract_text(self) -> str:
            nonlocal extract_calls
            extract_calls += 1
            return self._text

        def get(self, key: str) -> dict[str, object]:
            if key == "/Resources":
                return {"/Font": {"F1": object()}}
            return {}

    class _Reader:
        pages = [_Page("First page"), _Page("Second page")]

    text, metrics = PdfDocumentLoader()._extract_text_and_metrics(_Reader())

    _assert(text == "First page\nSecond page", f"unexpected native text: {text!r}")
    _assert(extract_calls == 2, "native text extraction should happen once per page")
    _assert(metrics.native_text_chars == len("First pageSecond page"), "native text chars should be counted")
    _assert(metrics.pages_with_fonts == 2, "font resources should be counted")
    _assert(not metrics.is_scanned_image_pdf, "native text PDFs should not require OCR")


def test_pdf_page_boundary_evidence_preserves_native_load_contract() -> None:
    extract_calls = 0

    class _Page:
        def __init__(self, text: str) -> None:
            self._text = text

        def extract_text(self) -> str:
            nonlocal extract_calls
            extract_calls += 1
            return self._text

        def get(self, key: str) -> dict[str, object]:
            if key == "/Resources" and self._text:
                return {"/Font": {"F1": object()}}
            return {}

    class _Reader:
        page_labels = ["i", "1", "2"]

        def __init__(self, path: str) -> None:
            self.path = path
            self.pages = [_Page("Alpha"), _Page("Beta"), _Page("")]

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        pdf_path = temp_path / "sample.pdf"
        pdf_bytes = b"fake native pdf bytes for boundary evidence regression"
        pdf_path.write_bytes(pdf_bytes)
        previous_reader = pdf_loader_module.PdfReader
        pdf_loader_module.PdfReader = _Reader
        try:
            loader = PdfDocumentLoader(base_dir=str(temp_path))
            raw_text = loader.load("sample")
            evidence = loader.load_page_boundary_evidence("sample")
        finally:
            pdf_loader_module.PdfReader = previous_reader

    expected_hash = hashlib.sha256(pdf_bytes).hexdigest()
    _assert(raw_text == "Alpha\nBeta", f"unexpected load() text: {raw_text!r}")
    _assert(
        extract_calls == 6,
        "load() and page boundary evidence should each extract native text once per page",
    )
    _assert(evidence.doc_name == "sample", "evidence should preserve doc_name")
    _assert(
        evidence.source_file_name == "sample.pdf",
        "evidence should preserve source file name",
    )
    _assert(evidence.source_sha256 == expected_hash, "evidence should include source PDF hash")
    _assert(evidence.page_count == 3, "evidence should report reader page count")
    _assert(len(evidence.boundaries) == 3, "evidence should include one boundary per page")
    _assert(
        [boundary.page_label for boundary in evidence.boundaries] == ["i", "1", "2"],
        "page labels should be copied when available",
    )
    _assert(
        [(boundary.char_start, boundary.char_end) for boundary in evidence.boundaries]
        == [(0, 5), (6, 10), (10, 10)],
        "page boundaries should map to load() raw-text offsets",
    )
    _assert(
        [boundary.source_sha256 for boundary in evidence.boundaries]
        == [expected_hash, expected_hash, expected_hash],
        "each page boundary should carry source identity",
    )


def test_apple_pdf_page_boundaries_use_native_text_without_ocr() -> None:
    """APPLE.pdf is a native-text PDF; page-boundary evidence must not require OCR."""
    loader = PdfDocumentLoader(ocr_enabled=False)
    evidence = loader.load_page_boundary_evidence("APPLE.pdf")

    _assert(evidence.source_file_name == "APPLE.pdf", "fixture should resolve APPLE.pdf")
    _assert(evidence.page_count > 1, "APPLE.pdf should have multiple native pages")
    _assert(
        len(evidence.boundaries) == evidence.page_count,
        "page-boundary evidence should include one boundary per PDF page",
    )
    _assert(
        any(boundary.text.strip() for boundary in evidence.boundaries),
        "native page-boundary evidence should include extracted text",
    )
    _assert(
        loader.last_ocr_provenance is None,
        "native APPLE.pdf boundary loading should not start OCR",
    )


def test_pdf_page_layout_evidence_includes_source_metadata() -> None:
    class _Page:
        images: list[object] = []

        def __init__(self, text: str) -> None:
            self._text = text

        def extract_text(self) -> str:
            return self._text

    class _Reader:
        def __init__(self, path: str) -> None:
            self.path = path
            self.pages = [_Page("Alpha"), _Page("")]

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        pdf_path = temp_path / "layout.pdf"
        pdf_bytes = b"fake native pdf bytes for layout evidence metadata"
        pdf_path.write_bytes(pdf_bytes)
        previous_reader = pdf_loader_module.PdfReader
        pdf_loader_module.PdfReader = _Reader
        try:
            loader = PdfDocumentLoader(base_dir=str(temp_path))
            evidence = loader.load_page_layout_evidence("layout")
        finally:
            pdf_loader_module.PdfReader = previous_reader

    expected_hash = hashlib.sha256(pdf_bytes).hexdigest()
    _assert(len(evidence) == 2, "layout evidence should include every page")
    _assert(
        [page.source_file_name for page in evidence] == ["layout.pdf", "layout.pdf"],
        "layout evidence should carry source file name on every page",
    )
    _assert(
        [page.source_sha256 for page in evidence] == [expected_hash, expected_hash],
        "layout evidence should carry source PDF hash on every page",
    )
    _assert(
        [page.evidence_schema_version for page in evidence] == [1, 1],
        "layout evidence should carry schema version on every page",
    )
    _assert(
        [page.native_text_chars for page in evidence] == [5, 0],
        "layout evidence should preserve native text metrics",
    )
    _assert(
        [page.analysis_cost_tier for page in evidence] == ["cheap_inventory", "cheap_inventory"],
        "layout inventory should record cheap inspection cost tier when OCR is disabled",
    )
    _assert(
        [page.ocr_pass_count for page in evidence] == [0, 0],
        "layout inventory should not record OCR passes when OCR is disabled",
    )
    _assert(
        all(page.cache_key and ":stage:inventory" in page.cache_key for page in evidence),
        "layout inventory should record stable inventory cache keys",
    )
    _assert(
        [page.high_cost_analysis_permitted for page in evidence] == [False, False],
        "layout inventory should not permit high-cost analysis by default",
    )


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
        )
        loader._render_page_images = lambda **kwargs: [Path(temp_dir) / "page.png"]

        text = loader.load("暗水幽灵")
        _assert(text == "OCR fallback text", f"unexpected OCR text: {text!r}")


def test_ocr_uses_memory_pages_without_file_cache() -> None:
    class _Reader:
        pages = [object()]

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        pdf_path = temp_path / "sample.pdf"
        pdf_path.write_bytes(b"not a real pdf; private OCR helper regression")
        count_file = temp_path / "ocr-count"
        cache_dir = temp_path / "ocr-cache"
        fake_tesseract = temp_path / "fake-tesseract"
        fake_tesseract.write_text(
            "#!/bin/sh\n"
            "if [ \"$1\" = \"--version\" ]; then printf 'tesseract 5.5.0\\n'; exit 0; fi\n"
            "count_file=\"$FAKE_TESSERACT_COUNT_FILE\"\n"
            "count=$(cat \"$count_file\" 2>/dev/null || printf '0')\n"
            "count=$((count + 1))\n"
            "printf '%s' \"$count\" > \"$count_file\"\n"
            "printf 'Memory OCR text\\n'\n",
            encoding="utf-8",
        )
        fake_tesseract.chmod(0o755)

        previous = os.environ.get("FAKE_TESSERACT_COUNT_FILE")
        os.environ["FAKE_TESSERACT_COUNT_FILE"] = str(count_file)
        try:
            loader = PdfDocumentLoader(
                base_dir=str(temp_path),
                ocr_enabled=True,
                tesseract_cmd=str(fake_tesseract),
                ocr_language="eng",
                ocr_page_limit=1,
            )
            loader._render_page_images = lambda **kwargs: [temp_path / "page.png"]

            first_pages = loader._load_pages_with_ocr(
                doc_name="sample",
                file_path=pdf_path,
                reader=_Reader(),
            )
            second_pages = loader._load_pages_with_ocr(
                doc_name="sample",
                file_path=pdf_path,
                reader=_Reader(),
            )

            _assert(first_pages == ["Memory OCR text"], "first OCR pass should return recognized text")
            _assert(second_pages == first_pages, "second OCR pass should reuse in-memory OCR pages")
            _assert(count_file.read_text(encoding="utf-8") == "4", "OCR candidates should run only once per PSM")
            _assert(not cache_dir.exists(), "OCR file cache must not create cache directory")
        finally:
            if previous is None:
                os.environ.pop("FAKE_TESSERACT_COUNT_FILE", None)
            else:
                os.environ["FAKE_TESSERACT_COUNT_FILE"] = previous


def test_ocr_candidate_selection_does_not_accept_first_non_empty_output() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        image_path = temp_path / "page.png"
        image_path.write_bytes(b"fake image")
        fake_tesseract = temp_path / "fake-tesseract"
        fake_tesseract.write_text(
            "#!/bin/sh\n"
            "if [ \"$1\" = \"--version\" ]; then printf 'tesseract 5.5.0\\n'; exit 0; fi\n"
            "case \"$*\" in\n"
            "  *'--psm 6'*) printf '國富論分工市場價格自然價格貨幣資本勞動土地租稅\\n' ;;\n"
            "  *'--psm 5'*) printf 'UAE RG Rl Qo JOGA ol 8 TAN ts SS NS RS ot\\n' ;;\n"
            "  *'--psm 11'*) printf '國\\n富\\n論\\n' ;;\n"
            "  *) printf 'UAE RG Rl Qo JOGA ol 8 TAN ts SS NS RS ot\\n' ;;\n"
            "esac\n",
            encoding="utf-8",
        )
        fake_tesseract.chmod(0o755)

        loader = PdfDocumentLoader(
            base_dir=str(temp_path),
            ocr_enabled=True,
            tesseract_cmd=str(fake_tesseract),
            ocr_language="chi_tra+eng",
        )

        text = loader._ocr_image_text(
            image_path=image_path,
            doc_name="國富論lite",
            page_index=0,
        )

        _assert("國富論分工" in text, "higher-quality psm 6 output should beat non-empty default garbage")
        _assert("UAE RG" not in text, "default garbage should not be accepted just because it is non-empty")


def test_ocr_candidate_selection_suppresses_rejected_text() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        image_path = temp_path / "page.png"
        image_path.write_bytes(b"fake image")
        fake_tesseract = temp_path / "fake-tesseract"
        fake_tesseract.write_text(
            "#!/bin/sh\n"
            "if [ \"$1\" = \"--version\" ]; then printf 'tesseract 5.5.0\\n'; exit 0; fi\n"
            "printf 'UAE RG Rl Qo JOGA ol 8 TAN ts SS NS RS ot HEME RRKSERRRERM\\n'\n",
            encoding="utf-8",
        )
        fake_tesseract.chmod(0o755)

        loader = PdfDocumentLoader(
            base_dir=str(temp_path),
            ocr_enabled=True,
            tesseract_cmd=str(fake_tesseract),
            ocr_language="chi_tra+eng",
        )

        text = loader._ocr_image_text(
            image_path=image_path,
            doc_name="國富論lite",
            page_index=0,
        )

        _assert(text == "", "rejected OCR candidate text must not enter raw-text handoff")


def test_ocr_quality_gate_rejects_non_empty_garbage_pages() -> None:
    class _Reader:
        pages = [object(), object(), object()]

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        pdf_path = temp_path / "sample.pdf"
        pdf_path.write_bytes(b"not a real pdf; private OCR quality regression")
        fake_tesseract = temp_path / "fake-tesseract"
        fake_tesseract.write_text(
            "#!/bin/sh\n"
            "if [ \"$1\" = \"--version\" ]; then printf 'tesseract 5.5.0\\n'; exit 0; fi\n"
            "printf 'UAE RG Rl Qo JOGA ol 8 TAN ts SS NS RS ot HEME RRKSERRRERM\\n'\n",
            encoding="utf-8",
        )
        fake_tesseract.chmod(0o755)

        loader = PdfDocumentLoader(
            base_dir=str(temp_path),
            ocr_enabled=True,
            tesseract_cmd=str(fake_tesseract),
            ocr_language="chi_tra+eng",
            ocr_page_limit=3,
        )
        loader._render_page_images = lambda **kwargs: [temp_path / "page.png"]

        try:
            loader._load_pages_with_ocr(
                doc_name="國富論lite",
                file_path=pdf_path,
                reader=_Reader(),
            )
        except RawTextOcrLowQualityError as error:
            _assert(error.doc_name == "國富論lite", "low-quality OCR error should preserve doc_name")
            _assert(
                (error.detail or "").startswith("document_quality_gate_failed:"),
                f"unexpected low-quality detail: {error.detail}",
            )
            return
        raise AssertionError("non-empty garbage OCR should fail the document quality gate")


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
    test_default_raw_data_dir_is_independent_of_cwd()
    test_scanned_pdf_classification_uses_text_image_and_font_signals()
    test_native_pdf_text_is_preserved_while_collecting_metrics()
    test_pdf_page_boundary_evidence_preserves_native_load_contract()
    test_pdf_page_layout_evidence_includes_source_metadata()
    test_real_scanned_pdf_inspection_for_dark_water_fixture()
    test_real_scanned_pdf_load_requires_ocr()
    test_real_scanned_pdf_can_use_explicit_ocr_fallback()
    test_ocr_uses_memory_pages_without_file_cache()
    test_ocr_candidate_selection_does_not_accept_first_non_empty_output()
    test_ocr_candidate_selection_suppresses_rejected_text()
    test_ocr_quality_gate_rejects_non_empty_garbage_pages()
    test_ocr_language_policy_uses_project_language_codes()
    test_pdf_loader_defaults_to_multilingual_ocr_language()
    print("OK: PDF document loader inspection tests passed")
