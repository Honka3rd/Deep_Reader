#!/usr/bin/env python3
"""Real three-mode PDF acceptance; temporary JSON, stub language/profile, no network."""
import argparse
from collections import Counter
from contextlib import redirect_stdout
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pypdf
from pypdf import PdfReader, PdfWriter
from pypdf.generic import NameObject, RectangleObject
from accept_selective_ocr import IsolatedStore, capture_stages, make_fixtures, TRANSCRIPT
from doc_loaders.pdf_document_loader import PdfDocumentLoader
from doc_loaders.pdf_extraction_contract import PdfExtractionError
from document_structure.structured_document_builder import StructuredDocumentBuilder
from test_pdf_page_inspection import page_fixture
from test_prepare_structured_reuse_before_raw_load import _build_pipeline


def counts(calls):
    result = Counter()
    for call in calls:
        command = call.args[0]
        executable = Path(command[0]).name
        if executable == "pdftoppm" and "-v" not in command:
            result["render"] += 1
        elif executable == "tesseract" and "stdout" in command:
            result["ocr"] += 1
    return dict(result)


def accept(source, *, modes, english_page=None, expected_pages=None, budget=False):
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    reports = []
    with tempfile.TemporaryDirectory(prefix="pdf-fix-03-") as directory:
        root = Path(directory)
        shutil.copyfile(source, root / "sample.pdf")
        loader = PdfDocumentLoader(root, ocr_enabled=True, ocr_language="eng+chi_sim+chi_tra",
                                   ocr_page_limit=0 if budget else None)
        store = IsolatedStore(root / "structured.json")
        pipeline = _build_pipeline(loader=loader, store=store)
        pipeline.structured_document_builder = StructuredDocumentBuilder()
        original = loader.extract
        for mode in modes:
            before = store.path.read_bytes() if store.path.exists() else None
            evidence = {}

            def extract(*args, **kwargs):
                start = time.perf_counter()
                calls_before = len(processes.call_args_list)
                try:
                    result = original(*args, **kwargs)
                finally:
                    evidence["extraction_ms"] = round((time.perf_counter() - start) * 1000, 1)
                    evidence["extraction_processes"] = counts(processes.call_args_list[calls_before:])
                spans = result.boundary_evidence.boundaries
                previous = 0
                for index, span in enumerate(spans):
                    assert span.page_index == index and span.char_start >= previous
                    assert result.text[span.char_start:span.char_end] == span.text
                    assert span.source_sha256 == digest
                    previous = span.char_end
                assert previous == len(result.text)
                if expected_pages is not None:
                    assert len(spans) == expected_pages
                if english_page is not None:
                    reference = PdfReader(source).pages[english_page - 1].extract_text()
                    # Retention, not a claim of exact OCR ground truth.
                    words = {word.lower().strip(".,:;!?()\"'") for word in reference.split() if len(word) >= 5 and word.isascii()}
                    actual = result.pages[english_page - 1].lower()
                    ratio = sum(word in actual for word in words) / max(1, len(words))
                    assert len(words) >= 10 and ratio >= .7, (mode, "English retention", ratio)
                    evidence["english_retention"] = round(ratio, 3)
                if source.stem in {"scan", "mixed-outline"} and mode != "fast":
                    assert TRANSCRIPT in " ".join(result.text.split()), "known scan transcript missing"
                if mode == "fast":
                    assert result.pages == tuple((page.extract_text() or "").strip() for page in PdfReader(source).pages)
                evidence.update(pages=len(spans), characters=len(result.text), ordered_spans=True,
                                nonblank_pages=len(result.pages) - len(result.empty_page_indexes))
                return result

            failure = None
            failure_detail = None
            start = time.perf_counter()
            with capture_stages() as logs, patch.object(loader, "extract", side_effect=extract), \
                    patch("subprocess.run", wraps=subprocess.run) as processes, redirect_stdout(sys.stderr):
                try:
                    assets = pipeline.prepare("sample", mode="base", pdf_extraction_mode=mode)
                    assert assets.structured_document_ready, assets.errors
                except PdfExtractionError as error:
                    failure = error.reason.value
                    failure_detail = str(error)
            report = {"source": source.name, "sha256": digest, "mode": mode,
                      "elapsed_ms": round((time.perf_counter() - start) * 1000, 1),
                      "stages_ms": {key: round(value, 1) for key, value in logs.stages.items()},
                      "processes": counts(processes.call_args_list), "failure": failure,
                      "failure_detail": failure_detail, **evidence}
            expected_failure = ("budget_exceeded" if budget and mode != "fast" else
                                "extraction_failed" if source.stem == "corrupt" else
                                "no_text" if source.stem == "blank" or (source.stem == "scan" and mode == "fast") else None)
            report["contract_passed"] = failure == expected_failure
            if mode == "fast":
                assert not report["processes"], report
            if failure:
                after = store.path.read_bytes() if store.path.exists() else None
                assert after == before, "failed extraction mutated persisted hierarchy"
                report["failure_preserved_storage"] = True
            else:
                assert assets.pdf_extraction_provenance == "applied", "mode switch reused another mode"
                document = store.load(None)
                assert "sections" not in document.to_dict() and "structure_nodes" not in document.to_dict()
                assert document.parse_provenance["pdf_extraction"]["identity"]["mode"] == mode
                saved = store.path.read_bytes()
                saves = store.save_calls
                with patch.object(loader, "extract", side_effect=AssertionError("warm extraction")), \
                        patch.object(pipeline.language_detector, "detect", side_effect=AssertionError("warm language")), \
                        patch.object(pipeline.profile_builder, "build", side_effect=AssertionError("warm profile")):
                    reused = pipeline.prepare("sample", mode="base", pdf_extraction_mode=mode)
                assert reused.pdf_extraction_provenance == "reused"
                assert saved == store.path.read_bytes() and saves == store.save_calls
                report["warm_reused_without_extraction"] = True
                if mode == "ocr":
                    assert evidence["extraction_processes"].get("render") == evidence["pages"], report
                    assert evidence["extraction_processes"].get("ocr", 0) == evidence["nonblank_pages"] * 4, report
                if mode == "auto" and source.stem in {"native", "scan", "mixed-outline"}:
                    assert evidence["extraction_processes"].get("render", 0) == (0 if source.stem == "native" else 1), report
            reports.append(report)
            print(json.dumps(report), flush=True)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == digest
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, required=True, help="Actual 13-page bilingual watermark fixture (read-only)")
    parser.add_argument("--synthetic-only", action="store_true")
    args = parser.parse_args()
    versions = {"python": platform.python_version(), "pypdf": pypdf.__version__}
    for executable, flag in (("tesseract", "--version"), ("pdftoppm", "-v")):
        process = subprocess.run([executable, flag], capture_output=True, text=True, check=True)
        versions[executable] = (process.stdout or process.stderr).splitlines()[0]
    print(json.dumps({"runtime": versions, "language_profile": "stubs", "storage": "temporary JSON"}), flush=True)
    with tempfile.TemporaryDirectory(prefix="pdf-mode-fixtures-") as directory:
        reports = []
        root = Path(directory)
        make_fixtures(root)
        stream = b"BT /F1 12 Tf 50 700 Td (Chapter One. Native text must remain unchanged and in order.) Tj ET"
        writer, page = page_fixture(stream, with_font=True)
        # Renderer fixtures need an indirect stream, unlike parser-only unit fixtures.
        page[NameObject("/Contents")] = writer._add_object(page["/Contents"])
        page.mediabox = RectangleObject((0, 0, 612, 792))
        page.cropbox = RectangleObject((0, 0, 612, 792))
        writer.pages[0].mediabox = RectangleObject((0, 0, 612, 792))
        writer.pages[0].cropbox = RectangleObject((0, 0, 612, 792))
        writer.write(root / "native.pdf")
        mixed = PdfWriter()
        mixed.add_page(PdfReader(root / "native.pdf").pages[0])
        mixed.add_page(PdfReader(root / "scan.pdf").pages[0])
        mixed.add_blank_page(width=612, height=792)
        mixed.add_outline_item("Chapter One", 0)
        mixed.add_outline_item("Chapter Two", 1)
        mixed.write(root / "mixed-outline.pdf")
        blank = PdfWriter()
        blank.add_blank_page(width=612, height=792)
        blank.write(root / "blank.pdf")
        (root / "corrupt.pdf").write_bytes(b"not a PDF")
        for name in ("native", "scan", "mixed-outline", "blank", "corrupt"):
            reports.extend(accept(root / f"{name}.pdf", modes=("fast", "ocr", "auto")))
        reports.extend(accept(root / "mixed-outline.pdf", modes=("fast", "ocr", "auto"), budget=True))
    if not args.synthetic_only:
        reports.extend(accept(args.pdf, modes=("fast", "ocr", "auto"), english_page=10, expected_pages=13))
    passed = all(report["contract_passed"] for report in reports)
    print(json.dumps({"result": "PASS" if passed else "FAIL", "actual_fixture": not args.synthetic_only,
                      "cases": len(reports), "passed": sum(report["contract_passed"] for report in reports)}), flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
