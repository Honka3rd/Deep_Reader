#!/usr/bin/env python3
"""Isolated OCR acceptance. Real PDF/OCR/structure/JSON, no LLM or live storage."""

import argparse
from collections import Counter
from contextlib import contextmanager, redirect_stdout
import hashlib
import json
import logging
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
from PIL import Image, ImageDraw, ImageFont
from pypdf import PdfReader, PdfWriter

from doc_loaders.document_loader_factory import DocumentLoaderFactory
from doc_loaders.pdf_document_loader import PdfDocumentLoader
from document_structure.structured_document_builder import StructuredDocumentBuilder
from document_structure.structured_document_store import StructuredDocumentStore
from logging_config import request_id_context
from test_pdf_page_inspection import page_fixture
from test_prepare_structured_reuse_before_raw_load import _build_pipeline


class IsolatedStore:
    """Ignore production storage configuration; all persistence stays in one temp file."""

    def __init__(self, path):
        self.path = path
        self.save_calls = 0

    def location(self, target):
        return str(self.path)

    def exists(self, target):
        return self.path.exists()

    def save(self, document, target):
        self.save_calls += 1
        StructuredDocumentStore.save(document, str(self.path))

    def load(self, target):
        return StructuredDocumentStore.load(str(self.path))


class StageCapture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.stages = Counter()
        self.processes = Counter()

    def emit(self, record):
        # Read the existing logger's structured arguments, never collect document text.
        if record.msg.startswith(("stage_finished stage=", "stage_failed stage=")):
            self.stages[record.args[0]] += record.args[2]
        elif record.msg.startswith("pdf_process request_id="):
            self.processes[f"{record.args[1]}.{record.args[2]}"] += 1


@contextmanager
def capture_stages():
    capture = StageCapture()
    loggers = [logging.getLogger(name) for name in (
        "document_preparation.document_preparation_pipeline",
        "doc_loaders.pdf_document_loader", "doc_loaders.pdf_execution_pass",
    )]
    old = [(logger, logger.level, logger.propagate) for logger in loggers]
    for logger in loggers:
        logger.setLevel(logging.INFO)
        logger.propagate = False
        logger.addHandler(capture)
    try:
        yield capture
    finally:
        for logger, level, propagate in old:
            logger.removeHandler(capture)
            logger.setLevel(level)
            logger.propagate = propagate


def measure(pipeline, store, phase):
    token = request_id_context.set(f"ocr-08-{phase}")
    runner = subprocess.run
    started = time.perf_counter()
    try:
        with capture_stages() as logs, patch(
            "doc_loaders.pdf_document_loader.subprocess.run", wraps=runner,
        ) as processes:
            with redirect_stdout(sys.stderr):
                assets = pipeline.prepare("sample", mode="base")
        elapsed = (time.perf_counter() - started) * 1000
        assert sum(logs.processes.values()) == processes.call_count
        result = {
            "ready": assets.structured_document_ready,
            "errors": assets.errors,
            "elapsed_ms": round(elapsed, 1),
            "stages_ms": {key: round(value, 1) for key, value in logs.stages.items()},
            "process_calls": dict(logs.processes),
            "save_calls_total": store.save_calls,
        }
        return assets, result
    finally:
        request_id_context.reset(token)


def accept(source, *, page=None, language="eng+chi_sim+chi_tra", ocr=True,
           expected=None, expected_text=None, required_ocr_pages=None,
           native_reference_pages=()):
    """page is a one-based excerpt, explicitly not whole-book acceptance."""
    source = Path(source)
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    with tempfile.TemporaryDirectory(prefix="ocr-08-") as directory:
        root = Path(directory)
        target = root / "sample.pdf"
        if page is None:
            shutil.copyfile(source, target)
        else:
            writer = PdfWriter()
            writer.add_page(PdfReader(source).pages[page - 1])
            writer.write(target)
        fixture_hash = hashlib.sha256(target.read_bytes()).hexdigest()
        store = IsolatedStore(root / "structured.json")
        loader = PdfDocumentLoader(root, ocr_enabled=ocr, ocr_language=language,
                                   tesseract_cmd="tesseract", pdf_renderer_cmd="pdftoppm")
        pipeline = _build_pipeline(loader=loader, store=store)
        pipeline.loader_factory = DocumentLoaderFactory(root)
        pipeline.loader_factory.pdf_loader = loader
        pipeline.structured_document_builder = StructuredDocumentBuilder()
        raw_load = loader.load_with_page_boundary_evidence
        evidence = {}

        def validate_boundaries(doc_name):
            raw, boundaries = raw_load(doc_name)
            previous_end = 0
            assert boundaries.source_sha256 == fixture_hash
            assert len(boundaries.boundaries) == len(PdfReader(target).pages)
            reference_reader = PdfReader(target)
            for index, boundary in enumerate(boundaries.boundaries):
                assert boundary.page_index == index
                assert boundary.char_start >= previous_end
                assert raw[boundary.char_start:boundary.char_end] == boundary.text
                assert boundary.source_sha256 == fixture_hash
                if index in native_reference_pages:
                    assert boundary.text == (reference_reader.pages[index].extract_text() or "")
                previous_end = boundary.char_end
            assert previous_end == len(raw)
            evidence.update(page_count=boundaries.page_count, raw_chars=len(raw),
                            boundaries="exact_ordered_spans",
                            text_sha256=hashlib.sha256(raw.encode()).hexdigest())
            if expected_text is not None:
                assert expected_text in " ".join(raw.split()), "known transcript missing"
            return raw, boundaries

        with patch.object(loader, "load_with_page_boundary_evidence", side_effect=validate_boundaries):
            assets, cold = measure(pipeline, store, "cold")
        ready = assets.structured_document_ready
        if expected is not None:
            assert ready == expected, cold
        warm = None
        if ready:
            assert not assets.errors and evidence, cold
            document = store.load(None)
            payload = document.to_dict()
            assert "sections" not in payload and "structure_nodes" not in payload
            assert document.chapters and document.raw_text == assets.raw_text
            if not cold["process_calls"].get("text_recovery.render", 0):
                native_pages = [item.extract_text() or "" for item in PdfReader(target).pages]
                assert assets.raw_text == "\n".join(text for text in native_pages if text)
                evidence["quality_check"] = "exact_native_text"
            else:
                evidence["quality_check"] = ("known_transcript" if expected_text
                                             else "runtime_gate_only_no_ground_truth")
            if required_ocr_pages is not None:
                assert cold["process_calls"].get("text_recovery.render", 0) == required_ocr_pages
                assert cold["process_calls"].get("text_recovery.ocr", 0) == required_ocr_pages * 4
            evidence["layout"] = document.parse_provenance.get("layout_evidence")
            evidence["parser"] = document.parse_provenance.get("effective_parser_mode")
            before = store.path.read_bytes()
            with patch.object(pipeline.loader_factory, "get", side_effect=AssertionError("warm raw load")), \
                    patch.object(pipeline.profile_builder, "build", side_effect=AssertionError("warm profile")):
                reused, warm = measure(pipeline, store, "warm")
            assert reused.structured_document_ready and reused.raw_text is None
            assert not warm["process_calls"] and store.save_calls == 1
            assert before == store.path.read_bytes()
        else:
            assert not store.path.exists() and store.save_calls == 0
            assert assets.errors
            evidence["quality_check"] = "rejected_no_persistence"
        assert hashlib.sha256(source.read_bytes()).hexdigest() == source_hash
        return {"source": source.name, "source_sha256": source_hash,
                "page_excerpt": page, "fixture_sha256": fixture_hash,
                "settings": {"ocr": ocr, "language": loader.ocr_language,
                             "dpi": loader.pdf_render_dpi},
                "cold": cold, "warm": warm, "evidence": evidence}


SCAN_LINES = (
    "Chapter Two. Evidence and interpretation.",
    "Careful readers compare evidence before accepting a claim.",
    "A reliable conclusion explains both the observation and its limits.",
    "We should consider alternative explanations before deciding.",
    "The next paragraph provides independent support for this argument.",
)
TRANSCRIPT = " ".join(SCAN_LINES)


def make_fixtures(root):
    _, native = page_fixture(
        b"BT /F1 3 Tf 5 70 Td (Chapter One. Native text must remain unchanged and in order.) Tj ET",
        with_font=True,
    )
    writer = PdfWriter()
    writer.add_page(native)
    writer.write(root / "native.pdf")
    font_path = next((p for p in (
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
    ) if p.is_file()), None)
    if font_path is None:
        raise RuntimeError("No acceptance fixture font available")
    canvas = Image.new("RGB", (1600, 2000), "white")
    drawing = ImageDraw.Draw(canvas)
    font = ImageFont.truetype(str(font_path), 32)
    for index, line in enumerate(SCAN_LINES):
        drawing.text((90, 120 + index * 90), line, fill="black", font=font)
    canvas.save(root / "scan.pdf", resolution=150.0)
    writer = PdfWriter()
    writer.add_page(native)
    writer.add_page(PdfReader(root / "scan.pdf").pages[0])
    writer.add_blank_page(width=100, height=100)
    writer.add_outline_item("Chapter One", 0)
    writer.add_outline_item("Chapter Two", 1)
    writer.write(root / "mixed-outline.pdf")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", action="append", default=[], type=Path)
    parser.add_argument("--sample", action="append", nargs=2, default=[], metavar=("PDF", "PAGE"))
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()
    print(json.dumps({"python": platform.python_version(), "pypdf": pypdf.__version__,
                      "scope": "real loader/builder/JSON; stub language/profile; no API/DB/LLM"}))
    for source in args.pdf:
        print(json.dumps(accept(source), ensure_ascii=True), flush=True)
    for source, page in args.sample:
        print(json.dumps(accept(source, page=int(page)), ensure_ascii=True), flush=True)
    if args.synthetic:
        with tempfile.TemporaryDirectory(prefix="ocr-08-fixtures-") as directory:
            root = Path(directory)
            make_fixtures(root)
            for name, count in (("native", 0), ("scan", 1), ("mixed-outline", 1)):
                result = accept(root / f"{name}.pdf", language="eng", expected=True,
                                expected_text=TRANSCRIPT if count else None,
                                required_ocr_pages=count,
                                native_reference_pages=(0, 2) if name == "mixed-outline" else ())
                if name == "native":
                    assert not result["cold"]["process_calls"]
                if name == "mixed-outline":
                    assert result["evidence"]["parser"] == "native_pdf_outline"
                    assert not any(key.startswith("layout_analysis") for key in result["cold"]["process_calls"])
                print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
