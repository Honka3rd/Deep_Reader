#!/usr/bin/env python3
"""Request-scoped cache identity, subprocess budgets and content-free diagnostics."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import replace
import io
import logging
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image
from pypdf import PageObject, PdfWriter

from doc_loaders.document_load_errors import RawTextOcrFailedError
from doc_loaders.document_loader_factory import DocumentLoaderFactory
from doc_loaders.pdf_document_loader import PdfDocumentLoader
from doc_loaders.pdf_execution_pass import PdfBudgetExceeded, PdfExecutionLimits, PdfExecutionPass
from doc_loaders.pdf_ocr_layout_fixtures import OCR_LAYOUT_FIXTURES
from doc_loaders.pdf_ocr_policy import PdfOcrPolicy
from logging_config import RequestContextFilter, request_id_context
from test_pdf_page_inspection import add_image, page_fixture
from test_prepare_structured_reuse_before_raw_load import _build_pipeline, _fake_document, _StructuredStore
from document_structure.structured_document_builder import StructuredDocumentBuilder


class ExecutionPassTests(unittest.TestCase):
    def test_limits_validate_negative_bool_and_nonfinite_values(self):
        for name in PdfExecutionLimits.__dataclass_fields__:
            for value in (-1, True, float("inf"), float("nan")):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    PdfExecutionLimits(**{name: value})

    def test_cache_is_lru_bounded_and_defensively_copied(self):
        state = PdfExecutionPass(PdfExecutionLimits(cache_entries=2, cache_bytes=2000))
        state.put("a", [1])
        state.put("b", [2])
        state.get("a", stage="test").append(9)
        state.put("c", [3])
        self.assertIsNone(state.get("b", stage="test"))
        self.assertEqual(state.get("a", stage="test"), [1])
        state.put("huge", "x" * 3000)
        self.assertIsNone(state.get("huge", stage="test"))
        self.assertLessEqual(state.cache_bytes, 2000)
        state.clear_cache()
        self.assertEqual(state.cache_bytes, 0)

    def test_unknown_keys_and_disabled_cache_never_hit(self):
        for limits in (PdfExecutionLimits(cache_entries=0), PdfExecutionLimits(cache_bytes=0)):
            state = PdfExecutionPass(limits)
            state.put("a", [1])
            state.put(None, [1])
            self.assertIsNone(state.get("a", stage="test"))
            self.assertIsNone(state.get(None, stage="test"))

    def test_deadline_clamps_timeout_and_rejects_late_result(self):
        now = [0.0]
        state = PdfExecutionPass(PdfExecutionLimits(text_seconds=2), clock=lambda: now[0])

        def run(*args, **kwargs):
            self.assertEqual(kwargs["timeout"], 2)
            now[0] = 3
            return subprocess.CompletedProcess([], 0, "", "")

        with self.assertRaisesRegex(PdfBudgetExceeded, "time_budget_exhausted"):
            state.run(run, [], purpose="text_recovery", stage="ocr", page=0, timeout=60)
        self.assertEqual(state.counters[("text_recovery", "ocr")], 1)

    def test_budget_timeout_differs_from_per_command_timeout(self):
        for seconds, expected in ((1, PdfBudgetExceeded), (120, subprocess.TimeoutExpired)):
            state = PdfExecutionPass(PdfExecutionLimits(text_seconds=seconds), clock=lambda: 0)
            runner = Mock(side_effect=subprocess.TimeoutExpired("PRIVATE_COMMAND", 1))
            with self.assertRaises(expected):
                state.run(runner, [], purpose="text_recovery", stage="ocr", page=0, timeout=60)
            self.assertEqual(runner.call_count, 1)

    def test_failed_attempts_count_and_purpose_budgets_are_independent(self):
        state = PdfExecutionPass(PdfExecutionLimits(text_render_attempts=1, layout_render_attempts=0))
        runner = Mock(side_effect=OSError("PRIVATE"))
        with self.assertRaises(OSError):
            state.run(runner, [], purpose="text_recovery", stage="render", page=0, timeout=60)
        for purpose in ("text_recovery", "layout_analysis"):
            with self.assertRaises(PdfBudgetExceeded):
                state.run(runner, [], purpose=purpose, stage="render", page=0, timeout=60)
        self.assertEqual(runner.call_count, 1)


class PdfRuntimePassTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.path = self.directory / "sample.pdf"
        self.engine = self.directory / "engine"
        self.renderer = self.directory / "renderer"
        self.engine.write_bytes(b"fixture engine v1")
        self.renderer.write_bytes(b"fixture renderer v1")
        self.loader = PdfDocumentLoader(base_dir=self.directory, ocr_enabled=True,
                                        ocr_language="eng", tesseract_cmd="engine", pdf_renderer_cmd="renderer")

    def write_pdf(self, *kinds, revision=""):
        writer = PdfWriter()
        for kind in kinds:
            content = ("BT /F1 3 Tf 10 60 Td (PRIVATE_BODY A useful native paragraph with enough readable words. "
                       + revision + ") Tj ET").encode() if kind == "native" else b"q 100 0 0 100 0 0 cm /Im Do Q"
            _, page = page_fixture(content, with_font=kind == "native")
            if kind != "native":
                add_image(page)
            writer.add_page(page)
        writer.write(self.path)

    def mock_process(self, stack):
        stack.enter_context(patch("doc_loaders.pdf_document_loader.shutil.which",
                                 side_effect=lambda cmd: str(self.engine if cmd == "engine" else self.renderer)))

        def run(command, **kwargs):
            if "--version" in command:
                output = "tesseract fixture-v1"
            elif "-singlefile" in command:
                image = Image.new("RGB", (20, 20), "white")
                image.putpixel((0, 0), (0, 0, 0))
                image.save(command[-1] + ".png")
                output = ""
            elif "tsv" in command:
                output = OCR_LAYOUT_FIXTURES["horizontal_scan_leader_lines"].tsv_text()
            else:
                output = "Recovered complete paragraph with enough readable words for quality."
            return subprocess.CompletedProcess(command, 0, output, "")

        return stack.enter_context(patch("doc_loaders.pdf_document_loader.subprocess.run", side_effect=run))

    def test_native_inspection_shared_across_raw_and_layout_only_in_one_pass(self):
        self.write_pdf("native", "native")
        with patch.object(PageObject, "extract_text", autospec=True, side_effect=PageObject.extract_text) as extract:
            with self.loader.preparation_scope() as state:
                self.loader.load("sample")
                self.loader.load_page_layout_evidence("sample")
                self.assertEqual(extract.call_count, 2)
                self.assertTrue(state.cache)
            self.assertFalse(state.cache)
            self.loader.load("sample")
        self.assertEqual(extract.call_count, 4)

    def test_raw_and_coordinate_results_reuse_but_never_cross_purposes(self):
        self.write_pdf("scan")
        with ExitStack() as stack:
            process = self.mock_process(stack)
            with self.loader.preparation_scope() as state:
                text = self.loader.load("sample")
                first = self.loader.load_page_layout_evidence("sample")
                calls = process.call_count
                self.assertEqual(self.loader.load("sample"), text)
                second = self.loader.load_page_layout_evidence("sample")
                self.assertEqual(process.call_count, calls)
                self.assertFalse(first[0].cache_hit)
                self.assertTrue(second[0].cache_hit)
                self.assertEqual(state.counters[("text_recovery", "render")], 1)
                self.assertEqual(state.counters[("text_recovery", "ocr")], 4)
                self.assertEqual(state.counters[("layout_analysis", "render")], 1)
                self.assertEqual(state.counters[("layout_analysis", "ocr")], 2)
                self.assertEqual(self.loader.last_ocr_pages, [text])
                self.assertEqual(self.loader.last_ocr_provenance["recovery_page_indexes"], [0])
            self.loader.load("sample")
            self.assertGreater(process.call_count, calls)

    def test_execution_key_covers_settings_versions_page_region_and_purpose(self):
        self.write_pdf("scan")
        with ExitStack() as stack:
            self.mock_process(stack)
            with self.loader.preparation_scope():
                base = self.loader._execution_key("a" * 64, 0, "layout_analysis", 0)
                for args in (("b" * 64, 0, "layout_analysis", 0), ("a" * 64, 1, "layout_analysis", 0),
                             ("a" * 64, 0, "text_recovery", 0), ("a" * 64, 0, "layout_analysis", 90)):
                    self.assertNotEqual(base, self.loader._execution_key(*args))
                for name, value in (("ocr_language", "chi_tra"), ("pdf_render_dpi", 150), ("ocr_timeout_seconds", 3)):
                    old = getattr(self.loader, name)
                    setattr(self.loader, name, value)
                    self.assertNotEqual(base, self.loader._execution_key("a" * 64, 0, "layout_analysis", 0))
                    setattr(self.loader, name, old)
                self.assertIn("full-page", base)

    def test_layout_cache_invalidates_on_binary_policy_language_and_dpi_changes(self):
        self.write_pdf("scan")
        changes = [lambda: self.engine.write_bytes(b"engine-v2"),
                   lambda: self.renderer.write_bytes(b"renderer-v2"),
                   lambda: setattr(self.loader, "ocr_language", "chi_tra"),
                   lambda: setattr(self.loader, "pdf_render_dpi", 150),
                   lambda: setattr(self.loader, "ocr_policy", PdfOcrPolicy(replace(self.loader.ocr_policy.config, min_native_chars=41)))]
        with ExitStack() as stack:
            process = self.mock_process(stack)
            with self.loader.preparation_scope():
                previous_key = self.loader.load_page_layout_evidence("sample")[0].cache_key
                for change in changes:
                    calls = process.call_count
                    change()
                    result = self.loader.load_page_layout_evidence("sample")
                    self.assertFalse(result[0].cache_hit)
                    self.assertNotEqual(previous_key, result[0].cache_key)
                    previous_key = result[0].cache_key
                    self.assertGreater(process.call_count, calls)

    def test_unknown_binary_identity_does_not_reuse_ocr_output(self):
        self.write_pdf("scan")
        with ExitStack() as stack:
            process = self.mock_process(stack)
            stack.enter_context(patch.object(self.loader, "_binary_identity", return_value=None))
            with self.loader.preparation_scope():
                self.loader.load_page_layout_evidence("sample")
                calls = process.call_count
                result = self.loader.load_page_layout_evidence("sample")
                self.assertFalse(result[0].cache_hit)
                self.assertIsNone(result[0].cache_key)
                self.assertGreater(process.call_count, calls)

    def test_source_replacement_invalidates_native_evidence(self):
        self.write_pdf("native")
        with self.loader.preparation_scope():
            first = self.loader.load("sample")
            self.write_pdf("native", revision="new edition")
            second = self.loader.load("sample")
        self.assertNotEqual(first, second)

    def test_required_text_render_zero_fails_without_publishing_partial_text(self):
        self.write_pdf("native", "scan")
        self.loader.execution_limits = replace(self.loader.execution_limits, text_render_attempts=0)
        with ExitStack() as stack:
            process = self.mock_process(stack)
            with self.assertRaisesRegex(RawTextOcrFailedError, "text_recovery_render_budget_exhausted"):
                self.loader.load("sample")
            self.assertFalse(any("-singlefile" in call.args[0] for call in process.call_args_list))
        self.assertIsNone(self.loader.last_ocr_pages)

    def test_required_text_ocr_exhaustion_rejects_even_a_good_first_candidate(self):
        self.write_pdf("scan")
        self.loader.execution_limits = replace(self.loader.execution_limits, text_ocr_attempts=1)
        with ExitStack() as stack:
            self.mock_process(stack)
            with self.assertRaisesRegex(RawTextOcrFailedError, "text_recovery_ocr_budget_exhausted"):
                self.loader.load("sample")
        self.assertIsNone(self.loader.last_ocr_provenance)

    def test_layout_exhaustion_retains_all_page_inventory(self):
        self.write_pdf("scan", "native", "scan")
        for changes, reason in (({"layout_render_attempts": 0}, "render"),
                                ({"layout_ocr_attempts": 1}, "ocr"),
                                ({"layout_seconds": 0}, "time")):
            with self.subTest(changes=changes), ExitStack() as stack:
                self.loader.execution_limits = replace(PdfExecutionLimits(), **changes)
                self.mock_process(stack)
                pages = self.loader.load_page_layout_evidence("sample")
            self.assertEqual(len(pages), 3)
            self.assertIn("PRIVATE_BODY", pages[1].ocr_text)
            self.assertEqual(pages[0].failure_reason, f"layout_analysis_{reason}_budget_exhausted")
            self.assertEqual(pages[2].analysis_stage, "inventory")

    def test_text_time_and_metadata_zero_budgets_fail_explicitly(self):
        self.write_pdf("scan")
        for changes, reason in (({"text_seconds": 0}, "time"), ({"metadata_attempts": 0}, "metadata")):
            with self.subTest(changes=changes), ExitStack() as stack:
                self.loader.execution_limits = replace(PdfExecutionLimits(), **changes)
                process = self.mock_process(stack)
                with self.assertRaisesRegex(RawTextOcrFailedError, f"text_recovery_{reason}_budget_exhausted"):
                    self.loader.load("sample")
                process.assert_not_called()

    def test_ocr_disabled_and_candidate_zero_cannot_be_bypassed_by_cache(self):
        self.write_pdf("scan")
        with ExitStack() as stack:
            process = self.mock_process(stack)
            with self.loader.preparation_scope():
                self.loader.load_page_layout_evidence("sample")
                calls = process.call_count
                self.assertEqual(self.loader.load_page_layout_evidence("sample", candidate_page_limit=0)[0].analysis_stage, "inventory")
                self.loader.ocr_enabled = False
                self.assertEqual(self.loader.load_page_layout_evidence("sample")[0].analysis_stage, "inventory")
                self.assertEqual(calls, process.call_count)

    def test_failed_pass_clears_cache_and_releases_context(self):
        self.write_pdf("scan")
        with self.loader.preparation_scope() as state:
            with patch.object(self.loader, "_load_pages_with_ocr", side_effect=RuntimeError("failure")):
                with self.assertRaises(RuntimeError):
                    self.loader.load("sample")
            self.assertFalse(state.cache)
        self.assertIsNone(self.loader._pass_context.get())

    def test_nested_scope_shares_budget_but_next_scope_starts_fresh(self):
        with self.loader.preparation_scope() as first:
            first.counters[("text_recovery", "ocr")] = 5
            with self.loader.preparation_scope() as nested:
                self.assertIs(nested, first)
        with self.loader.preparation_scope() as second:
            self.assertIsNot(first, second)
            self.assertFalse(second.counters)

    def test_cache_does_not_hide_source_replacement_or_page_limit_in_raw_recovery(self):
        self.write_pdf("scan")
        with ExitStack() as stack:
            process = self.mock_process(stack)
            with self.loader.preparation_scope():
                self.loader.load("sample")
                calls = process.call_count
                self.write_pdf("native", "scan")
                self.loader.load("sample")
                self.assertGreater(process.call_count, calls)
                self.loader.ocr_page_limit = 1
                with self.assertRaisesRegex(RawTextOcrFailedError, "excludes_required_page"):
                    self.loader.load("sample")

    def test_prepare_budget_failure_never_saves_partial_hierarchy(self):
        from doc_loaders.pdf_extraction_contract import PdfExtractionError
        self.write_pdf("native", "scan")
        self.loader.execution_limits = replace(self.loader.execution_limits, text_ocr_attempts=0)
        store = _StructuredStore(exists_value=False, document=_fake_document("sample"))
        pipeline = _build_pipeline(loader=self.loader, store=store)
        pipeline.loader_factory = DocumentLoaderFactory(base_dir=self.directory)
        pipeline.loader_factory.pdf_loader = self.loader
        with ExitStack() as stack:
            self.mock_process(stack)
            with self.assertRaises(PdfExtractionError) as caught:
                pipeline.prepare("sample", mode="base", pdf_extraction_mode="auto")
        self.assertIn("text_recovery_ocr_budget_exhausted", str(caught.exception.__cause__))
        self.assertEqual(store.save_calls, 0)
        self.assertEqual(pipeline.profile_builder.build_calls, 0)

    def test_prepare_optional_budget_failure_keeps_recovered_text_and_parser(self):
        self.write_pdf("scan")
        self.loader.execution_limits = replace(self.loader.execution_limits, layout_render_attempts=0)
        store = _StructuredStore(exists_value=False, document=_fake_document("sample"))
        pipeline = _build_pipeline(loader=self.loader, store=store)
        pipeline.loader_factory = DocumentLoaderFactory(base_dir=self.directory)
        pipeline.loader_factory.pdf_loader = self.loader
        pipeline.structured_document_builder = StructuredDocumentBuilder()
        with ExitStack() as stack:
            self.mock_process(stack)
            assets = pipeline.prepare("sample", mode="base", pdf_extraction_mode="auto")
        self.assertTrue(assets.structured_document_ready, assets.errors)
        self.assertIn("Recovered complete paragraph", assets.raw_text)
        self.assertIsNone(store.document.structure_error_code)
        self.assertEqual(store.save_calls, 1)

    def test_threads_do_not_share_cached_evidence_or_ocr_results(self):
        barrier = threading.Barrier(2)

        def worker(value):
            with self.loader.preparation_scope() as state:
                state.put("same", value)
                self.loader.last_ocr_pages = [value]
                barrier.wait(timeout=5)
                self.assertEqual(state.get("same", stage="test"), value)
                self.assertEqual(self.loader.last_ocr_pages, [value])
            self.assertFalse(state.cache)
            return self.loader.last_ocr_pages

        with ThreadPoolExecutor(2) as pool:
            results = list(pool.map(worker, ["request-a", "request-b"]))
        self.assertEqual(results, [["request-a"], ["request-b"]])
        self.assertIsNone(self.loader.last_ocr_pages)

    def test_prepare_and_manual_paths_use_real_factory_scope(self):
        self.write_pdf("native", "native")
        store = _StructuredStore(exists_value=False, document=_fake_document("sample"))
        pipeline = _build_pipeline(loader=self.loader, store=store)
        pipeline.loader_factory = DocumentLoaderFactory(base_dir=self.directory)
        pipeline.loader_factory.pdf_loader = self.loader
        pipeline.structured_document_builder = StructuredDocumentBuilder()
        with patch.object(PageObject, "extract_text", autospec=True, side_effect=PageObject.extract_text) as extract:
            assets = pipeline.prepare("sample", mode="base", pdf_extraction_mode="auto")
            self.assertTrue(assets.structured_document_ready, assets.errors)
            self.assertEqual(extract.call_count, 2)
            manual = pipeline.load_manual_structure_source_evidence("sample")
            self.assertEqual(len(manual.page_boundaries), 2)
            self.assertEqual(extract.call_count, 2)
        self.assertIsNone(self.loader._pass_context.get())

    def test_logs_correlate_and_exclude_source_ocr_and_subprocess_content(self):
        self.write_pdf("scan", "native")
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.addFilter(RequestContextFilter())
        handler.setFormatter(logging.Formatter("%(request_id)s %(message)s"))
        logger = logging.getLogger("doc_loaders")
        previous_level = logger.level
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)
        token = request_id_context.set("request-test-07")
        try:
            with ExitStack() as stack:
                self.mock_process(stack)
                with self.loader.preparation_scope():
                    self.loader.load_page_layout_evidence("sample")
                    self.loader.load_page_layout_evidence("sample")
            lines = stream.getvalue()
            for text in ("pdf_page_decision", "outcome=hit", "outcome=miss", "pdf_process", "duration_ms=", "pdf_pass_finished"):
                self.assertIn(text, lines)
            self.assertTrue(all(line.startswith("request-test-07") for line in lines.splitlines()))
            self.assertNotIn("PRIVATE_BODY", lines)
            self.assertNotIn("Recovered complete paragraph", lines)
            self.assertNotIn("level\tpage_num", lines)
        finally:
            request_id_context.reset(token)
            logger.removeHandler(handler)
            logger.setLevel(previous_level)
            handler.close()


if __name__ == "__main__":
    unittest.main()
