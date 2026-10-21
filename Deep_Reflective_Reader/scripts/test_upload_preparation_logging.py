#!/usr/bin/env python3
"""Logging regression with temporary files and fake preparation dependencies."""

import asyncio
import io
import logging
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient
import main
from logging_config import RequestContextFilter, log_stage, request_id_context
from test_prepare_structured_reuse_before_raw_load import (
    _build_pipeline, _fake_document, _StructuredStore, _TextLoader,
)


class LoggingTests(unittest.TestCase):
    def setUp(self):
        self.stream = io.StringIO()
        self.handler = logging.StreamHandler(self.stream)
        self.handler.addFilter(RequestContextFilter())
        self.handler.setFormatter(logging.Formatter("request_id=%(request_id)s %(message)s"))
        logging.getLogger().addHandler(self.handler)

    def tearDown(self):
        logging.getLogger().removeHandler(self.handler)
        self.handler.close()

    def test_upload_stages_correlate_without_content(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(main, "_RAW_DOCUMENT_BASE_DIR", Path(directory)):
            client = TestClient(main.app)
            response = client.post("/documents/upload", files={"file": ("Log Test.txt", b"PRIVATE_BODY_SENTINEL", "text/plain")})
            self.assertEqual(response.status_code, 201)
            request_id = response.headers["X-Request-ID"]
            lines = self.stream.getvalue().splitlines()
            for stage in ("upload_receive", "upload_multipart_parse", "upload_write", "upload_hash"):
                for event in ("stage_started", "stage_finished"):
                    matching = [line for line in lines if f"{event} stage={stage} " in line]
                    self.assertEqual(len(matching), 1)
                    self.assertIn(f"request_id={request_id}", matching[0])
            self.assertNotIn("PRIVATE_BODY_SENTINEL", self.stream.getvalue())
            self.assertIn("preparation_started=false", self.stream.getvalue())
            duplicate = client.post("/documents/upload", files={"file": ("Log Test.txt", b"new", "text/plain")})
            self.assertEqual(duplicate.status_code, 409)
            self.assertNotEqual(request_id, duplicate.headers["X-Request-ID"])
            self.assertIn("upload_rejected reason=duplicate_conflict", self.stream.getvalue())
            self.assertEqual(request_id_context.get(), "-")

    def test_prepare_stages_and_cache_preserve_behavior(self):
        loader = _TextLoader()
        store = _StructuredStore(exists_value=True, document=_fake_document("Log Test"))
        pipeline = _build_pipeline(loader=loader, store=store)
        with patch.object(main.qa_coordinator, "document_preparation_pipeline", pipeline):
            client = TestClient(main.app)
            response = client.post("/documents/prepare", json={"doc_name": "Log Test", "mode": "base", "force_rebuild": True})
            self.assertEqual(response.status_code, 200)
            request_id = response.headers["X-Request-ID"]
            for stage in ("prepare_pipeline", "raw_load", "language_detect", "profile_generate", "structured_build", "structured_save", "structured_readback"):
                self.assertIn(f"request_id={request_id} stage_finished stage={stage} ", self.stream.getvalue())
            self.assertIn("structured_ready=True", self.stream.getvalue())
            self.assertNotIn("force rebuild raw text", self.stream.getvalue())
            self.stream.truncate(0)
            self.stream.seek(0)
            reused = client.post("/documents/prepare", json={"doc_name": "Log Test", "mode": "base"})
            self.assertEqual(reused.status_code, 200)
            self.assertIn("structured_reused=True", self.stream.getvalue())
            self.assertNotIn("stage=raw_load", self.stream.getvalue())
            self.assertEqual(loader.load_calls, 1)

    def test_stage_failure_preserves_exception_without_detail(self):
        failure = RuntimeError("PRIVATE_ERROR_SENTINEL")
        with self.assertRaises(RuntimeError) as caught:
            with log_stage(logging.getLogger(__name__), "test_failure"):
                raise failure
        self.assertIs(caught.exception, failure)
        self.assertIn("stage_failed stage=test_failure", self.stream.getvalue())
        self.assertIn("duration_ms=", self.stream.getvalue())
        self.assertNotIn("PRIVATE_ERROR_SENTINEL", self.stream.getvalue())

    def test_request_context_isolated_across_async_and_worker_calls(self):
        async def run_one(value):
            token = request_id_context.set(value)
            try:
                await asyncio.sleep(0)
                await asyncio.to_thread(logging.getLogger(__name__).info, "context_probe=%s", value)
            finally:
                request_id_context.reset(token)

        async def run():
            await asyncio.gather(run_one("request-a"), run_one("request-b"))

        asyncio.run(run())
        for value in ("request-a", "request-b"):
            self.assertIn(f"request_id={value} context_probe={value}", self.stream.getvalue())
        self.assertEqual(request_id_context.get(), "-")


if __name__ == "__main__":
    unittest.main()
