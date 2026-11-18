"""EMBED-01: configured sinks with the actual installed embedding SDK, offline."""

import contextlib
import io
import json
import logging
import os
import socket
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
import openai

from embeddings.openai_embedder import OpenAIEmbedder
from logging_config import configure_logging, request_id_context, stage_id_context

SOURCE = [
    "PRIVATE_SOURCE_ONE " + "synthetic paragraph " * 30,
    "PRIVATE_SOURCE_TWO 中文 " + "另一段合成文字 " * 30,
]
KEY = "SYNTHETIC_KEY_NEVER_VALID"
TOKEN = "SYNTHETIC_URL_TOKEN"
BODY = "SYNTHETIC_PROVIDER_BODY"
HEADER = "SYNTHETIC_RESPONSE_HEADER"


class Key:
    def get(self):
        return KEY


class EmbeddingLoggingPrivacyTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = logging.getLogger()
        self.old_handlers = self.root.handlers[:]
        self.old_level = self.root.level
        # Protect the runner's handlers: configure_logging owns only this fixture's sinks.
        self.root.handlers = []
        self.http_logger = logging.getLogger("httpx")
        self.old_filters = self.http_logger.filters[:]
        self.stream = io.StringIO()
        self.stderr = contextlib.redirect_stderr(self.stream)
        self.stderr.__enter__()
        self.addCleanup(self.restore_logging)
        self.request_token = request_id_context.set("embedding-privacy-request")
        self.addCleanup(request_id_context.reset, self.request_token)

    def restore_logging(self):
        for handler in self.root.handlers[:]:
            self.root.removeHandler(handler)
            handler.close()
        self.root.handlers = self.old_handlers
        self.root.setLevel(self.old_level)
        self.http_logger.filters = self.old_filters
        self.stderr.__exit__(None, None, None)

    def configure(self, level, suffix=""):
        with patch.dict(
            os.environ,
            {
                "DEEP_READER_LOG_LEVEL": level,
                "DEEP_READER_LOG_DIR": str(Path(self.directory.name) / suffix),
            },
        ):
            return configure_logging()

    def assert_sinks(
        self, path, expected_calls, failed=False, level="DEBUG", stage="embedding.batch"
    ):
        for handler in self.root.handlers:
            handler.flush()
        for sink, content in [
            ("file", path.read_text()),
            ("stream", self.stream.getvalue()),
        ]:
            with self.subTest(sink=sink):
                for marker in SOURCE + [
                    "PRIVATE_SOURCE_ONE",
                    "PRIVATE_SOURCE_TWO",
                    KEY,
                    TOKEN,
                    BODY,
                    HEADER,
                ]:
                    self.assertNotIn(marker, content)
                self.assertNotIn("offline.invalid", content)
                self.assertEqual(
                    content.count("external_response endpoint=embeddings"),
                    expected_calls,
                )
                self.assertIn("request_id=embedding-privacy-request", content)
                self.assertIn("stage_id=", content)
                self.assertIn("duration_ms=", content)
                self.assertIn("stage_failed" if failed else "stage_finished", content)
                self.assertIn("stage=" + stage, content)
                if failed:
                    self.assertIn("error_type=BadRequestError", content)
                if expected_calls:
                    self.assertIn("status=400" if failed else "status=200", content)
                self.assertEqual(
                    "application_debug_safe count=2" in content, level == "DEBUG"
                )
                self.assertIn("application_info_safe", content)

    def sdk_case(self, level, kind, failed=False, texts=None):
        path = self.configure(level)
        calls = []
        texts = list(SOURCE if texts is None else texts)

        def respond(request):
            payload = json.loads(request.content)
            self.assertEqual(request.headers["authorization"], "Bearer " + KEY)
            calls.append(payload["input"])
            if failed:
                return httpx.Response(
                    400,
                    json={
                        "error": {
                            "message": BODY + " " + KEY + " " + SOURCE[0],
                            "type": "invalid_request_error",
                            "param": "input",
                            "code": "bad_input",
                        }
                    },
                    headers={"x-request-id": HEADER},
                )
            return httpx.Response(
                200,
                json={
                    "object": "list",
                    "model": payload["model"],
                    "data": [
                        {
                            "object": "embedding",
                            "index": i,
                            "embedding": [float(i), 1.0],
                        }
                        for i in range(len(payload["input"]))
                    ],
                    "usage": {"prompt_tokens": 1, "total_tokens": 1},
                },
                headers={"x-request-id": HEADER},
            )

        app_logger = logging.getLogger("app.embedding_logging_test")
        app_logger.debug("application_debug_safe count=2")
        app_logger.info("application_info_safe")
        with patch.object(
            socket.socket, "connect", side_effect=AssertionError("unexpected network")
        ):
            adapter = OpenAIEmbedder(Key())
            self.assertEqual(calls, [])
            with httpx.Client(
                transport=httpx.MockTransport(respond), params={"private": TOKEN}
            ) as client:
                # Query and response headers exercise raw HTTP/SDK logging exclusion.
                adapter.embed_model.api_base = "https://offline.invalid/v1"
                adapter.embed_model._http_client = client
                operation = {
                    "single": lambda: adapter.get_text_embedding(texts[0]),
                    "batch": lambda: adapter.get_text_embedding_batch(texts),
                    "probe": adapter.probe_vector_dimension,
                }[kind]
                if failed:
                    with self.assertRaises(openai.BadRequestError) as caught:
                        operation()
                    self.assertEqual(caught.exception.status_code, 400)
                    self.assertIn(
                        BODY, str(caught.exception)
                    )  # propagation is deliberately unchanged
                else:
                    value = operation()
                    if kind == "single":
                        self.assertEqual(value, [0.0, 1.0])
                    elif kind == "probe":
                        self.assertEqual(value, 2)
                    else:
                        self.assertEqual(
                            value, [[float(i % 100), 1.0] for i in range(len(texts))]
                        )
        expected = 1 if kind != "batch" else (len(texts) + 99) // 100
        self.assertEqual(len(calls), expected)
        wanted = (
            ["dimension probe"]
            if kind == "probe"
            else texts[:1] if kind == "single" else texts
        )
        self.assertEqual([text for batch in calls for text in batch], wanted)
        self.assertEqual(stage_id_context.get(), "-")
        self.assert_sinks(
            path,
            expected,
            failed,
            level,
            "embedding.batch" if kind == "batch" else "embedding.single",
        )

    def test_info_single_success(self):
        self.sdk_case("INFO", "single")

    def test_debug_single_success(self):
        self.sdk_case("DEBUG", "single")

    def test_info_batch_success(self):
        self.sdk_case("INFO", "batch")

    def test_debug_batch_success(self):
        self.sdk_case("DEBUG", "batch")

    def test_info_single_failure(self):
        self.sdk_case("INFO", "single", True)

    def test_debug_single_failure(self):
        self.sdk_case("DEBUG", "single", True)

    def test_info_batch_failure(self):
        self.sdk_case("INFO", "batch", True)

    def test_debug_batch_failure(self):
        self.sdk_case("DEBUG", "batch", True)

    def test_probe(self):
        self.sdk_case("DEBUG", "probe")

    def test_empty_batch(self):
        self.sdk_case("DEBUG", "batch", texts=[])

    def test_partitioned_batch_order(self):
        self.sdk_case(
            "DEBUG", "batch", texts=["synthetic entry " + str(i) for i in range(205)]
        )

    def test_configured_handlers_filter_children_even_if_sdk_enables_debug(self):
        path = self.configure("INFO")
        names = [
            "openai",
            "openai._base_client",
            "httpcore.connection",
            "httpx.child",
            "llama_index.core",
        ]
        for name in names:
            logger = logging.getLogger(name)
            previous = logger.level
            try:
                logger.setLevel(logging.DEBUG)
                for level in (
                    logging.DEBUG,
                    logging.INFO,
                    logging.WARNING,
                    logging.ERROR,
                ):
                    try:
                        raise RuntimeError(BODY)
                    except RuntimeError:
                        logger.log(level, "raw %s %s", SOURCE[0], KEY, exc_info=True)
            finally:
                logger.setLevel(previous)
        logging.getLogger("app.embedding_logging_test").info("application_info_safe")
        for h in self.root.handlers:
            h.flush()
        for content in (path.read_text(), self.stream.getvalue()):
            self.assertNotIn(SOURCE[0], content)
            self.assertNotIn(KEY, content)
            self.assertNotIn(BODY, content)
            self.assertIn("application_info_safe", content)

    def test_reconfiguration_observes_each_exchange_once_and_omits_url(self):
        self.configure("DEBUG", "first")
        path = self.configure("DEBUG", "second")
        with httpx.Client(
            transport=httpx.MockTransport(lambda request: httpx.Response(200))
        ) as client:
            client.post(
                "https://user:" + KEY + "@offline.invalid/v1/embeddings?token=" + TOKEN,
                headers={"Authorization": "Bearer " + KEY},
                json={"input": SOURCE},
            )
        for h in self.root.handlers:
            h.flush()
        for content in (path.read_text(), self.stream.getvalue()):
            self.assertEqual(content.count("external_response endpoint=embeddings"), 1)
            self.assertNotIn(TOKEN, content)
            self.assertNotIn(KEY, content)
            self.assertNotIn("offline.invalid", content)


if __name__ == "__main__":
    unittest.main()
