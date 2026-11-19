"""RETRIEVAL-02: real FAISS/QA output privacy without altering returned sources."""

import contextlib
import io
import logging
import os
import socket
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import faiss
import numpy as np
from llama_index.core.schema import TextNode
from fastapi.testclient import TestClient
from pydantic import ValidationError
from api_schemas import AskDocumentRequest
from retrieval.faiss_index_bundle import FaissIndexBundle
from retrieval.node_record import NodeRecord
from context.document_context_builder import DocumentContextBuilder
from context.token_budget_manager import TokenBudgetManager
from prompts.prompt_assembler import PromptAssembler
from logging_config import configure_logging
from test_context_capacity import make_case
from test_question_output_boundary import route_app

MARKER = "PRIVATE_RETRIEVAL_SYNTHETIC_BODY"
TEXT = (
    MARKER
    + " This synthetic document is stored as one complete searchable chunk. Its full source must be used by answering without ordinary diagnostics copying it. "
)
SOURCE = "PRIVATE_SOURCE_FILENAME.txt"
OBS = []


class Embedder:
    def __init__(self, vector=(0.0, 0.0)):
        self.vector = vector
        self.calls = []

    def get_text_embedding(self, text):
        self.calls.append(text)
        return self.vector


def bundle(texts, ids=None, vectors=None, linked=True):
    ids = list(range(len(texts))) if ids is None else ids
    vectors = (
        [[float(i), 0.0] for i in range(len(texts))] if vectors is None else vectors
    )
    records = {}
    for i, (key, text) in enumerate(zip(ids, texts)):
        metadata = {
            "chunk_index": i,
            "char_start": i * 100,
            "char_end": i * 100 + len(text),
            "source": SOURCE,
            "chapter": "Chapter " + str(i),
            "position": i,
        }
        if linked:
            metadata.update(
                prev_node_id="node" + str(ids[i - 1]) if i else None,
                next_node_id="node" + str(ids[i + 1]) if i + 1 < len(ids) else None,
            )
        records[key] = NodeRecord(
            TextNode(text=text, id_="node" + str(key), metadata=metadata), key
        )
    index = faiss.IndexIDMap(faiss.IndexFlatL2(2))
    if ids:
        index.add_with_ids(
            np.array(vectors, dtype="float32"), np.array(ids, dtype="int64")
        )
    return FaissIndexBundle(index, Embedder(), None, records, 2, "en", 1000, 2000, 100)


def quiet_search(b, q="Exact query 保留空格 ", k=3):
    with contextlib.redirect_stdout(io.StringIO()):
        return b.search(q, k)


class IsolatedTest(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.temp = self.stack.enter_context(tempfile.TemporaryDirectory())
        previous = os.getcwd()
        os.chdir(self.temp)
        self.addCleanup(os.chdir, previous)
        self.stack.enter_context(
            patch.dict(os.environ, {"DEEP_READER_STRUCTURED_STORAGE_BACKEND": "file"})
        )
        for name in (
            "socket.socket.connect",
            "socket.socket.connect_ex",
            "socket.create_connection",
        ):
            self.stack.enter_context(
                patch(name, side_effect=AssertionError("network forbidden"))
            )


class Controls(IsolatedTest):
    def test_real_l2_rank_scores_ids_and_original_query(self):
        b = bundle(
            ["First", "Second", "Third"],
            [13, 7, 42],
            [[2.0, -1.0], [0.0, 0.0], [-1.0, 0.0]],
        )
        q = "Exact query 保留空格 "
        hits = quiet_search(b, q)
        self.assertEqual([h.faiss_id for h in hits], [7, 42, 13])
        self.assertEqual([h.score for h in hits], [0.0, 1.0, 5.0])
        self.assertEqual(b.embedder.calls, [q])
        self.assertEqual(
            (hits[0].text, hits[0].source, hits[0].chapter, hits[0].position),
            ("Second", SOURCE, "Chapter 1", 1),
        )

    def test_valid_top_k_limits_and_faiss_sentinels(self):
        for k, expected in [(1, 1), (3, 3), (20, 5)]:
            with self.subTest(k=k):
                b = bundle(["source" + str(i) for i in range(5)])
                hits = quiet_search(b, k=k)
                self.assertEqual(len(hits), expected)
                self.assertEqual(len(b.embedder.calls), 1)
                self.assertTrue(all(h.faiss_id >= 0 for h in hits))

    def test_empty_index_has_no_hits_or_source_output(self):
        b = bundle([])
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            self.assertEqual(b.search("question", 20), [])
        self.assertEqual(b.embedder.calls, ["question"])
        self.assertNotIn(MARKER, stream.getvalue())
        self.assertNotIn(SOURCE, stream.getvalue())

    def test_normalized_text_filter_preserves_first_and_does_not_refill(self):
        b = bundle(["Same words", "Same\nwords", "Same    words", "Different words"])
        hits = quiet_search(b, k=3)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].text, "Same words")
        self.assertEqual(hits[0].faiss_id, 0)
        OBS.append(
            {
                "observation": "post-top-k dedupe underfills; maximum-count contract, not blocker",
                "raw_k": 3,
                "hits": len(hits),
                "distinct_fourth_exists": True,
            }
        )

    def test_get_node_and_bidirectional_maps_with_noncontiguous_ids(self):
        b = bundle(["left", "center", "right"], [42, 7, 83])
        self.assertIs(b.get_node_by_id(7), b.id_to_record[7])
        self.assertIsNone(b.get_node_by_id(999))
        self.assertEqual(
            b.node_key_to_faiss_id, {"node42": 42, "node7": 7, "node83": 83}
        )
        self.assertEqual(b.chunk_index_to_faiss_id, {0: 42, 1: 7, 2: 83})

    def test_neighbor_links_keep_source_order(self):
        b = bundle(["left", "center", "right"], [42, 7, 83])
        builder = DocumentContextBuilder(TokenBudgetManager(PromptAssembler()))
        self.assertEqual(
            builder.build_local_window(b, 7, 1), ["left", "center", "right"]
        )
        self.assertEqual(
            builder.build_local_window(b, 42, 2), ["left", "center", "right"]
        )

    def test_neighbor_chunk_fallback_keeps_source_order(self):
        b = bundle(["left", "center", "right"], [42, 7, 83], linked=False)
        builder = DocumentContextBuilder(TokenBudgetManager(PromptAssembler()))
        self.assertEqual(
            builder.build_local_window(b, 7, 1), ["left", "center", "right"]
        )
        self.assertEqual(builder.build_local_window(b, 999, 1), [])

    def test_profile_attachment_fluent_identity(self):
        b = bundle(["source"])
        profile = object()
        self.assertIs(b.set_profile(profile), b)
        self.assertIs(b.profile, profile)

    def test_public_request_k_boundary(self):
        for k in [1, 20]:
            self.assertEqual(
                AskDocumentRequest(doc_name="book", query="Question", top_k=k).top_k, k
            )
        for k in [0, -1, 21]:
            with self.subTest(k=k), self.assertRaises(ValidationError):
                AskDocumentRequest(doc_name="book", query="Question", top_k=k)


class Privacy(IsolatedTest):
    def configured(self, level, fn):
        out = io.StringIO()
        err = io.StringIO()
        root = logging.getLogger()
        old_handlers = root.handlers[:]
        old_level = root.level
        root.handlers = []
        http_logger = logging.getLogger("httpx")
        old_filters = http_logger.filters[:]
        try:
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(
                err
            ), patch.dict(
                os.environ,
                {
                    "DEEP_READER_LOG_LEVEL": level,
                    "DEEP_READER_LOG_DIR": str(Path(self.temp) / "logs" / level),
                },
            ):
                logpath = configure_logging()
                value = fn()
                for handler in root.handlers:
                    handler.flush()
                logged = logpath.read_text()
            return value, out.getvalue(), err.getvalue(), logged
        finally:
            for h in root.handlers:
                h.close()
            root.handlers = old_handlers
            root.setLevel(old_level)
            http_logger.filters[:] = old_filters

    def test_direct_search_omits_source_at_info_and_error(self):
        for level in ["INFO", "ERROR"]:
            with self.subTest(level=level):
                b = bundle([TEXT])
                hits, stdout, stderr, file = self.configured(
                    level, lambda: b.search("question", 3)
                )
                self.assertEqual(len(hits), 1)
                self.assertEqual(hits[0].text, TEXT.strip())
                self.assertEqual(hits[0].source, SOURCE)
                self.assertEqual(hits[0].score, 0.0)
                self.assertEqual(b.embedder.calls, ["question"])
                self.assertNotIn(MARKER, file + stderr)
                OBS.append(
                    {
                        "path": "direct",
                        "level": level,
                        "returned_text_unchanged": True,
                        "source_in_stdout": MARKER in stdout,
                        "complete_chunk_in_stdout": TEXT.strip() in stdout,
                        "source_name_in_stdout": SOURCE in stdout,
                        "source_in_file_or_stderr": MARKER in file + stderr,
                    }
                )
                self.assertNotIn(MARKER, stdout)
                self.assertNotIn(SOURCE, stdout + stderr + file)

    def test_actual_ask_route_omits_source_at_info_and_error(self):
        for level in ["INFO", "ERROR"]:
            with self.subTest(level=level):
                case = make_case(texts=[TEXT])
                case.bundle.id_to_record[0].metadata["source"] = SOURCE

                def run():
                    with TestClient(route_app(case.coordinator)) as client:
                        return client.post(
                            "/documents/ask",
                            json={
                                "doc_name": "synthetic",
                                "query": "Explain this sentence.",
                                "top_k": 3,
                                "session_id": "isolated",
                            },
                        )

                response, stdout, stderr, file = self.configured(level, run)
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["answer"], "Controlled answer")
                self.assertEqual(case.context.result.context_text, TEXT.strip())
                self.assertEqual(len(case.answer.calls), 1)
                self.assertIn(TEXT.strip(), case.answer.calls[0])
                self.assertNotIn(MARKER, response.text + stderr + file)
                OBS.append(
                    {
                        "path": "production ask function/schema in isolated ASGI",
                        "level": level,
                        "status": response.status_code,
                        "answer_calls": len(case.answer.calls),
                        "context_text_unchanged": True,
                        "full_short_document_in_stdout": TEXT.strip() in stdout,
                        "source_in_file_or_stderr": MARKER in file + stderr,
                        "source_in_http_response": MARKER in response.text,
                    }
                )
                self.assertNotIn(MARKER, stdout)
                self.assertNotIn(SOURCE, stdout + stderr + file)

    def test_query_and_metadata_markers_keep_exact_direct_and_qa_data(self):
        query = "PRIVATE_RETRIEVAL_QUERY Explain this sentence."
        for level in ("INFO", "ERROR", "DEBUG"):
            for path in ("direct", "qa"):
                with self.subTest(level=level, path=path):
                    if path == "direct":
                        b = bundle([TEXT])
                        value, stdout, stderr, file = self.configured(
                            level, lambda: b.search(query, 3)
                        )
                        self.assertEqual(b.embedder.calls, [query])
                        self.assertEqual(value[0].text, TEXT.strip())
                        self.assertEqual(value[0].source, SOURCE)
                    else:
                        case = make_case(query=query, texts=[TEXT])
                        case.bundle.id_to_record[0].metadata["source"] = SOURCE
                        with patch.object(
                            case.bundle.embedder,
                            "get_text_embedding",
                            wraps=case.bundle.embedder.get_text_embedding,
                        ) as embedding:

                            def run():
                                with TestClient(route_app(case.coordinator)) as client:
                                    return client.post(
                                        "/documents/ask",
                                        json={
                                            "doc_name": "synthetic",
                                            "query": query,
                                            "top_k": 3,
                                            "session_id": "isolated",
                                        },
                                    )

                            value, stdout, stderr, file = self.configured(level, run)
                            embedding.assert_called_once_with(query)
                        self.assertEqual(value.status_code, 200)
                        self.assertEqual(value.json()["answer"], "Controlled answer")
                        self.assertEqual(len(case.answer.calls), 1)
                        self.assertEqual(len(case.standardizer.calls), 1)
                        self.assertIn(TEXT.strip(), case.answer.calls[0])
                        self.assertIn(query, case.answer.calls[0])
                        self.assertEqual(case.context.result.context_text, TEXT.strip())
                        self.assertIn("ContextOrchestrator#scope:", stdout)
                    for marker in (MARKER, SOURCE, "PRIVATE_RETRIEVAL_QUERY"):
                        self.assertNotIn(marker, stdout + stderr + file)

    def test_empty_search_all_sinks_and_normal_application_logging(self):
        for level in ("INFO", "ERROR", "DEBUG"):
            with self.subTest(level=level):
                b = bundle([])
                query = "PRIVATE_RETRIEVAL_QUERY"

                def run():
                    hits = b.search(query, 20)
                    logging.getLogger("retrieval_test").log(
                        getattr(logging, level), "retrieval_control hits=%d", len(hits)
                    )
                    return hits

                hits, stdout, stderr, file = self.configured(level, run)
                self.assertEqual(hits, [])
                self.assertEqual(b.embedder.calls, [query])
                self.assertIn("retrieval_control hits=0", stderr)
                self.assertIn("retrieval_control hits=0", file)
                for marker in (MARKER, SOURCE, query):
                    self.assertNotIn(marker, stdout + stderr + file)


if __name__ == "__main__":
    unittest.main()
