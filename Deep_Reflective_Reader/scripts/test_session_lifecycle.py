"""SESSION-01: actual QA/ASGI lifecycle races with deterministic providers."""

import contextlib
import io
import json
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, asdict
from types import SimpleNamespace

import faiss
import numpy as np
from fastapi.testclient import TestClient
from llama_index.core.schema import TextNode

from language.language_code import LanguageCode
from profile.document_profile import DocumentProfile
from question.standardized.question_standardizer import QuestionStandardizer
from retrieval.faiss_index_bundle import FaissIndexBundle
from retrieval.node_record import NodeRecord
from retrieval.search_metadata import SearchMetadata
from session.session_manager import SessionManager, SessionUpdateResult
from scripts.test_context_capacity import make_case
from scripts.test_question_output_boundary import route_app

OLD = "Explain this sentence. OLD_A"
NEW_B = "Explain this sentence. NEW_B"
NEW_A = "Explain this sentence. NEW_A"
FOLLOW_B = "Explain this sentence. FOLLOW_B"


class QueryEmbedder:
    def get_text_embedding(self, text):
        return [7.0, 0.0] if "OLD_A" in text else [0.0, 0.0]


class EchoStandardizer:
    def __init__(self):
        self.calls = []

    def complete_text(self, prompt):
        self.calls.append(prompt)
        return json.dumps(
            dict(
                user_language="en",
                standardized_query=prompt.split("User query: ", 1)[1].strip(),
            )
        )


class BlockingAnswer:
    def __init__(self, block=True):
        self.block = block
        self.entered = threading.Event()
        self.release = threading.Event()
        self.calls = []

    def complete_text(self, prompt):
        self.calls.append(prompt)
        if "OLD_A" in prompt and self.block:
            self.entered.set()
            if not self.release.wait(10):
                raise RuntimeError("test_event_timeout")
        return "Controlled answer"


class TraceContext:
    def __init__(self, actual):
        self.actual = actual
        self.rows = {}
        self.lock = threading.Lock()

    def build(self, **kwargs):
        result = self.actual.build(**kwargs)
        with self.lock:
            self.rows[kwargs["query"]] = dict(
                mode=result.mode.value,
                input_active=kwargs.get("session_active_chunk_index"),
            )
        return result


def make_bundle(name, count):
    records = {}
    offset = 0
    for i in range(count):
        text = f"{name} SOURCE_{i}\n" + name * 5000
        node = TextNode(
            id_=f"{name}-{i}",
            text=text,
            metadata=dict(
                chunk_index=i,
                char_start=offset,
                char_end=offset + len(text),
                prev_node_id=f"{name}-{i-1}" if i else None,
                next_node_id=f"{name}-{i+1}" if i + 1 < count else None,
            ),
        )
        records[i] = NodeRecord(node, i)
        offset += len(text) + 1
    index = faiss.IndexIDMap(faiss.IndexFlatL2(2))
    index.add_with_ids(
        np.array([[float(i), 0.0] for i in range(count)], dtype="float32"),
        np.arange(count, dtype="int64"),
    )
    bundle = FaissIndexBundle(
        index, QueryEmbedder(), None, records, 2, "en", 1500, 3200, 500
    )
    bundle.set_profile(
        DocumentProfile(name, "Synthetic source summary", LanguageCode.EN)
    )
    return bundle


class Engine:
    def __init__(self, block=True):
        case = make_case()
        self.coordinator = case.coordinator
        self.bundles = {"A": make_bundle("A", 8), "B": make_bundle("B", 2)}
        self.coordinator.document_preparation_pipeline = SimpleNamespace(
            prepare_and_load=lambda name: SimpleNamespace(
                bundle=self.bundles[name],
                assets=SimpleNamespace(
                    structured_document_path=None,
                    bundle_ready=True,
                    faiss_ready=True,
                    profile_ready=True,
                ),
            )
        )
        self.coordinator.session_manager = SessionManager()
        self.standardizer = EchoStandardizer()
        case.context.actual.question_standardizer = QuestionStandardizer(
            self.standardizer
        )
        self.trace = TraceContext(case.context.actual)
        self.coordinator.context_orchestrator = self.trace
        self.answer = BlockingAnswer(block)
        self.coordinator.llm_provider = self.answer
        self.app = route_app(self.coordinator)

    def state(self, sid="shared"):
        return asdict(self.coordinator.get_session(sid))

    @staticmethod
    def post(client, doc, query, sid="shared"):
        return client.post(
            "/documents/ask", json=dict(doc_name=doc, query=query, session_id=sid)
        )


class SessionRaces(unittest.TestCase):
    def run_race(self, kind):
        engine = Engine()
        with contextlib.redirect_stdout(io.StringIO()), TestClient(
            engine.app
        ) as client, ThreadPoolExecutor(max_workers=2) as pool:
            future = pool.submit(engine.post, client, "A", OLD)
            try:
                self.assertTrue(engine.answer.entered.wait(5))
                if kind == "reset":
                    engine.coordinator.session_manager.reset_session("shared")
                else:
                    response = engine.post(
                        client,
                        "B" if kind != "same" else "A",
                        NEW_B if kind != "same" else NEW_A,
                        "other" if kind == "isolated" else "shared",
                    )
                    self.assertEqual(response.status_code, 200)
                    if kind == "aba":
                        self.assertEqual(
                            engine.post(client, "A", NEW_A).status_code, 200
                        )
                before = engine.state("other" if kind == "isolated" else "shared")
            finally:
                engine.answer.release.set()
            self.assertEqual(future.result(10).status_code, 200)
            after = engine.state("other" if kind == "isolated" else "shared")
            expected_calls = {"reset": 1, "aba": 3}.get(kind, 2)
            self.assertEqual(len(engine.answer.calls), expected_calls)
            self.assertEqual(len(engine.standardizer.calls), expected_calls)
            if kind != "same":
                self.assertEqual(
                    after, before, "late completion must preserve every current field"
                )
            else:
                self.assertEqual(after["recent_questions"], [NEW_A, OLD])
                self.assertEqual(after["recent_chunk_indices"], [0, 7])
                self.assertEqual(after["active_chunk_index"], 7)
            if kind == "switch":
                self.assertEqual(after["doc_name"], "B")
                self.assertEqual(after["active_chunk_index"], 0)
                self.assertEqual(after["recent_questions"], [NEW_B])
                self.assertEqual(engine.post(client, "B", FOLLOW_B).status_code, 200)
                self.assertEqual(
                    engine.trace.rows[FOLLOW_B]["mode"], "local_window_mode"
                )
                self.assertEqual(len(engine.answer.calls), 3)
            if kind == "isolated":
                self.assertEqual(engine.state()["recent_questions"], [OLD])
                self.assertEqual(engine.state()["active_chunk_index"], 7)

    def test_switch(self):
        self.run_race("switch")

    def test_aba(self):
        self.run_race("aba")

    def test_reset(self):
        self.run_race("reset")

    def test_independent_sessions(self):
        self.run_race("isolated")

    def test_same_generation_concurrent_completions_merge(self):
        self.run_race("same")

    def test_serial_switch_and_followup(self):
        engine = Engine(False)
        with contextlib.redirect_stdout(io.StringIO()), TestClient(
            engine.app
        ) as client:
            self.assertEqual(engine.post(client, "A", OLD).status_code, 200)
            self.assertEqual(engine.state()["active_chunk_index"], 7)
            self.assertEqual(engine.post(client, "B", NEW_B).status_code, 200)
            self.assertEqual(engine.state()["recent_questions"], [NEW_B])
            self.assertEqual(engine.state()["recent_chunk_indices"], [0])
            self.assertEqual(engine.post(client, "B", FOLLOW_B).status_code, 200)
            self.assertEqual(engine.trace.rows[FOLLOW_B]["mode"], "local_window_mode")
            self.assertEqual(len(engine.answer.calls), 3)
            self.assertEqual(len(engine.standardizer.calls), 3)


class SessionControls(unittest.TestCase):
    def setUp(self):
        self.manager = SessionManager()
        self.bundle = make_case().bundle

    def update(self, snapshot, question, ids=(0,)):
        hits = [
            SearchMetadata(i, str(i), "controlled source", float(k))
            for k, i in enumerate(ids)
        ]
        return self.manager.update_session(
            snapshot, SessionUpdateResult(question, self.bundle, hits)
        )

    def test_snapshot_identity_and_history_are_isolated(self):
        first = self.manager.get_or_create_session("one", "A")
        with self.assertRaises(FrozenInstanceError):
            first.doc_name = "B"
        with self.assertRaises(FrozenInstanceError):
            first.generation = "other"
        first.recent_questions.append("local-only")
        first.recent_chunk_indices.append(999)
        self.assertEqual(self.manager.get_session("one").recent_questions, [])
        self.assertEqual(self.manager.get_session("one").recent_chunk_indices, [])
        self.assertTrue(self.update(first, "valid"))
        self.assertEqual(self.manager.get_session("one").recent_questions, ["valid"])
        read = self.manager.get_session("one")
        read.recent_questions.clear()
        self.assertEqual(self.manager.get_session("one").recent_questions, ["valid"])

    def test_reset_and_aba_reject_old_helper_updates(self):
        first = self.manager.get_or_create_session("one", "A")
        other = self.manager.get_or_create_session("other", "B")
        self.update(other, "other")
        other_state = self.manager.get_session("other")
        self.manager.reset_session("one")
        reset = self.manager.get_session("one")
        self.assertNotEqual(reset.generation, first.generation)
        self.assertFalse(self.update(first, "stale"))
        self.assertEqual(self.manager.get_session("one"), reset)
        self.manager.get_or_create_session("one", "B")
        new = self.manager.get_or_create_session("one", "A")
        self.assertFalse(self.update(reset, "stale again"))
        self.assertEqual(self.manager.get_session("one"), new)
        self.assertEqual(self.manager.get_session("other"), other_state)
        self.assertIsNone(self.manager.get_session("missing"))
        self.manager.reset_session("missing")
        self.assertIsNone(self.manager.get_session("missing"))

    def test_history_limits_and_current_merge(self):
        for limit in (1, 2, 10):
            with self.subTest(limit=limit):
                self.manager = SessionManager(limit)
                snapshot = self.manager.get_or_create_session("one", "A")
                for i in range(15):
                    self.assertTrue(self.update(snapshot, f"q{i}", (i % 2,)))
                current = self.manager.get_session("one")
                self.assertEqual(
                    current.recent_questions, [f"q{i}" for i in range(15 - limit, 15)]
                )
                self.assertEqual(
                    current.recent_chunk_indices, [i % 2 for i in range(15 - limit, 15)]
                )
                self.assertEqual(current.active_chunk_index, 0)
                self.assertEqual(snapshot.recent_questions, [])
                self.assertEqual(current.generation, snapshot.generation)

    def test_empty_missing_and_record_index(self):
        snapshot = self.manager.get_or_create_session("one", "A")
        self.bundle.id_to_record[0].metadata["chunk_index"] = 17
        self.update(snapshot, "first")
        self.update(snapshot, "empty", ())
        self.update(snapshot, "missing", (999,))
        current = self.manager.get_session("one")
        self.assertEqual(current.active_chunk_index, 17)
        self.assertEqual(current.recent_chunk_indices, [17])
        self.assertEqual(current.recent_questions, ["first", "empty", "missing"])

    def test_reset_during_chunk_lookup_rejects_completion_without_long_lock(self):
        snapshot = self.manager.get_or_create_session("one", "A")
        entered, release = threading.Event(), threading.Event()
        records = self.bundle.id_to_record

        class BlockingRecords:
            def get(self, key):
                entered.set()
                if not release.wait(5):
                    raise RuntimeError("lookup_event_timeout")
                return records.get(key)

        self.bundle.id_to_record = BlockingRecords()
        with ThreadPoolExecutor(max_workers=2) as pool:
            future = pool.submit(self.update, snapshot, "late")
            try:
                self.assertTrue(entered.wait(5))
                pool.submit(self.manager.reset_session, "one").result(2)
                reset = self.manager.get_session("one")
            finally:
                release.set()
            self.assertFalse(future.result(5))
        self.assertEqual(self.manager.get_session("one"), reset)
        self.assertEqual(reset.recent_questions, [])

    def test_simultaneous_updates_do_not_lose_questions(self):
        snapshot = self.manager.get_or_create_session("one", "A")
        barrier = threading.Barrier(8)

        def submit(i):
            barrier.wait(timeout=5)
            return self.update(snapshot, str(i))

        with ThreadPoolExecutor(max_workers=8) as pool:
            self.assertEqual(list(pool.map(submit, range(8))), [True] * 8)
        current = self.manager.get_session("one")
        self.assertEqual(set(current.recent_questions), set(map(str, range(8))))
        self.assertEqual(len(current.recent_questions), 8)
        self.assertEqual(current.recent_chunk_indices, [0] * 8)


if __name__ == "__main__":
    unittest.main()
