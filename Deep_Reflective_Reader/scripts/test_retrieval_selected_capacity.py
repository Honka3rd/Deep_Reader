"""RETRIEVAL-05: selected context budget survives real cold/warm QA dispatch."""

import contextlib
import io
import os
import socket
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient
from llama_index.core.node_parser import SentenceSplitter

from config.app_DI_config import AppDIConfig
from config.faiss_storage_config import FaissStorageConfig
from fingerprint_handler import FingerprintHandler
from language.language_code import LanguageCode
from llm.offline_token_counter import OfflineTokenCounter
from llm.openai_llm_provider import OpenAILLMProvider
from profile.document_profile import DocumentProfile
from retrieval.faiss_index_builder import FaissIndexBuilder
from retrieval.faiss_index_store import FaissIndexStore
from retrieval.node_provider import NodeProvider
from session.session_manager import SessionUpdateResult
from test_context_capacity import make_case
from test_question_output_boundary import route_app

GLOBAL = "Summarize the whole document."
LOCAL = "Explain this sentence."


class SelectedCapacityTests(unittest.TestCase):
    def setUp(self):
        self.stack = self.enterContext(contextlib.ExitStack())
        self.temp = self.enterContext(tempfile.TemporaryDirectory())
        self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        self.stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
        for target in (
            "socket.socket.connect",
            "socket.socket.connect_ex",
            "socket.create_connection",
        ):
            self.stack.enter_context(
                patch(target, side_effect=AssertionError("network forbidden"))
            )
        self.counter = OfflineTokenCounter(
            os.environ.get("DEEP_READER_TEST_TOKENIZER_ASSETS")
        )
        self.stack.enter_context(
            patch("llm.openai_llm_provider.TOKEN_COUNTER", self.counter)
        )

    def exercise(
        self,
        *,
        paragraphs=20,
        query=GLOBAL,
        input_target=3200,
        model="gpt-4o-mini",
        active=False,
        expected_status=200,
        count_override=None,
        result_transform=None,
        text=None,
        language="zh",
    ):
        cfg = AppDIConfig(llm_model=model, target_max_input_tokens=input_target)
        text = text if text is not None else ("一" * 299 + "\n") * paragraphs
        detector = Mock()
        detector.detect.return_value = language
        embedder = Mock()
        embedder.probe_vector_dimension.return_value = 2
        embedder.get_text_embedding_batch.side_effect = lambda texts: [
            [1.0, 0.0] for _ in texts
        ]
        embedder.get_text_embedding.return_value = [1.0, 0.0]
        responses = model.startswith("gpt-4.1")
        adapter = self.stack.enter_context(
            patch(
                "llm.openai_llm_provider."
                + ("OpenAIResponses" if responses else "OpenAI")
            )
        ).return_value
        adapter.complete.return_value.text = "Controlled answer"
        adapter.complete.return_value.raw = (
            {
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "status": "completed",
                        "content": [
                            {"type": "output_text", "text": "Controlled answer"}
                        ],
                    }
                ],
            }
            if responses
            else {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": "Controlled answer"},
                    }
                ]
            }
        )
        provider = OpenAILLMProvider(
            SimpleNamespace(get=lambda: "synthetic-no-key"),
            model=model,
            target_max_output_tokens=cfg.target_max_output_tokens,
        )
        budgets = dict(
            target_max_input_tokens=cfg.target_max_input_tokens,
            target_max_output_tokens=cfg.target_max_output_tokens,
            target_max_context_tokens=cfg.target_max_context_tokens,
            input_budget_utilization_ratio=cfg.input_budget_utilization_ratio,
            context_budget_utilization_ratio=cfg.context_budget_utilization_ratio,
        )
        builder = FaissIndexBuilder(embedder, provider, batch_size=100, **budgets)
        parsed = NodeProvider(
            SentenceSplitter(
                chunk_size=cfg.chunk_size, chunk_overlap=cfg.chunk_overlap
            ),
            detector,
        ).parse(text, None)
        cold = builder.build_from_parsed_document(parsed)
        store = FaissIndexStore(embedder, provider, **budgets)
        storage = FaissStorageConfig("synthetic", base_root=Path(self.temp))
        fingerprint = FingerprintHandler(
            cfg.embedding_model, cfg.chunk_size, cfg.chunk_overlap
        )
        store.publish(
            cold,
            storage,
            text=text,
            fingerprint=fingerprint,
            verify_source=lambda: None,
        )

        def stored_bytes():
            return {
                p.name: p.read_bytes()
                for p in storage.base_dir.iterdir()
                if p.is_file()
            }

        before = stored_bytes()
        build_calls = embedder.get_text_embedding_batch.call_count
        warm = store.load_validated(storage, text, fingerprint)
        self.assertEqual(embedder.get_text_embedding_batch.call_count, build_calls)
        observations = []
        for phase, bundle in (("cold", cold), ("warm", warm)):
            with self.subTest(phase=phase, model=model, input_target=input_target):
                bundle.set_profile(
                    DocumentProfile(
                        "Synthetic", "Synthetic source summary", LanguageCode(language)
                    )
                )
                case = make_case(query=query, texts=["placeholder"])
                coordinator = case.coordinator
                coordinator.document_preparation_pipeline = SimpleNamespace(
                    prepare_and_load=lambda name: SimpleNamespace(
                        bundle=bundle,
                        assets=SimpleNamespace(
                            structured_document_path=None,
                            bundle_ready=True,
                            faiss_ready=True,
                            profile_ready=True,
                        ),
                    )
                )
                coordinator.llm_provider = provider
                if result_transform is not None:
                    original = coordinator.context_orchestrator.build
                    coordinator.context_orchestrator.build = (
                        lambda **kw: result_transform(original(**kw))
                    )
                sid = "selected-" + phase
                snapshot = coordinator.session_manager.get_or_create_session(
                    sid, "synthetic"
                )
                if active:
                    coordinator.session_manager.update_session(
                        snapshot,
                        SessionUpdateResult(
                            question="seed",
                            bundle=bundle,
                            results=bundle.search("seed", 1),
                        ),
                    )
                before_session = coordinator.session_manager.get_session(sid)
                adapter.complete.reset_mock()
                embeddings_before = embedder.get_text_embedding.call_count
                count = (
                    Mock(wraps=provider.count_input_tokens)
                    if count_override is None
                    else Mock(side_effect=lambda prompt: count_override(bundle))
                )
                with patch.object(provider, "count_input_tokens", count), TestClient(
                    route_app(coordinator)
                ) as client:
                    response = client.post(
                        "/documents/ask",
                        json={
                            "doc_name": "synthetic",
                            "query": query,
                            "session_id": sid,
                        },
                    )
                self.assertEqual(response.status_code, expected_status, response.json())
                self.assertEqual(count.call_count, 1)
                self.assertEqual(len(case.standardizer.calls), 1)
                self.assertEqual(len(case.scope_provider.calls), 0)
                self.assertEqual(
                    embedder.get_text_embedding.call_count - embeddings_before, 1
                )
                self.assertEqual(
                    embedder.get_text_embedding_batch.call_count, build_calls
                )
                self.assertEqual(before, stored_bytes())
                selected = case.context.result
                self.assertEqual(selected.standardized_question.original_query, query)
                self.assertEqual(
                    selected.standardized_question.standardized_query, query
                )
                prompt = count.call_args.args[0]
                measured = self.counter.count(model, prompt)
                if measured is None:
                    measured = len(prompt.encode("utf-8"))
                if expected_status == 200:
                    self.assertEqual(adapter.complete.call_count, 1)
                    self.assertEqual(response.json()["answer"], "Controlled answer")
                    self.assertEqual(adapter.complete.call_args.args[0], prompt)
                    dispatched = adapter.complete.call_args.kwargs.get(
                        "max_output_tokens",
                        adapter.complete.call_args.kwargs.get("max_completion_tokens"),
                    )
                    self.assertEqual(dispatched, bundle.reserved_output_tokens)
                    self.assertEqual(
                        coordinator.session_manager.get_session(sid).recent_questions,
                        before_session.recent_questions + [query],
                    )
                else:
                    adapter.complete.assert_not_called()
                    self.assertEqual(
                        response.json()["detail"], "context_capacity_exceeded"
                    )
                    self.assertEqual(
                        coordinator.session_manager.get_session(sid), before_session
                    )
                observations.append(
                    dict(
                        phase=phase,
                        mode=selected.mode.value,
                        selected=selected,
                        prompt=prompt,
                        measured=measured,
                        reserve=bundle.reserved_output_tokens,
                        cap=bundle.max_prompt_tokens,
                    )
                )
        return observations

    def test_legal_global_full_text_uses_selected_cap_cold_and_warm(self):
        for model in ("gpt-4o-mini", "gpt-4.1-mini"):
            for row in self.exercise(model=model):
                self.assertEqual(row["mode"], "full_text_mode")
                selected = row["selected"]
                self.assertEqual(
                    selected.metadata["full_text_budget_policy"],
                    "model_capability_ratio",
                )
                self.assertEqual(
                    selected.max_prompt_tokens,
                    selected.metadata["effective_input_budget"],
                )
                self.assertGreater(row["measured"] + row["reserve"], row["cap"])
                self.assertLessEqual(
                    row["measured"] + selected.metadata["effective_output_budget"],
                    selected.max_prompt_tokens,
                )

    def test_global_full_text_is_independent_of_retrieval_cap(self):
        small = self.exercise(input_target=3200)
        large = self.exercise(input_target=10000)
        for left, right in zip(small, large):
            self.assertEqual(left["prompt"], right["prompt"])
            self.assertEqual(
                left["selected"].max_prompt_tokens, right["selected"].max_prompt_tokens
            )
            self.assertEqual(
                left["selected"].metadata["effective_output_budget"],
                right["selected"].metadata["effective_output_budget"],
            )

    def test_small_global_and_local_full_text_preserve_distinct_caps(self):
        for query in (GLOBAL, LOCAL):
            for row in self.exercise(paragraphs=1, query=query):
                self.assertEqual(row["mode"], "full_text_mode")
                cap = 64000 if query == GLOBAL else 3200
                self.assertEqual(row["selected"].max_prompt_tokens, cap)

    def test_active_local_window_retains_configured_cap(self):
        for row in self.exercise(query=LOCAL, active=True, input_target=10000):
            self.assertEqual(row["mode"], "local_window_mode")
            self.assertEqual(row["selected"].max_prompt_tokens, 10000)
        self.exercise(
            query=LOCAL,
            active=True,
            input_target=10000,
            count_override=lambda bundle: bundle.max_prompt_tokens,
            expected_status=500,
        )

    def test_global_not_selected_as_full_text_retains_retrieval_cap(self):
        for row in self.exercise(paragraphs=200, input_target=10000):
            self.assertEqual(row["mode"], "retrieval_mode")
            self.assertEqual(row["selected"].metadata["scope"], "global")
            self.assertEqual(row["selected"].max_prompt_tokens, 10000)
        self.exercise(
            paragraphs=200,
            input_target=10000,
            count_override=lambda bundle: bundle.max_prompt_tokens,
            expected_status=500,
        )

    def test_precise_global_overflow_stops_answer_and_session(self):
        # Assembly's established heuristic ignores whitespace. Interleaved newlines
        # are cheap to tokenize but create a real exact-count underestimate.
        for row in self.exercise(text="一\n" * 30000, expected_status=500):
            self.assertEqual(row["mode"], "full_text_mode")
            self.assertGreater(
                row["measured"] + row["reserve"], row["selected"].max_prompt_tokens
            )

    def test_actual_output_is_reserved_once_at_selected_boundary(self):
        self.exercise(
            count_override=lambda bundle: 64000 - bundle.reserved_output_tokens
        )
        self.exercise(
            count_override=lambda bundle: 64001 - bundle.reserved_output_tokens,
            expected_status=500,
        )

    def test_unavailable_counter_keeps_conservative_selected_capacity(self):
        for row in self.exercise(count_override=lambda bundle: None):
            self.assertGreater(
                len(row["prompt"].encode("utf-8")) + row["reserve"], row["cap"]
            )
            self.assertLessEqual(
                len(row["prompt"].encode("utf-8")) + row["reserve"],
                row["selected"].max_prompt_tokens,
            )

    def test_legacy_none_cap_does_not_promote_diagnostic_metadata(self):
        def legacy(result):
            return replace(
                result,
                max_prompt_tokens=None,
                metadata={**result.metadata, "effective_input_budget": 999999},
            )

        self.exercise(result_transform=legacy, expected_status=500)
        self.exercise(paragraphs=1, result_transform=legacy)


if __name__ == "__main__":
    unittest.main()
