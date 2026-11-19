"""RETRIEVAL-04: default DI cold/warm QA versus explicit inadequate caps.

Real splitter, FAISS publication/readback, provider budgeting and production ask
route; only external provider ports and preparation's ready-bundle handoff are fixed.
All storage is temporary; no network or application data access.
"""

import contextlib
import io
import json
import os
import socket
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from app.qa_coordinator import QACoordinator
from config.app_DI_config import AppDIConfig
from config.container import ApplicationLookupContainer
from config.faiss_storage_config import FaissStorageConfig
from language.language_code import LanguageCode
from llm.offline_token_counter import OfflineTokenCounter
from profile.document_profile import DocumentProfile
from question.standardized.question_standardizer import QuestionStandardizer
from scripts.test_context_capacity import CapturedContext, FixedProvider
from scripts.test_question_output_boundary import route_app

TEXT = "Photosynthesis stores sunlight as chemical energy in plants."
QUERY = "Explain this sentence."


def deny_network(*args, **kwargs):
    raise AssertionError("test_network_forbidden")


class DefaultQATests(unittest.TestCase):
    def setUp(self):
        stack = self.enterContext(contextlib.ExitStack())
        for target in ("connect", "connect_ex"):
            stack.enter_context(patch.object(socket.socket, target, deny_network))
        stack.enter_context(patch("socket.create_connection", deny_network))
        stack.enter_context(patch("socket.getaddrinfo", deny_network))
        stack.enter_context(
            patch.dict(
                os.environ,
                {
                    "DEEP_READER_STRUCTURED_STORAGE_BACKEND": "file",
                },
            )
        )
        self.counter = OfflineTokenCounter(
            os.environ.get("DEEP_READER_TEST_TOKENIZER_ASSETS")
        )

    def exercise(self, cfg, *, expected_status=200, text=TEXT, language="en"):
        observations = []
        with tempfile.TemporaryDirectory() as temp, contextlib.ExitStack() as stack:
            previous = os.getcwd()
            os.chdir(temp)
            stack.callback(os.chdir, previous)
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
            responses = str(cfg.llm_model).startswith("gpt-4.1")
            adapter = stack.enter_context(
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
            stack.enter_context(
                patch("llm.openai_llm_provider.TOKEN_COUNTER", self.counter)
            )
            container = ApplicationLookupContainer.build(cfg)
            container.api_key_provider.override(
                SimpleNamespace(get=lambda: "SYNTHETIC_NOT_A_KEY")
            )
            embed = Mock()
            embed.probe_vector_dimension.return_value = 2
            embed.get_text_embedding.return_value = [1.0, 0.0]
            embed.get_text_embedding_batch.side_effect = lambda texts: [
                [1.0, 0.0] for _ in texts
            ]
            container.embedder.override(embed)
            detector = Mock()
            detector.detect.return_value = language
            container.document_language_detector.override(detector)
            standardizer = FixedProvider(
                json.dumps({"user_language": "en", "standardized_query": QUERY})
            )
            container.question_standardizer.override(QuestionStandardizer(standardizer))
            provider = container.llm_provider()
            builder, store = (
                container.faiss_index_builder(),
                container.faiss_index_store(),
            )
            parsed = container.node_provider().parse(text, None)
            cold = builder.build_from_parsed_document(parsed)
            config = FaissStorageConfig("synthetic", base_root=Path(temp) / "faiss")
            fingerprint = container.fingerprint_handler_factory()
            store.publish(
                cold,
                config,
                text=text,
                fingerprint=fingerprint,
                verify_source=lambda: None,
            )

            def stored_bytes():
                return {
                    p.name: p.read_bytes()
                    for p in config.base_dir.iterdir()
                    if p.is_file()
                }

            before = stored_bytes()
            batch_calls = embed.get_text_embedding_batch.call_count
            warm = store.load_validated(config, text, fingerprint)
            self.assertEqual(embed.get_text_embedding_batch.call_count, batch_calls)
            self.assertEqual(before, stored_bytes())
            for label, bundle in (("cold", cold), ("warm", warm)):
                with self.subTest(
                    path=label,
                    cap=cfg.target_max_input_tokens,
                    model=str(cfg.llm_model),
                ):
                    self.assertEqual(
                        bundle.max_prompt_tokens, cfg.target_max_input_tokens
                    )
                    self.assertEqual(
                        bundle.max_context_tokens, cfg.target_max_context_tokens
                    )
                    self.assertEqual(
                        bundle.reserved_output_tokens,
                        provider.get_default_output_budget(),
                    )
                    bundle.set_profile(
                        DocumentProfile(
                            "Synthetic", "Short source summary", LanguageCode(language)
                        )
                    )
                    coordinator = object.__new__(QACoordinator)
                    coordinator.app_config = cfg
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
                    captured = CapturedContext(container.context_orchestrator())
                    coordinator.context_orchestrator = captured
                    coordinator.prompt_assembler = container.prompt_assembler()
                    coordinator.llm_provider = provider
                    coordinator.session_manager = container.session_manager()
                    sid = "default-" + label
                    old_session = coordinator.session_manager.get_or_create_session(
                        sid, "synthetic"
                    )
                    adapter.complete.reset_mock()
                    standardizer.calls.clear()
                    query_embeddings = embed.get_text_embedding.call_count
                    with TestClient(route_app(coordinator)) as client:
                        response = client.post(
                            "/documents/ask",
                            json={
                                "doc_name": "synthetic",
                                "query": QUERY,
                                "session_id": sid,
                            },
                        )
                    self.assertEqual(response.status_code, expected_status)
                    self.assertEqual(len(standardizer.calls), 1)
                    self.assertEqual(before, stored_bytes())
                    self.assertEqual(
                        embed.get_text_embedding_batch.call_count, batch_calls
                    )
                    session = coordinator.session_manager.get_session(sid)
                    if expected_status != 200:
                        self.assertEqual(
                            response.json()["detail"], "context_capacity_exceeded"
                        )
                        adapter.complete.assert_not_called()
                        self.assertEqual(session, old_session)
                        continue
                    self.assertEqual(response.json()["answer"], "Controlled answer")
                    self.assertEqual(adapter.complete.call_count, 1)
                    self.assertEqual(
                        embed.get_text_embedding.call_count - query_embeddings, 1
                    )
                    self.assertEqual(session.recent_questions, [QUERY])
                    self.assertEqual(session.active_chunk_index, 0)
                    self.assertEqual(session.recent_chunk_indices, [0])
                    self.assertEqual(session.generation, old_session.generation)
                    prompt = adapter.complete.call_args.args[0]
                    allowance = adapter.complete.call_args.kwargs.get(
                        "max_output_tokens",
                        adapter.complete.call_args.kwargs.get("max_completion_tokens"),
                    )
                    self.assertEqual(allowance, bundle.reserved_output_tokens)
                    self.assertIn(text.strip(), captured.result.context_text)
                    self.assertIn(text.strip(), prompt)
                    tokens = self.counter.count(str(cfg.llm_model), prompt)
                    if tokens is None:
                        tokens = len(prompt.encode("utf-8"))
                    self.assertLessEqual(tokens + allowance, bundle.max_prompt_tokens)
                    observations.append(
                        dict(
                            phase=label,
                            cap=bundle.max_prompt_tokens,
                            output=allowance,
                            prompt_tokens=tokens,
                            answer_calls=adapter.complete.call_count,
                        )
                    )
        return observations

    def test_default_di_answers_short_source_cold_and_warm(self):
        self.exercise(AppDIConfig())

    def test_explicit_inadequate_cap_is_not_silently_increased(self):
        self.exercise(AppDIConfig(target_max_input_tokens=3200), expected_status=500)

    def test_explicit_viable_cap_remains_authoritative(self):
        self.exercise(AppDIConfig(target_max_input_tokens=6000))

    def test_chat_model_still_works_with_explicit_small_cap(self):
        self.exercise(
            AppDIConfig(llm_model="gpt-4o-mini", target_max_input_tokens=3200)
        )

    def test_grouped_and_serialized_budgets_preserve_defaults_and_overrides(self):
        default = AppDIConfig()
        self.assertEqual(default.target_max_input_tokens, 10000)
        self.assertEqual(str(default.llm_model), "gpt-4.1-mini")
        self.assertEqual(default.target_max_context_tokens, 1500)
        self.assertEqual(default.target_max_output_tokens, 500)
        for cfg in (default, replace(default, target_max_input_tokens=3200)):
            self.assertEqual(
                cfg.retrieval_token_budget().target_max_input_tokens,
                cfg.target_max_input_tokens,
            )
            self.assertEqual(
                cfg.to_container_config_dict()["target_max_input_tokens"],
                cfg.target_max_input_tokens,
            )


if __name__ == "__main__":
    unittest.main()
