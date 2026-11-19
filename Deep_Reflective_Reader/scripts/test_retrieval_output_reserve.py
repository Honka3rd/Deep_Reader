"""RETRIEVAL-03: ordinary dispatch allowance, real cold/warm QA budgeting."""

import contextlib
import io
import os
import socket
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from llama_index.core.node_parser import SentenceSplitter
from config.faiss_storage_config import FaissStorageConfig
from config.token_budget_resolver import resolve_effective_token_budgets
from fingerprint_handler import FingerprintHandler
from language.language_code import LanguageCode
from llm.llm_model_capabilities import LLMModelCapabilities
from llm.llm_provider import LLMProvider
from llm.offline_token_counter import OfflineTokenCounter
from llm.openai_llm_provider import OpenAILLMProvider
from profile.document_profile import DocumentProfile
from retrieval.faiss_index_builder import FaissIndexBuilder
from retrieval.faiss_index_store import FaissIndexStore
from retrieval.node_provider import NodeProvider
from test_context_capacity import make_case


class OutputReserveTests(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.temp = self.stack.enter_context(tempfile.TemporaryDirectory())
        self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        for name in ("socket.socket.connect", "socket.create_connection"):
            self.stack.enter_context(
                patch(name, side_effect=AssertionError("network forbidden"))
            )

    def provider(self, model="gpt-4o-mini", target=500):
        responses = model.startswith("gpt-4.1")
        cls = "OpenAIResponses" if responses else "OpenAI"
        adapter = self.stack.enter_context(
            patch("llm.openai_llm_provider." + cls)
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
        return (
            OpenAILLMProvider(
                SimpleNamespace(get=lambda: "synthetic-key"), model, target
            ),
            adapter,
        )

    def build(self, provider, target=500, input_target=128000, text=None):
        text = text if text is not None else ("一" * 299 + "\n") * 423 + "一" * 100
        detector = Mock()
        detector.detect.return_value = "zh"
        parsed = NodeProvider(
            SentenceSplitter(chunk_size=300, chunk_overlap=0), detector
        ).parse(text, None)
        embedder = Mock()
        embedder.probe_vector_dimension.return_value = 2
        embedder.get_text_embedding_batch.side_effect = lambda texts: [
            [1.0, 0.0] for _ in texts
        ]
        embedder.get_text_embedding.return_value = [1.0, 0.0]
        budgets = dict(
            target_max_input_tokens=input_target,
            target_max_output_tokens=target,
            target_max_context_tokens=input_target,
            input_budget_utilization_ratio=1,
            context_budget_utilization_ratio=1,
        )
        builder = FaissIndexBuilder(embedder, provider, batch_size=100, **budgets)
        cold = builder.build_from_parsed_document(parsed)
        store = FaissIndexStore(embedder, provider, **budgets)
        config = FaissStorageConfig("book", base_root=self.temp)
        fingerprint = FingerprintHandler("controlled", 300, 0)
        store.publish(
            cold, config, text=text, fingerprint=fingerprint, verify_source=lambda: None
        )
        files = {
            p.name: p.read_bytes() for p in config.base_dir.iterdir() if p.is_file()
        }
        count = embedder.get_text_embedding_batch.call_count
        warm = store.load_validated(config, text, fingerprint)
        self.assertEqual(embedder.get_text_embedding_batch.call_count, count)
        self.assertEqual(
            files,
            {p.name: p.read_bytes() for p in config.base_dir.iterdir() if p.is_file()},
        )
        self.assertEqual(builder.token_budgets, store.token_budgets)
        self.assertEqual(builder.token_budgets.target_max_output_tokens, target)
        return cold, warm, store, config, fingerprint, text

    def ask(self, provider, adapter, bundle, *, expect_capacity_rejection=False):
        bundle.set_profile(
            DocumentProfile("Synthetic", "Synthetic source summary", LanguageCode.ZH)
        )
        case = make_case(texts=["placeholder"])
        case.coordinator.document_preparation_pipeline = SimpleNamespace(
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
        case.coordinator.llm_provider = provider
        adapter.complete.reset_mock()
        if expect_capacity_rejection:
            from context.token_budget_manager import ContextCapacityError

            before = case.coordinator.session_manager.get_or_create_session(
                "pending", "synthetic"
            )
            with patch.object(
                provider, "count_input_tokens", wraps=provider.count_input_tokens
            ) as count:
                with self.assertRaisesRegex(
                    ContextCapacityError, "^context_capacity_exceeded$"
                ):
                    case.coordinator.ask(case.query, "synthetic", session_id="pending")
                count.assert_called_once()
                prompt = count.call_args.args[0]
            self.assertGreater(
                provider.count_input_tokens(prompt) + bundle.reserved_output_tokens,
                bundle.max_prompt_tokens,
            )
            adapter.complete.assert_not_called()
            self.assertEqual(
                case.coordinator.session_manager.get_session("pending"), before
            )
            return None
        case.coordinator.ask(case.query, "synthetic")
        adapter.complete.assert_called_once()
        options = adapter.complete.call_args.kwargs
        allowance = options.get(
            "max_output_tokens", options.get("max_completion_tokens")
        )
        self.assertEqual(bundle.reserved_output_tokens, allowance)
        self.assertEqual(allowance, provider.get_default_output_budget())
        prompt = adapter.complete.call_args.args[0]
        self.assertLessEqual(
            case.budget.estimate_tokens(prompt) + allowance, bundle.max_prompt_tokens
        )
        return prompt, allowance, case

    def test_near_limit_cold_warm_both_endpoints(self):
        for model, expected in [("gpt-4o-mini", 1638), ("gpt-4.1-mini", 3276)]:
            with self.subTest(model=model):
                provider, adapter = self.provider(model)
                cold, warm, *_ = self.build(provider)
                for bundle in (cold, warm):
                    prompt, allowance, case = self.ask(provider, adapter, bundle)
                    self.assertEqual(allowance, expected)
                    self.assertEqual(case.context.result.mode.value, "retrieval_mode")
                    self.assertLess(len(case.context.result.context_text), 127000)

    def test_explicit_small_input_cap_both_endpoints(self):
        # An explicit 3200 cap (not the app default) is below the Responses floor:
        # deliberate capacity failure,
        # not a smaller provider allowance or a dispatched over-budget request.
        from context.token_budget_manager import ContextCapacityError

        for model in ("gpt-4o-mini", "gpt-4.1-mini"):
            with self.subTest(model=model):
                provider, adapter = self.provider(model)
                cold, warm, *_ = self.build(provider, input_target=3200)
                for bundle in (cold, warm):
                    if model == "gpt-4o-mini":
                        self.ask(provider, adapter, bundle)
                    else:
                        with self.assertRaisesRegex(
                            ContextCapacityError, "context_capacity_exceeded"
                        ):
                            self.ask(provider, adapter, bundle)
                        adapter.complete.assert_not_called()

    def test_targets_at_and_above_floor_are_preserved(self):
        for model, floor in [("gpt-4o-mini", 1638), ("gpt-4.1-mini", 3276)]:
            for target in (floor, floor + 100):
                with self.subTest(model=model, target=target):
                    provider, adapter = self.provider(model, target)
                    cold, warm, *_ = self.build(
                        provider,
                        target,
                        input_target=10000,
                        text="A complete source paragraph.",
                    )
                    for bundle in (cold, warm):
                        _, allowance, _ = self.ask(provider, adapter, bundle)
                        self.assertEqual(allowance, target)

    def test_reload_uses_current_provider_not_persisted_budget(self):
        provider, _ = self.provider()
        _, _, store, config, fingerprint, text = self.build(
            provider, text="A complete source paragraph."
        )
        original = {
            p.name: p.read_bytes() for p in config.base_dir.iterdir() if p.is_file()
        }
        newer, adapter = self.provider("gpt-4.1-mini", 4000)
        store = FaissIndexStore(
            store.embedder,
            newer,
            target_max_input_tokens=10000,
            target_max_output_tokens=500,
            target_max_context_tokens=10000,
            input_budget_utilization_ratio=1,
            context_budget_utilization_ratio=1,
        )
        with patch.object(
            store.embedder,
            "get_text_embedding_batch",
            side_effect=AssertionError("warm embedding"),
        ):
            warm = store.load_validated(config, text, fingerprint)
            self.ask(newer, adapter, warm)
        self.assertEqual(warm.reserved_output_tokens, 4000)
        self.assertEqual(
            original,
            {p.name: p.read_bytes() for p in config.base_dir.iterdir() if p.is_file()},
        )

    def test_per_call_override_does_not_change_default_reserve(self):
        for model in ("gpt-4o-mini", "gpt-4.1-mini"):
            provider, adapter = self.provider(model)
            before = provider.get_default_output_budget()
            provider.complete_text_with_output_budget("controlled", 100)
            self.assertEqual(provider.get_default_output_budget(), before)
            provider.complete_text("controlled")
            options = adapter.complete.call_args.kwargs
            self.assertEqual(
                options.get("max_output_tokens", options.get("max_completion_tokens")),
                before,
            )
            self.assertEqual(adapter.complete.call_count, 2)

    def test_declared_reserve_is_deducted_once_and_preserves_targets(self):
        for context, independent, expected in [
            (10000, None, 9000),
            (None, 8000, 8000),
            (10000, 8000, 8000),
        ]:
            cap = LLMModelCapabilities(
                "fixture", "responses", context, 2000, independent
            )
            budget = resolve_effective_token_budgets(
                cap, 50000, 500, 40000, 1, 1, declared_output_tokens=1000
            )
            self.assertEqual(
                budget.effective_input_budget - budget.effective_output_budget, expected
            )
            self.assertEqual(budget.target_max_output_tokens, 500)
            limited = resolve_effective_token_budgets(
                cap, 3000, 500, 1500, 1, 1, declared_output_tokens=1000
            )
            self.assertEqual(
                (limited.effective_input_budget, limited.effective_context_budget),
                (3000, 1500),
            )

    def test_invalid_declarations_reject_without_clamping(self):
        cap = LLMModelCapabilities("fixture", "responses", 10000, 2000)
        for value in (True, False, 0, -1, 1.5, "1000", 2001):
            with self.subTest(value=value), self.assertRaisesRegex(
                ValueError, "^invalid_provider_output_budget$"
            ):
                resolve_effective_token_budgets(
                    cap, 5000, 500, 2000, declared_output_tokens=value
                )

    def test_legacy_fallback_and_no_capability_declaration(self):
        class Legacy(LLMProvider):
            def complete_text(self, prompt):
                return "controlled"

            def get_model_capabilities(self):
                return None

        self.assertIsNone(Legacy().get_default_output_budget())
        for provider in (
            Legacy(),
            SimpleNamespace(get_model_capabilities=lambda: None),
        ):
            cold, warm, *_ = self.build(provider, text="A complete source paragraph.")
            self.assertEqual(
                (cold.reserved_output_tokens, warm.reserved_output_tokens), (500, 500)
            )
        budgets = resolve_effective_token_budgets(
            None, 5000, 500, 2000, declared_output_tokens=1000
        )
        self.assertEqual(budgets.effective_output_budget, 1000)
        self.assertTrue(budgets.fallback_used)

    def test_final_prompt_counter_boundaries(self):
        from context.token_budget_manager import (
            ContextCapacityError,
            TokenBudgetManager,
        )

        for count in (False, 0, -1, 1.5, "12", 901):
            with self.subTest(count=count), self.assertRaisesRegex(
                ContextCapacityError, "^context_capacity_exceeded$"
            ):
                TokenBudgetManager.require_prompt_capacity(
                    "synthetic",
                    max_prompt_tokens=1000,
                    reserved_output_tokens=100,
                    token_counter=lambda _: count,
                )
        for count in (None, 900):
            TokenBudgetManager.require_prompt_capacity(
                "synthetic",
                max_prompt_tokens=1000,
                reserved_output_tokens=100,
                token_counter=lambda _: count,
            )
        with self.assertRaisesRegex(
            ContextCapacityError, "^context_capacity_exceeded$"
        ):
            TokenBudgetManager.require_prompt_capacity(
                "一" * 400,
                max_prompt_tokens=1000,
                reserved_output_tokens=100,
                token_counter=lambda _: None,
            )

    @unittest.skipUnless(
        os.environ.get("DEEP_READER_TEST_TOKENIZER_ASSETS"),
        "opt-in verified local tokenizer",
    )
    def test_exact_offline_count_near_each_model_context_limit(self):
        counter = OfflineTokenCounter(os.environ["DEEP_READER_TEST_TOKENIZER_ASSETS"])
        self.stack.enter_context(
            patch("llm.openai_llm_provider.TOKEN_COUNTER", counter)
        )
        for model in ("gpt-4o-mini", "gpt-4.1-mini"):
            with self.subTest(model=model):
                provider, adapter = self.provider(model)
                limit = provider.get_model_capabilities().context_window_tokens
                text = ("一" * 299 + "\n") * ((limit - 1000) // 300)
                cold, warm, *_ = self.build(provider, input_target=limit, text=text)
                self.assertLessEqual(
                    max(
                        counter.count(model, r.text())
                        for r in cold.id_to_record.values()
                    ),
                    300,
                )
                for bundle in (cold, warm):
                    if model == "gpt-4.1-mini":
                        # Newline-heavy exact prompt exceeds the heuristic at this
                        # size. Refusal must precede dispatch and session update.
                        self.ask(
                            provider, adapter, bundle, expect_capacity_rejection=True
                        )
                        continue
                    prompt, allowance, _ = self.ask(provider, adapter, bundle)
                    count = counter.count(model, prompt)
                    self.assertIsInstance(count, int)
                    self.assertLessEqual(count + allowance, limit)


if __name__ == "__main__":
    unittest.main()
