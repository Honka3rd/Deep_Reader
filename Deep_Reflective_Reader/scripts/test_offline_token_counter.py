"""Offline tokenizer, budget identity and optional immutable-book regressions."""

import base64
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import tiktoken
from llm import offline_token_counter as offline
from llm.openai_llm_provider import OpenAILLMProvider
from document_structure.enhancement_capacity import require_capacity_identity, validate_dispatch_identity
from document_structure.plan_prompt_budget import enforce_prompt_budget, EnhancementPromptBudgetError
from document_structure.semantic_plan_provider import SemanticPlanProvider


def no_network(stack):
    sentinels = [stack.enter_context(patch(name, side_effect=AssertionError("network forbidden")))
                 for name in ("socket.socket.connect", "socket.create_connection", "requests.get",
                              "tiktoken.load.read_file", "tiktoken.encoding_for_model", "tiktoken.get_encoding")]
    return sentinels


def provider():
    value = object.__new__(OpenAILLMProvider)
    value.model_capabilities = value.MODEL_CAPABILITIES_BY_NAME["gpt-4.1-mini"]
    value.effective_output_budget = 3276
    value.llm = Mock()
    return value


class OfflineCounterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        # Small real byte vocabulary for unit tests, NOT model-matched acceptance.
        self.data = b"".join(base64.b64encode(bytes([i])) + f" {i}\n".encode() for i in range(256))
        self.asset = replace(offline.encoding_asset_for_model("gpt-4.1-mini"),
                             size=len(self.data), sha256=hashlib.sha256(self.data).hexdigest())
        self.path = self.root / self.asset.filename
        self.path.write_bytes(self.data)
        self.counter = offline.OfflineTokenCounter(self.root)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.network = no_network(self.stack)
        self.stack.enter_context(patch.object(offline, "_O200K", self.asset))

    def tearDown(self):
        for call in self.network:
            call.assert_not_called()

    def test_supported_models_match_provider_inventory(self):
        self.assertEqual(set(offline.SUPPORTED_MODELS), set(OpenAILLMProvider.MODEL_CAPABILITIES_BY_NAME))
        for model in offline.SUPPORTED_MODELS:
            self.assertEqual(tiktoken.encoding_name_for_model(model), self.asset.name)
            self.assertEqual(self.counter.count(model, "text"), 4)

    def test_cold_warm_concurrent_share_one_verified_encoding(self):
        text = "literal <|endoftext|> \u4e2d\u6587 \U0001f600\ne\u0301"
        with patch.object(tiktoken, "Encoding", wraps=tiktoken.Encoding) as construct:
            with ThreadPoolExecutor(max_workers=8) as pool:
                counts = list(pool.map(lambda _: self.counter.count("gpt-4.1-mini", text), range(24)))
            self.assertEqual(counts, [len(text.encode())] * 24)
            self.assertEqual(self.counter.count("gpt-4o", text), counts[0])
            self.assertEqual(construct.call_count, 1)
        self.assertEqual(self.counter.count("gpt-4.1", ""), 0)

    def test_missing_corrupt_and_directory_are_not_repaired(self):
        for data in (b"bad", b"x" * len(self.data), self.data + b"extra"):
            self.path.write_bytes(data)
            self.assertIsNone(self.counter.count("gpt-4.1-mini", "PRIVATE_TEXT"))
            self.assertEqual(self.path.read_bytes(), data)
        self.path.unlink()
        self.assertIsNone(self.counter.count("gpt-4.1-mini", "text"))
        self.assertFalse(self.path.exists())
        self.path.mkdir()
        self.assertIsNone(self.counter.count("gpt-4.1-mini", "text"))
        self.path.rmdir()
        self.path.write_bytes(self.data)
        self.assertEqual(self.counter.count("gpt-4.1-mini", "text"), 4)

    def test_unsupported_and_wrong_mapping_fail_closed(self):
        with patch.object(Path, "open", side_effect=AssertionError("must not read")):
            self.assertIsNone(self.counter.count("PRIVATE_UNKNOWN_MODEL", "text"))
            self.assertIsNone(self.counter.identity("PRIVATE_UNKNOWN_MODEL"))
            with patch.object(tiktoken, "encoding_name_for_model", return_value="cl100k_base"):
                self.assertIsNone(self.counter.count("gpt-4.1-mini", "text"))

    def test_missing_dependency_init_and_count_failure_do_not_escape(self):
        with patch.object(Path, "open", side_effect=PermissionError("PRIVATE_PATH")):
            self.assertIsNone(self.counter.count("gpt-4.1-mini", "text"))
        with patch.dict(sys.modules, {"tiktoken": None}):
            self.assertIsNone(self.counter.count("gpt-4.1-mini", "text"))
            self.assertIsNone(self.counter.identity("gpt-4.1-mini"))
        with patch.object(tiktoken, "Encoding", side_effect=RuntimeError("PRIVATE_ERROR")):
            self.assertIsNone(self.counter.count("gpt-4.1-mini", "text"))
        self.assertEqual(self.counter.count("gpt-4.1-mini", "text"), 4)
        encoding = self.counter._encodings[self.asset.name]
        with patch.object(encoding, "encode", side_effect=ValueError("PRIVATE_ERROR")):
            self.assertIsNone(self.counter.count("gpt-4.1-mini", "text"))

    def test_log_stages_contain_no_text_paths_or_external_error(self):
        with self.assertLogs(offline.LOGGER, "INFO") as logs:
            self.counter.count("gpt-4.1-mini", "PRIVATE_SOURCE")
            self.counter.count("gpt-4.1-mini", "PRIVATE_SOURCE")
            self.counter.count("PRIVATE_MODEL", "PRIVATE_SOURCE")
        output = "\n".join(logs.output)
        for expected in ("stage=asset_verify", "stage=initialize", "stage=count", "cache=cold", "cache=memory", "elapsed_ms="):
            self.assertIn(expected, output)
        self.assertNotIn("PRIVATE", output)
        self.assertNotIn(str(self.root), output)

    def test_full_prompt_and_schema_use_same_counter_and_identity(self):
        p = provider()
        with patch("llm.openai_llm_provider.TOKEN_COUNTER", self.counter):
            request = SemanticPlanProvider(p)
            decision = enforce_prompt_budget("source <|endoftext|>", request)
            self.assertEqual(decision["input_estimate"], len(("source <|endoftext|>" + request.request_schema_text).encode()))
            self.assertEqual(decision["tokenizer_identity"], p.tokenizer_identity)
            self.assertEqual(decision["output_reserve"], 32768)
            self.assertEqual(p.effective_output_budget, 3276)
            self.assertEqual(decision["context_window_tokens"], 1047576)
            p.llm.complete.assert_not_called()

    def test_absent_assets_use_labelled_upper_bound_and_reject_overflow(self):
        self.path.unlink()
        p = provider()
        with patch("llm.openai_llm_provider.TOKEN_COUNTER", self.counter):
            decision = enforce_prompt_budget("\u4e2d" * 100, p)
            self.assertEqual(decision["input_estimate"], 300)
            self.assertEqual(decision["measurement"], "utf8_byte_upper_estimate_not_provider_usage")
            self.assertNotIn("tokenizer_identity", decision)
            with self.assertRaises(EnhancementPromptBudgetError):
                enforce_prompt_budget("\u4e2d" * 400000, p)
            p.llm.complete.assert_not_called()

    def test_changed_identity_or_fallback_rejects_stale_consent(self):
        from document_structure.enhancement_capacity import capacity_identity
        from document_structure.enhancement_resources import EnhancementResourceLimits
        from document_structure.ocr_cleanup_planner import cleanup_input_capacity, plan_cleanup_input
        from scripts.test_cleanup_shared_engine import snapshot
        p, value = provider(), snapshot()
        self.path.unlink()
        with patch("llm.openai_llm_provider.TOKEN_COUNTER", self.counter):
            prompt = "source"
            before = capacity_identity(prompt, enforce_prompt_budget(prompt, p), EnhancementResourceLimits())
            cleanup_before = cleanup_input_capacity(value, p)
            self.path.write_bytes(self.data)
            with require_capacity_identity(before), self.assertRaisesRegex(EnhancementPromptBudgetError, "configuration_changed"):
                validate_dispatch_identity(capacity_identity(prompt, enforce_prompt_budget(prompt, p), EnhancementResourceLimits()))
            result = plan_cleanup_input(snapshot=value, provider=p, operation_id="test", expected_capacity_fingerprint=cleanup_before)
            self.assertIsNotNone(result.outcome)
            self.assertEqual(result.outcome.reason, "cleanup_policy_changed")
            self.assertIsNone(result.proposal)
            # Identical counts still invalidate consent when the tokenizer recipe changes.
            before = cleanup_input_capacity(value, p)
            with patch.object(offline, "RECIPE_VERSION", "next_recipe"):
                self.assertNotEqual(before, cleanup_input_capacity(value, p))
                result = plan_cleanup_input(snapshot=value, provider=p, operation_id="test", expected_capacity_fingerprint=before)
                self.assertIsNone(result.proposal)
                self.assertEqual(result.outcome.reason, "cleanup_policy_changed")
            p.llm.complete.assert_not_called()


@unittest.skipUnless(os.environ.get("DEEP_READER_TEST_TOKENIZER_ASSETS"), "opt-in verified real tokenizer asset")
class RealEncodingTests(unittest.TestCase):
    def test_fresh_process_uses_verified_asset_not_ephemeral_cache(self):
        code = """
import socket, tempfile, os
attempts = []
def deny(*a, **k):
    attempts.append(True)
    raise AssertionError('network forbidden')
socket.socket.connect = deny
socket.create_connection = deny
with tempfile.TemporaryDirectory() as cache:
    os.environ['TIKTOKEN_CACHE_DIR'] = cache
    from llm.offline_token_counter import TOKEN_COUNTER
    cold = TOKEN_COUNTER.count('gpt-4.1-mini', 'Hello world <|endoftext|>')
    assert cold and cold == TOKEN_COUNTER.count('gpt-4.1-mini', 'Hello world <|endoftext|>')
    assert not os.listdir(cache)
assert not attempts
"""
        subprocess.run([sys.executable, "-c", code], check=True, timeout=30, env={**os.environ,
            "DEEP_READER_TOKENIZER_ASSET_DIR": os.environ["DEEP_READER_TEST_TOKENIZER_ASSETS"]})

    def test_upstream_recipe_and_multilingual_parity_offline(self):
        from tiktoken_ext import openai_public
        root = Path(os.environ["DEEP_READER_TEST_TOKENIZER_ASSETS"])
        asset = offline.encoding_asset_for_model("gpt-4.1-mini")
        data = (root / asset.filename).read_bytes()
        self.assertEqual(hashlib.sha256(data).hexdigest(), asset.sha256)
        ranks = {base64.b64decode(t): int(r) for t, r in (line.split() for line in data.splitlines())}
        with ExitStack() as stack:
            blocked = no_network(stack)
            with patch.object(openai_public, "load_tiktoken_bpe", return_value=ranks) as load:
                expected = tiktoken.Encoding(**openai_public.o200k_base())
                load.assert_called_once_with(asset.url, expected_hash=asset.sha256)
            actual = offline.OfflineTokenCounter(root)
            for text in ("", "Hello world", "\u4e2d\u6587\u6e2c\u8a66", "\u65e5\u672c\u8a9e", "\u0645\u0631\u062d\u0628\u0627", "\U0001f600 e\u0301\n<|endoftext|>\r\n", "'s '\t123456"):
                for model in offline.SUPPORTED_MODELS:
                    self.assertEqual(actual.count(model, text), len(expected.encode(text, disallowed_special=())))
            for call in blocked:
                call.assert_not_called()

    @unittest.skipUnless(os.environ.get("DEEP_READER_TEST_TOKENIZER_BOOKS"), "opt-in immutable book snapshots")
    def test_real_book_full_text_counts_preserve_backups(self):
        root = Path(os.environ["DEEP_READER_TEST_TOKENIZER_BOOKS"])
        for book in ("fa794acef055f235", "085747b65094d52f"):
            folder = root / book
            manifest = json.loads((folder / "manifest.json").read_text())
            before = {name: hashlib.sha256((folder / name).read_bytes()).hexdigest() for name in manifest["files"]}
            self.assertEqual(before, manifest["files"])
            snapshot = json.loads((folder / "snapshot.json").read_text())
            text = snapshot["canonical_text"]
            self.assertEqual(hashlib.sha256(text.encode()).hexdigest(), snapshot["text_sha256"])
            counter = offline.OfflineTokenCounter(os.environ["DEEP_READER_TEST_TOKENIZER_ASSETS"])
            with ExitStack() as stack:
                blocked = no_network(stack)
                with patch("llm.openai_llm_provider.TOKEN_COUNTER", counter):
                    p = provider()
                    request = SemanticPlanProvider(p)
                    cold = enforce_prompt_budget(text, request)
                    self.assertEqual(cold, enforce_prompt_budget(text, request))
                    self.assertIn("local_tokenizer", cold["measurement"])
                    self.assertGreater(cold["input_estimate"], 0)
                    self.assertLess(cold["input_estimate"], len(text.encode()))
                    p.llm.complete.assert_not_called()
                for call in blocked:
                    call.assert_not_called()
            self.assertEqual(before, {name: hashlib.sha256((folder / name).read_bytes()).hexdigest() for name in before})


class ColdProcessTests(unittest.TestCase):
    def test_fresh_import_and_missing_asset_never_access_network(self):
        code = """
import socket
attempts = []
def deny(*a, **k):
    attempts.append(True)
    raise AssertionError('network forbidden')
socket.socket.connect = deny
socket.create_connection = deny
from llm.offline_token_counter import OfflineTokenCounter
from llm.openai_llm_provider import OpenAILLMProvider
import tempfile
with tempfile.TemporaryDirectory() as root:
    assert OfflineTokenCounter(root).count('gpt-4.1-mini', 'source') is None
assert not attempts
"""
        subprocess.run([sys.executable, "-c", code], check=True, timeout=30)


if __name__ == "__main__":
    unittest.main()
