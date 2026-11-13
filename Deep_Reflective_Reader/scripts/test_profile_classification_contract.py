"""Complete-request capacity and classification validation, without provider IO."""
from __future__ import annotations

import json
import unittest
from dataclasses import replace

import tiktoken

from llm.llm_model_capabilities import LLMModelCapabilities, ENDPOINT_KIND_RESPONSES
from profile.document_profile_builder import DocumentProfileBuilder
from profile.document_profile import TextForm
from test_document_profile_parser_metadata import (
    _FakeLLMProvider, _build_prompt_text_normalization, _build_profile_prompt_policy,
)


PAYLOAD = dict(topic="fiction", summary="Accepted summary", document_language="en",
               text_form="novel", confidence=0.7)
ENCODING = tiktoken.get_encoding("cl100k_base")


class Provider(_FakeLLMProvider):
    def __init__(self, *, context=8000, reserve=2000, counter="tokens", outputs=None):
        super().__init__(outputs or [json.dumps(PAYLOAD)])
        self.context = context
        self.effective_output_budget = reserve
        self.counter = counter
        self.counts = 0

    def get_model_capabilities(self):
        if self.context is None:
            raise RuntimeError("capability unavailable")
        return LLMModelCapabilities("fixture", ENDPOINT_KIND_RESPONSES,
                                    self.context, 4000)

    def count_input_tokens(self, text):
        self.counts += 1
        if self.counter == "raise":
            raise RuntimeError("tokenizer unavailable")
        if self.counter == "none":
            return None
        if self.counter == "invalid":
            return True
        return len(ENCODING.encode(text, disallowed_special=()))


def builder(provider, **policy_overrides):
    return DocumentProfileBuilder(provider, _build_prompt_text_normalization(),
                                  replace(_build_profile_prompt_policy(), **policy_overrides))


class ClassificationContractTests(unittest.TestCase):
    def test_complete_prompt_capacity_matrix(self):
        texts = ["0123456789" * 3000, "中文閱讀知識與文化。" * 4000,
                 "A reader explores stories and ideas. " * 2000]
        for context, reserve in [(4000, 1000), (8000, 2000), (64000, 3000)]:
            for text in texts:
                with self.subTest(context=context, reserve=reserve, sample=text[:8]):
                    provider = Provider(context=context, reserve=reserve)
                    result = builder(provider).build(text, "en")
                    self.assertEqual(result.topic, "fiction")
                    self.assertEqual(len(provider.prompts), 1)
                    self.assertGreater(provider.counts, 0)
                    prompt = provider.prompts[0]
                    self.assertLessEqual(len(ENCODING.encode(prompt)), context - reserve - 256)
                    evidence = json.loads(prompt.split("Evidence package JSON:\n", 1)[1])
                    self.assertEqual(evidence["raw_text_length"], len(text))
                    self.assertTrue(text.startswith(evidence["head_excerpt"]))
                    self.assertTrue(text.endswith(evidence["tail_excerpt"]))
                    self.assertIn(evidence["middle_excerpt"], text)

    def test_dense_candidates_and_escaping_remain_valid_json(self):
        text = '\n'.join(f'Chapter {n}: "\\\\語言" <|endoftext|>' for n in range(1000))
        provider = Provider(context=4000, reserve=1000)
        result = builder(provider, evidence_heading_lines_limit=1000).build(text, "en")
        self.assertEqual(result.topic, "fiction")
        prompt = provider.prompts[0]
        evidence = json.loads(prompt.split("Evidence package JSON:\n", 1)[1])
        self.assertLessEqual(len(ENCODING.encode(prompt, disallowed_special=())), 2744)
        self.assertLess(len(evidence["candidate_lines"]), 1000)
        self.assertTrue(all(line in text.splitlines() for line in evidence["candidate_lines"]))

    def test_unavailable_counter_uses_conservative_bytes(self):
        for counter in ("none", "raise", "invalid"):
            with self.subTest(counter=counter):
                provider = Provider(counter=counter)
                builder(provider).build("中文0123456789" * 10000, "zh")
                self.assertLessEqual(len(provider.prompts[0].encode()), 5700)

    def test_mandatory_prompt_cannot_fit_no_dispatch_or_fallback(self):
        provider = Provider(context=2100, reserve=2000)
        with self.assertRaisesRegex(ValueError, "profile_prompt_budget"):
            builder(provider).build("A short document", "en")
        self.assertEqual(provider.prompts, [])

    def test_unknown_capacity_no_dispatch_or_fallback(self):
        provider = Provider(context=None)
        with self.assertRaisesRegex(ValueError, "profile_prompt_budget"):
            builder(provider).build("A short document", "en")
        self.assertEqual(provider.prompts, [])

    def test_actual_output_reserve_and_input_only_ceiling(self):
        provider = Provider(context=8000, reserve=500)
        provider.get_model_capabilities = lambda: LLMModelCapabilities(
            "fixture", ENDPOINT_KIND_RESPONSES, None, 4000, max_input_tokens=6000)
        builder(provider).build("0123456789" * 10000, "en")
        self.assertLessEqual(len(ENCODING.encode(provider.prompts[0])), 5700)
        # A maximum-output reservation here would reject a prompt that fits the
        # actual provider allowance. Fixed instructions need more than 500 tokens.
        provider = Provider(context=4500, reserve=500)
        builder(provider).build("A narrative", "en")
        self.assertEqual(len(provider.prompts), 1)

    def test_invalid_output_reserve_is_not_dispatched(self):
        for reserve in (-1, 4001, True, "500"):
            with self.subTest(reserve=reserve):
                provider = Provider(reserve=reserve)
                with self.assertRaisesRegex(ValueError, "profile_prompt_budget"):
                    builder(provider).build("Short narrative.", "en")
                self.assertEqual(provider.prompts, [])

    def test_fixed_metadata_overflow_does_not_truncate_or_dispatch(self):
        provider = Provider()
        subject = builder(provider)
        metadata = subject.metadata_extractor.extract(text="A narrative", document_language="en")
        metadata = replace(metadata, notes=["large deterministic note " * 10000])
        with self.assertRaisesRegex(ValueError, "profile_prompt_budget"):
            subject._build_profile_classification_prompt(
                evidence=subject._build_profile_evidence(text="A narrative", document_language="en"),
                deterministic_metadata=metadata,
            )
        self.assertEqual(provider.prompts, [])

    def test_fallback_requests_fit_too(self):
        provider = Provider(outputs=["malformed", "essay", "Fallback summary"])
        result = builder(provider).build("中文0123456789" * 10000, "zh")
        self.assertEqual(result.topic, "essay")
        self.assertEqual(result.summary, "Fallback summary")
        self.assertEqual(len(provider.prompts), 3)
        for prompt in provider.prompts:
            self.assertLessEqual(len(ENCODING.encode(prompt)), 5700)

    def test_valid_confidence_keeps_one_response(self):
        for value in ("absent", None, 0, 1, 0.45):
            with self.subTest(value=value):
                payload = dict(PAYLOAD)
                if value == "absent":
                    payload.pop("confidence")
                else:
                    payload["confidence"] = value
                provider = Provider(outputs=[json.dumps(payload), "fallback", "fallback summary"])
                result = builder(provider).build("Chapter One\nA narrative.", "en")
                self.assertEqual(len(provider.prompts), 1)
                self.assertEqual(result.topic, "fiction")
                self.assertEqual(result.summary, "Accepted summary")
                self.assertEqual(result.parser_metadata.text_form, TextForm.NOVEL)
                self.assertEqual(result.parser_metadata.confidence,
                                 None if value == "absent" else value)

    def test_invalid_confidence_falls_back_without_losing_deterministic_metadata(self):
        for value in (-0.01, 1.01, float("nan"), float("inf"), float("-inf"),
                      "0.5", True, False, [], {}):
            with self.subTest(value=value):
                provider = Provider(outputs=[json.dumps(dict(PAYLOAD, confidence=value)),
                                             "essay", "Fallback summary"])
                subject = builder(provider)
                text = "A clear English narrative."
                result = subject.build(text, "en")
                self.assertEqual(len(provider.prompts), 3)
                self.assertEqual(result.topic, "essay")
                self.assertEqual(result.parser_metadata,
                                 subject.metadata_extractor.extract(text=text, document_language="en"))


if __name__ == "__main__":
    unittest.main()
