"""Retired dispatch configuration is distinct from historical model metadata."""

import contextlib
import io
import json
import socket
import unittest
from unittest.mock import Mock, patch

from config.app_DI_config import AppDIConfig
from config.container import ApplicationLookupContainer
from llm.offline_token_counter import SUPPORTED_MODELS, encoding_asset_for_model
from llm.openai_llm_provider import OpenAILLMProvider, OpenAIModelName

ACTIVE = ("gpt-4.1", "gpt-4.1-mini", "gpt-4o", "gpt-4o-mini")
RETIRED = (
    "gpt-4.1-nano",
    "gpt_4.1_nano",
    OpenAIModelName.GPT_4_1_NANO,
    " GPT_4.1_NANO ",
    "Gpt-4.1-Nano",
)


class ModelAdmissionTests(unittest.TestCase):
    def setUp(self):
        guard = patch.object(
            socket.socket, "connect", side_effect=AssertionError("network forbidden")
        )
        guard.start()
        self.addCleanup(guard.stop)
        self.key = Mock()
        self.key.get.return_value = "SYNTHETIC_NOT_A_KEY"

    def test_retired_forms_reject_before_key_and_adapter(self):
        for selection in RETIRED:
            with self.subTest(selection=selection), patch(
                "llm.openai_llm_provider.OpenAIResponses"
            ) as responses, patch("llm.openai_llm_provider.OpenAI") as chat:
                with self.assertRaisesRegex(
                    ValueError,
                    "^llm_model_retired: gpt-4.1-nano; select an available model explicitly$",
                ) as caught:
                    OpenAILLMProvider(self.key, model=selection)
                self.assertNotIn(self.key.get.return_value, str(caught.exception))
                responses.assert_not_called()
                chat.assert_not_called()
                self.key.get.assert_not_called()

    def test_actual_config_container_denies_retired_without_dispatch(self):
        for selection in RETIRED:
            with self.subTest(selection=selection), patch(
                "llm.openai_llm_provider.OpenAIResponses"
            ) as responses, patch("llm.openai_llm_provider.OpenAI") as chat:
                config = AppDIConfig(llm_model=selection)
                container = ApplicationLookupContainer()
                container.config.from_dict(config.to_container_config_dict())
                with container.api_key_provider.override(self.key):
                    with self.assertRaisesRegex(ValueError, "llm_model_retired"):
                        container.llm_provider()
                responses.assert_not_called()
                chat.assert_not_called()
                self.key.get.assert_not_called()

    def test_active_canonical_enum_and_aliases_keep_selection(self):
        for name in ACTIVE:
            for selection in (
                name,
                OpenAIModelName(name),
                name.replace("-", "_"),
                " " + name.upper() + " ",
            ):
                with self.subTest(selection=selection), patch(
                    "llm.openai_llm_provider.OpenAIResponses"
                ) as responses, patch(
                    "llm.openai_llm_provider.OpenAI"
                ) as chat, contextlib.redirect_stdout(
                    io.StringIO()
                ):
                    provider = OpenAILLMProvider(self.key, model=selection)
                    selected = responses if name.startswith("gpt-4.1") else chat
                    other = chat if selected is responses else responses
                    selected.assert_called_once_with(
                        model=name, api_key=self.key.get.return_value
                    )
                    other.assert_not_called()
                    self.assertEqual(provider.model_name.value, name)
                    caps = provider.get_model_capabilities()
                    self.assertEqual(
                        (caps.context_window_tokens, caps.max_output_tokens),
                        (
                            (1047576, 32768)
                            if name.startswith("gpt-4.1")
                            else (128000, 16384)
                        ),
                    )
                    self.assertIsNone(caps.max_input_tokens)
                    self.assertEqual(
                        provider.effective_output_budget,
                        int(caps.max_output_tokens * 0.1),
                    )
                    selected.return_value.complete.assert_not_called()

    def test_container_preserves_valid_default_and_explicit_alias(self):
        for config, name in (
            (AppDIConfig(), "gpt-4.1-mini"),
            (AppDIConfig(llm_model="gpt_4o_mini"), "gpt-4o-mini"),
        ):
            with self.subTest(name=name), patch(
                "llm.openai_llm_provider.OpenAIResponses"
            ), patch("llm.openai_llm_provider.OpenAI"), contextlib.redirect_stdout(
                io.StringIO()
            ):
                container = ApplicationLookupContainer()
                container.config.from_dict(config.to_container_config_dict())
                with container.api_key_provider.override(self.key):
                    self.assertEqual(container.llm_provider().model_name.value, name)

    def test_unknown_and_blank_reject_without_adapter(self):
        for name in ("unknown-model", "", " ", "gpt-4.1-nano-2025-04-14"):
            with self.subTest(name=name), patch(
                "llm.openai_llm_provider.OpenAIResponses"
            ) as responses, patch("llm.openai_llm_provider.OpenAI") as chat:
                with self.assertRaises(ValueError):
                    OpenAILLMProvider(self.key, model=name)
                responses.assert_not_called()
                chat.assert_not_called()
                self.key.get.assert_not_called()

    def test_unavailable_is_not_advertised_as_supported_dispatch(self):
        with self.assertRaises(ValueError) as caught:
            OpenAILLMProvider(self.key, model="not-configured")
        supported = str(caught.exception).split("Supported models: ")[1]
        self.assertEqual(set(supported.split(", ")), set(ACTIVE))

    def test_historical_identifiers_capacities_and_offline_manifest_remain_readable(
        self,
    ):
        model = OpenAIModelName("gpt-4.1-nano")
        self.assertEqual(
            json.loads(json.dumps({"model": model})), {"model": "gpt-4.1-nano"}
        )
        self.assertEqual(
            AppDIConfig(llm_model=model).to_container_config_dict()["llm_model"], model
        )
        caps = OpenAILLMProvider.MODEL_CAPABILITIES_BY_NAME[model]
        self.assertEqual(
            (caps.context_window_tokens, caps.max_output_tokens), (1047576, 32768)
        )
        self.assertIn(model.value, SUPPORTED_MODELS)
        self.assertIsNotNone(encoding_asset_for_model(model.value))
        self.key.get.assert_not_called()


if __name__ == "__main__":
    unittest.main()
