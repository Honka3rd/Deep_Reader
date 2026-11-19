from collections.abc import Mapping
from enum import StrEnum
from openai import OpenAI as OpenAIClient
from llama_index.llms.openai import OpenAI
from llama_index.llms.openai.responses import OpenAIResponses

from auth.api_key_provider import APIKeyProvider
from llm.llm_model_capabilities import (
    ENDPOINT_KIND_CHAT_COMPLETIONS,
    ENDPOINT_KIND_RESPONSES,
    LLMModelCapabilities,
)
from llm.llm_provider import LLMProvider
from llm.offline_token_counter import TOKEN_COUNTER


class OpenAIModelName(StrEnum):
    """Canonical OpenAI model names used by this provider."""
    GPT_4_1 = "gpt-4.1"
    GPT_4_1_MINI = "gpt-4.1-mini"
    GPT_4_1_NANO = "gpt-4.1-nano"
    GPT_4O = "gpt-4o"
    GPT_4O_MINI = "gpt-4o-mini"


class OpenAILLMProvider(LLMProvider):
    """LLMProvider implementation backed by OpenAI via llama-index."""
    llm: OpenAI | OpenAIResponses
    model_name: OpenAIModelName
    model_capabilities: LLMModelCapabilities
    effective_output_budget: int
    output_budget_floor_ratio: float

    # Historical capability facts are not a live-dispatch allowlist. Keep these
    # identities available for stored provenance and offline tokenizer consumers.
    RETIRED_MODELS = frozenset({OpenAIModelName.GPT_4_1_NANO})
    MODEL_CAPABILITIES_BY_NAME: dict[OpenAIModelName, LLMModelCapabilities] = {
        OpenAIModelName.GPT_4_1: LLMModelCapabilities(
            model_name=OpenAIModelName.GPT_4_1.value,
            endpoint_kind=ENDPOINT_KIND_RESPONSES,
            context_window_tokens=1_047_576,
            max_output_tokens=32_768,
        ),
        OpenAIModelName.GPT_4_1_MINI: LLMModelCapabilities(
            model_name=OpenAIModelName.GPT_4_1_MINI.value,
            endpoint_kind=ENDPOINT_KIND_RESPONSES,
            context_window_tokens=1_047_576,
            max_output_tokens=32_768,
        ),
        OpenAIModelName.GPT_4_1_NANO: LLMModelCapabilities(
            model_name=OpenAIModelName.GPT_4_1_NANO.value,
            endpoint_kind=ENDPOINT_KIND_RESPONSES,
            context_window_tokens=1_047_576,
            max_output_tokens=32_768,
        ),
        OpenAIModelName.GPT_4O: LLMModelCapabilities(
            model_name=OpenAIModelName.GPT_4O.value,
            endpoint_kind=ENDPOINT_KIND_CHAT_COMPLETIONS,
            context_window_tokens=128_000,
            max_output_tokens=16_384,
        ),
        OpenAIModelName.GPT_4O_MINI: LLMModelCapabilities(
            model_name=OpenAIModelName.GPT_4O_MINI.value,
            endpoint_kind=ENDPOINT_KIND_CHAT_COMPLETIONS,
            context_window_tokens=128_000,
            max_output_tokens=16_384,
        ),
    }
    OUTPUT_BUDGET_FLOOR_RATIO = 0.10

    @classmethod
    def _resolve_model_name(cls, model: OpenAIModelName | str) -> OpenAIModelName:
        normalized = model.strip().lower()
        if not normalized:
            raise ValueError("Model name cannot be empty.")
        # Compatibility path: accept underscore-typed names (e.g. gpt_4.1_mini).
        alias = normalized.replace("_", "-")
        if alias in OpenAIModelName._value2member_map_:
            resolved = OpenAIModelName(alias)
            if resolved in cls.RETIRED_MODELS:
                raise ValueError(
                    "llm_model_retired: gpt-4.1-nano; select an available model explicitly"
                )
            return resolved

        supported = ", ".join(name.value for name in OpenAIModelName if name not in cls.RETIRED_MODELS)
        raise ValueError(
            f"Unsupported llm model '{model}'. Supported models: {supported}"
        )

    @classmethod
    def _resolve_model_capabilities(
        cls,
        model_name: OpenAIModelName,
    ) -> LLMModelCapabilities:
        capabilities = cls.MODEL_CAPABILITIES_BY_NAME.get(model_name)
        if capabilities is None:
            supported = ", ".join(model.value for model in OpenAIModelName)
            raise ValueError(
                f"Unsupported llm model '{model_name.value}'. "
                f"Supported models: {supported}"
            )
        return capabilities

    def __init__(
        self,
        api_key_provider: APIKeyProvider,
        model: OpenAIModelName | str = OpenAIModelName.GPT_4_1_MINI,
        target_max_output_tokens: int = 500,
    ):
        """Initialize object state and injected dependencies.

Args:
    api_key_provider: Api key provider.
    model: Model.
    target_max_output_tokens: App-level output-token target before model clamp.
"""
        self.model_name = self._resolve_model_name(model)
        self.model_capabilities = self._resolve_model_capabilities(self.model_name)
        api_key = api_key_provider.get()
        if target_max_output_tokens <= 0:
            raise ValueError("target_max_output_tokens must be > 0")
        self.output_budget_floor_ratio = self.OUTPUT_BUDGET_FLOOR_RATIO
        capability_floor_budget = max(
            1,
            int(self.model_capabilities.max_output_tokens * self.output_budget_floor_ratio),
        )
        self.effective_output_budget = min(
            self.model_capabilities.max_output_tokens,
            max(target_max_output_tokens, capability_floor_budget),
        )

        if self.model_capabilities.endpoint_kind == ENDPOINT_KIND_RESPONSES:
            self.llm = OpenAIResponses(
                model=self.model_name.value,
                api_key=api_key,
            )
        else:
            self.llm = OpenAI(
                model=self.model_name.value,
                api_key=api_key,
            )

        print(
            "OpenAILLMProvider#init:",
            f"model={self.model_capabilities.model_name}",
            f"endpoint_kind={self.model_capabilities.endpoint_kind}",
            f"context_window_tokens={self.model_capabilities.context_window_tokens}",
            f"max_input_tokens={self.model_capabilities.max_input_tokens}",
            f"max_output_tokens={self.model_capabilities.max_output_tokens}",
            f"target_max_output_tokens={target_max_output_tokens}",
            f"one_tenth_floor={capability_floor_budget}",
            f"effective_output_budget={self.effective_output_budget}",
        )

    def get_default_output_budget(self) -> int:
        """Expose the same allowance dispatched by ordinary completion."""
        return self.effective_output_budget

    def complete_text(self, prompt: str) -> str:
        """Call underlying completion backend and return generated text.

Args:
    prompt: Prompt text sent to completion model.

Returns:
    Trimmed text completion generated by the OpenAI model."""
        return self.complete_text_with_output_budget(prompt, self.get_default_output_budget())

    def complete_text_with_output_budget(self, prompt: str, output_tokens: int, *, response_schema: dict | None = None) -> str:
        """Per-call allowance; never mutate shared interaction/provider defaults."""
        if type(output_tokens) is not int or not 0 < output_tokens <= self.model_capabilities.max_output_tokens:
            raise ValueError("invalid_output_budget")
        request_options = {"timeout": 300.0} if output_tokens > self.effective_output_budget else {}
        if response_schema is not None:
            schema_format = {"name": "semantic_reading_plan", "strict": True, "schema": response_schema}
            if self.model_capabilities.endpoint_kind == ENDPOINT_KIND_RESPONSES:
                request_options["text"] = {"format": {"type": "json_schema", **schema_format}}
            else:
                request_options["response_format"] = {"type": "json_schema", "json_schema": schema_format}
        print("OpenAI LLM:prompt", len(prompt))
        if self.model_capabilities.endpoint_kind == ENDPOINT_KIND_RESPONSES:
            response = self.llm.complete(
                prompt,
                max_output_tokens=output_tokens,
                **request_options,
            )
        else:
            response = self.llm.complete(
                prompt,
                max_completion_tokens=output_tokens,
                **request_options,
            )
        return self._completed_text(response.raw, response.text)

    @staticmethod
    def _field(value, name, default=None):
        """Read SDK objects or llama-index raw dictionaries without coercion."""
        if isinstance(value, Mapping):
            return value.get(name, default)
        return getattr(value, name, default)

    def _completed_text(self, response, text) -> str:
        """Admit completed plain text using endpoint evidence, never text guesses.

        Run after the ordinary adapter returns and inside the once client lifetime.
        Failures contain fixed identifiers only; they do not authorize another call.
        """
        field = self._field
        if self.model_capabilities.endpoint_kind == ENDPOINT_KIND_RESPONSES:
            status = field(response, "status")
            if status != "completed":
                code = "completion_incomplete" if status in (
                    "incomplete", "failed", "cancelled", "queued", "in_progress"
                ) else "completion_status_unavailable"
                raise ValueError(code)
            if field(response, "error") is not None or field(response, "incomplete_details") is not None:
                raise ValueError("completion_incomplete")
            messages = [item for item in (field(response, "output") or [])
                        if field(item, "type") == "message"]
            if not messages:
                raise ValueError("completion_text_unavailable")
            has_text = False
            for message in messages:
                if field(message, "status") != "completed":
                    raise ValueError("completion_incomplete")
                if any(field(item, "type") == "refusal" for item in (field(message, "content") or [])):
                    raise ValueError("completion_refused")
                has_text |= any(
                    field(item, "type") == "output_text"
                    and isinstance(field(item, "text"), str) and field(item, "text").strip()
                    for item in (field(message, "content") or [])
                )
            if not has_text:
                raise ValueError("completion_text_unavailable")
        else:
            choices = field(response, "choices")
            if not choices:
                raise ValueError("completion_status_unavailable")
            choice = choices[0]
            reason = field(choice, "finish_reason")
            if reason != "stop":
                code = "completion_incomplete" if reason in (
                    "length", "content_filter", "tool_calls", "function_call"
                ) else "completion_status_unavailable"
                raise ValueError(code)
            message = field(choice, "message")
            if field(message, "refusal") is not None:
                raise ValueError("completion_refused")
            if field(message, "tool_calls") or field(message, "function_call"):
                raise ValueError("completion_text_unavailable")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("completion_text_unavailable")
        return text.strip()

    def get_model_capabilities(self) -> LLMModelCapabilities:
        """Return current model static capability metadata."""
        return self.model_capabilities

    def complete_text_once_with_output_budget(self, prompt: str, output_tokens: int) -> str:
        """One paid attempt after durable dispatch; no SDK or adapter retries.

        Reuse this provider's resolved endpoint/model/credentials. A request-local
        client avoids changing the ordinary interaction client's retry policy.
        """
        caps = self.model_capabilities
        if type(output_tokens) is not int or not 0 < output_tokens <= caps.max_output_tokens:
            raise ValueError("invalid_output_budget")
        timeout = 300.0 if output_tokens > self.effective_output_budget else self.llm.timeout
        with OpenAIClient(api_key=self.llm.api_key, base_url=self.llm.api_base,
                          default_headers=self.llm.default_headers, timeout=timeout,
                          max_retries=0) as client:
            if caps.endpoint_kind == ENDPOINT_KIND_RESPONSES:
                response = client.responses.create(model=caps.model_name, input=prompt,
                    max_output_tokens=output_tokens, temperature=self.llm.temperature)
                return self._completed_text(response, response.output_text)
            response = client.chat.completions.create(model=caps.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=output_tokens, temperature=self.llm.temperature)
            content = response.choices[0].message.content if response.choices else None
            return self._completed_text(response, content)

    def count_input_tokens(self, text: str) -> int | None:
        return TOKEN_COUNTER.count(self.model_capabilities.model_name, text)

    @property
    def tokenizer_identity(self) -> str | None:
        return TOKEN_COUNTER.identity(self.model_capabilities.model_name)
