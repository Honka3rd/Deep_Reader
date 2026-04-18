from enum import StrEnum
from llama_index.llms.openai import OpenAI
from llama_index.llms.openai.responses import OpenAIResponses

from auth.api_key_provider import APIKeyProvider
from llm.llm_model_capabilities import (
    ENDPOINT_KIND_CHAT_COMPLETIONS,
    ENDPOINT_KIND_RESPONSES,
    LLMModelCapabilities,
)
from llm.llm_provider import LLMProvider


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

    MODEL_CAPABILITIES_BY_NAME: dict[OpenAIModelName, LLMModelCapabilities] = {
        OpenAIModelName.GPT_4_1: LLMModelCapabilities(
            model_name=OpenAIModelName.GPT_4_1.value,
            endpoint_kind=ENDPOINT_KIND_RESPONSES,
            max_input_tokens=1_047_576,
            max_output_tokens=32_768,
        ),
        OpenAIModelName.GPT_4_1_MINI: LLMModelCapabilities(
            model_name=OpenAIModelName.GPT_4_1_MINI.value,
            endpoint_kind=ENDPOINT_KIND_RESPONSES,
            max_input_tokens=1_047_576,
            max_output_tokens=32_768,
        ),
        OpenAIModelName.GPT_4_1_NANO: LLMModelCapabilities(
            model_name=OpenAIModelName.GPT_4_1_NANO.value,
            endpoint_kind=ENDPOINT_KIND_RESPONSES,
            max_input_tokens=1_047_576,
            max_output_tokens=32_768,
        ),
        OpenAIModelName.GPT_4O: LLMModelCapabilities(
            model_name=OpenAIModelName.GPT_4O.value,
            endpoint_kind=ENDPOINT_KIND_CHAT_COMPLETIONS,
            max_input_tokens=128_000,
            max_output_tokens=16_384,
        ),
        OpenAIModelName.GPT_4O_MINI: LLMModelCapabilities(
            model_name=OpenAIModelName.GPT_4O_MINI.value,
            endpoint_kind=ENDPOINT_KIND_CHAT_COMPLETIONS,
            max_input_tokens=128_000,
            max_output_tokens=16_384,
        ),
    }

    @classmethod
    def _resolve_model_name(cls, model: OpenAIModelName | str) -> OpenAIModelName:
        if isinstance(model, OpenAIModelName):
            return model

        normalized = model.strip().lower()
        if not normalized:
            raise ValueError("Model name cannot be empty.")

        if normalized in OpenAIModelName._value2member_map_:
            return OpenAIModelName(normalized)

        # Compatibility path: accept underscore-typed names (e.g. gpt_4.1_mini).
        alias = normalized.replace("_", "-")
        if alias in OpenAIModelName._value2member_map_:
            return OpenAIModelName(alias)

        supported = ", ".join(model_name.value for model_name in OpenAIModelName)
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
        api_key = api_key_provider.get()
        self.model_name = self._resolve_model_name(model)
        self.model_capabilities = self._resolve_model_capabilities(self.model_name)
        if target_max_output_tokens <= 0:
            raise ValueError("target_max_output_tokens must be > 0")
        self.effective_output_budget = min(
            target_max_output_tokens,
            self.model_capabilities.max_output_tokens,
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
            f"max_input_tokens={self.model_capabilities.max_input_tokens}",
            f"max_output_tokens={self.model_capabilities.max_output_tokens}",
            f"effective_output_budget={self.effective_output_budget}",
        )

    def complete_text(self, prompt: str) -> str:
        """Call underlying completion backend and return generated text.

Args:
    prompt: Prompt text sent to completion model.

Returns:
    Trimmed text completion generated by the OpenAI model."""
        print("OpenAI LLM:prompt", len(prompt))
        if self.model_capabilities.endpoint_kind == ENDPOINT_KIND_RESPONSES:
            response = self.llm.complete(
                prompt,
                max_output_tokens=self.effective_output_budget,
            )
        else:
            response = self.llm.complete(
                prompt,
                max_completion_tokens=self.effective_output_budget,
            )
        return response.text.strip()

    def get_model_capabilities(self) -> LLMModelCapabilities:
        """Return current model static capability metadata."""
        return self.model_capabilities
