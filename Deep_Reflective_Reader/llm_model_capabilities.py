from dataclasses import dataclass


ENDPOINT_KIND_RESPONSES = "responses"
ENDPOINT_KIND_CHAT_COMPLETIONS = "chat_completions"
_SUPPORTED_ENDPOINT_KINDS = {
    ENDPOINT_KIND_RESPONSES,
    ENDPOINT_KIND_CHAT_COMPLETIONS,
}


@dataclass(frozen=True)
class LLMModelCapabilities:
    """Static capability metadata for one concrete LLM model."""
    model_name: str
    endpoint_kind: str
    max_input_tokens: int
    max_output_tokens: int

    def __post_init__(self) -> None:
        if self.endpoint_kind not in _SUPPORTED_ENDPOINT_KINDS:
            raise ValueError(
                "endpoint_kind must be one of "
                f"{sorted(_SUPPORTED_ENDPOINT_KINDS)}; got: {self.endpoint_kind}"
            )
        if self.max_input_tokens <= 0:
            raise ValueError("max_input_tokens must be > 0")
        if self.max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be > 0")
