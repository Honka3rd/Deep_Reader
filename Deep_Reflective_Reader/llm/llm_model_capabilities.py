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
    context_window_tokens: int | None
    max_output_tokens: int
    max_input_tokens: int | None = None

    def __post_init__(self) -> None:
        if self.endpoint_kind not in _SUPPORTED_ENDPOINT_KINDS:
            raise ValueError(
                "endpoint_kind must be one of "
                f"{sorted(_SUPPORTED_ENDPOINT_KINDS)}; got: {self.endpoint_kind}"
            )
        validate_capacity_limits(self)

    def input_token_limit(self, output_reserve: int) -> int:
        """Input-only ceiling; output is deducted from shared context exactly once."""
        return input_token_limit(self, output_reserve)

    def combined_token_limit(self, output_reserve: int) -> int:
        """Adapter for legacy QA budgets that subtract output downstream."""
        return self.input_token_limit(output_reserve) + output_reserve


def validate_capacity_limits(capabilities) -> None:
    context = getattr(capabilities, "context_window_tokens", None)
    input_limit = getattr(capabilities, "max_input_tokens", None)
    output = getattr(capabilities, "max_output_tokens", None)
    if context is None and input_limit is None:
        raise ValueError("input capacity unavailable")
    for name, value in (("context_window_tokens", context), ("max_input_tokens", input_limit),
                        ("max_output_tokens", output)):
        if (value is not None or name == "max_output_tokens") and (
            type(value) is not int or value <= 0
        ):
            raise ValueError(f"{name} must be a positive integer")


def input_token_limit(capabilities, output_reserve: int) -> int:
    validate_capacity_limits(capabilities)
    if type(output_reserve) is not int or not 0 <= output_reserve <= capabilities.max_output_tokens:
        raise ValueError("invalid output reserve")
    limits = []
    context = getattr(capabilities, "context_window_tokens", None)
    independent_input = getattr(capabilities, "max_input_tokens", None)
    if context is not None:
        limits.append(context - output_reserve)
    if independent_input is not None:
        limits.append(independent_input)
    return max(0, min(limits))
