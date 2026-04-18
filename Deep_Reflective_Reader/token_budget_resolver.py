from dataclasses import dataclass

from llm_model_capabilities import LLMModelCapabilities


@dataclass(frozen=True)
class EffectiveTokenBudgets:
    """Runtime token budgets after clamping app targets to model capabilities."""
    target_max_input_tokens: int
    target_max_output_tokens: int
    target_max_context_tokens: int
    effective_input_budget: int
    effective_output_budget: int


def resolve_effective_token_budgets(
    capabilities: LLMModelCapabilities,
    target_max_input_tokens: int,
    target_max_output_tokens: int,
    target_max_context_tokens: int,
) -> EffectiveTokenBudgets:
    """Compute effective input/output budgets from app targets and model limits."""
    if target_max_input_tokens <= 0:
        raise ValueError("target_max_input_tokens must be > 0")
    if target_max_output_tokens <= 0:
        raise ValueError("target_max_output_tokens must be > 0")
    if target_max_context_tokens <= 0:
        raise ValueError("target_max_context_tokens must be > 0")

    effective_input_budget = min(target_max_input_tokens, capabilities.max_input_tokens)
    effective_output_budget = min(target_max_output_tokens, capabilities.max_output_tokens)
    return EffectiveTokenBudgets(
        target_max_input_tokens=target_max_input_tokens,
        target_max_output_tokens=target_max_output_tokens,
        target_max_context_tokens=target_max_context_tokens,
        effective_input_budget=effective_input_budget,
        effective_output_budget=effective_output_budget,
    )
