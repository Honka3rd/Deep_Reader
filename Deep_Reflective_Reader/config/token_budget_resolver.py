import math
from dataclasses import dataclass

from llm.llm_model_capabilities import LLMModelCapabilities


@dataclass(frozen=True)
class EffectiveTokenBudgets:
    """Runtime token budgets after clamping app targets to model capabilities."""
    target_max_input_tokens: int
    target_max_output_tokens: int
    target_max_context_tokens: int
    input_budget_utilization_ratio: float
    context_budget_utilization_ratio: float
    effective_input_budget: int
    effective_output_budget: int
    effective_context_budget: int
    fallback_used: bool
    capability_model_name: str | None
    capability_endpoint_kind: str | None
    capability_max_input_tokens: int | None
    capability_max_output_tokens: int | None
    capability_context_window_tokens: int | None = None


def resolve_effective_token_budgets(
    capabilities: LLMModelCapabilities | None,
    target_max_input_tokens: int,
    target_max_output_tokens: int,
    target_max_context_tokens: int,
    input_budget_utilization_ratio: float = 0.2,
    context_budget_utilization_ratio: float = 0.9,
    *,
    declared_output_tokens: int | None = None,
) -> EffectiveTokenBudgets:
    """Compute effective budgets using model-capability ratio policy with target fallback."""
    if target_max_input_tokens <= 0:
        raise ValueError("target_max_input_tokens must be > 0")
    if target_max_output_tokens <= 0:
        raise ValueError("target_max_output_tokens must be > 0")
    if target_max_context_tokens <= 0:
        raise ValueError("target_max_context_tokens must be > 0")
    if not (0 < input_budget_utilization_ratio <= 1):
        raise ValueError("input_budget_utilization_ratio must be in (0, 1]")
    if not (0 < context_budget_utilization_ratio <= 1):
        raise ValueError("context_budget_utilization_ratio must be in (0, 1]")

    if declared_output_tokens is not None and (
        type(declared_output_tokens) is not int or declared_output_tokens <= 0
        or (capabilities is not None and declared_output_tokens > capabilities.max_output_tokens)
    ):
        raise ValueError("invalid_provider_output_budget")
    output_budget = target_max_output_tokens
    if declared_output_tokens is not None:
        output_budget = declared_output_tokens
    elif capabilities is not None:
        output_budget = min(output_budget, capabilities.max_output_tokens)

    if capabilities is None:
        # Input/context fall back to configured targets. A provider-declared
        # output allowance remains authoritative even without capability metadata.
        effective_input_budget = target_max_input_tokens
        effective_output_budget = output_budget
        effective_context_budget = target_max_context_tokens
        fallback_used = True
        capability_model_name = None
        capability_endpoint_kind = None
        capability_max_input_tokens = None
        capability_max_output_tokens = None
    else:
        model_based_input_budget = max(
            1,
            math.floor(capabilities.combined_token_limit(
                output_budget
            ) * input_budget_utilization_ratio),
        )
        effective_input_budget = min(target_max_input_tokens, model_based_input_budget)
        effective_output_budget = output_budget
        model_based_context_budget = max(
            1,
            math.floor(effective_input_budget * context_budget_utilization_ratio),
        )
        effective_context_budget = min(target_max_context_tokens, model_based_context_budget)
        fallback_used = False
        capability_model_name = capabilities.model_name
        capability_endpoint_kind = capabilities.endpoint_kind
        capability_max_input_tokens = capabilities.max_input_tokens
        capability_max_output_tokens = capabilities.max_output_tokens

    return EffectiveTokenBudgets(
        target_max_input_tokens=target_max_input_tokens,
        target_max_output_tokens=target_max_output_tokens,
        target_max_context_tokens=target_max_context_tokens,
        input_budget_utilization_ratio=input_budget_utilization_ratio,
        context_budget_utilization_ratio=context_budget_utilization_ratio,
        effective_input_budget=effective_input_budget,
        effective_output_budget=effective_output_budget,
        effective_context_budget=effective_context_budget,
        fallback_used=fallback_used,
        capability_model_name=capability_model_name,
        capability_endpoint_kind=capability_endpoint_kind,
        capability_max_input_tokens=capability_max_input_tokens,
        capability_max_output_tokens=capability_max_output_tokens,
        capability_context_window_tokens=(capabilities.context_window_tokens if capabilities else None),
    )
