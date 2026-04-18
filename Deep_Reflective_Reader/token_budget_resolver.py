import math
from dataclasses import dataclass

from llm_model_capabilities import LLMModelCapabilities


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


def resolve_effective_token_budgets(
    capabilities: LLMModelCapabilities | None,
    target_max_input_tokens: int,
    target_max_output_tokens: int,
    target_max_context_tokens: int,
    input_budget_utilization_ratio: float = 0.2,
    context_budget_utilization_ratio: float = 0.9,
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

    if capabilities is None:
        # Fallback path uses app-level target budgets directly.
        # This keeps a single source of truth in AppDIConfig.
        effective_input_budget = target_max_input_tokens
        effective_output_budget = target_max_output_tokens
        effective_context_budget = target_max_context_tokens
        fallback_used = True
        capability_model_name = None
        capability_endpoint_kind = None
        capability_max_input_tokens = None
        capability_max_output_tokens = None
    else:
        model_based_input_budget = max(
            1,
            math.floor(capabilities.max_input_tokens * input_budget_utilization_ratio),
        )
        effective_input_budget = min(target_max_input_tokens, model_based_input_budget)
        effective_output_budget = min(target_max_output_tokens, capabilities.max_output_tokens)
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
    )
