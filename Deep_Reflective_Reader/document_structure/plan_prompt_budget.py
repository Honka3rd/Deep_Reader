"""Fail closed before spending a planner call on an oversized full prompt."""

import json
import logging
import math
from llm.llm_model_capabilities import input_token_limit, validate_capacity_limits

LOGGER = logging.getLogger(__name__)


class EnhancementPromptBudgetError(ValueError):
    def __init__(self, reason, decision=None):
        super().__init__(reason)
        self.decision = dict(decision or {})


def enforce_prompt_budget(prompt, provider, *, safety_margin_ratio=0.05, minimum_margin_tokens=256,
                          transport_overhead_tokens=256):
    request_schema = getattr(provider, "request_schema_text", "")
    if isinstance(request_schema, str):
        prompt += request_schema
    if type(transport_overhead_tokens) is not int or transport_overhead_tokens < 0:
        raise ValueError("invalid transport overhead")
    if not isinstance(safety_margin_ratio, (int, float)) or isinstance(safety_margin_ratio, bool) or not 0 <= safety_margin_ratio < 1:
        raise ValueError("invalid safety margin ratio")
    if type(minimum_margin_tokens) is not int or minimum_margin_tokens < 0:
        raise ValueError("invalid minimum safety margin")
    decision = {"policy": "enhancement_prompt_budget_v3",
                "measurement": "utf8_byte_upper_estimate_not_provider_usage",
                "input_estimate": len(prompt.encode("utf-8"))}

    def reject(reason):
        LOGGER.warning("enhancement_prompt_budget %s", json.dumps({**decision, "decision": reason}))
        raise EnhancementPromptBudgetError(reason, decision)

    try:
        capabilities = provider.get_model_capabilities()
        validate_capacity_limits(capabilities)
        output_limit = capabilities.max_output_tokens
    except Exception:
        reject("enhancement_budget_unavailable")
    counter = getattr(provider, "count_input_tokens", None)
    if callable(counter):
        try:
            count = counter(prompt)
            if type(count) is int and count > 0:
                decision.update(input_estimate=count,
                                measurement="provider_matched_local_tokenizer_not_provider_usage")
        except Exception:
            pass
    # With no configured allowance, reserve the provider's entire output limit.
    output_reserve = getattr(provider, "effective_output_budget", None)
    if output_reserve is None:
        output_reserve = output_limit
    if type(output_reserve) is not int or not 0 < output_reserve <= output_limit:
        reject("enhancement_output_budget_invalid")
    input_limit = input_token_limit(capabilities, output_reserve)
    margin = max(minimum_margin_tokens, math.ceil(input_limit * safety_margin_ratio))
    available = max(0, input_limit - margin)
    decision.update(input_limit=input_limit, output_reserve=output_reserve,
                    model=getattr(capabilities, "model_name", None),
                    endpoint_kind=getattr(capabilities, "endpoint_kind", None),
                    context_window_tokens=getattr(capabilities, "context_window_tokens", None),
                    max_input_tokens=getattr(capabilities, "max_input_tokens", None),
                    max_output_tokens=output_limit,
                    safety_margin=margin, available_input=available)
    decision.update(transport_overhead_tokens=transport_overhead_tokens,
                    required_input_tokens=decision["input_estimate"] + transport_overhead_tokens)
    if decision["required_input_tokens"] > available:
        reject("enhancement_prompt_budget_exceeded")
    LOGGER.info("enhancement_prompt_budget %s", json.dumps({**decision, "decision": "accepted"}))
    return decision
