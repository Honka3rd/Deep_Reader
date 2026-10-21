"""Safe capacity projection and request-scoped dispatch configuration guard."""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict
import hashlib
import json

from document_structure.plan_prompt_budget import EnhancementPromptBudgetError
from document_structure.structure_plan_contract import PROMPT_VERSION

_expected = ContextVar("enhancement_capacity_identity", default=None)


def capacity_identity(prompt, budget, resource_limits):
    value = {"prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
             "prompt_version": PROMPT_VERSION, "budget": budget,
             "resource_limits": asdict(resource_limits)}
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


@contextmanager
def require_capacity_identity(identity):
    token = _expected.set(identity)
    try:
        yield
    finally:
        _expected.reset(token)


def validate_dispatch_identity(identity):
    expected = _expected.get()
    if expected is not None and expected != identity:
        raise EnhancementPromptBudgetError("enhancement_configuration_changed")


def capacity_projection(decision=None, *, reason=None, source_chars=None):
    decision = decision or {}
    reasons = {"enhancement_prompt_budget_exceeded": ("exceeded", "model_capacity_exceeded"),
               "local_resource_limit": ("resource_limited", "local_resource_limit")}
    status, code = reasons.get(reason, ("unavailable", "capacity_unavailable")) if reason else ("fits", None)
    measurement = decision.get("measurement", "")
    method = ("tokenizer" if "local_tokenizer" in measurement else
              "conservative_estimate" if decision.get("required_input_tokens") is not None else "unavailable")
    return dict(status=status, reason_code=code, model=decision.get("model"),
        context_window_tokens=decision.get("context_window_tokens"),
        max_input_tokens=decision.get("max_input_tokens"), max_output_tokens=decision.get("max_output_tokens"),
        output_reserve_tokens=decision.get("output_reserve"), safety_margin_tokens=decision.get("safety_margin"),
        required_input_tokens=decision.get("required_input_tokens"), available_input_tokens=decision.get("available_input"),
        transport_overhead_tokens=decision.get("transport_overhead_tokens"), count_method=method,
        tokenizer_identity=f"{decision.get('model')}:{method}" if method != "unavailable" else None,
        full_source=decision.get("full_source"), source_chars=source_chars,
        visible_source_chars=decision.get("visible_source_chars"),
        budget_policy_version=decision.get("policy"), prompt_version=PROMPT_VERSION)
