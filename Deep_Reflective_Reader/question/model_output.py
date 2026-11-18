"""Strict, non-content-bearing JSON boundary for question model replies."""
import json
import math
import re


class ModelJSONSyntaxError(ValueError):
    """Syntax alone may permit the scope classifier's legacy block envelope."""


def _finite_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("non_finite_number")
    return number


def _reject_constant(value: str) -> None:
    raise ValueError("non_finite_number")


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate_field")
        result[key] = value
    return result


def parse_model_object(response: str) -> dict[str, object]:
    """Reject ambiguous/non-finite JSON without retaining decoder diagnostics."""
    if not isinstance(response, str):
        raise ValueError("invalid_json_object")
    try:
        payload = json.loads(
            response, object_pairs_hook=_unique_object,
            parse_constant=_reject_constant, parse_float=_finite_float,
        )
    except json.JSONDecodeError:
        raise ModelJSONSyntaxError("invalid_json_object") from None
    except (ValueError, RecursionError):
        raise ValueError("invalid_json_object") from None
    if not isinstance(payload, dict):
        raise ValueError("invalid_json_object")
    return payload


def parse_scope_object(response: str) -> dict[str, object]:
    """Unwrap a complete fence or short prose label, never a nested JSON value.

    Supported prose labels end in a colon (e.g. Result:, Here is the result:).
    The complete remainder must be one JSON object, including trailing checks.
    """
    try:
        return parse_model_object(response)
    except ModelJSONSyntaxError:
        text = response.strip()
        fence = re.fullmatch(r"```(?:json)?\s*([\s\S]*)\s*```", text, re.IGNORECASE)
        label = re.fullmatch(r"[^\W\d_][\w \t-]{0,63}[:：]\s*([\s\S]+)", text)
        if fence is not None:
            payload = fence.group(1)
        elif label is not None:
            payload = label.group(1)
        else:
            raise ValueError("invalid_scope_envelope") from None
        return parse_model_object(payload)
