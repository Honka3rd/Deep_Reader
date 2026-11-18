"""Strict, non-content-bearing JSON boundary for question model replies."""
import json
import math


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
