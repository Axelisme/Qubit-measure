"""Comparison of already validated finite JSON values."""

from pydantic import JsonValue


def json_values_equal(left: JsonValue, right: JsonValue) -> bool:
    """Compare finite JSON recursively, distinguishing boolean from number.

    Numbers compare by numeric value (1 equals 1.0). Object key order does not
    matter, but key sets and list order do. Compare unknown fields as well.
    Inputs must already be valid finite JSON; neither input is changed.
    """
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            json_values_equal(value, right[key]) for key, value in left.items()
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            json_values_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return left == right
