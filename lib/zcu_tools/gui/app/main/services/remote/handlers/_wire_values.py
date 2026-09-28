"""Wire-stable value coercion helpers."""

from __future__ import annotations

import math

import numpy as np

_COMPLEX_TAG = "__complex__"


def context_wire_value(value: object) -> object:
    """Project complete context values; reject unsupported types and keys."""
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite context number")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, complex):
        return context_wire_value({"__complex__": [value.real, value.imag]})
    if isinstance(value, np.ndarray):
        return context_wire_value(value.tolist())
    if isinstance(value, np.generic):
        return context_wire_value(value.item())
    if isinstance(value, dict):
        result: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"unsupported context key: {type(key).__name__}")
            result[key] = context_wire_value(item)
        return result
    if isinstance(value, list):
        return [context_wire_value(item) for item in value]
    raise TypeError(f"unsupported context value: {type(value).__name__}")


def _complex_tag(value: complex) -> dict[str, list[float]]:
    return {_COMPLEX_TAG: [value.real, value.imag]}


def _is_complex_tag(value: object) -> bool:
    return (
        isinstance(value, dict)
        and set(value) == {_COMPLEX_TAG}
        and isinstance(value[_COMPLEX_TAG], (list, tuple))
        and len(value[_COMPLEX_TAG]) == 2
        and all(isinstance(p, (int, float)) for p in value[_COMPLEX_TAG])
    )


def _json_safe(value: object) -> object:
    """Make ``value`` JSON-safe without loss where the type has a wire encoding.

    ``complex`` → ``{"__complex__": [re, im]}`` (round-trips via
    ``_coerce_wire_value``). Otherwise return ``value`` if it round-trips through
    JSON as-is, else its ``repr`` (lossy last resort for opaque objects).
    """
    import json

    if isinstance(value, complex):
        return _complex_tag(value)
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return {"__repr__": repr(value)}


def _coerce_wire_value(value: object) -> object:
    """Inverse of :func:`_json_safe`'s structured tags for inbound wire values.

    A ``{"__complex__": [re, im]}`` tag becomes a Python ``complex``; every other
    value passes through untouched. Used on the writeback ``set`` input so an
    agent-supplied complex proposed_value applies as a real ``complex`` (the
    in-process md apply + MetaDict persistence both speak ``complex``)."""
    if _is_complex_tag(value):
        re, im = value[_COMPLEX_TAG]  # type: ignore[index]
        return complex(re, im)
    return value
