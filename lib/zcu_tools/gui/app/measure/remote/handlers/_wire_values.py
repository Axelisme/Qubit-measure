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


def _is_complex_tag(value: object) -> bool:
    return (
        isinstance(value, dict)
        and set(value) == {_COMPLEX_TAG}
        and isinstance(value[_COMPLEX_TAG], (list, tuple))
        and len(value[_COMPLEX_TAG]) == 2
        and all(isinstance(p, (int, float)) for p in value[_COMPLEX_TAG])
    )


def coerce_wire_value(value: object) -> object:
    """Decode an inbound complex tag; leave every other wire value unchanged.

    Writeback uses this before applying a proposed value to MetaDict, whose
    persistence format also supports complex numbers.
    """
    if _is_complex_tag(value):
        re, im = value[_COMPLEX_TAG]  # type: ignore[index]
        return complex(re, im)
    return value
