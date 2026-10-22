"""Wire-stable value coercion helpers."""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
from pydantic import JsonValue

from zcu_tools.core.complex_codec import decode_complex, encode_complex
from zcu_tools.experiment.stop_signal import parse_cancel_scope
from zcu_tools.gui.remote.errors import ErrorCode, RemoteError


def cancel_scope_arguments(params: Mapping[str, object]) -> tuple[str, int | None]:
    """Decode scope primitives and translate shared resolver syntax errors."""
    scope, expected = params.get("scope", "script"), params.get("expected")
    if not isinstance(scope, str) or (
        expected is not None and not isinstance(expected, int)
    ):
        raise RemoteError(ErrorCode.INVALID_PARAMS, "Invalid scope or expected type")
    try:
        return parse_cancel_scope(scope, expected)
    except ValueError as error:
        raise RemoteError(ErrorCode.INVALID_PARAMS, str(error)) from error


def context_wire_value(value: object) -> JsonValue:
    """Project complete values to JSON; tuples become lists and complex uses tags.

    Reject non-finite numbers, unsupported objects and non-string mapping keys.
    Nested tuple/list/array values retain their ordering and scalar contents.
    """
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite context number")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, complex):
        return context_wire_value(encode_complex(value))
    if isinstance(value, np.ndarray):
        return context_wire_value(value.tolist())
    if isinstance(value, np.generic):
        return context_wire_value(value.item())
    if isinstance(value, dict):
        result: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"unsupported context key: {type(key).__name__}")
            result[key] = context_wire_value(item)
        return result
    if isinstance(value, (list, tuple)):
        return [context_wire_value(item) for item in value]
    raise TypeError(f"unsupported context value: {type(value).__name__}")


def coerce_wire_value(value: object) -> object:
    """Decode an inbound complex tag; leave every other wire value unchanged.

    A dict containing __complex__ is decoded by the shared strict scalar
    codec. Invalid tags raise ValueError, and oversized numeric components
    can raise OverflowError. No strings or tuples are guessed as complex.
    Writeback uses this before accepting the proposed working-unit value.
    """
    if isinstance(value, dict) and "__complex__" in value:
        return decode_complex(value)
    return value
