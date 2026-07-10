"""Small readers for module-shaped cfg objects used by autofluxdep nodes."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from zcu_tools.gui.session.types import ExpContext


def ctx_md_float(ctx: Any | None, key: str) -> float | None:
    """Return a numeric MetaDict value from ``ctx`` when present."""
    if not isinstance(ctx, ExpContext):
        return None
    value = ctx.md.get(key)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def ctx_module(ctx: Any | None, *names: str) -> Any | None:
    """Return the first ModuleLibrary module found in ``ctx`` under ``names``."""
    if not isinstance(ctx, ExpContext):
        return None
    for name in names:
        try:
            module = ctx.ml.get_module(name)
        except (KeyError, ValueError):
            module = None
        if module is not None:
            return module
    return None


def nested_get(value: Any, *path: str) -> Any | None:
    cur = value
    for part in path:
        if isinstance(cur, Mapping):
            cur = cur.get(part)
        else:
            cur = getattr(cur, part, None)
        if cur is None:
            return None
    return cur


def pulse_gain(module: Any) -> float | None:
    value = nested_get(module, "gain")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def pulse_length(module: Any) -> float | None:
    value = nested_get(module, "waveform", "length")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def pulse_product(module: Any) -> float | None:
    length = pulse_length(module)
    gain = pulse_gain(module)
    if length is None or gain is None:
        return None
    return length * gain


def readout_pulse_freq(module: Any) -> float | None:
    value = nested_get(module, "pulse_cfg", "freq")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def readout_pulse_gain(module: Any) -> float | None:
    value = nested_get(module, "pulse_cfg", "gain")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


__all__ = [
    "ctx_md_float",
    "ctx_module",
    "nested_get",
    "pulse_gain",
    "pulse_length",
    "pulse_product",
    "readout_pulse_freq",
    "readout_pulse_gain",
]
