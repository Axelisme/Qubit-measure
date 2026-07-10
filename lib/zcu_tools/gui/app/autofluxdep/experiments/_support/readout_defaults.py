"""Readout seed helpers shared by readout-oriented autofluxdep nodes."""

from __future__ import annotations

from typing import Any

from zcu_tools.gui.app.autofluxdep.experiments._support.module_aliases import (
    READOUT_LIBRARY_ALIASES,
)
from zcu_tools.gui.app.autofluxdep.experiments._support.utils.module_values import (
    ctx_md_float,
    ctx_module,
    nested_get,
)


def _readout_pulse_float(module: Any, key: str) -> float | None:
    value = nested_get(module, "pulse_cfg", key)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def seed_readout_freq(ctx: Any | None, fallback: float) -> float:
    md_value = ctx_md_float(ctx, "r_f")
    if md_value is not None:
        return md_value
    module = ctx_module(ctx, *READOUT_LIBRARY_ALIASES)
    return _readout_pulse_float(module, "freq") or fallback


def seed_readout_gain(ctx: Any | None, fallback: float) -> float:
    module = ctx_module(ctx, *READOUT_LIBRARY_ALIASES)
    return _readout_pulse_float(module, "gain") or fallback
