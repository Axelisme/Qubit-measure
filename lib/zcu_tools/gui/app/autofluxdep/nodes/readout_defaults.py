"""Readout seed helpers shared by readout-oriented autofluxdep nodes."""

from __future__ import annotations

from typing import Any

from zcu_tools.gui.app.autofluxdep.nodes.defaults import (
    ctx_md_float,
    ctx_module,
    readout_pulse_freq,
    readout_pulse_gain,
)
from zcu_tools.gui.app.autofluxdep.nodes.module_aliases import READOUT_LIBRARY_ALIASES


def seed_readout_freq(ctx: Any | None, fallback: float) -> float:
    md_value = ctx_md_float(ctx, "r_f")
    if md_value is not None:
        return md_value
    module = ctx_module(ctx, *READOUT_LIBRARY_ALIASES)
    return readout_pulse_freq(module) or fallback


def seed_readout_gain(ctx: Any | None, fallback: float) -> float:
    module = ctx_module(ctx, *READOUT_LIBRARY_ALIASES)
    return readout_pulse_gain(module) or fallback
