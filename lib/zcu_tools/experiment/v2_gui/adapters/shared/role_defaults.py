"""Role-named default value helpers (thin wrappers over the generic factories).

Each wrapper encodes one notebook role's consistent fallback strategy in its
name, so adapters read like the notebook (default_pi / default_pi2 / ...) instead
of passing preferred_names lists. They delegate to the two-layer
make_*_default / make_*_ref_default helpers — no duplicated construction.

By contract these produce *value* defaults only and **never lock a field**;
locking (e.g. a probe's freq fixed to 0.0 because a sweep axis drives it) is the
adapter's job in cfg_spec() via spec.lock_literal(). See ADR-0009.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from typing_extensions import Literal, Optional, overload

from zcu_tools.gui.adapter import ModuleRefValue

from .module_ref_defaults import (
    make_pulse_readout_ref_default,
    make_pulse_ref_default,
    make_reset_ref_default,
)
from .module_value_defaults import make_pulse_default

if TYPE_CHECKING:
    from zcu_tools.gui.adapter import ExpContext

# Role preferred-name chains (library lookup order before fallback).
_PI_NAMES = ["pi_amp", "pi_len"]
_PI2_NAMES = ["pi2_amp", "pi2_len", "pi_amp", "pi_len"]


def default_pi(ctx: ExpContext) -> ModuleRefValue:
    """π pulse: prefer library pi_amp → pi_len, else blank pulse default."""
    return make_pulse_ref_default(ctx, _PI_NAMES)


def default_pi2(ctx: ExpContext) -> ModuleRefValue:
    """π/2 pulse: prefer pi2_amp → pi2_len, then degrade to pi, else blank."""
    return make_pulse_ref_default(ctx, _PI2_NAMES)


def default_qub_probe(ctx: ExpContext) -> ModuleRefValue:
    """Qubit probe pulse: a fresh blank pulse (freq usually locked by the adapter
    because a sweep axis drives it). Does not reuse a calibrated pi pulse."""
    return make_pulse_default(ctx)


@overload
def default_res_probe(
    ctx: ExpContext, *, optional: Literal[False] = ...
) -> ModuleRefValue: ...


@overload
def default_res_probe(
    ctx: ExpContext, *, optional: Literal[True]
) -> Optional[ModuleRefValue]: ...


def default_res_probe(
    ctx: ExpContext, *, optional: bool = False
) -> Optional[ModuleRefValue]:
    """Resonator probe readout: prefer library readout entries, else blank."""
    return make_pulse_readout_ref_default(ctx, optional=optional)


@overload
def default_reset(
    ctx: ExpContext, *, optional: Literal[False] = ...
) -> ModuleRefValue: ...


@overload
def default_reset(
    ctx: ExpContext, *, optional: Literal[True]
) -> Optional[ModuleRefValue]: ...


def default_reset(
    ctx: ExpContext, *, optional: bool = False
) -> Optional[ModuleRefValue]:
    """Reset module: prefer library reset entries, else blank (often optional)."""
    return make_reset_ref_default(ctx, optional=optional)
