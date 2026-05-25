"""Library-aware ModuleRefValue defaults: search ml first, fallback to value defaults.

Each make_*_ref_default:
  1. Searches ctx.ml for a matching named module.
  2. Returns None if optional and nothing found.
  3. Falls back to the corresponding make_*_default from module_value_defaults.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, overload

from typing_extensions import Literal, Optional

from zcu_tools.gui.adapter import ModuleRefValue

if TYPE_CHECKING:
    from zcu_tools.gui.adapter import ExpContext
from zcu_tools.program.v2.modules import AbsResetCfg, PulseReadoutCfg
from zcu_tools.program.v2.modules.pulse import PulseCfg

from .module_defaults import NamedModuleValue, select_named_module_value
from .module_value_defaults import (
    make_pulse_default,
    make_pulse_readout_default,
    make_readout_default,
    make_reset_default,
)


def _select(
    ctx: ExpContext,
    module_type: type,
    preferred_names: list[str],
) -> Optional[NamedModuleValue]:
    return select_named_module_value(
        ml=ctx.ml,
        module_type=module_type,
        preferred_names=preferred_names,
    )


@overload
def make_readout_ref_default(
    ctx: ExpContext,
    preferred_names: list[str] = ...,
    *,
    optional: Literal[False] = ...,
) -> ModuleRefValue: ...


@overload
def make_readout_ref_default(
    ctx: ExpContext,
    preferred_names: list[str] = ...,
    *,
    optional: Literal[True],
) -> Optional[ModuleRefValue]: ...


def make_readout_ref_default(
    ctx: ExpContext,
    preferred_names: list[str] = ["readout_rf", "readout", "res_readout"],
    *,
    optional: bool = False,
) -> Optional[ModuleRefValue]:
    selected = _select(ctx, PulseReadoutCfg, preferred_names)
    if selected is not None:
        return ModuleRefValue(chosen_key=selected.name, value=selected.value)
    if optional:
        return None
    return make_readout_default(ctx)


@overload
def make_pulse_readout_ref_default(
    ctx: ExpContext,
    preferred_names: list[str] = ...,
    *,
    optional: Literal[False] = ...,
) -> ModuleRefValue: ...


@overload
def make_pulse_readout_ref_default(
    ctx: ExpContext,
    preferred_names: list[str] = ...,
    *,
    optional: Literal[True],
) -> Optional[ModuleRefValue]: ...


def make_pulse_readout_ref_default(
    ctx: ExpContext,
    preferred_names: list[str] = ["readout_rf", "readout", "res_readout"],
    *,
    optional: bool = False,
) -> Optional[ModuleRefValue]:
    selected = _select(ctx, PulseReadoutCfg, preferred_names)
    if selected is not None:
        return ModuleRefValue(chosen_key=selected.name, value=selected.value)
    if optional:
        return None
    return make_pulse_readout_default(ctx)


@overload
def make_pulse_ref_default(
    ctx: ExpContext,
    preferred_names: list[str] = ...,
    *,
    optional: Literal[False] = ...,
) -> ModuleRefValue: ...


@overload
def make_pulse_ref_default(
    ctx: ExpContext,
    preferred_names: list[str] = ...,
    *,
    optional: Literal[True],
) -> Optional[ModuleRefValue]: ...


def make_pulse_ref_default(
    ctx: ExpContext,
    preferred_names: list[str] = ["pi_amp", "pi_len"],
    *,
    optional: bool = False,
) -> Optional[ModuleRefValue]:
    selected = _select(ctx, PulseCfg, preferred_names)
    if selected is not None:
        return ModuleRefValue(chosen_key=selected.name, value=selected.value)
    if optional:
        return None
    return make_pulse_default(ctx)


@overload
def make_reset_ref_default(
    ctx: ExpContext,
    preferred_names: list[str] = ...,
    *,
    optional: Literal[False] = ...,
) -> ModuleRefValue: ...


@overload
def make_reset_ref_default(
    ctx: ExpContext,
    preferred_names: list[str] = ...,
    *,
    optional: Literal[True],
) -> Optional[ModuleRefValue]: ...


def make_reset_ref_default(
    ctx: ExpContext,
    preferred_names: list[str] = ["reset_bath", "reset_10", "reset_120"],
    *,
    optional: bool = False,
) -> Optional[ModuleRefValue]:
    selected = _select(ctx, AbsResetCfg, preferred_names)
    if selected is not None:
        return ModuleRefValue(chosen_key=selected.name, value=selected.value)
    if optional:
        return None
    return make_reset_default(ctx)
