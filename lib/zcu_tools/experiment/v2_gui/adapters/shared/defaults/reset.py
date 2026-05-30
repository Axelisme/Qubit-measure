"""Default factories for the ``reset`` role (qubit reset module).

The reset module has several concrete shapes (none / pulse / two-pulse / bath);
each has its own blank builder. ``make_reset_default`` is the role's blank
(pulse-reset); ``make_reset_ref_default`` prefers a calibrated library reset.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from typing_extensions import Literal, Optional, overload

from zcu_tools.gui.adapter import CfgSectionValue, ModuleRefValue
from zcu_tools.gui.specs.reset import (
    make_bath_reset_spec,
    make_none_reset_spec,
    make_pulse_reset_spec,
    make_two_pulse_reset_spec,
)
from zcu_tools.program.v2.modules import AbsResetCfg

from ..ctx_helpers import md_scalar_float, md_scalar_int
from .helpers import make_default_value, patch_pulse_fields, select_named_module_value

if TYPE_CHECKING:
    from zcu_tools.gui.adapter import ExpContext

RESET_NAMES = ["reset_bath", "reset_10", "reset_120"]


def make_none_reset_default(ctx: ExpContext) -> ModuleRefValue:  # noqa: ARG001
    return ModuleRefValue(
        "<Custom:None Reset>", make_default_value(make_none_reset_spec())
    )


def make_pulse_reset_default(ctx: ExpContext) -> ModuleRefValue:
    q_f = md_scalar_float(ctx, "q_f", 4000.0)
    qub_ch = md_scalar_int(ctx, "qub_ch", 0)

    value = make_default_value(make_pulse_reset_spec())
    pulse_cfg = value.fields.get("pulse_cfg")
    if isinstance(pulse_cfg, CfgSectionValue):
        patch_pulse_fields(pulse_cfg, freq=q_f, ch=qub_ch, gain=0.2, length=1.0)
    return ModuleRefValue("<Custom:Pulse Reset>", value)


def make_two_pulse_reset_default(ctx: ExpContext) -> ModuleRefValue:
    q_f = md_scalar_float(ctx, "q_f", 4000.0)
    qub_ch = md_scalar_int(ctx, "qub_ch", 0)

    value = make_default_value(make_two_pulse_reset_spec())
    for key in ("pulse1_cfg", "pulse2_cfg"):
        pulse_cfg = value.fields.get(key)
        if isinstance(pulse_cfg, CfgSectionValue):
            patch_pulse_fields(pulse_cfg, freq=q_f, ch=qub_ch, gain=0.2, length=1.0)
    return ModuleRefValue("<Custom:Two-Pulse Reset>", value)


def make_bath_reset_default(ctx: ExpContext) -> ModuleRefValue:
    r_f = md_scalar_float(ctx, "r_f", 6000.0)
    q_f = md_scalar_float(ctx, "q_f", 4000.0)
    res_ch = md_scalar_int(ctx, "res_ch", 0)
    qub_ch = md_scalar_int(ctx, "qub_ch", 0)

    value = make_default_value(make_bath_reset_spec())
    cav = value.fields.get("cavity_tone_cfg")
    if isinstance(cav, CfgSectionValue):
        patch_pulse_fields(cav, freq=r_f, ch=res_ch, gain=0.1, length=1.0)
    qub = value.fields.get("qubit_tone_cfg")
    if isinstance(qub, CfgSectionValue):
        patch_pulse_fields(qub, freq=q_f, ch=qub_ch, gain=0.2, length=1.0)
    pi2 = value.fields.get("pi2_cfg")
    if isinstance(pi2, CfgSectionValue):
        patch_pulse_fields(pi2, freq=q_f, ch=qub_ch, gain=0.2, length=1.0)
    return ModuleRefValue("<Custom:Bath Reset>", value)


def make_reset_default(ctx: ExpContext) -> ModuleRefValue:
    """The reset role's blank — a pulse reset."""
    return make_pulse_reset_default(ctx)


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
    preferred_names: list[str] = RESET_NAMES,
    *,
    optional: bool = False,
) -> Optional[ModuleRefValue]:
    """Reference a calibrated library reset, else fall back to a blank pulse reset."""
    selected = select_named_module_value(
        ml=ctx.ml, module_type=AbsResetCfg, preferred_names=preferred_names
    )
    if selected is not None:
        return ModuleRefValue(chosen_key=selected.name, value=selected.value)
    if optional:
        return None
    return make_reset_default(ctx)
