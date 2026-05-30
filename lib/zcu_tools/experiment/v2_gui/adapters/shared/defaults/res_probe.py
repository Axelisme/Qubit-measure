"""Default factories for the ``res_probe`` role (resonator-side drive pulse).

A plain pulse on the resonator channel (res_ch / r_f), no ro_cfg — e.g. the CKP
res_pulse or AC-Stark stark_pulse1.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from typing_extensions import Literal, Optional, overload

from zcu_tools.gui.adapter import ModuleRefValue
from zcu_tools.gui.specs.pulse import make_pulse_spec
from zcu_tools.program.v2.modules.pulse import PulseCfg

from ..ctx_helpers import md_get_float, md_get_int
from .helpers import make_default_value, patch_pulse_fields, select_named_module_value

if TYPE_CHECKING:
    from zcu_tools.gui.adapter import ExpContext

RES_PROBE_NAMES = ["res_probe"]


def make_res_probe_default(
    ctx: ExpContext, *, gain: float = 0.05, length: float = 1.0
) -> ModuleRefValue:
    """Blank resonator probe pulse (res_ch / r_f), no library lookup."""
    r_f = md_get_float(ctx, "r_f", 6000.0)
    res_ch = md_get_int(ctx, "res_ch", 0)

    value = make_default_value(make_pulse_spec())
    patch_pulse_fields(value, freq=r_f, ch=res_ch, gain=gain, length=length)
    return ModuleRefValue("<Custom:Pulse>", value)


@overload
def make_res_probe_ref_default(
    ctx: ExpContext,
    preferred_names: list[str] = ...,
    *,
    optional: Literal[False] = ...,
) -> ModuleRefValue: ...


@overload
def make_res_probe_ref_default(
    ctx: ExpContext,
    preferred_names: list[str] = ...,
    *,
    optional: Literal[True],
) -> Optional[ModuleRefValue]: ...


def make_res_probe_ref_default(
    ctx: ExpContext,
    preferred_names: list[str] = RES_PROBE_NAMES,
    *,
    optional: bool = False,
) -> Optional[ModuleRefValue]:
    """Reference a library resonator probe pulse, else the blank one."""
    selected = select_named_module_value(
        ml=ctx.ml, module_type=PulseCfg, preferred_names=preferred_names
    )
    if selected is not None:
        return ModuleRefValue(chosen_key=selected.name, value=selected.value)
    if optional:
        return None
    return make_res_probe_default(ctx)
