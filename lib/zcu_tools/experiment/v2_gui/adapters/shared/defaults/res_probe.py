"""Default factories for the ``res_probe`` role (resonator-side drive pulse).

A plain pulse on the resonator channel (res_ch / r_f), no ro_cfg — e.g. the CKP
res_pulse or AC-Stark stark_pulse1.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from typing_extensions import Literal, overload

from zcu_tools.gui.app.main.adapter import ModuleRefValue
from zcu_tools.gui.app.main.specs.pulse import make_pulse_spec
from zcu_tools.program.v2.modules.pulse import PulseCfg

from ..ctx_helpers import md_scalar_float, md_scalar_int
from .helpers import make_default_value, patch_pulse_fields, select_named_module_value

if TYPE_CHECKING:
    from zcu_tools.gui.app.main.adapter import ExpContext

RES_PROBE_NAMES = ["res_probe"]


def make_res_probe_default(ctx: ExpContext) -> ModuleRefValue:
    """Blank resonator probe pulse (res_ch / r_f), no library lookup.

    Adapter-specific tuning is applied by the caller via ``.with_field(...)``.
    """
    r_f = md_scalar_float(ctx, "r_f", 6000.0)
    res_ch = md_scalar_int(ctx, "res_ch", 0)

    value = make_default_value(make_pulse_spec())
    patch_pulse_fields(value, freq=r_f, ch=res_ch, gain=0.05, length=1.0)
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
        return None  # optional ref disabled (ADR-0021)
    return make_res_probe_default(ctx)
