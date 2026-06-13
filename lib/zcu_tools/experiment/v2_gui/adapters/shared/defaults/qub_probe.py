"""Default factories for the ``qub_probe`` role (qubit drive/probe pulse).

Template for the per-role default file: each role exposes a *blank* factory
(``make_<role>_default``, sensible md-derived defaults, no library lookup) and a
*ref* factory (``make_<role>_ref_default``, prefers a named library entry, falls
back to the blank one). The ref factory's ``optional=True`` returns ``None`` when
the field is optional and nothing matches. See ADR-0009.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, overload

from zcu_tools.gui.app.main.adapter import ModuleRefValue
from zcu_tools.gui.app.main.specs.pulse import make_pulse_spec
from zcu_tools.program.v2.modules.pulse import PulseCfg

from ..ctx_helpers import md_scalar_float, md_scalar_int
from .helpers import make_default_value, patch_pulse_fields, select_named_module_value

if TYPE_CHECKING:
    from zcu_tools.gui.app.main.adapter import ExpContext

# Library names this role prefers when resolving a ref (user-chosen convention).
QUB_PROBE_NAMES = ["qub_probe"]


def make_qub_probe_default(ctx: ExpContext) -> ModuleRefValue:
    """Blank qubit probe pulse (qub_ch / q_f), no library lookup.

    Adapter-specific tuning (gain, length, …) is applied by the caller via
    ``.with_field(...)``.
    """
    q_f = md_scalar_float(ctx, "q_f", 4000.0)
    qub_ch = md_scalar_int(ctx, "qub_ch", 0)

    value = make_default_value(make_pulse_spec())
    patch_pulse_fields(value, freq=q_f, ch=qub_ch, gain=0.05, length=0.1)
    return ModuleRefValue("<Custom:Pulse>", value)


@overload
def make_qub_probe_ref_default(
    ctx: ExpContext,
    preferred_names: list[str] = ...,
    *,
    optional: Literal[False] = ...,
) -> ModuleRefValue: ...


@overload
def make_qub_probe_ref_default(
    ctx: ExpContext,
    preferred_names: list[str] = ...,
    *,
    optional: Literal[True],
) -> ModuleRefValue | None: ...


def make_qub_probe_ref_default(
    ctx: ExpContext,
    preferred_names: list[str] = QUB_PROBE_NAMES,
    *,
    optional: bool = False,
) -> ModuleRefValue | None:
    """Reference a library qubit probe pulse, else fall back to the blank one."""
    selected = select_named_module_value(
        ml=ctx.ml, module_type=PulseCfg, preferred_names=preferred_names
    )
    if selected is not None:
        return ModuleRefValue(chosen_key=selected.name, value=selected.value)
    if optional:
        return None  # optional ref disabled (ADR-0010)
    return make_qub_probe_default(ctx)
