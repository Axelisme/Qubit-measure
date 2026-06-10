"""Default factories for the ``pi_pulse`` role (calibrated qubit π pulse).

The blank is just a blank qubit pulse (a π pulse is a calibrated qubit pulse); the
ref prefers the library ``pi_amp`` / ``pi_len`` entries.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Optional, overload

from zcu_tools.gui.app.main.adapter import ModuleRefValue
from zcu_tools.program.v2.modules.pulse import PulseCfg

from .helpers import select_named_module_value
from .qub_probe import make_qub_probe_default

if TYPE_CHECKING:
    from zcu_tools.gui.app.main.adapter import ExpContext

PI_PULSE_NAMES = ["pi_amp", "pi_len"]


def make_pi_pulse_default(ctx: ExpContext) -> ModuleRefValue:
    """Blank π pulse — a blank qubit pulse (no library lookup)."""
    return make_qub_probe_default(ctx)


@overload
def make_pi_pulse_ref_default(
    ctx: ExpContext,
    preferred_names: list[str] = ...,
    *,
    optional: Literal[False] = ...,
) -> ModuleRefValue: ...


@overload
def make_pi_pulse_ref_default(
    ctx: ExpContext,
    preferred_names: list[str] = ...,
    *,
    optional: Literal[True],
) -> ModuleRefValue | None: ...


def make_pi_pulse_ref_default(
    ctx: ExpContext,
    preferred_names: list[str] = PI_PULSE_NAMES,
    *,
    optional: bool = False,
) -> ModuleRefValue | None:
    """Reference a library π pulse (pi_amp → pi_len), else the blank one."""
    selected = select_named_module_value(
        ml=ctx.ml, module_type=PulseCfg, preferred_names=preferred_names
    )
    if selected is not None:
        return ModuleRefValue(chosen_key=selected.name, value=selected.value)
    if optional:
        return None  # optional ref disabled (ADR-0010)
    return make_pi_pulse_default(ctx)
