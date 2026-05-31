"""Default factories for the ``pi2_pulse`` role (calibrated qubit π/2 pulse).

Blank is a blank qubit pulse; the ref prefers pi2 library entries, degrading to
the pi entries before falling back to blank.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from typing_extensions import Literal, Union, overload

from zcu_tools.gui.adapter import DisabledRefValue, ModuleRefValue
from zcu_tools.program.v2.modules.pulse import PulseCfg

from .helpers import select_named_module_value
from .qub_probe import make_qub_probe_default

if TYPE_CHECKING:
    from zcu_tools.gui.adapter import ExpContext

PI2_PULSE_NAMES = ["pi2_amp", "pi2_len", "pi_amp", "pi_len"]


def make_pi2_pulse_default(ctx: ExpContext) -> ModuleRefValue:
    """Blank π/2 pulse — a blank qubit pulse (no library lookup)."""
    return make_qub_probe_default(ctx)


@overload
def make_pi2_pulse_ref_default(
    ctx: ExpContext,
    preferred_names: list[str] = ...,
    *,
    optional: Literal[False] = ...,
) -> ModuleRefValue: ...


@overload
def make_pi2_pulse_ref_default(
    ctx: ExpContext,
    preferred_names: list[str] = ...,
    *,
    optional: Literal[True],
) -> Union[ModuleRefValue, DisabledRefValue]: ...


def make_pi2_pulse_ref_default(
    ctx: ExpContext,
    preferred_names: list[str] = PI2_PULSE_NAMES,
    *,
    optional: bool = False,
) -> Union[ModuleRefValue, DisabledRefValue]:
    """Reference a library π/2 pulse (pi2_* → pi_*), else the blank one."""
    selected = select_named_module_value(
        ml=ctx.ml, module_type=PulseCfg, preferred_names=preferred_names
    )
    if selected is not None:
        return ModuleRefValue(chosen_key=selected.name, value=selected.value)
    if optional:
        return DisabledRefValue()  # ADR-0012: present-but-disabled marker
    return make_pi2_pulse_default(ctx)
