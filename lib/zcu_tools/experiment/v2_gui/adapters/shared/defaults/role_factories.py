"""Single source of truth mapping a ``role_id`` to its L2 default factories.

A role has a *blank* factory (``make_<role>_default(ctx)`` — md-linked defaults,
never a library lookup, never ``None``) and optionally a *ref* factory
(``make_<role>_ref_default(ctx, *, optional=...)`` — prefers a named library
entry, falls back to the blank, and may return ``None`` when ``optional`` and
nothing matches). Both the GUI ``RoleCatalog`` registration (``registry.py``)
and the value-assembly ``CfgBuilder`` consume this table, so the role vocabulary
lives in exactly one place.

The ``CfgBuilder.role()`` verb selects between the two factories: a plain
``.role(path, role_id)`` calls the *ref* factory (library-aware, the common
case); ``prefer_blank=True`` forces the *blank* factory (e.g. a twotone readout
that must stay inline, never adopting a library ``readout_dpm``);
``optional=True`` takes the *ref* factory's optional path (library miss →
``None``). ``RoleCatalog`` always uses the *blank* factory (creating from a role
seeds a fresh entry, it never references an existing library entry).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Union

from .pi2_pulse import make_pi2_pulse_default, make_pi2_pulse_ref_default
from .pi_pulse import make_pi_pulse_default, make_pi_pulse_ref_default
from .qub_probe import make_qub_probe_default, make_qub_probe_ref_default
from .qub_waveform import make_qub_waveform_default, make_qub_waveform_ref_default
from .readout import (
    make_direct_readout_default,
    make_pulse_readout_default,
    make_readout_default,
    make_readout_dpm_default,
    make_readout_ref_default,
)
from .res_probe import make_res_probe_default, make_res_probe_ref_default
from .res_waveform import make_res_waveform_default, make_res_waveform_ref_default
from .reset import (
    make_bath_reset_default,
    make_none_reset_default,
    make_pulse_reset_default,
    make_reset_default,
    make_reset_ref_default,
    make_two_pulse_reset_default,
)

if TYPE_CHECKING:
    from zcu_tools.gui.app.main.adapter import (
        ExpContext,
        ModuleRefValue,
        WaveformRefValue,
    )

# A blank factory always produces a value (never None); a ref factory's optional
# path may return None (the disabled-optional ref, ADR-0010).
_RefNode = Union["ModuleRefValue", "WaveformRefValue"]
BlankFactory = Callable[["ExpContext"], _RefNode]
RefFactory = Callable[..., _RefNode | None]


@dataclass(frozen=True)
class RoleFactorySpec:
    """The L2 factory pair for one role.

    ``ref`` is ``None`` for roles that have no library-aware variant (a concrete
    shape that is always built inline — e.g. ``pulse_readout``, ``bath_reset``).
    Asking such a role for a ref/optional mount is a Fast-Fail at the call site.
    """

    blank: BlankFactory
    ref: RefFactory | None = None


# role_id -> factory pair. Order is informational only (RoleCatalog defines its
# own dropdown order in registry.py).
ROLE_FACTORIES: dict[str, RoleFactorySpec] = {
    # qubit / resonator probe pulses
    "qub_probe": RoleFactorySpec(make_qub_probe_default, make_qub_probe_ref_default),
    "res_probe": RoleFactorySpec(make_res_probe_default, make_res_probe_ref_default),
    "pi_pulse": RoleFactorySpec(make_pi_pulse_default, make_pi_pulse_ref_default),
    "pi2_pulse": RoleFactorySpec(make_pi2_pulse_default, make_pi2_pulse_ref_default),
    # readout (role "readout" is library-aware; the concrete shapes are blank-only)
    "readout": RoleFactorySpec(make_readout_default, make_readout_ref_default),
    "pulse_readout": RoleFactorySpec(make_pulse_readout_default),
    "direct_readout": RoleFactorySpec(make_direct_readout_default),
    "readout_dpm": RoleFactorySpec(make_readout_dpm_default),
    # reset (role "reset" is library-aware; the concrete shapes are blank-only)
    "reset": RoleFactorySpec(make_reset_default, make_reset_ref_default),
    "none_reset": RoleFactorySpec(make_none_reset_default),
    "pulse_reset": RoleFactorySpec(make_pulse_reset_default),
    "two_pulse_reset": RoleFactorySpec(make_two_pulse_reset_default),
    "bath_reset": RoleFactorySpec(make_bath_reset_default),
    # waveforms
    "qub_waveform": RoleFactorySpec(
        make_qub_waveform_default, make_qub_waveform_ref_default
    ),
    "res_waveform": RoleFactorySpec(
        make_res_waveform_default, make_res_waveform_ref_default
    ),
}
