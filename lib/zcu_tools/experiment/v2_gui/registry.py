"""Register the v2 experiment-adapter layer into the GUI framework.

The GUI defines the ``Registry`` (experiment adapters) and ``RoleCatalog``
(module/waveform role templates) interfaces; this module — in the experiment
layer, which may import gui — fills both. Wired once at startup by the entry
script (``run_gui.py``), which builds the empty containers and passes them in.

Role kinds (``RoleCatalog``):

- **md-aware** (``res_probe``, ``bath_reset``, …): the eval-aware
  ``make_<role>_default`` builders — seed a fresh blank with md-linked defaults.
- **blank** (``<discriminator>:blank``): a plain structural-zero blank of one
  concrete shape, for shapes that have no md-aware role (bare ``pulse``;
  ``drag``/``flat_top``/``gauss``/``arb`` waveforms) or when a literal blank is
  wanted. This is the single create path for *any* shape.

(The ``_ref`` factory variants are intentionally not used: creating from a role
seeds a fresh entry, it does not reference an existing library entry.)
"""

from __future__ import annotations

from typing import Callable, Union

from zcu_tools.gui.adapter import (
    ExpContext,
    ModuleRefValue,
    WaveformRefValue,
    make_default_value,
)
from zcu_tools.gui.cfg_schemas import _MODULE_SPEC_FACTORIES
from zcu_tools.gui.registry import Registry
from zcu_tools.gui.role_catalog import RoleCatalog, RoleEntry
from zcu_tools.gui.specs import make_waveform_spec_by_style

from .adapters.fake.freq import FakeFreqAdapter
from .adapters.lookback import LookbackAdapter
from .adapters.onetone.flux_dep import OneToneFluxDepAdapter
from .adapters.onetone.freq import OneToneFreqAdapter
from .adapters.onetone.power_dep import OneTonePowerDepAdapter
from .adapters.shared import (
    make_bath_reset_default,
    make_direct_readout_default,
    make_none_reset_default,
    make_pi2_pulse_default,
    make_pi_pulse_default,
    make_pulse_readout_default,
    make_pulse_reset_default,
    make_qub_probe_default,
    make_qub_waveform_default,
    make_readout_dpm_default,
    make_res_probe_default,
    make_res_waveform_default,
    make_two_pulse_reset_default,
)
from .adapters.twotone.flux_dep import FluxDepAdapter
from .adapters.twotone.freq import FreqAdapter
from .adapters.twotone.power_dep import PowerDepAdapter
from .adapters.twotone.rabi.amp_rabi import AmpRabiAdapter
from .adapters.twotone.rabi.len_rabi import LenRabiAdapter
from .adapters.twotone.ro_optimize import (
    RoOptAutoAdapter,
    RoOptFreqAdapter,
    RoOptFreqGainAdapter,
    RoOptLengthAdapter,
    RoOptPowerAdapter,
)
from .adapters.twotone.time_domain.t1 import T1Adapter
from .adapters.twotone.time_domain.t2echo import T2EchoAdapter
from .adapters.twotone.time_domain.t2ramsey import T2RamseyAdapter

# --- experiment adapters -------------------------------------------------

ADAPTERS = {
    "lookback": LookbackAdapter,
    "fake/freq": FakeFreqAdapter,
    "onetone/freq": OneToneFreqAdapter,
    "onetone/power_dep": OneTonePowerDepAdapter,
    "onetone/flux_dep": OneToneFluxDepAdapter,
    "twotone/freq": FreqAdapter,
    "twotone/power_dep": PowerDepAdapter,
    "twotone/flux_dep": FluxDepAdapter,
    "twotone/rabi/amp_rabi": AmpRabiAdapter,
    "twotone/rabi/len_rabi": LenRabiAdapter,
    "twotone/ro_optimize/freq": RoOptFreqAdapter,
    "twotone/ro_optimize/power": RoOptPowerAdapter,
    "twotone/ro_optimize/length": RoOptLengthAdapter,
    "twotone/ro_optimize/freq_gain": RoOptFreqGainAdapter,
    "twotone/ro_optimize/auto": RoOptAutoAdapter,
    "twotone/t1": T1Adapter,
    "twotone/t2ramsey": T2RamseyAdapter,
    "twotone/t2echo": T2EchoAdapter,
}


def register_all(registry: Registry) -> None:
    for name, cls in ADAPTERS.items():
        registry.register(name, cls)


# --- role catalog --------------------------------------------------------

# Blank shapes that have no md-aware role of their own. Module discriminators are
# the create-able ml module types; waveform discriminators are the styles.
_BLANK_MODULE_DISCRIMINATORS = list(_MODULE_SPEC_FACTORIES.keys())
_BLANK_WAVEFORM_DISCRIMINATORS = [
    "const",
    "cosine",
    "gauss",
    "drag",
    "flat_top",
    "arb",
]


def _blank_module_factory(
    disc: str,
) -> Callable[[ExpContext], ModuleRefValue]:
    def _make(_ctx: ExpContext) -> ModuleRefValue:
        value = make_default_value(_MODULE_SPEC_FACTORIES[disc]())
        return ModuleRefValue(f"<Custom:{disc}>", value)

    return _make


def _blank_waveform_factory(
    disc: str,
) -> Callable[[ExpContext], WaveformRefValue]:
    def _make(_ctx: ExpContext) -> WaveformRefValue:
        value = make_default_value(make_waveform_spec_by_style(disc))
        return WaveformRefValue(f"<Custom:{disc}>", value)

    return _make


def _blank_entries() -> list[RoleEntry]:
    entries: list[RoleEntry] = []
    factory: Callable[[ExpContext], Union[ModuleRefValue, WaveformRefValue]]
    for disc in _BLANK_MODULE_DISCRIMINATORS:
        factory = _blank_module_factory(disc)
        entries.append(RoleEntry(f"{disc}:blank", f"Blank: {disc}", "module", factory))
    for disc in _BLANK_WAVEFORM_DISCRIMINATORS:
        factory = _blank_waveform_factory(disc)
        entries.append(
            RoleEntry(f"{disc}:blank", f"Blank: {disc}", "waveform", factory)
        )
    return entries


# Insertion order = dropdown order. Readout/probe first, then pulses, then
# resets, then waveforms.
ROLE_ENTRIES: list[RoleEntry] = [
    RoleEntry("res_probe", "Resonator probe", "module", make_res_probe_default),
    RoleEntry("pulse_readout", "Pulse readout", "module", make_pulse_readout_default),
    RoleEntry(
        "readout_dpm", "Optimized readout (DPM)", "module", make_readout_dpm_default
    ),
    RoleEntry(
        "direct_readout", "Direct readout", "module", make_direct_readout_default
    ),
    RoleEntry("qub_probe", "Qubit probe pulse", "module", make_qub_probe_default),
    RoleEntry("pi_pulse", "Pi pulse", "module", make_pi_pulse_default),
    RoleEntry("pi2_pulse", "Pi/2 pulse", "module", make_pi2_pulse_default),
    RoleEntry("none_reset", "No reset", "module", make_none_reset_default),
    RoleEntry("pulse_reset", "Pulse reset", "module", make_pulse_reset_default),
    RoleEntry(
        "two_pulse_reset", "Two-pulse reset", "module", make_two_pulse_reset_default
    ),
    RoleEntry("bath_reset", "Bath reset", "module", make_bath_reset_default),
    RoleEntry(
        "qub_waveform", "Qubit drive waveform", "waveform", make_qub_waveform_default
    ),
    RoleEntry(
        "res_waveform", "Res-probe waveform", "waveform", make_res_waveform_default
    ),
]


# md-aware roles first, then the structural-blank roles (dropdown groups
# "named" roles on top, raw blanks below).
ALL_ROLE_ENTRIES: list[RoleEntry] = [*ROLE_ENTRIES, *_blank_entries()]


def register_all_roles(catalog: RoleCatalog) -> None:
    for entry in ALL_ROLE_ENTRIES:
        catalog.register(entry)
