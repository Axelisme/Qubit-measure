"""Register the v2 experiment-adapter layer into the GUI framework.

The GUI defines the ``Registry`` (experiment adapters) and ``RoleCatalog``
(module/waveform role templates) interfaces; this module — in the experiment
layer, which may import gui — fills both. Wired once at startup by the entry
script (``run_measure_gui.py``), which builds the empty containers and passes them in.

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

from collections.abc import Callable

from zcu_tools.gui.app.main.adapter import ExpContext
from zcu_tools.gui.app.main.cfg_schemas import _MODULE_SPEC_FACTORIES
from zcu_tools.gui.app.main.registry import Registry
from zcu_tools.gui.app.main.role_catalog import RoleCatalog, RoleEntry, RoleItemKind
from zcu_tools.gui.app.main.specs import make_waveform_spec_by_style
from zcu_tools.gui.cfg import (
    ReferenceValue,
    make_default_value,
)

from .adapters._support import ROLE_FACTORIES
from .adapters.fake.freq import FakeFreqAdapter
from .adapters.lookback import LookbackAdapter
from .adapters.onetone.flux_dep import OneToneFluxDepAdapter
from .adapters.onetone.freq import OneToneFreqAdapter
from .adapters.onetone.power_dep import OneTonePowerDepAdapter
from .adapters.singleshot import (
    CheckAdapter,
    GEAdapter,
    MistFreqAdapter,
    MistPowerAdapter,
    MistPowerFreqAdapter,
    SsAcStarkAdapter,
    SsLenRabiAdapter,
    SsT1Adapter,
    SsT1ToneAdapter,
    SsT1ToneSweepFreqAdapter,
    SsT1ToneSweepGainAdapter,
)
from .adapters.twotone.flux_dep import FluxDepAdapter
from .adapters.twotone.freq import FreqAdapter
from .adapters.twotone.power_dep import PowerDepAdapter
from .adapters.twotone.rabi.amp_rabi import AmpRabiAdapter
from .adapters.twotone.rabi.len_rabi import LenRabiAdapter
from .adapters.twotone.reset.bath import (
    BathFreqGainAdapter,
    BathLengthAdapter,
    BathPhaseAdapter,
)
from .adapters.twotone.reset.check import RabiCheckAdapter
from .adapters.twotone.reset.dual_tone import (
    DualToneFreqAdapter,
    DualToneLengthAdapter,
    DualTonePowerAdapter,
)
from .adapters.twotone.reset.single_tone import (
    SingleToneFreqAdapter,
    SingleToneLengthAdapter,
)
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
    "twotone/reset/single_tone/freq": SingleToneFreqAdapter,
    "twotone/reset/single_tone/length": SingleToneLengthAdapter,
    "twotone/reset/dual_tone/freq": DualToneFreqAdapter,
    "twotone/reset/dual_tone/power": DualTonePowerAdapter,
    "twotone/reset/dual_tone/length": DualToneLengthAdapter,
    "twotone/reset/bath/freq_gain": BathFreqGainAdapter,
    "twotone/reset/bath/length": BathLengthAdapter,
    "twotone/reset/bath/phase": BathPhaseAdapter,
    "twotone/reset/check": RabiCheckAdapter,
    "twotone/ro_optimize/freq": RoOptFreqAdapter,
    "twotone/ro_optimize/power": RoOptPowerAdapter,
    "twotone/ro_optimize/length": RoOptLengthAdapter,
    "twotone/ro_optimize/freq_gain": RoOptFreqGainAdapter,
    "twotone/ro_optimize/auto": RoOptAutoAdapter,
    "twotone/t1": T1Adapter,
    "twotone/t2ramsey": T2RamseyAdapter,
    "twotone/t2echo": T2EchoAdapter,
    "singleshot/ge": GEAdapter,
    "singleshot/check": CheckAdapter,
    "singleshot/len_rabi": SsLenRabiAdapter,
    "singleshot/t1": SsT1Adapter,
    "singleshot/t1_tone": SsT1ToneAdapter,
    "singleshot/t1_tone_sweep_gain": SsT1ToneSweepGainAdapter,
    "singleshot/t1_tone_sweep_freq": SsT1ToneSweepFreqAdapter,
    "singleshot/ac_stark": SsAcStarkAdapter,
    "singleshot/mist/freq": MistFreqAdapter,
    "singleshot/mist/power": MistPowerAdapter,
    "singleshot/mist/power_freq": MistPowerFreqAdapter,
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
) -> Callable[[ExpContext], ReferenceValue]:
    def _make(_ctx: ExpContext) -> ReferenceValue:
        value = make_default_value(_MODULE_SPEC_FACTORIES[disc]())
        return ReferenceValue(f"<Custom:{disc}>", value)

    return _make


def _blank_waveform_factory(
    disc: str,
) -> Callable[[ExpContext], ReferenceValue]:
    def _make(_ctx: ExpContext) -> ReferenceValue:
        value = make_default_value(make_waveform_spec_by_style(disc))
        return ReferenceValue(f"<Custom:{disc}>", value)

    return _make


def _blank_entries() -> list[RoleEntry]:
    entries: list[RoleEntry] = []
    factory: Callable[[ExpContext], ReferenceValue]
    for disc in _BLANK_MODULE_DISCRIMINATORS:
        factory = _blank_module_factory(disc)
        entries.append(RoleEntry(f"{disc}:blank", f"Blank: {disc}", "module", factory))
    for disc in _BLANK_WAVEFORM_DISCRIMINATORS:
        factory = _blank_waveform_factory(disc)
        entries.append(
            RoleEntry(f"{disc}:blank", f"Blank: {disc}", "waveform", factory)
        )
    return entries


# Catalog dropdown entries: (role_id, label, item_kind, default_name). Insertion
# order = dropdown order. Readout/probe first, then pulses, then resets, then
# waveforms. The factory is the role's *blank* builder, taken from the shared
# ROLE_FACTORIES table (single source) — creating from a role always seeds a fresh
# entry, never a library reference (the blank builder never adopts). The
# library-aware roles "readout" / "reset" therefore appear here too: their blank
# IS the pulse shape, so they double as the "create an inline pulse readout/reset"
# templates (ModuleInit only affects fresh adapter cfg materialization, not the catalog).
# default_name is the create dialog's naming-convention suggestion.
_CATALOG_ROLES: list[tuple[str, str, RoleItemKind, str]] = [
    ("res_probe", "Resonator probe", "module", "readout_rf"),
    ("readout", "Pulse readout", "module", "readout_rf"),
    ("readout_dpm", "Optimized readout (DPM)", "module", "readout_dpm"),
    ("direct_readout", "Direct readout", "module", "readout_direct"),
    ("qub_probe", "Qubit probe pulse", "module", "qub_pulse"),
    ("pi_pulse", "Pi pulse", "module", "pi_amp"),
    ("pi2_pulse", "Pi/2 pulse", "module", "pi2_amp"),
    ("none_reset", "No reset", "module", "reset_none"),
    ("reset", "Pulse reset", "module", "reset_10"),
    ("two_pulse_reset", "Two-pulse reset", "module", "reset_120"),
    ("bath_reset", "Bath reset", "module", "reset_bath"),
    ("qub_waveform", "Qubit drive waveform", "waveform", "qub_flat"),
    ("res_waveform", "Res-probe waveform", "waveform", "ro_waveform"),
]

ROLE_ENTRIES: list[RoleEntry] = [
    RoleEntry(role_id, label, kind, ROLE_FACTORIES[role_id].blank, default_name)
    for role_id, label, kind, default_name in _CATALOG_ROLES
]


# md-aware roles first, then the structural-blank roles (dropdown groups
# "named" roles on top, raw blanks below).
ALL_ROLE_ENTRIES: list[RoleEntry] = [*ROLE_ENTRIES, *_blank_entries()]


def register_all_roles(catalog: RoleCatalog) -> None:
    for entry in ALL_ROLE_ENTRIES:
        catalog.register(entry)
