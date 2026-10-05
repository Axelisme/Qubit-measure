"""Explicit user-owned measure catalog; importing it performs no registration."""

from typing import Any

from zcu_tools.gui.app.measure.role_catalog import RoleCatalog
from zcu_lab.roles import register_all_roles

from zcu_tools.gui.app.measure.adapter.base import BaseAdapter
from zcu_tools.gui.app.measure.registry import Registry

from zcu_lab.v2.fake.freq.gui import FakeFreqAdapter
from zcu_lab.v2.jpa.auto_optimize.gui import JpaAutoOptimizeAdapter
from zcu_lab.v2.jpa.check.gui import JpaCheckAdapter
from zcu_lab.v2.jpa.flux.gui import JpaFluxAdapter
from zcu_lab.v2.jpa.flux_onetone.gui import JpaFluxOneToneAdapter
from zcu_lab.v2.jpa.freq.gui import JpaFreqAdapter
from zcu_lab.v2.jpa.power.gui import JpaPowerAdapter
from zcu_lab.v2.lookback.gui import LookbackAdapter
from zcu_lab.v2.onetone.flux_dep.gui import OneToneFluxDepAdapter
from zcu_lab.v2.onetone.freq.gui import OneToneFreqAdapter
from zcu_lab.v2.onetone.power_dep.gui import OneTonePowerDepAdapter
from zcu_lab.v2.singleshot.check.gui import CheckAdapter
from zcu_lab.v2.singleshot.ge.gui import GEAdapter
from zcu_lab.v2.singleshot.mist.freq.gui import MistFreqAdapter
from zcu_lab.v2.singleshot.mist.power.gui import MistPowerAdapter
from zcu_lab.v2.singleshot.mist.power_freq.gui import MistPowerFreqAdapter
from zcu_lab.v2.singleshot.ac_stark.gui import SsAcStarkAdapter
from zcu_lab.v2.singleshot.amp_rabi.gui import SsAmpRabiAdapter
from zcu_lab.v2.singleshot.len_rabi.gui import SsLenRabiAdapter
from zcu_lab.v2.singleshot.reset_check.gui import SsResetCheckAdapter
from zcu_lab.v2.singleshot.t1.t1.gui import SsT1Adapter
from zcu_lab.v2.singleshot.t1.t1_with_tone.gui import SsT1ToneAdapter
from zcu_lab.v2.singleshot.t1.t1_with_tone_sweep.gui import SsT1ToneSweepFreqAdapter
from zcu_lab.v2.singleshot.t1.t1_with_tone_sweep.gui import SsT1ToneSweepGainAdapter
from zcu_lab.v2.twotone.ckp.gui import CKPAdapter
from zcu_lab.v2.twotone.fluxdep.gui import FluxDepAdapter
from zcu_lab.v2.twotone.freq.gui import FreqAdapter
from zcu_lab.v2.twotone.power_dep.gui import PowerDepAdapter
from zcu_lab.v2.twotone.rabi.amp_rabi.gui import AmpRabiAdapter
from zcu_lab.v2.twotone.rabi.len_rabi.gui import LenRabiAdapter
from zcu_lab.v2.twotone.reset.bath.freq.gui import BathFreqGainAdapter
from zcu_lab.v2.twotone.reset.bath.length.gui import BathLengthAdapter
from zcu_lab.v2.twotone.reset.bath.phase.gui import BathPhaseAdapter
from zcu_lab.v2.twotone.reset.rabi_check.gui import RabiCheckAdapter
from zcu_lab.v2.twotone.reset.dual_tone.freq.gui import DualToneFreqAdapter
from zcu_lab.v2.twotone.reset.dual_tone.length.gui import DualToneLengthAdapter
from zcu_lab.v2.twotone.reset.dual_tone.power.gui import DualTonePowerAdapter
from zcu_lab.v2.twotone.reset.single_tone.freq.gui import SingleToneFreqAdapter
from zcu_lab.v2.twotone.reset.single_tone.length.gui import SingleToneLengthAdapter
from zcu_lab.v2.twotone.ro_optimize.auto_optimize.gui import RoOptAutoAdapter
from zcu_lab.v2.twotone.ro_optimize.freq.gui import RoOptFreqAdapter
from zcu_lab.v2.twotone.ro_optimize.freq_gain.gui import RoOptFreqGainAdapter
from zcu_lab.v2.twotone.ro_optimize.length.gui import RoOptLengthAdapter
from zcu_lab.v2.twotone.ro_optimize.power.gui import RoOptPowerAdapter
from zcu_lab.v2.twotone.time_domain.t1.gui import T1Adapter
from zcu_lab.v2.twotone.time_domain.t2echo.gui import T2EchoAdapter
from zcu_lab.v2.twotone.time_domain.t2ramsey.gui import T2RamseyAdapter

ADAPTERS: dict[str, type[BaseAdapter[Any, Any, Any, Any]]] = {
    "lookback": LookbackAdapter,
    "fake/freq": FakeFreqAdapter,
    "onetone/freq": OneToneFreqAdapter,
    "onetone/power_dep": OneTonePowerDepAdapter,
    "onetone/flux_dep": OneToneFluxDepAdapter,
    "twotone/freq": FreqAdapter,
    "twotone/ckp": CKPAdapter,
    "twotone/power_dep": PowerDepAdapter,
    "twotone/flux_dep": FluxDepAdapter,
    "twotone/rabi/amp_rabi": AmpRabiAdapter,
    "twotone/rabi/len_rabi": LenRabiAdapter,
    "twotone/reset/single_tone/freq": SingleToneFreqAdapter,
    "twotone/reset/single_tone/length": SingleToneLengthAdapter,
    "twotone/reset/dual_tone/freq": DualToneFreqAdapter,
    "twotone/reset/dual_tone/length": DualToneLengthAdapter,
    "twotone/reset/dual_tone/power": DualTonePowerAdapter,
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
    "singleshot/reset_check": SsResetCheckAdapter,
    "singleshot/len_rabi": SsLenRabiAdapter,
    "singleshot/amp_rabi": SsAmpRabiAdapter,
    "singleshot/t1": SsT1Adapter,
    "singleshot/t1_tone": SsT1ToneAdapter,
    "singleshot/t1_tone_sweep_gain": SsT1ToneSweepGainAdapter,
    "singleshot/t1_tone_sweep_freq": SsT1ToneSweepFreqAdapter,
    "singleshot/ac_stark": SsAcStarkAdapter,
    "singleshot/mist/freq": MistFreqAdapter,
    "singleshot/mist/power": MistPowerAdapter,
    "singleshot/mist/power_freq": MistPowerFreqAdapter,
    "jpa/freq": JpaFreqAdapter,
    "jpa/flux": JpaFluxAdapter,
    "jpa/power": JpaPowerAdapter,
    "jpa/auto_optimize": JpaAutoOptimizeAdapter,
    "jpa/flux_onetone": JpaFluxOneToneAdapter,
    "jpa/check": JpaCheckAdapter,
}


def register_all(registry: Registry, *, roles: RoleCatalog | None = None) -> None:
    """Register adapters into the caller-owned registry.

    Pass a caller-owned RoleCatalog only at startup to register program/module
    roles. Omit roles on reload so existing role identities remain fixed.
    Duplicate entries raise the corresponding catalog registration error.
    """
    if roles is not None:
        register_all_roles(roles)
    for name, cls in ADAPTERS.items():
        registry.register(name, cls)
