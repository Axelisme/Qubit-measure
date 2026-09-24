"""Explicit, reloadable experiment catalog; startup roles live in role_registry."""

from zcu_tools.gui.app.main.registry import Registry

from .adapters.fake.freq import FakeFreqAdapter
from .adapters.jpa import (
    JpaAutoOptimizeAdapter,
    JpaCheckAdapter,
    JpaFluxAdapter,
    JpaFluxOneToneAdapter,
    JpaFreqAdapter,
    JpaPowerAdapter,
)
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
    SsAmpRabiAdapter,
    SsLenRabiAdapter,
    SsT1Adapter,
    SsT1ToneAdapter,
    SsT1ToneSweepFreqAdapter,
    SsT1ToneSweepGainAdapter,
)
from .adapters.twotone.ckp import CKPAdapter
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

ADAPTERS = {
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


def register_all(registry: Registry) -> None:
    for name, cls in ADAPTERS.items():
        registry.register(name, cls)
