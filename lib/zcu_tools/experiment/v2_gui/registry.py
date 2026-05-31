from __future__ import annotations

from zcu_tools.gui.registry import Registry

from .adapters.fake.freq import FakeFreqAdapter
from .adapters.lookback import LookbackAdapter
from .adapters.onetone.flux_dep import OneToneFluxDepAdapter
from .adapters.onetone.freq import OneToneFreqAdapter
from .adapters.onetone.power_dep import OneTonePowerDepAdapter
from .adapters.twotone.flux_dep import FluxDepAdapter
from .adapters.twotone.freq import FreqAdapter
from .adapters.twotone.power_dep import PowerDepAdapter
from .adapters.twotone.rabi.amp_rabi import AmpRabiAdapter
from .adapters.twotone.rabi.len_rabi import LenRabiAdapter
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
    "twotone/power_dep": PowerDepAdapter,
    "twotone/flux_dep": FluxDepAdapter,
    "twotone/rabi/amp_rabi": AmpRabiAdapter,
    "twotone/rabi/len_rabi": LenRabiAdapter,
    "twotone/t1": T1Adapter,
    "twotone/t2ramsey": T2RamseyAdapter,
    "twotone/t2echo": T2EchoAdapter,
}


def register_all(registry: Registry) -> None:
    for name, cls in ADAPTERS.items():
        registry.register(name, cls)
