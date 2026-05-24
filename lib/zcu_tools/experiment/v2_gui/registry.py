from __future__ import annotations

from zcu_tools.gui.registry import Registry

from .adapters.fake import FakeAdapter
from .adapters.lookback import LookbackAdapter
from .adapters.onetone.fakefreq import FakeFreqAdapter
from .adapters.onetone.flux_dep import OneToneFluxDepAdapter
from .adapters.onetone.freq import OneToneFreqAdapter
from .adapters.onetone.power_dep import OneTonePowerDepAdapter

ADAPTERS = {
    "fake": FakeAdapter,
    "lookback": LookbackAdapter,
    "onetone/fake_freq": FakeFreqAdapter,
    "onetone/freq": OneToneFreqAdapter,
    "onetone/power_dep": OneTonePowerDepAdapter,
    "onetone/flux_dep": OneToneFluxDepAdapter,
}


def register_all(registry: Registry) -> None:
    for name, cls in ADAPTERS.items():
        registry.register(name, cls)
