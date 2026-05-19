from __future__ import annotations

from zcu_tools.gui.registry import Registry

from .adapters.fake import FakeAdapter
from .adapters.onetone.freq import FakeFreqAdapter

ADAPTERS = {
    "fake": FakeAdapter,
    "onetone/freq": FakeFreqAdapter,
}


def register_all(registry: Registry) -> None:
    for name, cls in ADAPTERS.items():
        registry.register(name, cls)
