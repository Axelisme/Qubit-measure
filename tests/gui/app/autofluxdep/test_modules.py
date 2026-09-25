"""Module-space tests — Node-produced modules flowing to downstream Nodes.

Proves the parallel module space: a provider declares ``provides_modules`` and
emits a module via ``patch.set_module``; a downstream provider declaring
``ModuleDep`` reads it latest-available (this point → previous point → ml preset
→ default). This is the opt_readout flow, in the module space instead of info.
"""

from __future__ import annotations

from zcu_tools.gui.app.autofluxdep.nodes.io import Patch
from zcu_tools.gui.app.autofluxdep.nodes.spec import ModuleDep, Need
from zcu_tools.gui.app.autofluxdep.orchestrator import (
    InfoStore,
    Orchestrator,
    resolve_provider_snapshot,
)

from ._helpers import make_builder, place


class _FakeML:
    """A minimal ModuleSource: a dict of preset modules."""

    def __init__(self, presets: dict) -> None:
        self._presets = presets

    def get_module(self, name: str):
        return self._presets.get(name)


# --- resolve_provider_snapshot module resolution order ---


def test_module_resolves_node_produced_over_ml_preset():
    p = place(make_builder("n", optional_modules=(ModuleDep("readout"),)))
    info = InfoStore(module_point={"readout": "tuned"})
    ml = _FakeML({"readout": "preset"})
    snap = resolve_provider_snapshot(p, info, ml).snapshot
    assert snap is not None
    assert snap.module("readout") == "tuned"  # Node-produced wins


def test_module_falls_back_to_ml_preset():
    p = place(make_builder("n", optional_modules=(ModuleDep("readout"),)))
    info = InfoStore()  # no Node produced it
    ml = _FakeML({"readout": "preset"})
    snap = resolve_provider_snapshot(p, info, ml).snapshot
    assert snap is not None
    assert snap.module("readout") == "preset"


def test_module_falls_back_to_declared_default():
    p = place(
        make_builder(
            "n", optional_modules=(ModuleDep("readout", default=lambda: "fallback"),)
        )
    )
    snap = resolve_provider_snapshot(p, InfoStore(), _FakeML({})).snapshot
    assert snap is not None
    assert snap.module("readout") == "fallback"


def test_required_module_missing_everywhere_skips_node():
    p = place(make_builder("n", requires_modules=(ModuleDep("readout"),)))
    # no Node, no ml preset, no default → skip
    assert resolve_provider_snapshot(p, InfoStore(), _FakeML({})).snapshot is None


def test_module_prev_point_when_not_produced_this_point():
    p = place(make_builder("n", optional_modules=(ModuleDep("readout"),)))
    info = InfoStore(module_prev={"readout": "from_prev"})
    snap = resolve_provider_snapshot(p, info, _FakeML({})).snapshot
    assert snap is not None
    assert snap.module("readout") == "from_prev"


def test_module_now_only_does_not_use_prev_point():
    p = place(
        make_builder(
            "n",
            optional_modules=(
                ModuleDep("readout", need=Need.NOW, default=lambda: None),
            ),
        )
    )
    info = InfoStore(module_prev={"readout": "from_prev"})
    snap = resolve_provider_snapshot(p, info, _FakeML({})).snapshot

    assert snap is not None
    assert snap.module("readout") is None


# --- end-to-end: producer Node → downstream Node reads tuned module ---


def test_produced_module_flows_to_downstream_node():
    # ro_optimize-like producer emits a tuned readout module; a consumer reads it.
    seen: list = []

    def tune(_env, snap):
        base = snap.module("readout")  # reads base/ml/default
        return Patch(modules={"readout": f"tuned({base})"})  # produces tuned

    def consume(_env, snap):
        seen.append(snap.module("readout"))
        return Patch()

    producer = place(
        make_builder(
            "ro",
            provides_modules=("readout",),
            optional_modules=(ModuleDep("readout", default=lambda: "base"),),
            produce_fn=tune,
        )
    )
    consumer = place(
        make_builder(
            "cons", optional_modules=(ModuleDep("readout"),), produce_fn=consume
        )
    )
    ml = _FakeML({"readout": "preset"})

    # producer first → consumer sees THIS point's tuned
    Orchestrator([producer, consumer], ml=ml).run([0.0, 1.0])

    # point0: ro reads ml preset → tuned(preset); consumer sees tuned(preset)
    # point1: ro's module_point empty at ro time, module_prev has tuned(preset)
    #         → tuned(tuned(preset)); consumer sees that.
    assert seen == ["tuned(preset)", "tuned(tuned(preset))"]
