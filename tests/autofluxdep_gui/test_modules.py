"""Module-space tests — Node-produced modules flowing to downstream Nodes.

Proves the parallel module space: a Node declares ``provides_modules`` and emits
a module via ``patch.set_module``; a downstream Node declaring ``ModuleDep``
reads it latest-available (this point → previous point → ml preset → default).
This is the opt_readout flow, now in the module space instead of info.
"""

from __future__ import annotations

from zcu_tools.gui.app.autofluxdep.controller import Controller
from zcu_tools.gui.app.autofluxdep.event_bus import EventBus
from zcu_tools.gui.app.autofluxdep.nodes.io import Patch
from zcu_tools.gui.app.autofluxdep.nodes.spec import (
    ModuleDep,
    NodeInstance,
    NodeSpec,
)
from zcu_tools.gui.app.autofluxdep.orchestrator import InfoStore, project_snapshot
from zcu_tools.gui.app.autofluxdep.state import AutoFluxDepState


class _FakeML:
    """A minimal ModuleSource: a dict of preset modules."""

    def __init__(self, presets: dict) -> None:
        self._presets = presets

    def get_module(self, name: str):
        return self._presets.get(name)


# --- project_snapshot module resolution order ---


def test_module_resolves_node_produced_over_ml_preset():
    spec = NodeSpec(name="n", provides=(), optional_modules=(ModuleDep("readout"),))
    info = InfoStore(module_point={"readout": "tuned"})
    ml = _FakeML({"readout": "preset"})
    snap = project_snapshot(NodeInstance(spec), info, ml)
    assert snap is not None
    assert snap.module("readout") == "tuned"  # Node-produced wins


def test_module_falls_back_to_ml_preset():
    spec = NodeSpec(name="n", provides=(), optional_modules=(ModuleDep("readout"),))
    info = InfoStore()  # no Node produced it
    ml = _FakeML({"readout": "preset"})
    snap = project_snapshot(NodeInstance(spec), info, ml)
    assert snap is not None
    assert snap.module("readout") == "preset"


def test_module_falls_back_to_declared_default():
    spec = NodeSpec(
        name="n",
        provides=(),
        optional_modules=(ModuleDep("readout", default=lambda: "fallback"),),
    )
    snap = project_snapshot(NodeInstance(spec), InfoStore(), _FakeML({}))
    assert snap is not None
    assert snap.module("readout") == "fallback"


def test_required_module_missing_everywhere_skips_node():
    spec = NodeSpec(name="n", provides=(), requires_modules=(ModuleDep("readout"),))
    # no Node, no ml preset, no default → skip
    assert project_snapshot(NodeInstance(spec), InfoStore(), _FakeML({})) is None


def test_module_prev_point_when_not_produced_this_point():
    spec = NodeSpec(name="n", provides=(), optional_modules=(ModuleDep("readout"),))
    info = InfoStore(module_prev={"readout": "from_prev"})
    snap = project_snapshot(NodeInstance(spec), info, _FakeML({}))
    assert snap is not None
    assert snap.module("readout") == "from_prev"


# --- end-to-end: producer Node → downstream Node reads tuned module ---


def test_produced_module_flows_to_downstream_node():
    # ro_optimize-like producer emits a tuned readout module; a consumer reads it.
    producer = NodeSpec(
        name="ro",
        provides=(),
        provides_modules=("readout",),
        optional_modules=(ModuleDep("readout", default=lambda: "base"),),
    )
    consumer = NodeSpec(
        name="cons", provides=(), optional_modules=(ModuleDep("readout"),)
    )
    seen: list = []

    def run_node(node: NodeInstance, snap, _tools) -> Patch:
        if node.name == "ro":
            base = snap.module("readout")  # reads base/ml/default
            return Patch(modules={"readout": f"tuned({base})"})  # produces tuned
        seen.append(snap.module("readout"))
        return Patch()

    state = AutoFluxDepState(flux_values=[0.0, 1.0])
    ctrl = Controller(state, EventBus())
    ctrl.add_node(producer)  # producer first → consumer sees THIS point's tuned
    ctrl.add_node(consumer)
    ml = _FakeML({"readout": "preset"})

    ctrl.dry_run(run_node, ml=ml)

    # point0: ro reads ml preset → tuned(preset); consumer sees tuned(preset)
    # point1: ro reads THIS-point? no, module_point cleared at begin; it reads
    #         prev tuned... actually ro's own ModuleDep resolves latest: at
    #         point1 module_point empty at ro time, module_prev has tuned(preset)
    #         → tuned(tuned(preset)); consumer sees that.
    assert seen == ["tuned(preset)", "tuned(tuned(preset))"]
