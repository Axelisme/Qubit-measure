"""Skeleton tests — prove the dependency model resolves with latest-available.

No hardware, no Qt: a fake per-Node ``run_node`` returns each Node's declared
provides so we can assert latest-available resolution (this point else previous
point else default), user-ordered execution (no topo sort), required-missing
skip, and optional fallback.
"""

from __future__ import annotations

from zcu_tools.gui.app.autofluxdep.controller import Controller
from zcu_tools.gui.app.autofluxdep.event_bus import EventBus
from zcu_tools.gui.app.autofluxdep.nodes.io import Patch, Snapshot
from zcu_tools.gui.app.autofluxdep.nodes.qubit_freq import QUBIT_FREQ_SPEC
from zcu_tools.gui.app.autofluxdep.nodes.spec import (
    Dependency,
    NodeInstance,
    NodeSpec,
)
from zcu_tools.gui.app.autofluxdep.orchestrator import (
    InfoStore,
    Orchestrator,
    project_snapshot,
)
from zcu_tools.gui.app.autofluxdep.state import AutoFluxDepState

# --- latest-available resolution ---


def test_resolve_required_missing_no_default_returns_none():
    spec = NodeSpec(name="n", provides=(), requires=(Dependency("x"),))
    assert project_snapshot(NodeInstance(spec), InfoStore()) is None


def test_resolve_required_missing_with_default_keeps_node():
    spec = NodeSpec(
        name="n", provides=(), requires=(Dependency("x", default=lambda: 9),)
    )
    assert project_snapshot(NodeInstance(spec), InfoStore()) == {"x": 9}


def test_resolve_prefers_this_point_over_prev():
    spec = NodeSpec(name="n", provides=(), requires=(Dependency("x"),))
    info = InfoStore(point={"x": "now"}, prev={"x": "before"})
    assert project_snapshot(NodeInstance(spec), info) == {"x": "now"}


def test_resolve_falls_back_to_prev_point():
    spec = NodeSpec(name="n", provides=(), requires=(Dependency("x"),))
    info = InfoStore(prev={"x": "before"})  # not in this point
    assert project_snapshot(NodeInstance(spec), info) == {"x": "before"}


def test_resolve_optional_uses_default_when_absent_everywhere():
    spec = NodeSpec(
        name="n", provides=(), optional=(Dependency("k", default=lambda: 42),)
    )
    assert project_snapshot(NodeInstance(spec), InfoStore()) == {"k": 42}


def test_resolve_optional_prefers_value_over_default():
    spec = NodeSpec(
        name="n", provides=(), optional=(Dependency("k", default=lambda: 42),)
    )
    info = InfoStore(prev={"k": 7})
    assert project_snapshot(NodeInstance(spec), info) == {"k": 7}


def test_resolve_stored_none_is_a_value_not_missing():
    # a producer that wrote None should not trigger the default
    spec = NodeSpec(
        name="n", provides=(), optional=(Dependency("k", default=lambda: 42),)
    )
    info = InfoStore(point={"k": None})
    assert project_snapshot(NodeInstance(spec), info) == {"k": None}


# --- user-ordered execution (no topo sort) ---


def test_nodes_run_in_given_order():
    seen: list = []

    def run_node(node: NodeInstance, _snap, _tools) -> Patch:
        seen.append(node.name)
        return Patch()

    nodes = [NodeInstance(NodeSpec(name=n, provides=())) for n in ("c", "a", "b")]
    Orchestrator(nodes, run_node).run([0.0])
    assert seen == ["c", "a", "b"]  # exactly the declared order


def test_consumer_before_producer_reads_prev_point():
    # consumer is ordered BEFORE producer; with latest-available it reads the
    # previous point's value (None at point0), this is allowed (no DAG error).
    producer = NodeSpec(name="prod", provides=("x",))
    consumer = NodeSpec(
        name="cons", provides=(), optional=(Dependency("x", default=lambda: None),)
    )
    seen: list = []

    def run_node(node: NodeInstance, snap, _tools) -> Patch:
        if node.name == "prod":
            return Patch({"x": f"x@{node_idx[0]}"})
        seen.append(snap.get("x"))
        return Patch()

    node_idx = [0]

    def pre_point(idx, _flux, _info, _tools):
        node_idx[0] = idx

    # consumer FIRST, producer second
    Orchestrator([NodeInstance(consumer), NodeInstance(producer)], run_node).run(
        [0.0, 1.0, 2.0], pre_point=pre_point
    )

    # point0: no prev → default None; point1: prev x@0; point2: prev x@1
    assert seen == [None, "x@0", "x@1"]


def test_prev_snapshot_carries_across_points():
    producer = NodeSpec(name="prod", provides=("x",))
    consumer = NodeSpec(
        name="cons", provides=(), optional=(Dependency("x", default=lambda: None),)
    )
    seen: list = []

    def run_node(node: NodeInstance, snap, _tools) -> Patch:
        if node.name == "prod":
            return Patch({"x": node_idx[0]})
        seen.append(snap.get("x"))
        return Patch()

    node_idx = [0]

    def pre_point(idx, _flux, _info, _tools):
        node_idx[0] = idx

    # producer FIRST: consumer reads THIS point's value
    Orchestrator([NodeInstance(producer), NodeInstance(consumer)], run_node).run(
        [0.0, 1.0, 2.0], pre_point=pre_point
    )
    assert seen == [0, 1, 2]


# --- the worked qubit_freq Node ---


def test_qubit_freq_reports_raw_only():
    assert set(QUBIT_FREQ_SPEC.provides) == {"qubit_freq", "fit_detune", "fit_kappa"}


def test_qubit_freq_declares_fit_kappa_smoothed():
    smooth = {k: m for k, m in QUBIT_FREQ_SPEC.smooth_specs()}
    assert smooth == {"fit_kappa": "ewma"}


def test_qubit_freq_build_cfg_reads_snapshot_values_and_module():
    snap = Snapshot(
        {"predict_freq": 5000.0, "fit_kappa": 0.05},
        modules={"readout": "RO"},
    )
    assert QUBIT_FREQ_SPEC.build_cfg is not None
    cfg = QUBIT_FREQ_SPEC.build_cfg(snap, params={"reps": 500}, tools=None)
    assert cfg is not None
    assert cfg["modules"]["qub_pulse"]["freq"] == 5000.0
    assert cfg["modules"]["readout"] == "RO"  # the module, read via snapshot
    assert cfg["reps"] == 500


def test_qubit_freq_declares_readout_module_dep():
    assert [m.name for m in QUBIT_FREQ_SPEC.optional_modules] == ["readout"]


def test_controller_dry_run_end_to_end():
    state = AutoFluxDepState(flux_values=[0.0, 0.1])
    ctrl = Controller(state, EventBus())
    ctrl.add_node(QUBIT_FREQ_SPEC, detune_sweep=None, reps=1000)

    def run_node(node: NodeInstance, _snap, _tools) -> Patch:
        # fit_kappa must be numeric — the orchestrator auto-smooths it.
        return Patch(
            {
                k: (0.05 if k == "fit_kappa" else f"{node.name}:{k}")
                for k in node.spec.provides
            }
        )

    def pre_point(idx, _flux, info, _tools):
        info.point["predict_freq"] = 5000.0 + idx  # seed the required dep

    info = ctrl.dry_run(run_node, pre_point=pre_point)
    assert info.point["qubit_freq"] == "qubit_freq:qubit_freq"
    # the smoothed fit_kappa is projected into the smoothed store under same key
    assert info.point_smoothed["fit_kappa"] == 0.05
