"""Derivation tests — consumer-declared smoothing collected by the orchestrator.

A consumer declares smoothing on the dependency itself: ``Dependency("t1",
smooth="ewma")``. The orchestrator collects every such declaration, dedups by
key, builds one SmoothingService, and runs it after the Nodes each point. The
smoothed value is projected under the SAME key into ``info.point_smoothed`` — a
smoothing consumer reads the smoothed estimate, a plain consumer of the same key
reads the raw value. The producer Node reports RAW only.
"""

from __future__ import annotations

import pytest
from zcu_tools.gui.app.autofluxdep.controller import Controller
from zcu_tools.gui.app.autofluxdep.derivation import (
    SmoothConflictError,
    SmoothingService,
    SmoothRule,
)
from zcu_tools.gui.app.autofluxdep.event_bus import EventBus
from zcu_tools.gui.app.autofluxdep.nodes.io import Patch
from zcu_tools.gui.app.autofluxdep.nodes.spec import (
    Dependency,
    NodeInstance,
    NodeSpec,
)
from zcu_tools.gui.app.autofluxdep.state import AutoFluxDepState

# --- SmoothingService construction + dedup ---


def test_from_specs_builds_one_rule_per_key():
    svc = SmoothingService.from_specs([("t1", "ewma")])
    assert svc.rules == (SmoothRule("t1", "ewma"),)
    assert svc.provides() == ("t1",)


def test_from_specs_dedups_identical_declarations():
    svc = SmoothingService.from_specs([("t1", "ewma"), ("t1", "ewma")])
    assert svc.rules == (SmoothRule("t1", "ewma"),)


def test_from_specs_conflicting_mode_raises():
    with pytest.raises(SmoothConflictError, match="t1"):
        SmoothingService.from_specs([("t1", "ewma"), ("t1", "step_weighted")])


# --- SmoothingService.derive ---


def test_derive_emits_smoothed_and_skips_absent_raw():
    svc = SmoothingService.from_specs([("t1", "ewma")])
    assert svc.derive({"flux_idx": 0, "t1": 10.0}) == {"t1": 10.0}
    assert svc.derive({"flux_idx": 1, "t1": 20.0}) == {"t1": 15.0}
    assert svc.derive({"flux_idx": 2}) == {}  # raw absent → no emit, no advance
    assert svc.derive({"flux_idx": 3, "t1": 30.0}) == {"t1": 22.5}  # vs 15


# --- orchestrator auto-collects, same-key smoothed projection ---


def test_smoothed_and_raw_consumers_coexist_under_same_key():
    # producer reports raw t1; one consumer wants it smoothed, another raw.
    producer = NodeSpec(name="t1node", provides=("t1",))
    smoothed_consumer = NodeSpec(
        name="sc",
        provides=(),
        optional=(Dependency("t1", smooth="ewma", default=lambda: None),),
    )
    raw_consumer = NodeSpec(
        name="rc", provides=(), optional=(Dependency("t1", default=lambda: None),)
    )
    raw = {0: 10.0, 1: 20.0, 2: 5.0}
    smoothed_seen: list = []
    raw_seen: list = []

    def run_node(node: NodeInstance, snap, _tools) -> Patch:
        if node.name == "t1node":
            return Patch({"t1": raw[node_idx[0]]})  # HONEST raw
        if node.name == "sc":
            smoothed_seen.append(snap.get("t1"))
        else:
            raw_seen.append(snap.get("t1"))
        return Patch()

    node_idx = [0]

    def pre_point(idx, _flux, _info, _tools):
        node_idx[0] = idx

    state = AutoFluxDepState(flux_values=[0.0, 1.0, 2.0])
    ctrl = Controller(state, EventBus())
    # producer first so consumers see this point's value
    ctrl.add_node(producer)
    ctrl.add_node(smoothed_consumer)
    ctrl.add_node(raw_consumer)

    info = ctrl.dry_run(run_node, pre_point=pre_point)

    # smoothed consumer reads the PREVIOUS point's smoothed t1 (this point's is
    # derived only after all Nodes run):
    #   point0: no prev → default None
    #   point1: prev smoothed = 10 (raw 10 at point0)
    #   point2: prev smoothed = 15 (0.5*10 + 0.5*20 at point1)
    assert smoothed_seen == [None, 10.0, 15.0]
    # raw consumer reads THIS point's raw t1: 10, 20, 5
    assert raw_seen == [10.0, 20.0, 5.0]
    assert info.point["t1"] == 5.0
    assert info.point_smoothed["t1"] == 10.0  # 0.5*15 + 0.5*5 at point2


def test_orchestrator_raises_on_conflicting_declarations():
    a = NodeSpec(
        name="a",
        provides=(),
        optional=(Dependency("x", smooth="ewma", default=lambda: None),),
    )
    b = NodeSpec(
        name="b",
        provides=(),
        optional=(Dependency("x", smooth="step_weighted", default=lambda: None),),
    )
    state = AutoFluxDepState(flux_values=[0.0])
    ctrl = Controller(state, EventBus())
    ctrl.add_node(a)
    ctrl.add_node(b)

    def run_node(node: NodeInstance, _snap, _tools) -> Patch:
        return Patch()

    with pytest.raises(SmoothConflictError):
        ctrl.dry_run(run_node)
