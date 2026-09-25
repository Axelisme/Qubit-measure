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
from zcu_tools.gui.app.autofluxdep.derivation import (
    SmoothConflictError,
    SmoothingService,
    SmoothRule,
)
from zcu_tools.gui.app.autofluxdep.nodes.io import Patch
from zcu_tools.gui.app.autofluxdep.nodes.spec import Dependency
from zcu_tools.gui.app.autofluxdep.orchestrator import Orchestrator

from ._helpers import make_builder, place

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
    raw = {0: 10.0, 1: 20.0, 2: 5.0}
    smoothed_seen: list = []
    raw_seen: list = []

    def produce_t1(env, _snap):
        return Patch({"t1": raw[env.flux_idx]})  # HONEST raw

    def read_smoothed(_env, snap):
        smoothed_seen.append(snap.get("t1"))
        return Patch()

    def read_raw(_env, snap):
        raw_seen.append(snap.get("t1"))
        return Patch()

    producer = place(make_builder("t1node", provides=("t1",), produce_fn=produce_t1))
    smoothed_consumer = place(
        make_builder(
            "sc",
            optional=(Dependency("t1", smooth="ewma", default=lambda: None),),
            produce_fn=read_smoothed,
        )
    )
    raw_consumer = place(
        make_builder(
            "rc",
            optional=(Dependency("t1", default=lambda: None),),
            produce_fn=read_raw,
        )
    )

    # producer first so consumers see this point's value
    info = Orchestrator([producer, smoothed_consumer, raw_consumer]).run(
        [0.0, 1.0, 2.0]
    )

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


def test_smoothed_projection_carries_last_good_across_absent_raw_point():
    raw = {0: 10.0, 2: 30.0}
    smoothed_seen: list = []

    def produce_t1(env, _snap):
        if env.flux_idx not in raw:
            return Patch()
        return Patch({"t1": raw[env.flux_idx]})

    def read_smoothed(_env, snap):
        smoothed_seen.append(snap.get("t1"))
        return Patch()

    producer = place(make_builder("t1node", provides=("t1",), produce_fn=produce_t1))
    consumer = place(
        make_builder(
            "consumer",
            optional=(Dependency("t1", smooth="ewma", default=lambda: None),),
            produce_fn=read_smoothed,
        )
    )

    info = Orchestrator([producer, consumer]).run([0.0, 1.0, 2.0])

    assert smoothed_seen == [None, 10.0, 10.0]
    assert info.prev_smoothed["t1"] == 10.0
    assert info.point_smoothed["t1"] == 20.0


def test_orchestrator_raises_on_conflicting_declarations():
    a = place(
        make_builder(
            "a", optional=(Dependency("x", smooth="ewma", default=lambda: None),)
        )
    )
    b = place(
        make_builder(
            "b",
            optional=(Dependency("x", smooth="step_weighted", default=lambda: None),),
        )
    )
    with pytest.raises(SmoothConflictError):
        Orchestrator([a, b])  # conflict detected at construction (__post_init__)
