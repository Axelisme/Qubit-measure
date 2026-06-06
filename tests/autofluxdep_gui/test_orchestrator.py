"""Dependency-model tests — the orchestrator resolves with latest-available.

No hardware, no Qt: small ad-hoc Builders (``make_builder``) whose Nodes script
a ``produce`` let us assert latest-available resolution (this point else previous
point else default), user-ordered execution (no topo sort), required-missing
skip, and optional fallback. The orchestrator is a pure requirement resolver: it
builds each provider's Node and calls ``produce``, never an injected callback.
"""

from __future__ import annotations

from zcu_tools.gui.app.autofluxdep.nodes.io import Patch
from zcu_tools.gui.app.autofluxdep.nodes.spec import Dependency
from zcu_tools.gui.app.autofluxdep.orchestrator import (
    InfoStore,
    Orchestrator,
    project_snapshot,
)

from ._helpers import make_builder, place

# --- latest-available resolution (project_snapshot reads a placed provider) ---


def test_resolve_required_missing_no_default_returns_none():
    p = place(make_builder("n", requires=(Dependency("x"),)))
    assert project_snapshot(p, InfoStore()) is None


def test_resolve_required_missing_with_default_keeps_node():
    p = place(make_builder("n", requires=(Dependency("x", default=lambda: 9),)))
    assert project_snapshot(p, InfoStore()) == {"x": 9}


def test_resolve_prefers_this_point_over_prev():
    p = place(make_builder("n", requires=(Dependency("x"),)))
    info = InfoStore(point={"x": "now"}, prev={"x": "before"})
    assert project_snapshot(p, info) == {"x": "now"}


def test_resolve_falls_back_to_prev_point():
    p = place(make_builder("n", requires=(Dependency("x"),)))
    info = InfoStore(prev={"x": "before"})  # not in this point
    assert project_snapshot(p, info) == {"x": "before"}


def test_resolve_optional_uses_default_when_absent_everywhere():
    p = place(make_builder("n", optional=(Dependency("k", default=lambda: 42),)))
    assert project_snapshot(p, InfoStore()) == {"k": 42}


def test_resolve_optional_prefers_value_over_default():
    p = place(make_builder("n", optional=(Dependency("k", default=lambda: 42),)))
    info = InfoStore(prev={"k": 7})
    assert project_snapshot(p, info) == {"k": 7}


def test_resolve_stored_none_is_a_value_not_missing():
    # a producer that wrote None should not trigger the default
    p = place(make_builder("n", optional=(Dependency("k", default=lambda: 42),)))
    info = InfoStore(point={"k": None})
    assert project_snapshot(p, info) == {"k": None}


# --- user-ordered execution (no topo sort) ---


def test_nodes_run_in_given_order():
    seen: list = []

    def record(name):
        def fn(_env, _snap):
            seen.append(name)
            return Patch()

        return fn

    providers = [place(make_builder(n, produce_fn=record(n))) for n in ("c", "a", "b")]
    Orchestrator(providers).run([0.0])
    assert seen == ["c", "a", "b"]  # exactly the declared order


def test_consumer_before_producer_reads_prev_point():
    # consumer ordered BEFORE producer; with latest-available it reads the
    # previous point's value (None at point0), allowed (no DAG error).
    seen: list = []

    def produce_x(env, _snap):
        return Patch({"x": f"x@{env.flux_idx}"})

    def read_x(_env, snap):
        seen.append(snap.get("x"))
        return Patch()

    producer = place(make_builder("prod", provides=("x",), produce_fn=produce_x))
    consumer = place(
        make_builder(
            "cons", optional=(Dependency("x", default=lambda: None),), produce_fn=read_x
        )
    )
    # consumer FIRST, producer second
    Orchestrator([consumer, producer]).run([0.0, 1.0, 2.0])
    # point0: no prev → default None; point1: prev x@0; point2: prev x@1
    assert seen == [None, "x@0", "x@1"]


def test_prev_snapshot_carries_across_points():
    seen: list = []

    def produce_x(env, _snap):
        return Patch({"x": env.flux_idx})

    def read_x(_env, snap):
        seen.append(snap.get("x"))
        return Patch()

    producer = place(make_builder("prod", provides=("x",), produce_fn=produce_x))
    consumer = place(
        make_builder(
            "cons", optional=(Dependency("x", default=lambda: None),), produce_fn=read_x
        )
    )
    # producer FIRST: consumer reads THIS point's value
    Orchestrator([producer, consumer]).run([0.0, 1.0, 2.0])
    assert seen == [0, 1, 2]


def test_skipped_provider_does_not_run():
    # a required dep missing everywhere → the provider is skipped (Node not built)
    ran = []

    def fn(_env, _snap):
        ran.append(True)
        return Patch()

    p = place(make_builder("n", requires=(Dependency("missing"),), produce_fn=fn))
    Orchestrator([p]).run([0.0])
    assert ran == []  # never built / produced


# --- on_node (auto-follow) fires per resolved provider, skips skipped ones ---


def test_on_node_fires_for_each_resolved_provider_in_order():
    seen: list = []
    a = place(make_builder("a"))
    b = place(make_builder("b"))
    Orchestrator([a, b]).run([0.0, 1.0], on_node=lambda n, i: seen.append((n, i)))
    # both providers, both flux points, in list order
    assert seen == [("a", 0), ("b", 0), ("a", 1), ("b", 1)]


def test_on_node_does_not_fire_for_a_skipped_provider():
    seen: list = []
    runnable = place(make_builder("ok"))
    skipped = place(make_builder("skip", requires=(Dependency("missing"),)))
    Orchestrator([runnable, skipped]).run([0.0], on_node=lambda n, i: seen.append(n))
    # the skipped provider (required dep missing) never fires on_node
    assert seen == ["ok"]
