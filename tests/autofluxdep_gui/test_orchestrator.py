"""Dependency-model tests — the orchestrator resolves with latest-available.

No hardware, no Qt: small ad-hoc Builders (``make_builder``) whose Nodes script
a ``produce`` let us assert latest-available resolution (this point else previous
point else default), user-ordered execution (no topo sort), required-missing
skip, and optional fallback. The orchestrator is a pure requirement resolver: it
builds each provider's Node and calls ``produce``, never an injected callback.
"""

from __future__ import annotations

from zcu_tools.gui.app.autofluxdep.nodes.io import Patch
from zcu_tools.gui.app.autofluxdep.nodes.spec import Dependency, ModuleDep
from zcu_tools.gui.app.autofluxdep.orchestrator import (
    InfoStore,
    Orchestrator,
    project_snapshot,
    resolve_provider_snapshot,
)

from ._helpers import make_builder, place


class _ModuleSource:
    def __init__(self, modules: dict[str, object]) -> None:
        self._modules = modules

    def get_module(self, name: str) -> object | None:
        return self._modules.get(name)


# --- latest-available resolution (project_snapshot reads a placed provider) ---


def test_resolve_required_missing_no_default_returns_none():
    p = place(make_builder("n", requires=(Dependency("x"),)))
    assert project_snapshot(p, InfoStore()) is None


def test_resolve_provider_snapshot_returns_structured_skip_reason():
    p = place(
        make_builder(
            "n",
            requires=(Dependency("x"),),
            requires_modules=(ModuleDep("readout"),),
        )
    )
    resolution = resolve_provider_snapshot(p, InfoStore(), _ModuleSource({}))
    assert resolution.snapshot is None
    assert resolution.skip_reason is not None
    assert resolution.skip_reason.missing_info_keys == ("x",)
    assert resolution.skip_reason.missing_modules == ("readout",)


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


def test_module_dep_uses_library_alias_order():
    readout = object()
    stale_readout = object()
    p = place(
        make_builder(
            "n",
            optional_modules=(
                ModuleDep(
                    "readout",
                    aliases=("readout_rf", "readout"),
                    default=lambda: None,
                ),
            ),
        )
    )

    snap = project_snapshot(
        p,
        InfoStore(),
        ml=_ModuleSource({"readout": stale_readout, "readout_rf": readout}),
    )

    assert snap is not None
    assert snap.module("readout") is readout


def test_module_dep_prefers_produced_module_over_library_alias():
    produced = object()
    library = object()
    p = place(
        make_builder(
            "n",
            optional_modules=(
                ModuleDep(
                    "readout",
                    aliases=("readout_rf", "readout"),
                    default=lambda: None,
                ),
            ),
        )
    )

    snap = project_snapshot(
        p,
        InfoStore(module_point={"readout": produced}),
        ml=_ModuleSource({"readout_rf": library}),
    )

    assert snap is not None
    assert snap.module("readout") is produced


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


def test_on_skip_fires_with_reason_for_skipped_provider():
    seen: list = []
    skipped = place(make_builder("skip", requires=(Dependency("missing"),)))
    Orchestrator([skipped]).run(
        [0.0],
        on_skip=lambda n, i, reason: seen.append((n, i, reason.missing_info_keys)),
    )
    assert seen == [("skip", 0, ("missing",))]


def test_on_node_row_fires_after_validate_before_merge():
    seen: list = []

    def produce_x(_env, _snap):
        return Patch({"x": 1})

    provider = place(make_builder("prod", provides=("x",), produce_fn=produce_x))

    def on_node_row(name, idx, patch, info):
        seen.append((name, idx, patch.values(), "x" in info.point))

    info = Orchestrator([provider]).run([0.0], on_node_row=on_node_row)

    assert seen == [("prod", 0, {"x": 1}, False)]
    assert info.point["x"] == 1


def test_on_node_failed_fires_for_validate_patch_error():
    def bad_patch(_env, _snap):
        return Patch({"undeclared": 1})

    provider = place(make_builder("bad", provides=("x",), produce_fn=bad_patch))
    seen: list = []
    orch = Orchestrator([provider])
    orch.run([0.0], on_node_failed=lambda n, i, exc, stage: seen.append((n, i, stage)))

    assert seen == [("bad", 0, "validate_patch")]
    assert orch.run_error is not None
    assert orch.run_error_stage == "validate_patch"


# --- cooperative cancellation (should_stop) ---


def test_should_stop_before_a_flux_point_exits_early():
    # should_stop returns True once two points have completed → the sweep must
    # break before flux idx 2 and never start it.
    done_points: list[int] = []

    def on_point(idx, _flux, _info):
        done_points.append(idx)

    p = place(make_builder("n"))
    info = Orchestrator([p]).run(
        [0.0, 1.0, 2.0, 3.0],
        on_point=on_point,
        should_stop=lambda: len(done_points) >= 2,
    )
    # only points 0 and 1 ran; the sweep stopped before point 2
    assert done_points == [0, 1]
    # the returned InfoStore reflects the last completed point, not point 2/3
    assert info.point["flux_idx"] == 1


def test_should_stop_within_a_point_skips_remaining_providers():
    # should_stop flips True after the first provider of a point runs → the
    # second provider of that point must not run (orchestrator breaks the
    # provider loop). on_point still fires for that partially-run point (the
    # provider-loop break does not skip it), but the next flux point never
    # starts (should_stop is re-checked at its top).
    ran: list[str] = []
    flip = {"stop": False}

    def first(_env, _snap):
        ran.append("a")
        flip["stop"] = True  # request stop after the first provider
        return Patch()

    def second(_env, _snap):
        ran.append("b")
        return Patch()

    a = place(make_builder("a", produce_fn=first))
    b = place(make_builder("b", produce_fn=second))
    points: list[int] = []
    Orchestrator([a, b]).run(
        [0.0, 1.0],
        on_point=lambda idx, _f, _i: points.append(idx),
        should_stop=lambda: flip["stop"],
    )
    # only the first provider of point 0 ran; "b" was skipped by the break
    assert ran == ["a"]
    # the partially-run point 0 still completed (on_point fired); point 1 never
    # started because should_stop is True at its top
    assert points == [0]


def test_on_flux_committed_does_not_fire_for_stopped_partial_point():
    ran: list[str] = []
    flip = {"stop": False}

    def first(_env, _snap):
        ran.append("a")
        flip["stop"] = True
        return Patch()

    def second(_env, _snap):
        ran.append("b")
        return Patch()

    a = place(make_builder("a", produce_fn=first))
    b = place(make_builder("b", produce_fn=second))
    committed: list[int] = []
    points: list[int] = []
    Orchestrator([a, b]).run(
        [0.0, 1.0],
        on_point=lambda idx, _f, _i: points.append(idx),
        on_flux_committed=lambda idx, _f, _i: committed.append(idx),
        should_stop=lambda: flip["stop"],
    )

    assert ran == ["a"]
    assert points == [0]
    assert committed == []


def test_stop_set_inside_last_provider_produce_does_not_commit_row_or_flux():
    flip = {"stop": False}

    def last(_env, _snap):
        flip["stop"] = True
        return Patch({"x": 1})

    provider = place(make_builder("last", provides=("x",), produce_fn=last))
    rows: list[tuple[str, int]] = []
    committed: list[int] = []

    info = Orchestrator([provider]).run(
        [0.0],
        on_node_row=lambda name, idx, _patch, _info: rows.append((name, idx)),
        on_flux_committed=lambda idx, _f, _i: committed.append(idx),
        should_stop=lambda: flip["stop"],
    )

    assert rows == []
    assert committed == []
    assert "x" not in info.point
