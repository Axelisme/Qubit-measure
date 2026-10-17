"""Observe complete shadow planning through the documented pure collaborator seam."""

from dataclasses import replace

import numpy as np
import pytest
from zcu_tools.simulate.evolution.cache import PrepareCache, PreparedPayload
from zcu_tools.simulate.evolution.declarations import (
    AlgorithmIdentity,
    Eligibility,
    GuardDeclaration,
    KernelDeclaration,
    MemoryDeclaration,
    ReuseDeclaration,
    StageCosts,
)
from zcu_tools.simulate.evolution.profiles import (
    FamilyProfile,
    OneCoreProfile,
    ProfileDomain,
    ProfileProvenance,
)
from zcu_tools.simulate.evolution.shadow import (
    SearchLimits,
    ShadowCandidate,
    ShadowRequest,
    shadow_plan,
)

from ._engine import Payload
from ._tables import idle_table


def family(
    name: str, build: float, apply: float, *, bind: float = 0, resident: int = 0
):
    identity = AlgorithmIdentity(name, "synthetic-v1")
    declaration = KernelDeclaration(
        identity,
        Eligibility(max_tones=0),
        GuardDeclaration(),
        "exact-secular",
        ReuseDeclaration(name),
    )
    estimate = FamilyProfile(
        identity,
        StageCosts(0.5, build, bind, apply, 0, 0, 0),
        MemoryDeclaration(
            resident, 3 if resident else 0, 2 if resident else 1, 4 if bind else 0
        ),
    )
    return declaration, estimate


def profile(*families: FamilyProfile) -> OneCoreProfile:
    return OneCoreProfile(
        "synthetic-not-calibrated",
        "v1",
        ProfileDomain("tiny-neutral", (1, 8), (1e-6, 1e-2)),
        ProfileProvenance(
            "synthetic-no-measured-commit",
            ("no-runtime-measurements",),
            "logical one-core model",
            "synthetic:worked-example",
            "unmeasured synthetic estimates",
        ),
        tuple(families),
    )


def option(row: int, declaration: KernelDeclaration, key: bytes, *, prefix: str = ""):
    return ShadowCandidate(
        f"{prefix}{declaration.identity.name}-{row}",
        row,
        declaration,
        (declaration.reuse.namespace, "whole-model", 1e-3),
        key,
        "accepted",
    )


def scenario(n: int = 6):
    direct, direct_cost = family("z-direct", 0, 5)
    shared, shared_cost = family("a-shared", 6, 1, bind=2, resident=8)
    safe = tuple(option(row, direct, bytes([row])) for row in range(n))
    alternatives = tuple(
        replace(option(row, shared, b"shared"), binding_key=b"source:port0:controls")
        for row in range(n)
    )
    return idle_table([1.0] * n), safe, alternatives, profile(direct_cost, shared_cost)


@pytest.mark.parametrize("workers", [1, 2, 6])
@pytest.mark.parametrize("warm", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
def test_whole_graph_workers_cache_and_order_without_numerical_work(
    workers, warm, reverse
):
    table, safe, alternatives, costs = scenario()
    cache = PrepareCache(128)
    if warm:
        cache.add(
            alternatives[0].namespace,
            PreparedPayload(
                Payload(3), (b"shared", b"unused"), "synthetic-v1", ("not-a-row-guard",)
            ),
        )
    before = cache.inventory()
    candidates = safe + alternatives
    if reverse:
        table = replace(
            table, timeline=table.timeline[::-1], start_ns=table.start_ns[::-1]
        )
        candidates = tuple(replace(c, row=5 - c.row) for c in candidates[::-1])
    result = shadow_plan(
        table,
        candidates,
        tuple(c.candidate_id for c in safe[::-1]),
        costs,
        before,
        ShadowRequest(
            dimension=3,
            budget=1e-3,
            workers=workers,
            memory_allowance=256,
            state_bytes=12,
        ),
    )
    expected_shared = warm or workers != 6
    expected_wall = (
        (2 + 6 / workers if warm else 8 + 6 / workers) if expected_shared else 5
    )
    assert result.selected.wall_seconds == expected_wall
    assert [c.declaration.identity.name for c in result.selected.cover] == [
        "a-shared" if expected_shared else "z-direct"
    ] * 6
    assert len(result.selected.cover) == 6
    assert result.selected.lower_bound_seconds <= result.selected.wall_seconds
    assert result.incumbent.wall_seconds == 30 / workers
    assert not result.production_safe
    if expected_shared:
        tasks = result.selected.tasks
        assert len([t for t in tasks if t.stage == "build"]) == 1
        assert len([t for t in tasks if t.stage == "bind"]) == 1
        assert sum(t.seconds for t in tasks if t.stage == "build") == (0 if warm else 6)
        assert (
            result.selected.required_bytes == (48 if warm else 8) + 4 + 12 + workers * 3
        )
        assert result.selected.cache_allocations == (
            (before[0].allocation_id,) if warm else ()
        )
    assert cache.inventory() == before
    assert cache.live_bytes == 0
    if warm:
        # A snapshot consumed by planning cannot protect an old allocation from LRU eviction.
        cache.add(
            ("other", "whole-model", 1e-3), PreparedPayload(Payload(6), (b"other",))
        )
        assert all(
            entry.namespace != alternatives[0].namespace for entry in cache.inventory()
        )


@pytest.mark.parametrize("tentative", [False, True])
@pytest.mark.parametrize(
    "layout", ["whole", "split", "namespaces", "wrong-namespace", "wrong-revision"]
)
def test_multi_key_cache_acquisition_is_charged_per_allocation_not_per_key(
    layout, tentative
):
    direct, direct_cost = family("direct", 0, 100)
    cached, cached_cost = family("cached", 10, 1, bind=2, resident=8)
    cached = replace(cached, guard=GuardDeclaration(("model-build",), True))
    cached_cost = replace(
        cached_cost, stages=replace(cached_cost.stages, task_overhead=3)
    )
    safe = tuple(option(row, direct, bytes([row])) for row in range(2))
    alternatives = tuple(
        replace(
            option(row, cached, key),
            binding_key=b"same-input-different-prepare-key",
            guard="tentative" if tentative else "accepted",
        )
        for row, key in enumerate((b"a", b"b"))
    )
    cache = PrepareCache(128)
    namespace = alternatives[0].namespace
    revision = "synthetic-v1"
    if layout == "wrong-namespace":
        namespace = (namespace[0], "another-model", namespace[2])
    elif layout == "wrong-revision":
        revision = "old-revision"
    if layout == "split":
        for key in (b"a", b"b"):
            cache.add(namespace, PreparedPayload(Payload(1), (key,), revision))
    elif layout == "namespaces":
        # Identical keys in different allocations/namespaces are not one acquisition.
        other = (namespace[0], "another-model", namespace[2])
        alternatives = (alternatives[0], replace(alternatives[1], namespace=other))
        for space in (namespace, other):
            cache.add(space, PreparedPayload(Payload(1), (b"a", b"b"), revision))
    else:
        cache.add(namespace, PreparedPayload(Payload(1), (b"a", b"b"), revision))
    before = cache.inventory()
    result = shadow_plan(
        idle_table([1.0, 1.0]),
        safe + alternatives,
        tuple(c.candidate_id for c in safe),
        profile(direct_cost, cached_cost),
        before,
        ShadowRequest(
            dimension=3, budget=1e-3, workers=1, memory_allowance=128, state_bytes=0
        ),
    )
    # One acquisition costs 3; two independent binds cost 2+3 each;
    # two independent applies cost 1+3 each. Cold builds instead cost 10+3 each.
    hit = layout not in ("wrong-namespace", "wrong-revision")
    expected = 21 if layout == "whole" else 24 if hit else 44
    assert result.selected.wall_seconds == expected + (7 if tentative else 0)
    assert result.selected.cover == alternatives
    builds = [t for t in result.selected.tasks if t.stage == "build"]
    binds = [t for t in result.selected.tasks if t.stage == "bind"]
    applies = [t for t in result.selected.tasks if t.stage == "apply"]
    guards = [t for t in result.selected.tasks if t.stage == "guard"]
    assert len(builds) == (1 if layout == "whole" else 2)
    assert len(binds) == len(applies) == 2
    assert [t.seconds for t in binds] == [5, 5]
    assert [t.seconds for t in applies] == [4, 4]
    for candidate in alternatives:
        build = next(t for t in builds if candidate.candidate_id in t.candidate_ids)
        bind = next(t for t in binds if candidate.candidate_id in t.candidate_ids)
        apply = next(t for t in applies if candidate.candidate_id in t.candidate_ids)
        assert build.task_id in bind.dependencies
        assert bind.task_id in apply.dependencies
        if tentative:
            guard = next(t for t in guards if candidate.candidate_id in t.candidate_ids)
            assert build.task_id in guard.dependencies
            assert guard.task_id in apply.dependencies
    assert result.selected.tentative == (
        tuple(c.candidate_id for c in alternatives) if tentative else ()
    )
    assert result.selected.cache_allocations == (
        tuple(e.allocation_id for e in before) if hit else ()
    )
    assert cache.inventory() == before
    assert cache.live_bytes == 0


def test_reuse_partition_moves_escape_row_local_minima_and_preserve_timeline_dependencies():
    table = replace(
        idle_table([1.0] * 4),
        timeline=np.array([0, 0, 1, 1]),
        start_ns=np.array([0.0, 1.0, 0.0, 1.0]),
    )
    direct, d = family("direct", 0, 4)
    pairs, p = family("pairs", 3, 1)
    whole, w = family("whole", 5, 1)
    safe = tuple(option(row, direct, bytes([row])) for row in range(4))
    candidates = (
        safe
        + tuple(option(row, pairs, bytes([row % 2])) for row in range(4))
        + tuple(option(row, whole, b"whole") for row in range(4))
    )
    results = []
    for ordered in (candidates, candidates[::-1]):
        result = shadow_plan(
            table,
            ordered,
            tuple(c.candidate_id for c in safe),
            profile(d, p, w),
            (),
            ShadowRequest(
                dimension=3, budget=1e-3, workers=1, memory_allowance=100, state_bytes=0
            ),
        )
        assert [c.declaration.identity.name for c in result.selected.cover] == [
            "whole"
        ] * 4
        assert result.selected.wall_seconds == 9
        assert result.selected.lower_bound_seconds == 9
        applies = [task for task in result.selected.tasks if task.stage == "apply"]
        assert applies[0].task_id in applies[1].dependencies
        assert applies[2].task_id in applies[3].dependencies
        results.append(result)
    assert results[0] == results[1]


def test_memory_infeasible_option_retains_complete_accepted_incumbent():
    table, safe, alternatives, costs = scenario()
    result = shadow_plan(
        table,
        safe + alternatives,
        tuple(c.candidate_id for c in safe),
        costs,
        (),
        ShadowRequest(
            dimension=3, budget=1e-3, workers=2, memory_allowance=20, state_bytes=12
        ),
    )
    assert result.selected.cover == safe
    assert result.incumbent.feasible
    assert result.selected.required_bytes <= 20
    assert any("memory" in reason for reason in result.reasons)


def test_cache_hit_cannot_validate_build_or_actual_input_guard():
    table, safe, alternatives, costs = scenario(2)
    declaration = replace(
        alternatives[0].declaration,
        guard=GuardDeclaration(("model-build", "certified-input"), True, False),
    )
    pending = tuple(
        replace(c, declaration=declaration, guard="tentative") for c in alternatives
    )
    cache = PrepareCache(128)
    cache.add(
        pending[0].namespace,
        PreparedPayload(
            Payload(1), (b"shared",), "synthetic-v1", ("accepted-for-old-rows",)
        ),
    )
    result = shadow_plan(
        table,
        safe + pending,
        tuple(c.candidate_id for c in safe),
        costs,
        cache.inventory(),
        ShadowRequest(
            dimension=3, budget=1e-3, workers=1, memory_allowance=128, state_bytes=0
        ),
    )
    assert result.selected.tentative == tuple(c.candidate_id for c in pending)
    assert not result.production_safe
    assert result.incumbent.tentative == ()
    rejected = tuple(replace(c, guard="rejected") for c in pending)
    fallback = shadow_plan(
        table,
        safe + rejected,
        tuple(c.candidate_id for c in safe),
        costs,
        cache.inventory(),
        ShadowRequest(
            dimension=3, budget=1e-3, workers=1, memory_allowance=128, state_bytes=0
        ),
    )
    assert fallback.selected.cover == safe


def test_fixed_limits_and_ties_do_not_expand_timeline_cartesian_product():
    table = idle_table([1.0] * 40)
    direct, estimate = family("direct", 0, 1)
    safe = tuple(option(row, direct, b"key", prefix="z-") for row in range(40))
    tied = tuple(
        replace(c, candidate_id=c.candidate_id.replace("z-", "a-")) for c in safe
    )
    results = [
        shadow_plan(
            table,
            ordered,
            tuple(c.candidate_id for c in safe),
            profile(estimate),
            (),
            ShadowRequest(
                dimension=3,
                budget=1e-3,
                workers=6,
                memory_allowance=128,
                state_bytes=0,
                limits=SearchLimits(2, 2, 2, 8),
            ),
        )
        for ordered in (safe + tied, (safe + tied)[::-1])
    ]
    assert results[0] == results[1]
    assert results[0].evaluated <= 8
    assert results[0].rounds <= 2
    assert results[0].exhausted
    assert all(c.candidate_id.startswith("a-") for c in results[0].selected.cover)
    assert len(results[0].selected.cover) == 40


@pytest.mark.parametrize(
    "defect", ["domain", "revision", "incumbent", "allowance", "overlap", "eligibility"]
)
def test_invalid_request_cannot_claim_a_safe_cover(defect):
    table, safe, alternatives, costs = scenario(2)
    dimension, allowance = 3, 128
    if defect == "domain":
        dimension = 9
    elif defect == "revision":
        safe = tuple(
            replace(
                c,
                declaration=replace(
                    c.declaration,
                    identity=replace(c.declaration.identity, revision="unsupported"),
                ),
            )
            for c in safe
        )
    elif defect == "incumbent":
        safe = safe[:1]
    elif defect == "allowance":
        allowance = 0
    elif defect == "overlap":
        table = replace(table, timeline=np.array([0, 0]), start_ns=np.zeros(2))
    else:
        safe = tuple(
            replace(
                c,
                declaration=replace(
                    c.declaration, eligibility=Eligibility(min_tones=1)
                ),
            )
            for c in safe
        )
    with pytest.raises(ValueError, match="domain|incumbent|interval"):
        shadow_plan(
            table,
            safe + alternatives,
            tuple(c.candidate_id for c in safe),
            costs,
            (),
            ShadowRequest(
                dimension=dimension,
                budget=1e-3,
                workers=1,
                memory_allowance=allowance,
                state_bytes=12,
            ),
        )
