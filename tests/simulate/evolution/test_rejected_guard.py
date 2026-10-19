"""Rejected model-only guard domains retain lower priority than real consumers."""

from contextlib import closing
from dataclasses import replace
from threading import Event

import numpy as np
import pytest
from zcu_tools.simulate.evolution import Engine, EnvelopeSpec
from zcu_tools.simulate.evolution.kernel import GuardPreparation, GuardResult

from ._engine import Payload, RecordingKernel, batch, source, track


class GuardDomain(RecordingKernel):
    """Represent immutable frame/certificates separately from fresh row decisions."""

    guard_builds = True

    def __init__(self):
        super().__init__("dyson")
        self.built: list[float] = []
        self.admitted: list[float] = []
        self.accepted = {10.0, 30.0}
        self.domain_covers = True
        self.size = 16

    def guard_prepared(self, table, rows, context):
        self.built.extend(table.duration_ns[rows].tolist())
        return self.result(Payload(self.size // 16), table, rows, context)

    def admit_prepared(self, prepared, table, rows, context):
        self.admitted.extend(table.duration_ns[rows].tolist())
        if not self.domain_covers:
            return None
        return self.result(prepared, table, rows, context)

    def result(self, prepared, table, rows, context):
        keys = self.prepare_keys(table, rows, context)
        return GuardResult(
            np.isin(table.duration_ns[rows], list(self.accepted)),
            (GuardPreparation(keys, prepared),),
        )


class FreeFallback(RecordingKernel):
    def __init__(self):
        super().__init__("floquet_period")

    def prepare(self, keys, table, rows, context):
        return Payload(0)


def driven(duration, timelines=1):
    return replace(
        batch(timelines, duration=duration),
        tracks=(track(timelines, EnvelopeSpec("constant"), duration=duration),),
    )


@pytest.mark.parametrize("timelines", [1, 2])
def test_rejected_domain_reused_but_current_rows_and_domain_rechecked(timelines):
    guard = GuardDomain()
    src = source()
    with closing(Engine(2, 128, kernels=(guard, FreeFallback()))) as engine:
        first = engine.evolve(src, (driven(20, timelines),), cancel=Event())
        built = len(guard.built)
        second = engine.evolve(src, (driven(20, timelines),), cancel=Event())
        assert len(guard.built) == built, "rejected frame/certificates must survive"
        assert guard.admitted
        np.testing.assert_array_equal(second.densities, first.densities)
        assert second.paths.segments_by_kernel == {"floquet_period": timelines}

        guard.accepted.add(20.0)
        accepted = engine.evolve(src, (driven(20, timelines),), cancel=Event())
        assert accepted.paths.segments_by_kernel == {"dyson": timelines}
        assert len(guard.built) == built, "old rejection is not a permanent veto"
        guard.domain_covers = False
        engine.evolve(src, (driven(20, timelines),), cancel=Event())
        assert len(guard.built) > built, "domain misses must still rebuild"
        guard.domain_covers = True
        rebuilt = len(guard.built)
        engine.evolve(src, (driven(20, timelines),), budget=2e-3, cancel=Event())
        assert len(guard.built) > rebuilt, "budget namespaces remain distinct"


def test_rejected_domains_never_evict_existing_residents():
    guard = GuardDomain()
    src = source()
    with closing(Engine(1, 32, kernels=(guard, FreeFallback()))) as engine:
        engine.evolve(src, (driven(10),), cancel=Event())
        guard.size = 32
        for duration in (20, 21, 22, 20):
            engine.evolve(src, (driven(duration),), cancel=Event())
        assert guard.built.count(20) == 2, "no room: rejected domains must stream"
        engine.evolve(src, (driven(10),), cancel=Event())
        assert guard.built.count(10) == 1, "rejected scans cannot evict old residents"
        # A smaller rejected domain fits in the remaining capacity and is reused.
        guard.size = 16
        for _ in range(2):
            engine.evolve(src, (driven(23),), cancel=Event())
        assert guard.built.count(23) == 1, "idle capacity may retain rejected domains"


def test_accepted_work_evicts_rejected_domains_before_ordinary_lru():
    guard = GuardDomain()
    src = source()
    with closing(Engine(1, 32, kernels=(guard, FreeFallback()))) as engine:
        for duration in (10, 20, 20, 30, 10):
            engine.evolve(src, (driven(duration),), cancel=Event())
        assert guard.built.count(20) == 1, "the rejected domain must first be resident"
        assert guard.built.count(10) == 1, "a recent rejection is still first to evict"
        engine.evolve(src, (driven(20),), cancel=Event())
        assert guard.built.count(20) == 2, "accepted allocations replaced rejection"


@pytest.mark.parametrize("timelines", [1, 2])
def test_unretained_rejected_handoffs_release_before_fallback_apply(timelines):
    """Cache admission, not the classification record, owns unused handoffs."""
    from weakref import ref

    class ObservedGuard(GuardDomain):
        def __init__(self):
            super().__init__()
            self.products = []

        def result(self, prepared, table, rows, context):
            self.products.append(ref(prepared))
            return super().result(prepared, table, rows, context)

    guard = ObservedGuard()

    class Fallback(RecordingKernel):
        def __init__(self):
            super().__init__("floquet_period")
            self.live_rejections = []

        def apply(self, prepared, key_index, table, rows, states, context):
            self.live_rejections.append(sum(p() is not None for p in guard.products))
            return super().apply(prepared, key_index, table, rows, states, context)

    fallback = Fallback()
    with closing(Engine(1, 16, kernels=(guard, fallback))) as engine:
        result = engine.evolve(source(), (driven(20, timelines),), cancel=Event())
    assert result.paths.segments_by_kernel == {"floquet_period": timelines}
    assert guard.products, "exercise numerical guard handoff construction"
    assert fallback.live_rejections == [0] * len(fallback.calls), (
        "a rejected offer must release unused handoffs before fallback apply"
    )
