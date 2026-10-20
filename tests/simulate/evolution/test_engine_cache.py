"""Engine resident reuse, capacity and payload release across timelines and evolves."""

import weakref
from contextlib import closing
from dataclasses import replace
from threading import Event

import numpy as np
import pytest
from zcu_tools.simulate.evolution import Engine, EnvelopeSpec, FluxSchedule, ModelSource
from zcu_tools.simulate.evolution.cache import RESIDENT_METADATA_BYTES
from zcu_tools.simulate.evolution.kernel import GuardResult, KernelContext

from ._engine import Payload, RecordingKernel, TimelineKernel, batch, run, source, track


def test_shared_cache_namespace_budget_eviction_and_singletons():
    k = RecordingKernel("t4")
    # Include metadata for the intended resident working set.
    engine = Engine(1, 16 + RESIDENT_METADATA_BYTES, kernels=(k,))
    src = source()
    try:
        for duration in [3.0, 3.0, 4.0, 3.0]:
            engine.evolve(src, (batch(2, duration=duration),), cancel=Event())
        assert k.preparations == 3
        engine.evolve(src, (batch(2),), budget=1e-6, cancel=Event())
        assert k.preparations == 4
        other = ModelSource(
            src.initial_density,
            lambda level: replace(src.model(level), identity="different-source"),
        )
        engine.evolve(other, (batch(2),), cancel=Event())
        assert k.preparations == 5
        for _ in range(2):
            engine.evolve(src, (batch(1, duration=8),), cancel=Event())
        assert k.preparations == 6, "the second singleton evolve must reuse its payload"
    finally:
        engine.close()
        engine.close()
    with pytest.raises(RuntimeError, match="closed"):
        engine.evolve(src, (batch(),), cancel=Event())


@pytest.mark.parametrize("workers", [1, 2])
def test_singleton_residents_reuse_payloads_across_evolves(workers):
    kernel = TimelineKernel("t4")
    src = source()
    # Include metadata for the intended resident working set.
    with closing(
        Engine(workers, 128 + 8 * RESIDENT_METADATA_BYTES, kernels=(kernel,))
    ) as engine:
        first = engine.evolve(src, (batch(8),), cancel=Event())
        assert kernel.preparations == 8
        second = engine.evolve(src, (batch(8),), cancel=Event())
        assert kernel.preparations == 8, "resident singletons must not rebuild"
        np.testing.assert_array_equal(second.densities, first.densities)
        assert second.paths == first.paths


def test_oversize_local_payload_does_not_evict_fitting_singleton():
    class SizedKernel(RecordingKernel):
        def prepare(self, keys, table, rows, context: KernelContext):
            super().prepare(keys, table, rows, context)
            return Payload(2 if table.duration_ns[rows[0]] == 4 else 1)

    kernel = SizedKernel("t4")
    src = source()
    # Include metadata for the intended resident working set.
    with closing(Engine(1, 16 + RESIDENT_METADATA_BYTES, kernels=(kernel,))) as engine:
        engine.evolve(src, (batch(1),), cancel=Event())
        for point in range(2):
            # Distinct points avoid result memo after including retention metadata.
            timeline = replace(batch(1, duration=4), points=np.array([point]))
            engine.evolve(src, (timeline,), cancel=Event())
        engine.evolve(src, (batch(1),), cancel=Event())
        assert kernel.preparations == 3, (
            "oversize products must stream without eviction"
        )


def test_singleton_stream_preserves_current_shared_hit_and_fills_only_spare_bytes():
    kernel = RecordingKernel("t4")
    src = source()
    shared = batch(2)
    stream = replace(batch(3), points=np.arange(2, 5), end_ns=np.array([4.0, 5.0, 6.0]))
    # Include metadata for the intended resident working set.
    with closing(
        Engine(
            1, 32 + 2 * RESIDENT_METADATA_BYTES, kernels=(kernel,), chunks_per_worker=1
        )
    ) as engine:
        engine.evolve(src, (shared,), cancel=Event())
        engine.evolve(src, (shared, stream), cancel=Event())
        assert kernel.preparations == 4
        engine.evolve(src, (shared,), cancel=Event())
        assert kernel.preparations == 4, "streaming misses must not evict shared hits"
        engine.evolve(src, (batch(1, duration=4),), cancel=Event())
        assert kernel.preparations == 4, "the fitting local product must persist"


def test_new_source_singleton_replaces_previous_source_resident():
    kernel = RecordingKernel("t4")
    src = source()
    other = ModelSource(
        src.initial_density,
        lambda level: replace(src.model(level), identity="next-flux"),
    )
    # Include metadata for the intended resident working set.
    with closing(Engine(1, 16 + RESIDENT_METADATA_BYTES, kernels=(kernel,))) as engine:
        for current in (src, src, other, other, src):
            engine.evolve(current, (batch(1),), cancel=Event())
        assert kernel.preparations == 3, "old flux residents must yield to new ones"


@pytest.mark.parametrize(
    ("repeated", "name"),
    [(False, "t4"), (False, "dop853"), (True, "t4"), (True, "flux_dop853")],
)
def test_timeline_local_payloads_are_released_before_the_next_timeline(repeated, name):
    from zcu_tools.simulate.evolution.kernel import GuardPreparation

    class LifetimeKernel(TimelineKernel):
        guard_builds = True

        def __init__(self):
            super().__init__(name)
            self.live = 0
            self.peak_live = 0

        def release(self, count):
            self.live -= count

        def prepare(self, keys, table, rows, context: KernelContext):
            payload = super().prepare(keys, table, rows, context)
            self.live += len(keys)
            self.peak_live = max(self.peak_live, self.live)
            weakref.finalize(payload, self.release, len(keys))
            return payload

        def guard_prepared(self, table, rows, context: KernelContext):
            keys, first = np.unique(
                self.prepare_keys(table, rows, context), return_index=True
            )
            payload = self.prepare(keys, table, rows[first], context)
            return GuardResult(
                self.guard(table, rows, context),
                (GuardPreparation(keys, payload),),
            )

    n = 8
    b = batch(n)
    if repeated:
        b = replace(
            b,
            flux=FluxSchedule(
                2 * np.arange(n + 1),
                np.tile([0.0, 1.0], n),
                np.tile([[0], [1]], (n, 1)),
            ),
        )
    if name != "t4":
        b = replace(
            b, tracks=tuple(track(n, EnvelopeSpec("constant")) for _ in range(3))
        )
    kernel = LifetimeKernel()
    src = source()
    result = run(Engine(1, 0, kernels=(kernel,), chunks_per_worker=1), src, (b,))
    np.testing.assert_array_equal(
        result.densities, np.repeat(src.initial_density[None], n, axis=0)
    )
    assert kernel.preparations == n, "same-timeline repeats must prepare only once"
    assert kernel.peak_live == 1, (
        "local payloads must not accumulate in A-stage or chunks"
    )
    assert kernel.live == 0, "the evolve call must release all local payloads"


def test_local_singletons_keep_shared_apply_waves_and_release_each_payload():
    from threading import get_ident

    caller = get_ident()
    for workers in (1, 2):

        class Idle(TimelineKernel):
            def __init__(self):
                super().__init__("t4")
                self.live = 0
                self.peak = 0
                self.guarded = set()

            def guard(self, table, rows, context: KernelContext):
                self.guarded.add(get_ident())
                return super().guard(table, rows, context)

            def release(self):
                self.live -= 1

            def prepare(self, keys, table, rows, context: KernelContext):
                payload = super().prepare(keys, table, rows, context)
                self.live += 1
                self.peak = max(self.peak, self.live)
                weakref.finalize(payload, self.release)
                return payload

        idle, driven = Idle(), RecordingKernel("dop853")
        b = replace(
            batch(8),
            tracks=tuple(
                track(8, EnvelopeSpec("constant"), duration=1) for _ in range(3)
            ),
        )
        result = run(
            Engine(workers, 0, kernels=(idle, driven), chunks_per_worker=1),
            source(),
            (b,),
        )
        assert result.paths.segments_by_kernel == {"t4": 8, "dop853": 8}
        assert idle.guarded == {caller}
        assert any(len(rows) > 1 for _, rows, _ in driven.calls)
        assert idle.preparations == 8
        assert idle.peak <= workers
        assert idle.live == 0


class InventoryKernel(RecordingKernel):
    """Observe whole-allocation ownership without retaining numerical payloads."""

    def __init__(self, name="t4"):
        super().__init__(name)
        self.payloads: list[weakref.ReferenceType[Payload]] = []
        self.live_before_build: list[int] = []

    @property
    def live_bytes(self):
        return sum(payload.nbytes for ref in self.payloads if (payload := ref()))

    def prepare(self, keys, table, rows, context: KernelContext):
        self.live_before_build.append(self.live_bytes)
        payload = super().prepare(keys, table, rows, context)
        self.payloads.append(weakref.ref(payload))
        return payload


@pytest.mark.parametrize("changed_model", [False, True])
def test_complete_inventory_releases_stale_on_offer_before_next_build(changed_model):
    kernel = InventoryKernel()
    src = source()
    other = (
        ModelSource(
            src.initial_density,
            lambda level: replace(src.model(level), identity="adjacent-flux"),
        )
        if changed_model
        else src
    )
    # Include metadata for the intended resident working set.
    with closing(
        Engine(
            1, 32 + 2 * RESIDENT_METADATA_BYTES, kernels=(kernel,), chunks_per_worker=2
        )
    ) as engine:
        seed = replace(batch(4), end_ns=np.array([3.0, 3.0, 7.0, 7.0]))
        engine.evolve(src, (seed,), cancel=Event())
        assert kernel.live_bytes == 32
        misses = replace(batch(3), end_ns=np.array([4.0, 5.0, 6.0]))
        engine.evolve(other, (misses,), cancel=Event())
        # One inline worker may own a 16-byte active build above the account.
        # Once offered, it must release stale before the next build starts.
        assert kernel.live_before_build[2:] == [32, 32, 32], (
            "resident plus pending must fit after every offer"
        )
        assert kernel.live_bytes <= 32


def test_inventory_keeps_unused_residents_when_new_products_fit():
    kernel = InventoryKernel()
    src = source()
    # Include metadata for the intended resident working set.
    with closing(
        Engine(1, 64 + 4 * RESIDENT_METADATA_BYTES, kernels=(kernel,))
    ) as engine:
        engine.evolve(src, (batch(2),), cancel=Event())
        unused = kernel.payloads[0]
        engine.evolve(src, (batch(2, duration=4),), cancel=Event())
        assert unused() is not None, "spare capacity must preserve stale reuse"
        again = replace(batch(2), points=np.arange(2) + 10)
        engine.evolve(src, (again,), cancel=Event())
        assert kernel.preparations == 2


def test_inventory_protects_whole_borrowed_batches_and_bounds_pending():
    kernel = InventoryKernel()
    src = source()
    # Keep one whole three-key batch, including its metadata, protected.
    with closing(
        Engine(1, 48 + RESIDENT_METADATA_BYTES, kernels=(kernel,), chunks_per_worker=2)
    ) as engine:
        # One indivisible allocation owns both keys, even when only 3 ns is used.
        seed = replace(batch(6), end_ns=np.array([3.0, 3.0, 7.0, 7.0, 8.0, 8.0]))
        engine.evolve(src, (seed,), cancel=Event())
        borrowed = kernel.payloads[0]
        assert borrowed() is not None
        mixed = replace(batch(4), end_ns=np.array([3.0, 4.0, 5.0, 6.0]))
        engine.evolve(src, (mixed,), cancel=Event())
        assert borrowed() is not None, "a partial hit must keep its entire batch"
        assert kernel.preparations == 6, "borrowed keys must not rebuild"
        assert kernel.live_bytes <= 48, "resident plus pending share one budget"
        assert max(kernel.live_before_build) <= 48


def test_prefix_pending_shares_capacity_with_future_suffix_hits():
    idle, driven = InventoryKernel(), InventoryKernel("dyson")
    src = source()
    # Include metadata for two separately retained products.
    with closing(
        Engine(
            1,
            32 + 2 * RESIDENT_METADATA_BYTES,
            kernels=(idle, driven),
            chunks_per_worker=1,
        )
    ) as engine:
        # This hit belongs to the later suffix, not the representative prefix.
        engine.evolve(src, (batch(2),), cancel=Event())
        suffix = idle.payloads[0]
        pulses = tuple(
            track(2, EnvelopeSpec("constant"), start=start, duration=1)
            for start in (1, 2)
        )
        timeline = replace(batch(2, duration=6), tracks=pulses)
        result = engine.evolve(src, (timeline,), cancel=Event())
        assert result.paths.segments_by_kernel == {"t4": 4, "dyson": 4}
        assert suffix() is not None, "prefix work must protect later suffix hits"
        assert idle.preparations == 2, "prefix work must not evict suffix keys"
        assert idle.live_bytes + driven.live_bytes <= 32, (
            "prefix products, suffix residents and worker offers share capacity"
        )
