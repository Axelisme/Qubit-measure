"""Evolve-local payload release and shared apply waves through Engine."""

import weakref
from dataclasses import replace

import numpy as np
import pytest
from zcu_tools.simulate.evolution import Engine, EnvelopeSpec, FluxSchedule
from zcu_tools.simulate.evolution.kernel import GuardResult, KernelContext

from ._engine import RecordingKernel, TimelineKernel, batch, run, source, track


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
    result = run(Engine(1, kernels=(kernel,), chunks_per_worker=1), src, (b,))
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
            Engine(workers, kernels=(idle, driven), chunks_per_worker=1),
            source(),
            (b,),
        )
        assert result.paths.segments_by_kernel == {"t4": 8, "dop853": 8}
        assert idle.guarded == {caller}
        assert any(len(rows) > 1 for _, rows, _ in driven.calls)
        assert idle.preparations == 8
        assert idle.peak <= workers
        assert idle.live == 0
