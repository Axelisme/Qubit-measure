"""Probe-window changes reuse exact model-only Floquet frames through Engine."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import replace
from threading import Barrier, Event

import numpy as np
import pytest
from zcu_tools.simulate.evolution import (
    DriveTrack,
    Engine,
    EnvelopeSpec,
    FluxSchedule,
    ModelSource,
    SecularDrive,
    SecularModel,
    TimelineBatch,
)
from zcu_tools.simulate.evolution.kernel import ApplyContext, KernelContext
from zcu_tools.simulate.evolution.kernels.floquet import FloquetKernel
from zcu_tools.simulate.evolution.workplan import WorkPlan

from .._engine import RecordingKernel


class CountingFloquet(FloquetKernel):
    """Observe public construction/admission seams without replacing numerics."""

    def __init__(self):
        self.builds = 0
        self.sizes: list[int] = []
        self.guards = []

    def prepare(self, keys, table, rows, context):
        self.builds += len(keys)
        return super().prepare(keys, table, rows, context)

    def guard_prepared(self, table, rows, context):
        result = super().guard_prepared(table, rows, context)
        self.sizes.extend(p.prepared.nbytes for p in result.preparations)
        self.guards.append((table, rows, result))
        return result


def source() -> ModelSource:
    zero = np.zeros((2, 2), complex)
    drive = np.array([[0.0, 0.02], [0.02, 0.0]], complex)
    model = SecularModel(
        (0,),
        "certificate-extension",
        np.array([0.0, 0.13]),
        np.eye(2, dtype=complex),
        (SecularDrive("stark", drive, zero), SecularDrive("probe", drive, zero)),
        np.array([[0.0, 0.001], [0.0, 0.0]]),
        np.zeros((2, 2)),
    )
    vector = np.array([1.0, 1j]) / np.sqrt(2)
    return ModelSource(
        np.outer(vector, vector.conj()), lambda level: replace(model, level=level)
    )


def window(frequencies, *, point=0, gain=0.2, phase=0.3, duration=16.0):
    count = len(frequencies)
    tracks = tuple(
        DriveTrack(
            port,
            np.zeros(count),
            np.full(count, duration),
            np.broadcast_to(frequency, (count,)).copy(),
            np.full(count, phase),
            np.full(count, amplitude),
            (EnvelopeSpec("constant"),),
            np.zeros(count, dtype=np.int64),
            np.ones(count, dtype=bool),
        )
        for port, frequency, amplitude in (
            ("stark", 0.13, gain),
            ("probe", frequencies, 0.0001),
        )
    )
    return TimelineBatch(
        np.arange(count) + point,
        tracks,
        FluxSchedule(np.arange(count + 1), np.zeros(count), np.zeros((count, 1), int)),
        np.full(count, duration),
    )


def acquire(engine, src, batch):
    result = engine.evolve(src, (batch,), cancel=Event())
    assert result.paths.segments_by_kernel == {"floquet": len(batch.points)}
    return result.densities


@pytest.mark.parametrize("count", [1, 3])
def test_shifted_probe_window_extends_union_without_rebuilding_frame(count):
    solver = CountingFloquet()
    src = source()
    direct = RecordingKernel("dop853", accepts=False)
    with (
        closing(Engine(1, 1_000_000, kernels=(solver, direct))) as engine,
        closing(Engine(1, 0, kernels=(FloquetKernel(), direct))) as reference,
    ):
        initial = window(np.arange(count) * 0.004)
        acquire(engine, src, initial)
        assert solver.builds == 1
        shifted = window(
            np.arange(count) * 0.004 + 0.004, point=10, phase=0.8, duration=8.0
        )
        actual = acquire(engine, src, shifted)
        expected = acquire(reference, src, shifted)
        np.testing.assert_array_equal(actual, expected)
        assert solver.builds == 1, "new probe bins must extend the cached frame"
        # Both the old and the new bins remain resident, with fresh row decisions.
        for batch in (
            replace(initial, points=initial.points + 20),
            replace(shifted, points=shifted.points + 30),
        ):
            np.testing.assert_array_equal(
                acquire(engine, src, batch), acquire(reference, src, batch)
            )
        assert solver.builds == 1, "extension must retain the complete bin union"


def test_parallel_windows_replace_cache_charge_instead_of_accumulating_frames():
    src = source()
    direct = RecordingKernel("dop853", accepts=False)
    # Measure the opaque allocation at the kernel seam. Allow two full frames
    # and several small certificates, but not a third charge for the same frame.
    probe = CountingFloquet()
    with closing(Engine(1, 0, kernels=(probe, direct))) as sizing:
        acquire(sizing, src, window([0.0]))
    limit = 2 * probe.sizes[0] + 8_000
    solver = CountingFloquet()
    with (
        closing(Engine(2, limit, kernels=(solver, direct))) as engine,
        closing(Engine(1, 0, kernels=(FloquetKernel(), direct))) as reference,
    ):
        acquire(engine, src, window([0.0], gain=0.19))
        acquire(engine, src, window([0.0], point=10))
        assert solver.builds == 2
        # Distinct bins share one cached frame while apply uses two workers.
        shifted = window([0.004, 0.008], point=20, phase=1.2)
        np.testing.assert_array_equal(
            acquire(engine, src, shifted), acquire(reference, src, shifted)
        )
        assert solver.builds == 2, "parallel new bins must reuse the cached frame"
        for batch in (
            window([0.0], point=30, gain=0.19),
            window([0.0, 0.008], point=40),
        ):
            np.testing.assert_array_equal(
                acquire(engine, src, batch), acquire(reference, src, batch)
            )
        assert solver.builds == 2, (
            "replacement must neither evict the other frame nor lose old bins"
        )


class CountingWorkPlan(WorkPlan):
    """Observe certificate construction without inspecting published payloads."""

    def __init__(self):
        super().__init__()
        self.builds = 0

    def fact(self, key, build):
        def observed():
            self.builds += 1
            return build()

        return super().fact(key, observed)


def test_concurrent_extensions_share_complete_union_and_preserve_certificates():
    solver = CountingFloquet()
    src = source()
    with closing(
        Engine(1, 0, kernels=(solver, RecordingKernel("dop853", accepts=False)))
    ) as engine:
        acquire(engine, src, window([0.0, 0.004, 0.008]))
    table, rows, initial = solver.guards[-1]
    original = initial.preparations[0].prepared
    original_bytes = original.nbytes
    changed = replace(table, tone_frequency_ghz=table.tone_frequency_ghz + [0.0, 0.004])
    barrier = Barrier(2)
    with closing(CountingWorkPlan()) as plan, closing(WorkPlan()) as cold_plan:
        context = KernelContext(src, 1e-3, plan)

        def run():
            barrier.wait(timeout=10)
            return solver.extend_prepared(original, changed, context)

        with ThreadPoolExecutor(max_workers=2) as pool:
            a, b = list(pool.map(lambda _: run(), range(2)))
        assert plan.builds == 1, "new certificate publication must be single-flight"
        assert solver.builds == 1, "extensions must borrow the existing frame"
        assert original_bytes < a.nbytes < 2 * original_bytes
        assert a.nbytes == b.nbytes, "both callers must receive the complete union"
        cold_context = KernelContext(src, 1e-3, cold_plan)
        cold = solver.guard_prepared(changed, rows, cold_context)
        assert solver.builds == 2
        for prepared in (a, b):
            # The union admits both windows. Its numerical products agree with
            # the old payload on old rows and a fresh build on the shifted rows.
            for current, reference in ((table, initial), (changed, cold)):
                admitted = solver.admit_prepared(prepared, current, rows, context)
                assert admitted is not None
                np.testing.assert_array_equal(admitted.accepted, reference.accepted)
                states = np.repeat(src.initial_density[None], len(rows), axis=0)
                expected_states = states.copy()
                indices = np.zeros(len(rows), dtype=np.int64)
                solver.apply(
                    prepared,
                    indices,
                    current,
                    rows,
                    states,
                    ApplyContext(1e-3, admitted.plan),
                )
                solver.apply(
                    reference.preparations[0].prepared,
                    indices,
                    current,
                    rows,
                    expected_states,
                    ApplyContext(1e-3, reference.plan),
                )
                np.testing.assert_array_equal(states, expected_states)
            assert solver.extend_prepared(prepared, changed, context) is prepared
        assert plan.builds == 1, "covered domains must not rebuild certificates"
        assert original.nbytes == original_bytes
        assert solver.admit_prepared(original, changed, rows, context) is None, (
            "extension must not mutate the original payload's domain"
        )


def test_discarded_frame_releases_arrays_while_work_plan_remains_open():
    """A frame with no active binding must not survive as an evolve-long fact."""
    import gc
    import tracemalloc

    # Warm numerical libraries and obtain a real table at the kernel seam before
    # tracing. The warm payload is not part of the measured allocation interval.
    observer = CountingFloquet()
    src = source()
    with closing(
        Engine(1, 0, kernels=(observer, RecordingKernel("dop853", accepts=False)))
    ) as engine:
        acquire(engine, src, window([0.0]))
    table, _, _ = observer.guards[-1]
    solver = FloquetKernel()
    with closing(WorkPlan()) as plan:
        context = KernelContext(src, 1e-3, plan)
        rows = np.array([0])
        keys = solver.prepare_keys(table, rows, context)
        gc.collect()
        tracemalloc.start()
        try:
            before = tracemalloc.get_traced_memory()[0]
            prepared = solver.prepare(keys, table, rows, context)
            charged = prepared.nbytes
            del prepared
            gc.collect()
            retained = tracemalloc.get_traced_memory()[0] - before
        finally:
            tracemalloc.stop()
        assert retained < charged // 2, (
            f"discarded {charged}-byte frame still retains {retained} bytes "
            "outside resident admission"
        )
