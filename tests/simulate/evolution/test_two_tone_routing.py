"""Constant two-tone cost routing at the Engine/Kernel boundary."""

from dataclasses import replace

import numpy as np
import pytest
from zcu_tools.simulate.evolution import Engine, EnvelopeSpec, ModelSource
from zcu_tools.simulate.evolution.kernel import GuardResult, KernelContext
from zcu_tools.simulate.evolution.kernels.dop853 import DOP853Plan
from zcu_tools.simulate.evolution.kernels.floquet import FloquetAdmission, FloquetPlan

from ._engine import RecordingKernel, batch, run, track
from .contract.kernel_suite import synthetic_model


class GuardRecorder(RecordingKernel):
    """Record guard attempts by duration; keep all dynamics as identity."""

    def __init__(self, name, attempts, *, accepts=True):
        super().__init__(name, accepts=accepts)
        self.attempts = attempts

    def guard(self, table, rows, context: KernelContext):
        self.attempts.extend((self.name, float(table.duration_ns[r])) for r in rows)
        return super().guard(table, rows, context)


def two_tone_batch(duration, gain, point):
    pulse = replace(
        track(1, EnvelopeSpec("constant"), duration=duration),
        frequency_ghz=np.array([0.5]),
        amplitude=np.array([gain]),
    )
    probe = replace(pulse, frequency_ghz=np.array([0.425]), amplitude=np.array([1e-4]))
    return replace(
        batch(1, duration=duration), points=np.array([point]), tracks=(pulse, probe)
    )


def route(batches, *, workers=1, chunks=1, reject=None):
    model = synthetic_model(16)
    src = ModelSource(np.diag([1.0] + [0.0] * 15).astype(complex), lambda level: model)
    attempts = []
    kernels = tuple(
        GuardRecorder(name, attempts, accepts=name != reject)
        for name in ("floquet", "dop853")
    )
    result = run(
        Engine(workers, 0, chunks_per_worker=chunks, kernels=kernels), src, batches
    )
    return result, attempts


class PlannedFloquet(RecordingKernel):
    """Identity dynamics that require the guard's exact per-row decisions."""

    guard_builds = True

    def __init__(self):
        super().__init__("floquet")
        self.admissions = {}

    def guard_prepared(self, table, rows, context: KernelContext):
        admissions = {int(row): FloquetAdmission("modal") for row in rows}
        self.admissions.update(admissions)
        return GuardResult(np.ones(len(rows), dtype=bool), plan=FloquetPlan(admissions))

    def apply(self, prepared, key_index, table, rows, states, context):
        plan = context.plan
        assert isinstance(plan, FloquetPlan)
        for row in rows:
            assert row in plan.admissions, "local plans must retain shared admissions"
            assert plan.admissions[row] is self.admissions[row]
        assert {int(table.timeline[row]) for row in plan.admissions} == set(
            table.timeline[rows]
        ), "deferred dispatch must keep plans local to its timeline"
        return super().apply(prepared, key_index, table, rows, states, context)


class PlannedDirect(RecordingKernel):
    """Identity dynamics that require the dispatch's direct-solver plan."""

    def __init__(self):
        super().__init__("dop853")

    def apply(self, prepared, key_index, table, rows, states, context):
        assert isinstance(context.plan, DOP853Plan)
        assert np.all(context.plan.frame[rows] == "lab")
        return super().apply(prepared, key_index, table, rows, states, context)


@pytest.mark.parametrize("workers", [1, 2])
def test_deferred_floquet_retains_shared_admissions_alongside_direct_plans(workers):
    n = 3
    constant = EnvelopeSpec("constant")
    shared = replace(
        track(n, constant, duration=512),
        frequency_ghz=np.full(n, 0.5),
        phase_rad=np.arange(n) * 0.1,
    )
    local = replace(
        shared, start_ns=np.full(n, 512.0), duration_ns=np.arange(n) + 513.0
    )
    direct = track(n, constant, start=1030, duration=1)
    timeline = replace(
        batch(n, duration=1031),
        tracks=(
            shared,
            replace(shared, frequency_ghz=np.full(n, 0.425)),
            local,
            replace(local, frequency_ghz=np.full(n, 0.425)),
            direct,
            direct,
            direct,
        ),
    )
    floquet, dop853 = PlannedFloquet(), PlannedDirect()
    result = run(
        Engine(workers, 0, kernels=(floquet, dop853, RecordingKernel("t4"))),
        ModelSource(
            np.eye(16, dtype=np.complex128) / 16, lambda level: synthetic_model(16)
        ),
        (timeline,),
    )
    assert result.paths.segments_by_kernel["floquet"] == 2 * n
    assert result.paths.segments_by_kernel["dop853"] == n
    assert sum(len(rows) for _, rows, _ in floquet.calls) == 2 * n
    assert sum(len(rows) for _, rows, _ in dop853.calls) == n


def test_short_weak_direct_and_long_strong_floquet():
    result, attempts = route((two_tone_batch(16, 0.02, 0), two_tone_batch(512, 0.5, 1)))
    assert result.paths.segments_by_kernel == {"dop853": 1, "floquet": 1}
    assert sorted(attempts) == [("dop853", 16.0), ("floquet", 512.0)]


def test_cheaper_guard_rejection_tries_other_two_tone_family():
    # Both orders retain numerical admission. A failed guard is not acceptance.
    result, attempts = route((two_tone_batch(16, 0.02, 0),), reject="dop853")
    assert attempts == [("dop853", 16.0), ("floquet", 16.0)]
    assert result.paths.segments_by_kernel == {"floquet": 1}
    result, attempts = route((two_tone_batch(512, 0.5, 0),), reject="floquet")
    assert attempts == [("floquet", 512.0), ("dop853", 512.0)]
    assert result.paths.segments_by_kernel == {"dop853": 1}


@pytest.mark.parametrize(
    ("workers", "chunks", "reverse"), [(1, 1, False), (2, 7, True), (2, 3, False)]
)
def test_two_tone_cost_order_independent_of_workers_chunks_and_row_order(
    workers, chunks, reverse
):
    batches = tuple(
        two_tone_batch(16 if i % 2 == 0 else 512, 0.02 if i % 2 == 0 else 0.5, i)
        for i in range(8)
    )
    result, attempts = route(
        batches[::-1] if reverse else batches, workers=workers, chunks=chunks
    )
    assert result.paths.segments_by_kernel == {"dop853": 4, "floquet": 4}
    assert sorted(attempts) == [("dop853", 16.0)] * 4 + [("floquet", 512.0)] * 4
