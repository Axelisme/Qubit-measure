"""Constant two-tone cost routing at the Engine/Kernel boundary."""

from dataclasses import replace

import numpy as np
import pytest
from zcu_tools.simulate.evolution import Engine, EnvelopeSpec, ModelSource

from ._engine import RecordingKernel, batch, run, track
from .contract.kernel_suite import synthetic_model


class GuardRecorder(RecordingKernel):
    """Record guard attempts by duration; keep all dynamics as identity."""

    def __init__(self, name, attempts, *, accepts=True):
        super().__init__(name, accepts=accepts)
        self.attempts = attempts

    def guard(self, table, rows, source, budget):
        self.attempts.extend((self.name, float(table.duration_ns[r])) for r in rows)
        return super().guard(table, rows, source, budget)


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
