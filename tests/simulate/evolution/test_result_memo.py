"""Exact-input reuse through Engine and its injected kernel collaborator seam."""

from contextlib import closing
from dataclasses import replace
from threading import Event

import numpy as np
import pytest
from zcu_tools.simulate.evolution import (
    Cancelled,
    Engine,
    EnvelopeSpec,
    FluxExpansion,
    FluxSchedule,
    FluxWaveform,
    ModelSource,
    TimelineBatch,
)
from zcu_tools.simulate.evolution.kernel import KernelContext

from ._engine import RecordingKernel, batch, source, track
from .contract.kernel_suite import synthetic_model


class CountingKernel(RecordingKernel):
    """Identity dynamics with observations of all numerical entry points."""

    def __init__(self, name="t4", *, cancel=None):
        super().__init__(name, cancel=cancel)
        self.guards = 0

    def guard(self, table, rows, context: KernelContext):
        self.guards += 1
        return super().guard(table, rows, context)

    @property
    def counts(self):
        return self.guards, self.preparations, len(self.calls)


def test_repeat_skips_all_kernel_work_and_returns_isolated_bitwise_results():
    kernel = CountingKernel()
    with closing(Engine(1, 100_000, kernels=(kernel,))) as engine:
        first = engine.evolve(source(), (batch(2),), cancel=Event())
        expected = first.densities.tobytes()
        counts = kernel.counts
        # An equivalent new source/batch object must hit by content, not id().
        second = engine.evolve(source(), (batch(2),), cancel=Event())
        assert kernel.counts == counts
        assert second.densities.tobytes() == expected
        assert second.points.tobytes() == first.points.tobytes()
        assert second.paths == first.paths
        # Even re-enabling NumPy writes must not corrupt the stored result.
        for result in (first, second):
            result.densities.setflags(write=True)
            result.densities[:] = 0
            result.points.setflags(write=True)
            result.points[:] = 99
        third = engine.evolve(source(), (batch(2),), cancel=Event())
        assert third.densities.tobytes() == expected
        assert third.points.tolist() == [0, 1]
        assert kernel.counts == counts


@pytest.mark.parametrize(
    "changed",
    [
        "points",
        "end_ns",
        "start_ns",
        "duration_ns",
        "frequency_ghz",
        "phase_rad",
        "amplitude",
        "active",
        "envelope_index",
        "samples",
        "sample_ns",
        "tracks",
        "flux",
        "waveform",
        "waveform_absent",
        "budget",
        "initial",
        "source",
        "expansion",
        "batch_order",
    ],
)
def test_complete_input_changes_miss_after_an_exact_hit(changed):
    envelope = EnvelopeSpec("sampled", np.array([0.2, 0.8]), np.array([0.0, 1.0]))
    tr = track(1, envelope)
    tr = replace(
        tr, envelopes=(envelope, replace(envelope, samples=envelope.samples * 0.9))
    )
    waveform = FluxWaveform(np.array([0, 2]), np.array([0.0, 3.0]), np.zeros((2, 1)))
    b = replace(batch(1), tracks=(tr,), flux_waveform=waveform)
    other = replace(b, points=np.array([1]))
    model = synthetic_model()
    expansion = FluxExpansion(np.diag(model.energies_ghz).astype(complex), ())
    src = ModelSource(
        source().initial_density, lambda level: replace(model, level=level), expansion
    )
    kernels = tuple(CountingKernel(name) for name in ("t4", "dop853", "flux_dop853"))
    with closing(Engine(1, 100_000, kernels=kernels)) as engine:
        engine.evolve(src, (b, other), cancel=Event())
        counts = tuple(k.counts for k in kernels)
        engine.evolve(src, (b, other), cancel=Event())
        assert tuple(k.counts for k in kernels) == counts
        budget = 1e-3
        b = _changed_batch(b, waveform, changed)
        if changed == "budget":
            budget *= 0.5
        elif changed in ("initial", "source", "expansion"):
            initial = (
                np.diag([0.3, 0.2, 0.5]).astype(complex)
                if changed == "initial"
                else src.initial_density
            )
            if changed == "source":
                model = replace(model, identity="another-source")
            if changed == "expansion":
                expansion = replace(expansion, static=expansion.static * 1.01)
            src = ModelSource(
                initial, lambda level: replace(model, level=level), expansion
            )
        batches = (other, b) if changed == "batch_order" else (b, other)
        engine.evolve(src, batches, budget=budget, cancel=Event())
        assert tuple(k.counts for k in kernels) != counts


def _changed_batch(
    b: TimelineBatch, waveform: FluxWaveform, changed: str
) -> TimelineBatch:
    tr = b.tracks[0]
    envelope = tr.envelopes[0]
    if changed in (
        "start_ns",
        "duration_ns",
        "frequency_ghz",
        "phase_rad",
        "amplitude",
    ):
        tr = replace(tr, **{changed: getattr(tr, changed) + 0.1})
        b = replace(b, tracks=(tr,))
    elif changed == "active":
        b = replace(b, tracks=(replace(tr, active=np.array([False])),))
    elif changed == "envelope_index":
        b = replace(b, tracks=(replace(tr, envelope_index=np.array([1])),))
    elif changed in ("samples", "sample_ns"):
        env = replace(envelope, **{changed: getattr(envelope, changed) * 0.9})
        b = replace(b, tracks=(replace(tr, envelopes=(env, tr.envelopes[1])),))
    elif changed == "tracks":
        b = replace(b, tracks=())
    elif changed == "points":
        b = replace(b, points=np.array([2]))
    elif changed == "end_ns":
        b = replace(b, end_ns=b.end_ns + 1)
    elif changed == "flux":
        b = replace(
            b,
            flux=FluxSchedule(
                np.array([0, 2]), np.array([0.0, 1.0]), np.array([[0], [1]])
            ),
        )
    elif changed == "waveform":
        b = replace(b, flux_waveform=replace(waveform, flux=waveform.flux + 0.001))
    elif changed == "waveform_absent":
        b = replace(b, flux_waveform=None)
    return b


def test_cancelled_work_is_not_admitted_and_hits_still_check_cancel():
    cancel = Event()
    kernel = CountingKernel(cancel=cancel)
    src, b = source(), batch(1)
    with closing(Engine(1, 100_000, kernels=(kernel,))) as engine:
        with pytest.raises(Cancelled):
            engine.evolve(src, (b,), cancel=cancel)
        kernel.cancel = None
        cancel.clear()
        result = engine.evolve(src, (b,), cancel=cancel)
        assert len(kernel.calls) == 2
        counts = kernel.counts
        repeated = engine.evolve(src, (b,), cancel=cancel)
        assert kernel.counts == counts
        assert repeated.densities.tobytes() == result.densities.tobytes()
        cancel.set()
        with pytest.raises(Cancelled):
            engine.evolve(src, (b,), cancel=cancel)
        assert kernel.counts == counts


def test_result_lru_touches_hits_and_does_not_admit_oversize_results():
    # 100 three-level results need 15,200 array bytes plus small diagnostics.
    # 40,000 bytes holds two of these, but not three, including metadata.
    kernel = CountingKernel()
    src = source()
    a, b, c = (batch(100, duration=d) for d in (1.0, 2.0, 3.0))
    with closing(Engine(1, 40_000, kernels=(kernel,))) as engine:
        engine.evolve(src, (a,), cancel=Event())
        engine.evolve(src, (b,), cancel=Event())
        counts = kernel.counts
        engine.evolve(src, (a,), cancel=Event())
        assert kernel.counts == counts
        engine.evolve(src, (c,), cancel=Event())
        counts = kernel.counts
        engine.evolve(src, (a,), cancel=Event())
        assert kernel.counts == counts
        engine.evolve(src, (b,), cancel=Event())
        assert kernel.counts != counts
        # Oversize admission must not evict the just-used small entry.
        engine.evolve(src, (batch(300),), cancel=Event())
        counts = kernel.counts
        engine.evolve(src, (b,), cancel=Event())
        assert kernel.counts == counts


def test_disabled_result_memo_keeps_executing():
    kernel = CountingKernel()
    with closing(Engine(1, 0, kernels=(kernel,))) as engine:
        engine.evolve(source(), (batch(1),), cancel=Event())
        counts = kernel.counts
        engine.evolve(source(), (batch(1),), cancel=Event())
        assert kernel.counts != counts
