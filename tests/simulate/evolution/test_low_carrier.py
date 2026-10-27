"""Low-carrier constant tones through the public Engine, not device recipes."""

from dataclasses import replace

import numpy as np
import pytest
from zcu_tools.simulate.evolution import Engine, EnvelopeSpec, ModelSource
from zcu_tools.simulate.evolution.kernel import (
    ApplyContext,
    KernelContext,
    SegmentTable,
)
from zcu_tools.simulate.evolution.kernels.floquet_period import FloquetPeriodKernel

from ._engine import RecordingKernel, batch, run, track
from .contract.kernel_suite import KernelCase, reference, synthetic_model


@pytest.mark.parametrize("frequency", [0.038, -0.05, 0.22, 0.294])
@pytest.mark.parametrize("coherent", [False, True])
def test_low_carrier_strong_tones_match_reference(frequency: float, coherent: bool):
    model = synthetic_model(3)
    model = replace(
        model,
        energies_ghz=np.array([0.0, 0.05, 0.4]),
        rates_per_ns=model.rates_per_ns * 1e-3,
        dephasing_per_ns=model.dephasing_per_ns * 1e-3,
    )
    vector = np.array([1, 1j, -1]) / np.sqrt(3)
    initial = np.outer(vector, vector.conj()) if coherent else np.diag([0.5, 0.3, 0.2])
    src = ModelSource(initial.astype(complex), lambda level: model)
    duration = 40.7
    pulse = replace(
        track(1, EnvelopeSpec("constant"), duration=duration),
        frequency_ghz=np.array([frequency]),
        amplitude=np.array([0.2]),
        phase_rad=np.array([5.9]),
    )
    later = replace(pulse, start_ns=np.array([duration]), phase_rad=np.array([-0.7]))
    timeline = replace(batch(1, duration=2 * duration), tracks=(pulse, later))
    result = run(Engine(1), src, (timeline,))
    assert result.paths.segments_by_kernel == {"floquet_period": 2}
    assert result.paths.exceeded_by_kernel == {}

    # The independent strict Lindblad reference includes phase translation at
    # the second pulse's absolute start and a tail crossing a period boundary.
    table = SegmentTable(
        timeline=np.zeros(2, dtype=np.int64),
        start_ns=np.array([0.0, duration]),
        duration_ns=np.full(2, duration),
        level=np.zeros((2, 1), dtype=np.int64),
        tones=np.ones(2, dtype=np.int64),
        tone_port=np.zeros((2, 1), dtype=np.int64),
        tone_frequency_ghz=np.full((2, 1), frequency),
        tone_amplitude=np.full((2, 1), 0.2),
        tone_theta_rad=np.array([5.9, -0.7 + 2 * np.pi * frequency * duration])[
            :, None
        ],
        tone_envelope=np.zeros((2, 1), dtype=np.int64),
        tone_envelope_offset_ns=np.zeros((2, 1)),
        envelopes=(EnvelopeSpec("constant"),),
    )
    case = KernelCase(FloquetPeriodKernel(), src, table, initial[None].astype(complex))
    expected = reference(case, 1, reference(case, 0, initial.astype(complex)))
    np.testing.assert_allclose(result.densities[0], expected, atol=1e-2, rtol=0)
    assert result.paths.max_trace_error < 1e-8


@pytest.mark.parametrize("workers,chunks", [(1, 3), (2, 1), (2, 3)])
def test_gain_scan_is_bitwise_invariant_to_workers_and_chunks(workers, chunks):
    model = synthetic_model(3)
    model = replace(
        model,
        energies_ghz=np.array([0.0, 0.8, 1.6]),
        rates_per_ns=model.rates_per_ns * 1e-3,
        dephasing_per_ns=model.dephasing_per_ns * 1e-3,
    )
    src = ModelSource(np.ones((3, 3), dtype=np.complex128) / 3, lambda level: model)
    count = 40
    pulse = replace(
        track(count, EnvelopeSpec("constant"), duration=23.7),
        frequency_ghz=np.full(count, 0.05),
        amplitude=0.003 + np.arange(count) * 1e-5,
        phase_rad=np.full(count, 1.73),
    )
    timeline = replace(batch(count, duration=23.7), tracks=(pulse,))

    def evolve(worker_count, chunk_count):
        return run(
            Engine(
                worker_count,
                chunks_per_worker=chunk_count,
                kernels=(
                    RecordingKernel("dyson", accepts=False),
                    FloquetPeriodKernel(),
                ),
            ),
            src,
            (timeline,),
        )

    reference = evolve(1, 1)
    actual = evolve(workers, chunks)
    assert actual.paths.segments_by_kernel == {"floquet_period": count}
    np.testing.assert_array_equal(actual.densities, reference.densities)


def test_many_low_carrier_keys_do_not_all_stay_live():
    import weakref
    from dataclasses import dataclass

    from ._engine import RecordingKernel

    live = 0
    peak = 0

    def release(count: int):
        nonlocal live
        live -= count

    @dataclass
    class Payload:
        nbytes: int

    class ObservedPeriod(FloquetPeriodKernel):
        def guard(self, table, rows, context: KernelContext):
            return np.ones(len(rows), dtype=bool)

        def prepare(self, keys, table, rows, context: KernelContext):
            nonlocal live, peak
            live += len(keys)
            peak = max(peak, live)
            payload = Payload(len(keys))
            weakref.finalize(payload, release, len(keys))
            return payload

        def apply(
            self, prepared, key_index, table, rows, states, context: ApplyContext
        ):
            return np.zeros(len(rows), dtype=bool)

    count = 17
    pulse = replace(
        track(count, EnvelopeSpec("constant"), duration=1.0),
        frequency_ghz=np.linspace(0.038, 0.062, count),
        amplitude=np.full(count, 0.2),
    )
    # Each key is repeated in its timeline. A-stage must not retain the whole
    # low-carrier sweep merely because each key occurs twice.
    later = replace(pulse, start_ns=np.ones(count))
    timeline = replace(batch(count, duration=2.0), tracks=(pulse, later))
    result = run(
        Engine(
            1,
            chunks_per_worker=1,
            kernels=(
                RecordingKernel("dyson", accepts=False),
                ObservedPeriod(),
            ),
        ),
        ModelSource((np.eye(3) / 3).astype(complex), lambda level: synthetic_model(3)),
        (timeline,),
    )
    assert result.paths.segments_by_kernel == {"floquet_period": 2 * count}
    assert peak < count
    assert live == 0


def test_long_plateau_reuses_period_across_durations():
    frequency = 0.05
    from ._engine import RecordingKernel

    model = synthetic_model(3)
    model = replace(
        model,
        energies_ghz=np.array([0.0, abs(frequency), 0.9]),
        rates_per_ns=model.rates_per_ns * 0.01,
        dephasing_per_ns=model.dephasing_per_ns * 0.01,
    )
    vector = np.array([1, 1j, -1]) / np.sqrt(3)
    initial = np.outer(vector, vector.conj())
    src = ModelSource(initial, lambda level: model)
    durations = np.array([37.3, 81.7, 163.1])
    pulse = replace(
        track(3, EnvelopeSpec("constant")),
        frequency_ghz=np.full(3, frequency),
        duration_ns=durations,
        amplitude=np.full(3, 0.2),
        phase_rad=np.full(3, 1.3),
    )
    timeline = replace(batch(3), tracks=(pulse,), end_ns=durations)
    prepared_keys = []

    class ObservedPeriod(FloquetPeriodKernel):
        def prepare(self, keys, table, rows, context: KernelContext):
            prepared_keys.extend(keys)
            return super().prepare(keys, table, rows, context)

    result = run(
        Engine(
            1,
            kernels=(
                RecordingKernel("dyson", accepts=False),
                ObservedPeriod(),
                RecordingKernel("t6"),
            ),
        ),
        src,
        (timeline,),
    )
    assert result.paths.segments_by_kernel == {"floquet_period": 3}
    assert len(prepared_keys) == 1
    table = SegmentTable(
        timeline=np.arange(3),
        start_ns=np.zeros(3),
        duration_ns=durations,
        level=np.zeros((3, 1), dtype=np.int64),
        tones=np.ones(3, dtype=np.int64),
        tone_port=np.zeros((3, 1), dtype=np.int64),
        tone_frequency_ghz=pulse.frequency_ghz[:, None],
        tone_amplitude=pulse.amplitude[:, None],
        tone_theta_rad=pulse.phase_rad[:, None],
        tone_envelope=np.zeros((3, 1), dtype=np.int64),
        tone_envelope_offset_ns=np.zeros((3, 1)),
        envelopes=(EnvelopeSpec("constant"),),
    )
    case = KernelCase(FloquetPeriodKernel(), src, table, result.densities)
    expected = np.array([reference(case, row, initial) for row in range(3)])
    np.testing.assert_allclose(result.densities, expected, atol=1e-2, rtol=0)
    assert result.paths.max_trace_error < 1e-8
