"""Hilbert-period C5 channel with phase translation and secular splitting."""

from dataclasses import replace

import numpy as np
import pytest
from zcu_tools.simulate.evolution import EnvelopeSpec, ModelSource
from zcu_tools.simulate.evolution.kernel import SegmentTable

from ..contract.kernel_suite import KernelCase, KernelContract, synthetic_model


@pytest.fixture(
    params=[
        (3, False),
        (4, False),
        (3, True),
        (4, True),
        (3, "threshold"),
        (4, "threshold"),
    ]
)
def kernel_case(request: pytest.FixtureRequest) -> KernelCase:
    from zcu_tools.simulate.evolution.kernels.floquet_period import FloquetPeriodKernel

    size, low_frequency = request.param
    model = synthetic_model(size)
    if low_frequency:
        model = replace(
            model,
            energies_ghz=np.arange(size) * 0.4,
            rates_per_ns=model.rates_per_ns * 1e-3,
            dephasing_per_ns=model.dephasing_per_ns * 1e-3,
        )
    source = ModelSource(
        np.eye(size, dtype=np.complex128) / size,
        lambda level: replace(model, level=level, identity=f"model-{level}"),
    )
    count = 6
    table = SegmentTable(
        timeline=np.arange(count),
        start_ns=np.arange(count, dtype=float) * 3.7,
        duration_ns=np.array([0.0, 0.47, 0.47, 0.39, 0.28, 0.11]),
        level=np.zeros((count, 1), dtype=np.int64),
        tones=np.ones(count, dtype=np.int64),
        tone_port=np.zeros((count, 1), dtype=np.int64),
        tone_frequency_ghz=np.array([8.0, 8.0, 8.0, -8.0, -8.0, 8.0])[:, None],
        tone_amplitude=np.array([0.15, 0.15, 0.15, -0.15, 0.0006, 0.0006])[:, None],
        # row 2's tail crosses the period boundary from its translated origin.
        tone_theta_rad=np.array([0.3, 0.0, 5.9, -5.9, 2.41, -3.1])[:, None],
        tone_envelope=np.zeros((count, 1), dtype=np.int64),
        tone_envelope_offset_ns=np.full((count, 1), 7.0),
        envelopes=(EnvelopeSpec("constant"),),
    )
    if low_frequency:
        table = replace(
            table,
            duration_ns=np.array([0.0, 0.7, 8.7, 8.7, 20.7, 20.7]),
            tone_frequency_ghz=np.array([0.3, 0.3, 0.3, -0.5, 0.582, 0.582])[:, None],
        )
    if low_frequency == "threshold":
        # Long strong rows exercise the calibrated upper splitting allowance.
        model = replace(
            model,
            rates_per_ns=model.rates_per_ns * 4,
            dephasing_per_ns=model.dephasing_per_ns * 4,
        )
        source = ModelSource(
            np.eye(size, dtype=np.complex128) / size,
            lambda level: replace(model, level=level, identity=f"model-{level}"),
        )
        table = replace(
            table,
            duration_ns=np.array([0.0, 0.7, 80.7, 8.7, 20.7, 20.7]),
            tone_frequency_ghz=np.array([0.3, 0.3, 0.3, -0.5, 0.582, 0.582])[:, None],
        )
    vectors = np.exp(1j * np.arange(count)[:, None] * np.arange(size)) / np.sqrt(size)
    states = vectors[:, :, None] * vectors[:, None, :].conj()
    states[-1] = np.eye(size) / size
    return KernelCase(FloquetPeriodKernel(), source, table, states)


class TestFloquetPeriod(KernelContract):
    pass


def test_key_excludes_apply_only_parameters(kernel_case: KernelCase) -> None:
    case = kernel_case
    rows = np.array([1, 2], dtype=np.int64)
    keys = case.kernel.prepare_keys(case.table, rows)
    assert keys[0] == keys[1]
    for field in (
        "duration_ns",
        "start_ns",
        "tone_theta_rad",
        "tone_envelope_offset_ns",
    ):
        values = getattr(case.table, field).copy()
        values[2] += 0.31
        changed = replace(case.table, **{field: values})
        np.testing.assert_array_equal(case.kernel.prepare_keys(changed, rows), keys)
    for field in ("level", "tone_port", "tone_frequency_ghz", "tone_amplitude"):
        values = getattr(case.table, field).copy()
        values[2] += 1
        changed = replace(case.table, **{field: values})
        other = case.kernel.prepare_keys(changed, rows)
        assert other[0] != other[1], field


def test_guard_rejects_low_carrier_shapes_and_large_split_estimate(
    kernel_case: KernelCase,
) -> None:
    case = kernel_case
    rows = np.arange(len(case.states), dtype=np.int64)

    class NoDensity(ModelSource):
        @property
        def initial_density(self):
            raise AssertionError("guard must not read density")

    source = NoDensity(case.states[0], case.source.model)
    assert case.kernel.guard(case.table, rows, source, case.budget).all()
    low = replace(case.table, tone_frequency_ghz=np.full((len(rows), 1), 0.01))
    assert not case.kernel.guard(low, rows, source, case.budget).any()
    sampled = EnvelopeSpec("sampled", np.ones(2, complex), np.array([0.0, 10.0]))
    shaped = replace(case.table, envelopes=(sampled,))
    assert not case.kernel.guard(shaped, rows, source, case.budget).any()
    long = replace(case.table, duration_ns=np.full(len(rows), 1e12))
    assert not case.kernel.guard(long, rows, source, case.budget).any()
    idle = replace(
        case.table,
        tones=np.zeros(len(rows), dtype=np.int64),
        tone_port=np.full((len(rows), 1), -1, dtype=np.int64),
        tone_envelope=np.full((len(rows), 1), -1, dtype=np.int64),
    )
    assert not case.kernel.guard(idle, rows, source, case.budget).any()


def test_engine_dispatches_rejected_dyson_rows_to_hilbert_period() -> None:
    from zcu_tools.simulate.evolution import Engine
    from zcu_tools.simulate.evolution.kernels.floquet_period import FloquetPeriodKernel
    from zcu_tools.simulate.evolution.kernels.t6 import T6Kernel

    from ..test_engine import RecordingKernel, batch, run, source, track

    pulse = replace(
        track(2, EnvelopeSpec("constant"), duration=0.3),
        frequency_ghz=np.full(2, 8.0),
        amplitude=np.full(2, 0.5),
    )
    later = replace(pulse, start_ns=np.full(2, 0.3), phase_rad=np.full(2, 0.7))
    timeline = replace(batch(2, duration=0.6), tracks=(pulse, later))
    engine = Engine(
        1,
        100000,
        kernels=(
            RecordingKernel("dyson", accepts=False),
            FloquetPeriodKernel(),
            T6Kernel(),
        ),
    )
    result = run(engine, source(), (timeline,))
    assert result.paths.segments_by_kernel == {"floquet_period": 4}
    assert result.paths.exceeded_by_kernel == {}


def test_engine_dispatches_low_carrier_strong_and_unknown_inputs() -> None:
    from zcu_tools.simulate.evolution import Engine

    from ..test_engine import batch, run, track

    model = synthetic_model(3)
    model = replace(
        model,
        energies_ghz=np.arange(3) * 0.4,
        rates_per_ns=model.rates_per_ns * 1e-3,
        dephasing_per_ns=model.dephasing_per_ns * 1e-3,
    )
    source = ModelSource(np.diag([1.0, 0.0, 0.0]).astype(complex), lambda level: model)
    pulse = replace(
        track(1, EnvelopeSpec("constant"), duration=8.7),
        frequency_ghz=np.full(1, 0.4),
        amplitude=np.full(1, 0.15),
    )
    later = replace(pulse, start_ns=np.full(1, 8.7), phase_rad=np.full(1, 5.9))
    result = run(
        Engine(1, 100000),
        source,
        (replace(batch(1, duration=17.4), tracks=(pulse, later)),),
    )
    assert result.paths.segments_by_kernel == {"floquet_period": 2}
    assert result.paths.exceeded_by_kernel == {}

    weak = replace(pulse, amplitude=np.full(1, 1e-7))
    result = run(
        Engine(1, 100000),
        source,
        (replace(batch(1, duration=8.7), tracks=(weak,)),),
    )
    assert result.paths.segments_by_kernel == {"dyson": 1}
