"""Full-Hamiltonian plateau channels against an independent dense Lindblad ODE."""

from dataclasses import replace

import numpy as np
import pytest
from zcu_tools.simulate.evolution import Engine, EnvelopeSpec, ModelSource
from zcu_tools.simulate.evolution.kernel import (
    ApplyContext,
    KernelContext,
    SegmentTable,
)
from zcu_tools.simulate.evolution.kernels.slow_envelope import SlowEnvelopeKernel
from zcu_tools.simulate.evolution.workplan import WorkPlan

from .._engine import batch, run, track
from ..contract.kernel_suite import KernelCase, reference, synthetic_model


def plateau_case(frequency: float, *, segments: int = 20) -> KernelCase:
    """21-level weak-drive fixture with both relaxation and pure dephasing."""
    model = synthetic_model(21)
    model = replace(
        model,
        drives=tuple(
            replace(
                drive,
                in_phase_ghz=drive.in_phase_ghz * 0.02,
                quadrature_ghz=drive.quadrature_ghz * 0.02,
            )
            for drive in model.drives
        ),
        rates_per_ns=model.rates_per_ns * 1e-3,
        dephasing_per_ns=model.dephasing_per_ns * 1e-3,
    )
    initial = np.diag(np.r_[0.8, 0.15, 0.05, np.zeros(18)]).astype(complex)
    source = ModelSource(
        initial, lambda level: model, slow_envelope_segments_per_period=segments
    )
    table = SegmentTable(
        timeline=np.array([0]),
        start_ns=np.array([17.0]),
        duration_ns=np.array([50.0]),
        level=np.zeros((1, 1), dtype=np.int64),
        tones=np.ones(1, dtype=np.int64),
        tone_port=np.zeros((1, 1), dtype=np.int64),
        tone_frequency_ghz=np.array([[frequency]]),
        tone_amplitude=np.array([[0.2]]),
        tone_theta_rad=np.array([[1.1]]),
        tone_envelope=np.zeros((1, 1), dtype=np.int64),
        tone_envelope_offset_ns=np.zeros((1, 1)),
        envelopes=(EnvelopeSpec("constant"),),
    )
    return KernelCase(SlowEnvelopeKernel(), source, table, initial[None])


@pytest.mark.parametrize("frequency", [0.001, 0.003, 0.01, 0.03, -0.01, 0.0])
def test_populations_match_dense_ode(frequency: float) -> None:
    case = plateau_case(frequency)
    rows = np.array([0])
    context = KernelContext(case.source, case.budget, WorkPlan())
    assert case.kernel.guard(case.table, rows, context).all()
    prepared = case.kernel.prepare(
        case.kernel.prepare_keys(case.table, rows, context), case.table, rows, context
    )
    actual = case.states.copy()
    case.kernel.apply(
        prepared, rows, case.table, rows, actual, ApplyContext(case.budget)
    )
    expected = reference(case, 0, case.states[0])
    error = np.max(np.abs(actual[0].diagonal().real - expected.diagonal().real))
    assert error <= 1e-3
    np.testing.assert_allclose(np.trace(actual[0]), 1.0, atol=1e-10)

    pulse = replace(
        track(1, EnvelopeSpec("constant"), duration=50.0),
        frequency_ghz=np.array([frequency]),
        amplitude=np.array([0.2]),
        phase_rad=np.array([1.1]),
    )
    timeline = replace(batch(1, duration=50.0), tracks=(pulse,))
    result = run(Engine(1, 0), case.source, (timeline,))
    expected_route = "floquet_period" if abs(frequency) >= 0.03 else "slow_envelope"
    assert result.paths.segments_by_kernel == {expected_route: 1}
    np.testing.assert_allclose(
        result.densities[0].diagonal().real, expected.diagonal().real, atol=1e-3, rtol=0
    )


def test_segment_policy_refines_the_channel_and_changes_prepare_identity() -> None:
    coarse = plateau_case(0.01, segments=1)
    fine = plateau_case(0.01, segments=200)
    rows = np.array([0])
    results, keys = [], []
    for case in (coarse, fine):
        context = KernelContext(case.source, case.budget, WorkPlan())
        key = case.kernel.prepare_keys(case.table, rows, context)
        keys.append(key)
        prepared = case.kernel.prepare(key, case.table, rows, context)
        state = case.states.copy()
        case.kernel.apply(
            prepared, rows, case.table, rows, state, ApplyContext(case.budget)
        )
        results.append(state[0].diagonal().real)
    expected = reference(fine, 0, fine.states[0]).diagonal().real
    assert keys[0][0] != keys[1][0]
    assert np.max(np.abs(results[1] - expected)) < np.max(np.abs(results[0] - expected))


def test_guard_rejects_a_table_without_carrier_columns() -> None:
    case = plateau_case(0.001)
    table = replace(
        case.table,
        tones=np.zeros(1, dtype=np.int64),
        **{
            name: getattr(case.table, name)[:, :0]
            for name in (
                "tone_port",
                "tone_frequency_ghz",
                "tone_amplitude",
                "tone_theta_rad",
                "tone_envelope",
                "tone_envelope_offset_ns",
            )
        },
    )
    context = KernelContext(case.source, case.budget, WorkPlan())
    assert not case.kernel.guard(table, np.array([0]), context).any()


@pytest.mark.parametrize("segments", [0, -1, True, 1.5])
def test_invalid_segment_policy_is_rejected(segments) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        ModelSource(
            np.eye(3, dtype=np.complex128) / 3,
            lambda level: synthetic_model(3),
            slow_envelope_segments_per_period=segments,
        )
