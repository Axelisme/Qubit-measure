"""Modal Dyson agrees with independent order-by-order time integration.

A zero Stark operator makes the Floquet frame exact without reading the opaque
prepared payload. The reference integrates the three Dyson orders separately,
not the full dynamics, so it detects roundoff-scale evaluation regressions.
"""

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from zcu_tools.simulate.evolution import (
    EnvelopeSpec,
    ModelSource,
    SecularDrive,
    SecularModel,
    kernel,
)
from zcu_tools.simulate.evolution.kernel import ApplyContext
from zcu_tools.simulate.evolution.kernels.floquet import FloquetKernel


def _integrated_orders(
    energies: NDArray[np.float64],
    drive: SecularDrive,
    state: NDArray[np.complex128],
    duration: float,
    frequency: float,
    gain: float,
    theta: float,
) -> NDArray[np.complex128]:
    if duration == 0:
        return state.copy()
    size = len(energies)
    free = np.diag(energies)

    def rhs(time: float, flat: NDArray[np.complex128]) -> NDArray[np.complex128]:
        orders = flat.reshape(3, size, size)
        phase = 2 * np.pi * frequency * time + theta
        probe = gain * (
            np.cos(phase) * drive.in_phase_ghz + np.sin(phase) * drive.quadrature_ghz
        )
        result = -2j * np.pi * (free @ orders - orders @ free)
        result[1:] += -2j * np.pi * (probe @ orders[:-1] - orders[:-1] @ probe)
        return result.ravel()

    initial = np.zeros((3, size, size), dtype=np.complex128)
    initial[0] = state
    solution = solve_ivp(
        rhs,
        (0, duration),
        initial.ravel(),
        method="DOP853",
        rtol=2e-12,
        atol=2e-14,
        max_step=0.05,
    )
    assert solution.success
    return solution.y[:, -1].reshape(3, size, size).sum(axis=0)


@pytest.mark.parametrize(
    ("gap", "frequency"),
    [
        (0.0, 0.0),  # Three coincident poles, including repeated population modes.
        (0.13, 0.13),  # Exact carrier resonance.
        # In one apply, diagonal modal endpoints coincide while off-diagonal
        # gaps exceed 0.5/duration. DC also gives repeated inner poles, so the
        # independent reference checks both stable branches and dense contraction.
        (0.13, 0.0),
        ((0.5 - 1e-8) / (2 * np.pi), 0.0),
        ((0.5 + 1e-8) / (2 * np.pi), 0.0),
        ((1e-5 - 1e-8) / (2 * np.pi), 0.0),
        ((1e-5 + 1e-8) / (2 * np.pi), 0.0),
        ((1e-6 - 1e-9) / (2 * np.pi), 0.0),
        ((1e-6 + 1e-9) / (2 * np.pi), 0.0),
    ],
    ids=[
        "coincident",
        "resonant",
        "mixed-endpoints-and-inner-poles",
        "endpoint-below",
        "endpoint-above",
        "first-below",
        "first-above",
        "repeated-below",
        "repeated-above",
    ],
)
def test_modal_apply_matches_independent_dyson_orders(
    gap: float, frequency: float
) -> None:
    size, count = 3, 3
    energies = np.array([0.0, 0.0, gap])
    x = np.array([[0.01, 0.02, 0.03], [0.02, -0.01, 0.01], [0.03, 0.01, 0.0]])
    y = 0.01j * np.array([[0, 1, -1], [-1, 0, 2], [1, -2, 0]])
    probe = SecularDrive("probe", x.astype(complex), y)
    zero = np.zeros((size, size), dtype=np.complex128)
    model = SecularModel(
        (0,),
        "modal-reference",
        energies,
        np.eye(size, dtype=complex),
        (SecularDrive("stark", zero, zero), probe),
        np.zeros((size, size)),
        np.zeros((size, size)),
    )
    vector = np.exp(1j * np.array([0.2, -0.8, 1.1])) / np.sqrt(size)
    states = np.stack(
        [np.outer(vector, vector.conj()), np.diag([0.2, 0.3, 0.5]), np.eye(size) / size]
    ).astype(np.complex128)
    source = ModelSource(states[0], lambda level: model)
    table = kernel.SegmentTable(
        timeline=np.arange(count),
        start_ns=np.array([0.0, 7.0, 19.0]),
        duration_ns=np.array([1.0, 1.7, 0.0]),
        level=np.zeros((count, 1), dtype=np.int64),
        tones=np.full(count, 2),
        tone_port=np.tile([0, 1], (count, 1)),
        tone_frequency_ghz=np.tile([3.7, frequency], (count, 1)),
        tone_amplitude=np.column_stack((np.full(count, 0.2), [0.02, -0.015, 0.01])),
        tone_theta_rad=np.array([[0.7, -0.3], [-0.2, 1.1], [0.5, 0.2]]),
        tone_envelope=np.zeros((count, 2), dtype=np.int64),
        tone_envelope_offset_ns=np.zeros((count, 2)),
        envelopes=(EnvelopeSpec("constant"),),
    )
    floquet = FloquetKernel()
    rows = np.arange(count, dtype=np.int64)
    guarded = floquet.guard_prepared(table, rows, source, 1e-3)
    assert guarded.accepted.all()
    keys, representatives, inverse = np.unique(
        floquet.prepare_keys(table, rows), return_index=True, return_inverse=True
    )
    prepared = floquet.prepare(keys, table, rows[representatives], source, 1e-3)
    actual = states.copy()
    exceeded = floquet.apply(
        prepared, inverse, table, rows, actual, ApplyContext(1e-3, guarded.plan)
    )
    expected = np.stack(
        [
            _integrated_orders(
                energies,
                probe,
                states[row],
                table.duration_ns[row],
                frequency,
                table.tone_amplitude[row, 1],
                table.tone_theta_rad[row, 1],
            )
            for row in rows
        ]
    )
    assert not exceeded.any()
    assert np.isfinite(actual).all()
    # The existing small-gap formula has cancellation error around 1.5e-12.
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-11)
    np.testing.assert_allclose(
        actual, actual.conj().transpose(0, 2, 1), rtol=0, atol=3e-13
    )
    np.testing.assert_allclose(
        np.trace(actual, axis1=1, axis2=2), 1, rtol=0, atol=3e-13
    )
