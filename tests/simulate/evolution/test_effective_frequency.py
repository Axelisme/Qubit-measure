"""Coherent waiting frequencies must not depend on the selected algorithm."""

from contextlib import closing
from dataclasses import replace
from threading import Event

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
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
from zcu_tools.simulate.evolution.kernels.dop853 import DOP853Kernel
from zcu_tools.simulate.evolution.kernels.t4 import T4

from ._engine import RecordingKernel


def strict_coherences(times: NDArray[np.float64]) -> NDArray[np.complex128]:
    """Independent pure-state Schrodinger oracle, without a production RHS.

    Remove the bare 7.5 GHz phase exactly, retain both counter-rotating drives,
    then restore the lab-frame g-e coherence at the requested waiting times.
    """

    def rhs(time: float, state: NDArray[np.complex128]) -> NDArray[np.complex128]:
        drive = 0.01 * (
            0.12 * np.cos(2 * np.pi * 7.58 * time)
            + 0.08 * np.cos(2 * np.pi * 8.5 * time)
        )
        off_diagonal = drive * np.exp(-2j * np.pi * 7.5 * time)
        return (
            -2j
            * np.pi
            * np.array([off_diagonal * state[1], off_diagonal.conjugate() * state[0]])
        )

    solution = solve_ivp(
        rhs,
        (0.0, float(times[-1])),
        np.full(2, 1 / np.sqrt(2), dtype=np.complex128),
        method="DOP853",
        t_eval=times,
        rtol=2e-11,
        atol=2e-13,
        max_step=0.5,
    )
    assert solution.success, solution.message
    return solution.y[0] * solution.y[1].conjugate() * np.exp(2j * np.pi * 7.5 * times)


@pytest.mark.parametrize("chained", [False, True], ids=["independent", "touching"])
def test_two_tone_wait_preserves_strict_frequency_at_short_and_long_times(
    chained: bool,
):
    times = np.array([32.0, 128.0])
    x = np.array([[0.0, 0.01], [0.01, 0.0]], dtype=np.complex128)
    model = SecularModel(
        (0,),
        "coherent-two-tone-wait",
        np.array([0.0, 7.5]),
        np.eye(2, dtype=np.complex128),
        tuple(SecularDrive(f"p{port}", x, np.zeros_like(x)) for port in range(2)),
        np.zeros((2, 2)),
        np.zeros((2, 2)),
    )
    tracks = tuple(
        DriveTrack(
            f"p{port}",
            np.zeros(2),
            times,
            np.full(2, frequency),
            np.zeros(2),
            np.full(2, gain),
            (EnvelopeSpec("constant"),),
            np.zeros(2, dtype=np.int64),
            np.ones(2, dtype=np.bool_),
        )
        for port, frequency, gain in ((0, 7.58, 0.12), (1, 8.5, 0.08))
    )
    # An extra schedule boundary creates touching rows without changing the
    # Hamiltonian or physical reference. This also covers lab chains.
    flux = (
        FluxSchedule(
            np.array([0, 2, 4]),
            np.array([0.0, 16.0, 0.0, 16.0]),
            np.zeros((4, 1), dtype=np.int64),
        )
        if chained
        else FluxSchedule(np.arange(3), np.zeros(2), np.zeros((2, 1), dtype=np.int64))
    )
    batch = TimelineBatch(np.arange(2), tracks, flux, times)
    with (
        closing(
            ModelSource(
                np.full((2, 2), 0.5, dtype=np.complex128),
                lambda level: replace(model, level=level),
            )
        ) as source,
        closing(
            Engine(
                1,
                0,
                kernels=(
                    T4(),
                    DOP853Kernel(),
                    RecordingKernel("floquet", accepts=False),
                ),
            )
        ) as engine,
    ):
        # Omit optional approximations, not the direct solver's numerical guard.
        result = engine.evolve(source, (batch,), budget=1e-3, cancel=Event())
    assert result.paths.segments_by_kernel == {"dop853": 4 if chained else 2}
    reference = strict_coherences(times)
    # A vanished fringe has no frequency; angle(0) must not count as agreement.
    assert np.all(np.abs(result.densities[:, 0, 1]) > 0.1)
    assert np.all(np.abs(reference) > 0.1)
    phase_error = np.unwrap(np.angle(result.densities[:, 0, 1] / reference))
    frequency_error_ghz = phase_error / (2 * np.pi * times)
    # Endpoint averages catch a biased f01. The window slope also catches an
    # accumulating phase error that a fixed endpoint offset could conceal.
    np.testing.assert_array_less(np.abs(frequency_error_ghz), 10e-6)
    window_error_ghz = np.diff(phase_error) / (2 * np.pi * np.diff(times))
    np.testing.assert_array_less(np.abs(window_error_ghz), 10e-6)
    assert abs(frequency_error_ghz[1] - frequency_error_ghz[0]) <= 10e-6
