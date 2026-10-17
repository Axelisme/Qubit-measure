"""Shaped evolution must resolve envelope motion and secular dissipation."""

from threading import Event

import numpy as np
import pytest
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


def circular_pulse(
    frequency: float,
    times: np.ndarray,
    samples: np.ndarray,
    *,
    n: int = 1,
    rate: float = 0,
) -> tuple[ModelSource, TimelineBatch]:
    """A circular resonant two-level drive of strength 0.05 GHz."""
    x = (0.05 * np.array([[0, 1], [1, 0]])).astype(np.complex128)
    y = (0.05 * np.array([[0, 1j], [-1j, 0]])).astype(np.complex128)
    model = SecularModel(
        (0,),
        "circular-resonant",
        np.array([min(0.0, frequency), max(0.0, frequency)]),
        np.eye(2, dtype=complex),
        (SecularDrive("drive", x, y if frequency > 0 else -y),),
        np.array([[0.0, rate], [0.0, 0.0]]),
        np.zeros((2, 2)),
    )
    source = ModelSource(np.diag([1.0, 0.0]).astype(complex), lambda _: model)
    envelope = EnvelopeSpec("sampled", samples.astype(complex), times)
    track = DriveTrack(
        "drive",
        np.zeros(n),
        np.full(n, times[-1]),
        np.full(n, frequency),
        np.full(n, 0.37),
        np.ones(n),
        (envelope,),
        np.zeros(n, dtype=np.int64),
        np.ones(n, dtype=bool),
    )
    batch = TimelineBatch(
        np.arange(n),
        (track,),
        FluxSchedule(np.arange(n + 1), np.zeros(n), np.zeros((n, 1), dtype=np.int64)),
        np.full(n, times[-1]),
    )
    return source, batch


def evolve(source: ModelSource, batch: TimelineBatch) -> np.ndarray:
    engine = Engine(1, 0)
    try:
        return engine.evolve(source, (batch,), cancel=Event()).densities
    finally:
        engine.close()


@pytest.mark.parametrize("frequency", [0.05, -0.05])
def test_subperiod_ramp_preserves_resonant_population_transfer(
    frequency: float,
) -> None:
    # The exact rotation is 2*pi*strength*integral(envelope). A small terminal
    # slope prevents plateau splitting; the triangle is almost invisible at
    # the full-period midpoint. Its total area is 2.08 ns.
    source, batch = circular_pulse(
        frequency, np.array([0.0, 2.0, 4.0, 20.0]), np.array([0, 1, 0, 0.01])
    )
    density = evolve(source, batch)
    expected_population = np.sin(2 * np.pi * 0.05 * 2.08) ** 2
    assert density[0, 1, 1].real == pytest.approx(expected_population, abs=1e-3)


def test_repeated_shaped_pulse_resolves_dissipation_during_rotation() -> None:
    times, samples = np.array([0.0, 50.0, 100.0]), np.array([0.0, 1.0, 0.0])
    rate = 0.05
    source, batch = circular_pulse(0.05, times, samples, n=20, rate=rate)

    # Independent resonant Bloch equations. The rotating frame removes the
    # carrier exactly, and amplitude damping commutes with that frame change.
    def bloch(time: float, state: np.ndarray) -> list[float]:
        w, v = state
        omega = 2 * np.pi * 0.05 * np.interp(time, times, samples)
        return [-2 * omega * v + rate * (1 - w), 2 * omega * w - rate * v / 2]

    solution = solve_ivp(bloch, (0, 100), [1.0, 0.0], rtol=1e-10, atol=1e-12)
    expected_population = (1 - solution.y[0, -1]) / 2
    density = evolve(source, batch)
    np.testing.assert_allclose(density[:, 1, 1].real, expected_population, atol=1e-2)
