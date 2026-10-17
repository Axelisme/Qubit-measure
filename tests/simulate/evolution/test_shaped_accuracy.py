"""Shaped evolution must resolve envelope motion within a carrier period."""

from threading import Event

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


@pytest.mark.parametrize("frequency", [0.05, -0.05])
def test_subperiod_ramp_preserves_resonant_population_transfer(frequency: float) -> None:
    # A circular resonant drive commutes with itself in the interaction frame.
    # Its exact rotation angle is 2*pi*strength*integral(envelope), independent
    # of carrier phase. The narrow triangle has area 2 ns, but its value at
    # the only full-period midpoint is zero.
    strength = 0.05
    x = strength * np.array([[0, 1], [1, 0]], dtype=complex)
    y = strength * np.array([[0, 1j], [-1j, 0]], dtype=complex)
    model = SecularModel(
        (0,),
        "circular-resonant",
        np.array([0.0, frequency]),
        np.eye(2, dtype=complex),
        (SecularDrive("drive", x, y),),
        np.zeros((2, 2)),
        np.zeros((2, 2)),
    )
    source = ModelSource(np.diag([1.0, 0.0]).astype(complex), lambda _: model)
    envelope = EnvelopeSpec(
        "sampled",
        np.array([0, 1, 0, 0], dtype=complex),
        np.array([0.0, 2.0, 4.0, 20.0]),
    )
    track = DriveTrack(
        "drive",
        np.zeros(1),
        np.array([20.0]),
        np.array([frequency]),
        np.array([0.37]),
        np.ones(1),
        (envelope,),
        np.zeros(1, dtype=np.int64),
        np.ones(1, dtype=bool),
    )
    batch = TimelineBatch(
        np.array([0]),
        (track,),
        FluxSchedule(np.array([0, 1]), np.zeros(1), np.zeros((1, 1), dtype=np.int64)),
        np.array([20.0]),
    )
    engine = Engine(1, 0)
    try:
        result = engine.evolve(source, (batch,), cancel=Event())
    finally:
        engine.close()
    expected_population = np.sin(2 * np.pi * strength * 2) ** 2
    assert result.densities[0, 1, 1].real == pytest.approx(expected_population, abs=1e-3)
