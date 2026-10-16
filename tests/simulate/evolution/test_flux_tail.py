"""Continuous within-level flux routing at the public engine boundary."""

from threading import Event

import numpy as np
from scipy.integrate import solve_ivp
from zcu_tools.simulate.evolution import (
    DriveTrack,
    Engine,
    EnvelopeSpec,
    FluxExpansion,
    FluxSchedule,
    FluxTerm,
    FluxWaveform,
    ModelSource,
    SecularDrive,
    SecularModel,
    TimelineBatch,
)

from .contract.kernel_suite import assert_budget


def test_driven_tail_within_one_level_uses_continuous_hamiltonian():
    dc, duration = 0.2, 10.0
    sensitivity = np.diag([0.0, 10.0, 0.0]).astype(complex)
    static = (
        np.diag([0.0, 0.1, 0.4]).astype(complex) - np.sin(2 * np.pi * dc) * sensitivity
    )
    x = np.zeros((3, 3), complex)
    x[0, 1] = x[1, 0] = 0.5
    y = np.zeros((3, 3), complex)
    y[0, 1], y[1, 0] = 0.5j, -0.5j
    drive = SecularDrive("drive", x, y)

    def h0(flux):
        return static + np.sin(2 * np.pi * flux) * sensitivity

    def build(level):
        return SecularModel(
            level,
            str(level),
            h0(dc + 0.001 * level[0]).diagonal().real,
            np.eye(3, dtype=complex),
            (drive,),
            np.zeros((3, 3)),
            np.zeros((3, 3)),
        )

    initial = np.diag([1.0, 0.0, 0.0]).astype(complex)
    source = ModelSource(
        initial, build, FluxExpansion(static, (FluxTerm(0, "sin_2pi", sensitivity),))
    )
    # Both endpoints exceed delta/2 and round to level 1. No lattice event
    # occurs during this drive even though its tail still changes H(flux(t)).
    tail = np.array([0.0014, 0.0006])
    frequency, gain = float(h0(dc + 0.001)[1, 1].real), 0.04
    batch = TimelineBatch(
        np.array([0]),
        (
            DriveTrack(
                "drive",
                np.array([0.0]),
                np.array([duration]),
                np.array([frequency]),
                np.array([0.0]),
                np.array([gain]),
                (EnvelopeSpec("constant"),),
                np.array([0]),
                np.array([True]),
            ),
        ),
        FluxSchedule(np.array([0, 1]), np.array([0.0]), np.array([[1]])),
        np.array([duration]),
        flux_waveform=FluxWaveform(
            np.array([0, 2]), np.array([0.0, duration]), (dc + tail)[:, None]
        ),
    )

    def rhs(time, state):
        flux = dc + np.interp(time, [0.0, duration], tail)
        angle = 2 * np.pi * frequency * time
        h = h0(flux) + gain * (np.cos(angle) * x + np.sin(angle) * y)
        rho = state.reshape(3, 3)
        return (-2j * np.pi * (h @ rho - rho @ h)).ravel()

    expected = (
        solve_ivp(
            rhs,
            (0, duration),
            initial.ravel(),
            method="DOP853",
            rtol=1e-11,
            atol=1e-13,
            max_step=0.01,
        )
        .y[:, -1]
        .reshape(3, 3)
    )
    engine = Engine(1, 0)
    try:
        result = engine.evolve(source, (batch,), cancel=Event())
    finally:
        engine.close()
    assert result.paths.segments_by_kernel == {"flux_dop853": 1}
    assert_budget(result.densities, expected[None], 1e-3)
