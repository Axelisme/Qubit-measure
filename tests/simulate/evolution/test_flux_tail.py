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
    engine = Engine(1)
    try:
        result = engine.evolve(source, (batch,), cancel=Event())
    finally:
        engine.close()
    assert result.paths.segments_by_kernel == {"flux_dop853": 1}
    assert_budget(result.densities, expected[None], 1e-3)


def test_event_sides_preserve_density_at_boundary_and_after_short_plateaus():
    # Diagonal H and drive commute. A coherent initial state exposes the
    # integrated energy gap directly, without a second numerical solver.
    static_gap, sensitivity, drive_gap = 0.15, 0.4, 0.04
    static = np.diag([0.0, static_gap]).astype(complex)
    flux_operator = np.diag([0.0, sensitivity]).astype(complex)
    drive = SecularDrive(
        "drive", np.diag([0.0, drive_gap]).astype(complex), np.zeros((2, 2), complex)
    )

    def build(level: tuple[int, ...]) -> SecularModel:
        return SecularModel(
            level,
            f"commuting-{level}",
            np.array(
                [0.0, static_gap + sensitivity * np.sin(np.pi * 0.001 * level[0])]
            ),
            np.eye(2, dtype=complex),
            (drive,),
            np.zeros((2, 2)),
            np.zeros((2, 2)),
        )

    initial = np.full((2, 2), 0.5, dtype=complex)
    source = ModelSource(
        initial,
        build,
        FluxExpansion(static, (FluxTerm(0, "sin_pi", flux_operator),)),
    )
    # The first pair of jumps is inside a row. The jump at 1 ns is both
    # point 0's readout boundary and point 1's internal row boundary.
    # Point 1 then sees a short high plateau and a jump at its own endpoint.
    events = np.array([0.375, 0.625, 1.0, 1.125, 2.0])
    left = np.nextafter(events, -np.inf)
    times = np.r_[0.0, np.column_stack((left, events)).ravel()]
    flux = np.array([0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.5])
    ends = np.array([1.0, 2.0])
    batch = TimelineBatch(
        np.array([0, 1]),
        (
            DriveTrack(
                "drive",
                np.zeros(2),
                np.full(2, 2.0),
                np.zeros(2),
                np.zeros(2),
                np.ones(2),
                (EnvelopeSpec("constant"),),
                np.zeros(2, dtype=np.int64),
                np.ones(2, dtype=bool),
            ),
        ),
        FluxSchedule(
            np.array([0, 1, 2]), np.zeros(2), np.zeros((2, 1), dtype=np.int64)
        ),
        ends,
        flux_waveform=FluxWaveform(
            np.array([0, len(times), 2 * len(times)]),
            np.tile(times, 2),
            np.tile(flux, 2)[:, None],
        ),
        segment_boundaries=((), (1.0,)),
    )

    # sin(pi*flux) is one on each high plateau. Each linear 0 <-> 0.5
    # bridge has mean 2/pi, including bridges ending exactly at readout.
    plateau_area = np.array(
        [left[1] - events[0], left[1] - events[0] + left[3] - events[2]]
    )
    bridge_area = (2 / np.pi) * np.array(
        [(events[:3] - left[:3]).sum(), (events - left).sum()]
    )
    area = (static_gap + drive_gap) * ends + sensitivity * (plateau_area + bridge_area)
    expected = np.full((2, 2, 2), 0.5, dtype=complex)
    expected[:, 0, 1] = 0.5 * np.exp(2j * np.pi * area)
    expected[:, 1, 0] = expected[:, 0, 1].conj()

    engine = Engine(1)
    try:
        result = engine.evolve(source, (batch,), budget=1e-6, cancel=Event())
    finally:
        engine.close()
        source.close()
    np.testing.assert_allclose(result.densities, expected, atol=1e-6, rtol=0)
