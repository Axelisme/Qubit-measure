"""Undriven flux transport, phase and gauge through the public Engine seam."""

from dataclasses import replace
from threading import Event

import numpy as np
import pytest
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
    SegmentShift,
    TimelineBatch,
)

from ._engine import RecordingKernel, batch, run, source, track


def test_undriven_motion_has_its_own_domain_without_changing_driven_routes():
    kernels = tuple(
        RecordingKernel(name)
        for name in ("t4", "adiabatic_map", "flux_dop853", "dyson")
    )
    fixed = batch(1, duration=4)
    moving = replace(
        fixed,
        points=np.array([1]),
        flux_waveform=FluxWaveform(
            np.array([0, 3]),
            np.array([0.0, 2.0, 4.0]),
            np.array([[0.0], [0.0002], [0.0]]),
        ),
    )
    plateau = replace(
        fixed,
        points=np.array([2]),
        tracks=(track(1, EnvelopeSpec("constant"), duration=4),),
    )
    driven = replace(moving, points=np.array([3]), tracks=plateau.tracks)
    original = source()
    src = ModelSource(
        original.initial_density,
        original.model,
        FluxExpansion(np.diag([0.0, 0.1, 0.4]).astype(complex), ()),
    )
    result = run(Engine(1, kernels=kernels), src, (fixed, moving, plateau, driven))
    assert result.paths.segments_by_kernel == {
        "t4": 1,
        "adiabatic_map": 1,
        "dyson": 1,
        "flux_dop853": 1,
    }


def rotating_source(initial):
    """Conical spin with a complex Berry phase and arbitrary model gauge."""
    x = np.array([[0, 1], [1, 0]], complex)
    y = np.array([[0, -1j], [1j, 0]], complex)
    static = np.diag([-0.3, 0.3]).astype(complex)
    expansion = FluxExpansion(
        static,
        (
            FluxTerm(0, "cos_2pi", (0.15 + 0j) * x),
            FluxTerm(0, "sin_2pi", (0.15 + 0j) * y),
        ),
    )
    dc_energy, dc_basis = np.linalg.eigh(static + 0.15 * x)
    # Circular resonant pulses are expressed in fixed DC coordinates.
    physical_x = dc_basis @ (0.5 * x) @ dc_basis.conj().T
    physical_y = dc_basis @ (-0.5 * y) @ dc_basis.conj().T
    zero = np.zeros((2, 2))

    def build(level):
        phi = level[0] * 0.001
        energy, basis = np.linalg.eigh(
            static + 0.15 * (np.cos(2 * np.pi * phi) * x + np.sin(2 * np.pi * phi) * y)
        )
        basis = basis * np.exp(1j * np.array([7.0, -13.0]) * phi)
        return SecularModel(
            level,
            "conical:" + str(level),
            energy,
            basis,
            (
                SecularDrive(
                    "drive",
                    basis.conj().T @ physical_x @ basis,
                    basis.conj().T @ physical_y @ basis,
                ),
                SecularDrive("silent", zero.astype(complex), zero.astype(complex)),
            ),
            zero,
            zero,
        )

    return ModelSource(initial, build, expansion), float(dc_energy[1] - dc_energy[0])


def ramp_batch(duration=600.0, *, endpoint=1.0, boundaries=()):
    return TimelineBatch(
        np.array([0]),
        (),
        FluxSchedule(np.array([0, 1]), np.array([0.0]), np.array([[0]])),
        np.array([duration]),
        flux_waveform=FluxWaveform(
            np.array([0, 2]), np.array([0.0, duration]), np.array([[0.0], [endpoint]])
        ),
        segment_boundaries=(boundaries,) if boundaries else (),
    )


def silent_track(duration):
    return DriveTrack(
        "silent",
        np.array([0.0]),
        np.array([duration]),
        np.array([0.2]),
        np.array([0.0]),
        np.array([1.0]),
        (EnvelopeSpec("constant"),),
        np.array([0]),
        np.array([True]),
    )


def evolve(src, timeline, *, reference=False):
    if reference:
        timeline = replace(
            timeline, tracks=(*timeline.tracks, silent_track(float(timeline.end_ns[0])))
        )
    engine = Engine(1)
    try:
        return engine.evolve(src, (timeline,), budget=1e-6, cancel=Event())
    finally:
        engine.close()


@pytest.mark.parametrize(
    "initial", [np.diag([0.3, 0.7]).astype(complex), np.full((2, 2), 0.5, complex)]
)
def test_slow_ramp_matches_direct_density_with_complex_gauge(initial):
    src, _ = rotating_source(initial)
    timeline = ramp_batch(endpoint=0.65)
    actual = evolve(src, timeline)
    reference = evolve(src, timeline, reference=True)
    assert reference.paths.segments_by_kernel == {"flux_dop853": 1}
    # Adiabatic omission at this speed is below 4e-3, including coherences.
    np.testing.assert_allclose(actual.densities, reference.densities, atol=4e-3, rtol=0)
    assert actual.paths.segments_by_kernel == {"adiabatic_map": 1}


def test_ramp_partition_and_quantized_bath_gauges_do_not_add_t5_projections():
    src, _ = rotating_source(np.full((2, 2), 0.5, complex))
    timeline = ramp_batch()
    split = replace(
        timeline,
        flux=FluxSchedule(
            np.array([0, 3]),
            np.array([0.0, 200.0, 400.0]),
            np.array([[0], [333], [667]]),
        ),
    )
    actual = evolve(src, split)
    unsplit = evolve(src, timeline)
    np.testing.assert_allclose(actual.densities, unsplit.densities, atol=2e-6, rtol=0)
    assert actual.paths.segments_by_kernel == {"adiabatic_map": 3}


def test_piecewise_ramp_integrates_continuous_phase_and_row_bath_decay():
    static = np.diag([0.0, 0.2]).astype(complex)
    coupling = np.diag([0.0, 0.1]).astype(complex)
    rates = np.array([[0.0, 0.002], [0.001, 0.0]])
    gamma = np.array([[0.0, 0.003], [0.003, 0.0]])

    def build(level):
        return SecularModel(
            level,
            str(level),
            np.array([0.0, 0.3]),
            np.eye(2, dtype=complex),
            (),
            rates,
            gamma,
        )

    src = ModelSource(
        np.full((2, 2), 0.5, complex),
        build,
        FluxExpansion(static, (FluxTerm(0, "cos_2pi", coupling),)),
    )
    timeline = replace(
        ramp_batch(10.0, endpoint=0.25),
        flux_waveform=FluxWaveform(
            np.array([0, 3]),
            np.array([0.0, 2.0, 10.0]),
            np.array([[0.0], [0.2], [0.25]]),
        ),
        segment_shifts=(
            (
                SegmentShift(
                    0.0,
                    10.0,
                    np.array([0.0, 0.013]),
                    np.array([[0.0, 0.002], [0.002, 0.0]]),
                ),
            ),
        ),
    )
    # Exact integral of cos(2*pi*Phi) on the two linear waveform pieces.
    integral = 2 * np.sin(0.4 * np.pi) / (0.4 * np.pi)
    integral += 8 * (1 - np.sin(0.4 * np.pi)) / (0.1 * np.pi)
    expected = np.full((2, 2), 0.5, complex)
    expected[0, 1] *= np.exp(2j * np.pi * (2.13 + 0.1 * integral) - 0.065)
    expected[1, 0] = expected[0, 1].conjugate()
    actual = evolve(src, timeline)
    np.testing.assert_allclose(actual.densities[0], expected, atol=1e-9, rtol=0)


@pytest.mark.parametrize("phase", [0.0, np.pi / 2])
def test_ramsey_pulses_resolve_the_same_geometric_and_dynamic_phase_as_direct(phase):
    src, frequency = rotating_source(np.diag([1.0, 0.0]).astype(complex))
    pulse_duration, ramp_duration = 2.0, 600.0
    end = 2 * pulse_duration + ramp_duration
    pulses = tuple(
        DriveTrack(
            "drive",
            np.array([start]),
            np.array([pulse_duration]),
            np.array([frequency]),
            np.array([angle]),
            np.array([0.125]),
            (EnvelopeSpec("constant"),),
            np.array([0]),
            np.array([True]),
        )
        for start, angle in ((0.0, 0.0), (pulse_duration + ramp_duration, phase))
    )
    timeline = replace(
        ramp_batch(end),
        tracks=pulses,
        flux_waveform=FluxWaveform(
            np.array([0, 4]),
            np.array([0.0, pulse_duration, pulse_duration + ramp_duration, end]),
            np.array([[0.0], [0.0], [1.0], [1.0]]),
        ),
    )
    actual = evolve(src, timeline)
    reference = evolve(src, timeline, reference=True)
    # Two Ramsey quadratures constrain phase, not just a fringe at an extremum.
    np.testing.assert_allclose(actual.densities, reference.densities, atol=4e-3, rtol=0)
    assert actual.paths.segments_by_kernel.get("adiabatic_map") == 1
