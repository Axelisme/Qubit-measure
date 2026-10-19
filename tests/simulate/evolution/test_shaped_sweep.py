"""Low-reuse shaped sweeps through Engine, including guarded fallbacks."""

from dataclasses import replace
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
from zcu_tools.simulate.evolution.kernels.dop853 import DOP853Kernel


@pytest.mark.parametrize("frequency", [0.05, -0.05, -0.294, 5.0])
@pytest.mark.parametrize("coherent", [False, True])
def test_single_use_ramps_use_spectral_periods_and_match_direct_evolution(
    frequency: float, coherent: bool
) -> None:
    x = np.array([[0, 0.012], [0.012, 0]], dtype=complex)
    model = SecularModel(
        (0,),
        "shaped-sweep",
        np.array([0.0, abs(frequency)]),
        np.eye(2, dtype=complex),
        (SecularDrive("drive", x, np.zeros_like(x)),),
        np.array([[0.0, 1e-5], [0.0, 0.0]]),
        np.zeros((2, 2)),
    )
    initial = np.array([[0.6, 0.2j], [-0.2j, 0.4]]) if coherent else np.diag([0.6, 0.4])
    source = ModelSource(initial.astype(complex), lambda _: model)
    duration = 100.3 if abs(frequency) < 1 else 20.3
    # Each frequency key occurs once. The second envelope has fast unresolved
    # intra-period motion, so its row must retain the existing direct fallback.
    smooth = EnvelopeSpec(
        "sampled", np.array([0.0, 1.0], dtype=complex), np.array([0.0, duration])
    )
    sharp = EnvelopeSpec(
        "sampled",
        np.array([0.0, 1.0, 0.0, 0.01], dtype=complex),
        np.array([0.0, 0.01, 0.02, duration]),
    )
    track = DriveTrack(
        "drive",
        np.zeros(2),
        np.full(2, duration),
        np.array([frequency, frequency * 1.001]),
        np.array([0.37, -0.8]),
        np.full(2, 0.3),
        (smooth, sharp),
        np.arange(2),
        np.ones(2, dtype=bool),
    )
    batch = TimelineBatch(
        np.arange(2),
        (track,),
        FluxSchedule(np.arange(3), np.zeros(2), np.zeros((2, 1), dtype=np.int64)),
        np.full(2, duration),
    )
    engine = Engine(1, 0)
    direct = Engine(1, 0, kernels=(DOP853Kernel(),))
    try:
        actual = engine.evolve(source, (batch,), cancel=Event())
        # A complex two-tone description exercises the ordinary direct branch
        # without patching the dispatcher or reaching into kernel internals.
        reference_track = replace(track, amplitude=track.amplitude / 2)
        reference_batch = replace(batch, tracks=(reference_track, reference_track))
        expected = direct.evolve(
            source, (reference_batch,), budget=1e-6, cancel=Event()
        )
    finally:
        engine.close()
        direct.close()
    assert actual.paths.segments_by_kernel == {"shaped_sambe": 1, "dop853": 1}
    np.testing.assert_allclose(actual.densities, expected.densities, atol=1e-3, rtol=0)
    assert actual.paths.max_trace_error < 1e-6


@pytest.mark.parametrize("frequency", [0.294, -5.0])
def test_different_duration_ramps_match_independent_preparations(
    frequency: float,
) -> None:
    """Sharing a spectral family must not share duration-dependent playback."""
    x = np.array([[0, 0.012], [0.012, 0]], dtype=complex)
    model = SecularModel(
        (0,),
        "duration-ramps",
        np.array([0.0, abs(frequency)]),
        np.eye(2, dtype=complex),
        (SecularDrive("drive", x, np.zeros_like(x)),),
        np.array([[0.0, 1e-5], [0.0, 0.0]]),
        np.zeros((2, 2)),
    )
    source = ModelSource(np.array([[0.6, 0.2j], [-0.2j, 0.4]]), lambda _: model)
    durations = np.array([100.3, 150.7])
    envelopes = tuple(
        EnvelopeSpec(
            "sampled", np.array([0.0, 1.0], dtype=complex), np.array([0.0, duration])
        )
        for duration in durations
    )

    def batch(indices: np.ndarray) -> TimelineBatch:
        count = len(indices)
        track = DriveTrack(
            "drive",
            np.zeros(count),
            durations[indices],
            np.full(count, frequency),
            np.full(count, 0.37),
            np.full(count, 0.3),
            envelopes,
            indices,
            np.ones(count, dtype=bool),
        )
        return TimelineBatch(
            indices,
            (track,),
            FluxSchedule(
                np.arange(count + 1),
                np.zeros(count),
                np.zeros((count, 1), dtype=np.int64),
            ),
            durations[indices],
        )

    engine = Engine(1, 0)
    try:
        joint = engine.evolve(source, (batch(np.arange(2)),), cancel=Event())
    finally:
        engine.close()
    independent = []
    for point in range(2):
        engine = Engine(1, 0)
        try:
            result = engine.evolve(source, (batch(np.array([point])),), cancel=Event())
        finally:
            engine.close()
        assert result.paths.segments_by_kernel == {"shaped_sambe": 1}
        independent.append(result.densities[0])
    assert joint.paths.segments_by_kernel == {"shaped_sambe": 2}
    np.testing.assert_allclose(joint.densities, independent, atol=1e-12, rtol=0)


@pytest.mark.parametrize("frequency", [0.05, -0.05, 0.3, -0.3, 1.7])
@pytest.mark.parametrize("shape", ["drag", "zero-crossing"])
def test_smooth_complex_envelopes_use_sambe_and_match_dop(
    frequency: float, shape: str
) -> None:
    """Complex phase follows the envelope, carrier substeps and fractional tail."""
    x = np.array([[0, 0.04], [0.04, 0]], dtype=complex)
    y = np.array([[0, -0.02j], [0.02j, 0]], dtype=complex)
    model = SecularModel(
        (0,),
        "complex-shaped-sweep",
        np.array([0.0, abs(frequency)]),
        np.eye(2, dtype=complex),
        (SecularDrive("drive", x, y),),
        np.array([[0.0, 1e-5], [0.0, 0.0]]),
        np.zeros((2, 2)),
    )
    source = ModelSource(np.array([[0.6, 0.2j], [-0.2j, 0.4]]), lambda _: model)
    duration = 64.7
    times = np.linspace(0, duration, 65)
    coordinate = (times - duration / 2) / (duration / 4)
    gaussian = np.exp(-(coordinate**2) / 2)
    if shape == "drag":
        values = gaussian * (1 - 0.5j * coordinate)
    else:
        # A continuous signed complex envelope passes exactly through zero.
        # Tiny transverse offsets must not turn its phase jump into a drive.
        values = coordinate * (0.5 + 0.4j)
    smooth = EnvelopeSpec("sampled", np.asarray(values, dtype=np.complex128), times)
    near_zero = EnvelopeSpec(
        "sampled", np.asarray(values + 1e-14j, dtype=np.complex128), times
    )
    sharp = EnvelopeSpec(
        "sampled",
        np.array([0.0, 1.0j, -1.0, 0.01j]),
        np.array([0.0, 0.01, 0.02, duration]),
    )
    track = DriveTrack(
        "drive",
        np.zeros(3),
        np.full(3, duration),
        np.full(3, frequency),
        np.full(3, 0.73),
        np.full(3, -0.3),
        (smooth, near_zero, sharp),
        np.arange(3),
        np.ones(3, dtype=bool),
    )
    batch = TimelineBatch(
        np.arange(3),
        (track,),
        FluxSchedule(np.arange(4), np.zeros(3), np.zeros((3, 1), dtype=np.int64)),
        np.full(3, duration),
    )
    engine = Engine(1, 0)
    direct = Engine(1, 0, kernels=(DOP853Kernel(),))
    try:
        actual = engine.evolve(source, (batch,), cancel=Event())
        expected = direct.evolve(source, (batch,), budget=1e-7, cancel=Event())
    finally:
        engine.close()
        direct.close()
    assert actual.paths.segments_by_kernel == {"shaped_sambe": 2, "dop853": 1}
    np.testing.assert_allclose(actual.densities, expected.densities, atol=1e-2, rtol=0)
    np.testing.assert_allclose(actual.densities[0], actual.densities[1], atol=1e-10)
    assert np.isfinite(actual.densities).all()
    assert actual.paths.max_trace_error < 1e-6


@pytest.mark.parametrize("frequency", [0.3, -0.3])
def test_complex_fractional_tail_preserves_quadrature(frequency: float) -> None:
    """A row shorter than one period still applies the complex drive in its tail."""
    x = np.array([[0, 0.04], [0.04, 0]], dtype=complex)
    model = SecularModel(
        (0,),
        "complex-tail",
        np.array([0.0, abs(frequency)]),
        np.eye(2, dtype=complex),
        (SecularDrive("drive", x, np.zeros_like(x)),),
        np.zeros((2, 2)),
        np.zeros((2, 2)),
    )
    source = ModelSource(np.diag([1.0, 0.0]).astype(complex), lambda _: model)
    duration = 2.7
    envelope = EnvelopeSpec(
        "sampled", np.array([1j, 0.999j]), np.array([0.0, duration])
    )
    track = DriveTrack(
        "drive",
        np.zeros(1),
        np.full(1, duration),
        np.full(1, frequency),
        np.full(1, 0.73),
        np.full(1, 0.6),
        (envelope,),
        np.zeros(1, dtype=np.int64),
        np.ones(1, dtype=bool),
    )
    batch = TimelineBatch(
        np.arange(1),
        (track,),
        FluxSchedule(np.arange(2), np.zeros(1), np.zeros((1, 1), dtype=np.int64)),
        np.full(1, duration),
    )
    engine = Engine(1, 0)
    direct = Engine(1, 0, kernels=(DOP853Kernel(),))
    try:
        actual = engine.evolve(source, (batch,), cancel=Event())
        expected = direct.evolve(source, (batch,), budget=1e-7, cancel=Event())
    finally:
        engine.close()
        direct.close()
    assert actual.paths.segments_by_kernel == {"shaped_sambe": 1}
    np.testing.assert_allclose(actual.densities, expected.densities, atol=1e-3, rtol=0)


def test_real_and_complex_ramps_keep_their_resolution_across_cache_hits() -> None:
    """A rotated envelope cannot borrow the real ramp's coarser preparation."""
    x = np.array([[0, 0.04], [0.04, 0]], dtype=complex)
    model = SecularModel(
        (0,),
        "rotated-ramp-cache",
        np.array([0.0, 0.3]),
        np.eye(2, dtype=complex),
        (SecularDrive("drive", x, np.zeros_like(x)),),
        np.zeros((2, 2)),
        np.zeros((2, 2)),
    )
    source = ModelSource(np.array([[0.6, 0.2j], [-0.2j, 0.4]]), lambda _: model)
    duration = 64.7

    def batch(rotation: complex) -> TimelineBatch:
        envelope = EnvelopeSpec(
            "sampled",
            np.array([-1.0, 1.0], dtype=np.complex128) * rotation,
            np.array([0.0, duration]),
        )
        track = DriveTrack(
            "drive",
            np.zeros(1),
            np.full(1, duration),
            np.full(1, 0.3),
            np.full(1, 0.73),
            np.full(1, 0.3),
            (envelope,),
            np.zeros(1, dtype=np.int64),
            np.ones(1, dtype=bool),
        )
        return TimelineBatch(
            np.arange(1),
            (track,),
            FluxSchedule(np.arange(2), np.zeros(1), np.zeros((1, 1), dtype=np.int64)),
            np.full(1, duration),
        )

    engine = Engine(1, 64 * 1024**2)
    uncached = Engine(1, 0)
    try:
        for rotation in (1.0, 1.0j, 1.0, 1.0j):
            actual = engine.evolve(source, (batch(rotation),), cancel=Event())
            expected = uncached.evolve(source, (batch(rotation),), cancel=Event())
            assert actual.paths.segments_by_kernel == {"shaped_sambe": 1}
            np.testing.assert_allclose(
                actual.densities, expected.densities, atol=1e-12, rtol=0
            )
    finally:
        engine.close()
        uncached.close()
