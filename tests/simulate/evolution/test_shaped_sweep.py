"""Low-reuse shaped sweeps through Engine, including guarded fallbacks."""

from contextlib import closing
from dataclasses import replace
from threading import Event
from unittest.mock import patch

import numpy as np
import pytest
from numpy.typing import NDArray
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
from zcu_tools.simulate.evolution.kernels import shaped_sambe


def _triangle_track(
    starts: NDArray[np.float64], durations: NDArray[np.float64]
) -> DriveTrack:
    count = len(durations)
    return DriveTrack(
        "drive",
        starts,
        durations,
        np.full(count, 2.0),
        np.full(count, 0.37),
        np.ones(count),
        tuple(
            EnvelopeSpec(
                "sampled",
                np.array([0.1, 0.7, 0.1], dtype=complex),
                np.array([0.0, length / 2, length]),
            )
            for length in durations
        ),
        np.arange(count),
        np.ones(count, dtype=bool),
    )


@pytest.mark.parametrize("workers", [1, 2])
def test_shared_period_fallback_builds_one_table(workers: int) -> None:
    """Distinct duration guards can refuse rows that need the same band table."""
    x = np.array([[0, 0.15], [0.15, 0]], dtype=complex)
    model = SecularModel(
        (0,),
        "shared-fallback",
        np.array([0.0, 1.8]),
        np.eye(2, dtype=complex),
        (SecularDrive("drive", x, np.zeros_like(x)),),
        np.zeros((2, 2)),
        np.zeros((2, 2)),
    )
    durations = np.r_[30.0, np.arange(200.0, 280.0, 10.0)]
    count = len(durations)
    batch = TimelineBatch(
        np.arange(count),
        (_triangle_track(np.zeros(count), durations),),
        FluxSchedule(np.arange(count + 1), np.zeros(count), np.zeros((count, 1), int)),
        durations,
    )
    with (
        closing(
            ModelSource(np.diag([1.0, 0.0]).astype(complex), lambda _: model)
        ) as src,
        closing(Engine(workers, backend="cpu")) as engine,
        closing(Engine(1, backend="cpu", families=("dop853",))) as direct,
    ):
        result = engine.evolve(src, (batch,), telemetry=True)
        expected = direct.evolve(src, (batch,), budget=1e-7)

    sambe = result.paths.routing["shaped_sambe"]
    assert sambe.rejected_rows == 8
    # All nine duration keys reach spectral construction; eight products are discarded.
    assert sambe.prepare_key_builds == 9
    assert result.paths.segments_by_kernel == {"shaped_sambe": 1, "period_table": 8}
    assert result.paths.routing["period_table"].prepare_key_builds == 1
    np.testing.assert_allclose(result.densities, expected.densities, atol=2e-3, rtol=0)


@pytest.mark.parametrize(
    "second_duration", [145.0, 30.0], ids=["period-first", "mixed"]
)
def test_eager_sambe_preserves_timeline_spectral_reuse(second_duration: float) -> None:
    x = np.array([[0, 0.0012], [0.0012, 0]], dtype=complex)
    model = SecularModel(
        (0,),
        "spectral-reuse",
        np.array([0.0, 1.8]),
        np.eye(2, dtype=complex),
        (SecularDrive("drive", x, np.zeros_like(x)),),
        np.zeros((2, 2)),
        np.zeros((2, 2)),
    )
    first = np.array([140.0, 160.0])
    second = np.array([second_duration, 170.0])
    batch = TimelineBatch(
        np.arange(2),
        (_triangle_track(np.zeros(2), first), _triangle_track(first, second)),
        FluxSchedule(np.arange(3), np.zeros(2), np.zeros((2, 1), dtype=np.int64)),
        first + second,
    )
    with (
        closing(
            ModelSource(np.diag([1.0, 0.0]).astype(complex), lambda _: model)
        ) as src,
        closing(Engine(2, backend="cpu")) as engine,
        closing(Engine(1, backend="cpu", families=("dop853",))) as direct,
    ):
        # Public telemetry counts duration keys, not the coefficient work they
        # can share. Observe the real builder without substituting its result.
        builder = "_coefficients"
        with patch.object(
            shaped_sambe, builder, wraps=getattr(shaped_sambe, builder)
        ) as build:
            result = engine.evolve(src, (batch,), telemetry=True)
        expected = direct.evolve(src, (batch,), budget=1e-7)

    assert result.paths.segments_by_kernel == {"shaped_sambe": 4}
    assert result.paths.routing["shaped_sambe"].prepare_key_builds == 4
    np.testing.assert_allclose(result.densities, expected.densities, atol=2e-3, rtol=0)
    assert build.call_count <= 2  # One coefficient build per timeline suffices.


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
    engine = Engine(1)
    direct = Engine(1, families=("dop853",))
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

    engine = Engine(1)
    try:
        joint = engine.evolve(source, (batch(np.arange(2)),), cancel=Event())
    finally:
        engine.close()
    independent = []
    for point in range(2):
        engine = Engine(1)
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
    engine = Engine(1)
    direct = Engine(1, families=("dop853",))
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
    engine = Engine(1)
    direct = Engine(1, families=("dop853",))
    try:
        actual = engine.evolve(source, (batch,), cancel=Event())
        expected = direct.evolve(source, (batch,), budget=1e-7, cancel=Event())
    finally:
        engine.close()
        direct.close()
    assert actual.paths.segments_by_kernel == {"shaped_sambe": 1}
    np.testing.assert_allclose(actual.densities, expected.densities, atol=1e-3, rtol=0)
