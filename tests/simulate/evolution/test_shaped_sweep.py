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
