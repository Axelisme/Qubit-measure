"""Long accepted frames must not demand sub-roundoff local channel errors."""

import numpy as np
import pytest
from zcu_tools.simulate.evolution import (
    EnvelopeSpec,
    ModelSource,
    SecularDrive,
    SecularModel,
    kernel,
)
from zcu_tools.simulate.evolution.kernels.floquet import FloquetKernel


@pytest.mark.parametrize("duration", [512.0, 2000.0])
def test_long_periodic_frame_preserves_analytic_coherence(duration: float) -> None:
    zero = np.zeros((2, 2), dtype=np.complex128)
    # A diagonal probe commutes with H0, so its integrated phase is analytic.
    probe = np.diag([0.0, 0.1]).astype(np.complex128)
    model = SecularModel(
        (0,),
        "long-periodic-frame",
        np.array([0.0, 50.0]),
        np.eye(2, dtype=complex),
        (SecularDrive("stark", zero, zero), SecularDrive("probe", probe, zero)),
        np.zeros((2, 2)),
        np.zeros((2, 2)),
    )
    initial = np.full((2, 2), 0.5, dtype=np.complex128)
    source = ModelSource(initial, lambda level: model)
    frequency, gain, theta = 0.11, 0.01, 0.3
    table = kernel.SegmentTable(
        timeline=np.array([0]),
        start_ns=np.array([0.0]),
        duration_ns=np.array([duration]),
        level=np.zeros((1, 1), dtype=np.int64),
        tones=np.array([2]),
        tone_port=np.array([[0, 1]]),
        tone_frequency_ghz=np.array([[7.29, frequency]]),
        tone_amplitude=np.array([[0.5, gain]]),
        tone_theta_rad=np.array([[0.0, theta]]),
        tone_envelope=np.zeros((1, 2), dtype=np.int64),
        tone_envelope_offset_ns=np.zeros((1, 2)),
        envelopes=(EnvelopeSpec("constant"),),
    )
    floquet = FloquetKernel()
    rows = np.array([0])
    assert floquet.guard(table, rows, source, 1e-3).all()
    prepared = floquet.prepare(
        floquet.prepare_keys(table, rows), table, rows, source, 1e-3
    )
    actual = initial[None].copy()
    try:
        floquet.apply(prepared, rows, table, rows, actual, 1e-3)
    except RuntimeError as error:
        raise AssertionError(
            "Accepted long frame must integrate above roundoff"
        ) from error
    angle = 2 * np.pi * 50.0 * duration + gain * 0.1 / frequency * (
        np.sin(2 * np.pi * frequency * duration + theta) - np.sin(theta)
    )
    expected = initial.copy()
    expected[0, 1] *= np.exp(1j * angle)
    expected[1, 0] = expected[0, 1].conjugate()
    np.testing.assert_allclose(actual[0], expected, rtol=0, atol=2e-8)
