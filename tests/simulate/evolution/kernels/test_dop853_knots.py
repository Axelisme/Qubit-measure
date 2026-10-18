"""Sampled-envelope accuracy at the public DOP853 prepare/apply seam.

An isolated, hardware-scale feature in a long idle must not be skipped when
adaptive steps grow. Commuting drives give an independent pulse-area oracle.
"""

import numpy as np
import pytest
from scipy.linalg import expm
from zcu_tools.simulate.evolution import (
    EnvelopeSpec,
    ModelSource,
    SecularDrive,
    SecularModel,
)
from zcu_tools.simulate.evolution.kernel import SegmentTable
from zcu_tools.simulate.evolution.kernels.dop853 import DOP853Kernel

from tests.simulate.evolution.contract.kernel_suite import kernel_context


@pytest.mark.parametrize("tones", [1, 3])
@pytest.mark.parametrize("budget", [1e-3, 1e-8])
def test_isolated_complex_sampled_feature_preserves_pulse_area(
    tones: int, budget: float
) -> None:
    x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    model = SecularModel(
        (0,),
        "commuting",
        np.zeros(2),
        np.eye(2, dtype=np.complex128),
        (SecularDrive("drive", x, 0.4 * x),),
        np.zeros((2, 2)),
        np.zeros((2, 2)),
    )
    initial = np.diag([1, 0]).astype(complex)
    source = ModelSource(initial, lambda _: model)
    times = np.array([0, 400, 401, 402, 700, 701, 703, 1000.0])
    samples = np.array([0, 0, 1 + 0.5j, 0, 0, -0.3j, 0, 0], dtype=np.complex128)
    gain = np.array([0.17, -0.11, 0.07])[:tones]
    offsets = np.array([0, 0.37, 1.13])[:tones]
    table = SegmentTable(
        timeline=np.array([0]),
        start_ns=np.array([23.0]),
        duration_ns=np.array([1000.0]),
        level=np.array([[0]]),
        tones=np.array([tones]),
        tone_port=np.zeros((1, tones), dtype=np.int64),
        tone_frequency_ghz=np.zeros((1, tones)),
        tone_amplitude=gain[None],
        tone_theta_rad=np.zeros((1, tones)),
        tone_envelope=np.zeros((1, tones), dtype=np.int64),
        tone_envelope_offset_ns=offsets[None],
        envelopes=(EnvelopeSpec("sampled", samples, times),),
    )
    kernel = DOP853Kernel()
    rows = np.array([0], dtype=np.int64)
    prepared = kernel.prepare(
        kernel.prepare_keys(table, rows), table, rows, source, budget
    )
    actual = initial[None].copy()
    kernel.apply(
        prepared, rows, table, rows, actual, kernel_context(kernel.name, table, budget)
    )
    # Each shifted feature remains wholly inside the integration support.
    area = np.trapezoid(samples.real + 0.4 * samples.imag, times) * gain.sum()
    unitary = expm(-2j * np.pi * area * x)
    expected = unitary @ initial @ unitary.conj().T
    np.testing.assert_allclose(actual[0], expected, atol=budget, rtol=0)


@pytest.mark.parametrize("budget", [1e-3, 1e-8])
def test_dense_smooth_envelope_matches_uncompressed_reference(budget: float) -> None:
    from scipy.integrate import solve_ivp

    times = np.linspace(0, 20, 1001)
    gaussian = np.exp(-0.5 * ((times - 10) / 4) ** 2)
    samples = (gaussian + 0.4j * (times - 10) / 4 * gaussian).astype(np.complex128)
    x = 0.11 * np.array([[0, 1], [1, 0]], dtype=np.complex128)
    y = 0.11 * np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    energy = np.array([0, 0.047])
    model = SecularModel(
        (0,),
        "smooth",
        energy,
        np.eye(2, dtype=np.complex128),
        (SecularDrive("drive", x, y),),
        np.zeros((2, 2)),
        np.zeros((2, 2)),
    )
    initial = np.diag([1, 0]).astype(np.complex128)
    source = ModelSource(initial, lambda _: model)
    table = SegmentTable(
        timeline=np.array([0]),
        start_ns=np.array([0.0]),
        duration_ns=np.array([20.0]),
        level=np.array([[0]]),
        tones=np.array([1]),
        tone_port=np.array([[0]]),
        tone_frequency_ghz=np.array([[0.05]]),
        tone_amplitude=np.array([[0.7]]),
        tone_theta_rad=np.array([[0.3]]),
        tone_envelope=np.array([[0]]),
        tone_envelope_offset_ns=np.array([[0.0]]),
        envelopes=(EnvelopeSpec("sampled", samples, times),),
    )
    kernel = DOP853Kernel()
    rows = np.array([0], dtype=np.int64)
    prepared = kernel.prepare(
        kernel.prepare_keys(table, rows), table, rows, source, budget
    )
    actual = initial[None].copy()
    kernel.apply(
        prepared, rows, table, rows, actual, kernel_context(kernel.name, table, budget)
    )

    def rhs(time, flat):
        rho = flat.reshape(2, 2)
        # Independent uncompressed playback, not EnvelopeSpec.simplified/values.
        envelope = np.interp(time, times, samples.real) + 1j * np.interp(
            time, times, samples.imag
        )
        amplitude = 0.7 * envelope * np.exp(1j * (0.3 + 2 * np.pi * 0.05 * time))
        h = np.diag(energy) + amplitude.real * x + amplitude.imag * y
        return (-2j * np.pi * (h @ rho - rho @ h)).ravel()

    solution = solve_ivp(
        rhs,
        (0, 20),
        initial.ravel(),
        method="DOP853",
        rtol=1e-10,
        atol=1e-12,
        max_step=0.02,
    )
    assert solution.success
    np.testing.assert_allclose(
        actual[0], solution.y[:, -1].reshape(2, 2), atol=budget, rtol=0
    )
