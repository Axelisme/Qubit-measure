"""Workflow analysis contract with actual native projections and optimizers."""

import numpy as np
import pytest

from zcu_lab.v2.twotone.freq.core import FreqResult
from zcu_lab.v2.twotone.rabi.len_rabi.core import LenRabiResult
from zcu_lab.v2.twotone.time_domain.t1.core import T1Result
from zcu_lab.workflows.analysis import (
    analyze_len_rabi,
    analyze_qubit_freq,
    analyze_t1,
)


def test_frequency_fit_recovers_center_and_preserves_canonical_input() -> None:
    freqs = np.linspace(4900.0, 5100.0, 101)
    signals = (1.0 / (1.0 + ((freqs - 5003.0) / 12.0) ** 2)).astype(np.complex128)
    result = FreqResult(freqs, signals)
    original = result.signals.copy()

    fit = analyze_qubit_freq(result)

    assert fit.frequency_mhz == pytest.approx(5003.0, abs=0.5)
    # Background projection changes the raw width; FWHM belongs to the fitted curve.
    half_height = 0.5 * (np.max(fit.fitted) + np.min(fit.fitted))
    half_height_freqs = freqs[fit.fitted >= half_height]
    curve_width = float(half_height_freqs[-1] - half_height_freqs[0])
    assert fit.width_mhz == pytest.approx(curve_width, abs=4.0)
    assert np.isfinite(fit.width_error_mhz)
    assert np.isfinite(fit.error_mhz)
    assert fit.error_mhz >= 0.0
    assert fit.fitted.shape == result.signals.shape
    assert fit.quality.normalized_residual_rms is not None
    assert fit.quality.normalized_residual_rms < 0.05
    assert fit.quality.relative_parameter_errors == {}
    np.testing.assert_array_equal(result.signals, original)


@pytest.mark.parametrize("amplitude", [0.8, -0.8])
def test_rabi_fit_recovers_pi_length_for_both_zero_drive_extrema(
    amplitude: float,
) -> None:
    lengths = np.linspace(0.0, 1.6, 101)
    signals = (
        (1.0 + 0.4j) * (0.3 + amplitude * np.cos(2 * np.pi * lengths / 0.8))
    ).astype(np.complex128)
    result = LenRabiResult(lengths, signals)
    original = result.signals.copy()

    fit = analyze_len_rabi(result)

    assert fit.pi_length_us == pytest.approx(0.4, abs=0.01)
    assert np.isfinite(fit.error_us)
    assert fit.error_us >= 0.0
    assert fit.fitted.shape == result.signals.shape
    assert fit.quality.normalized_residual_rms is not None
    assert fit.quality.normalized_residual_rms < 0.01
    assert fit.quality.relative_parameter_errors == {}
    np.testing.assert_array_equal(result.signals, original)


def test_t1_fit_keeps_native_microseconds_and_actual_residual_diagnostics() -> None:
    times = np.linspace(0.0, 80.0, 41)
    signals = ((1.0 + 0.2j) * (0.1 + 0.9 * np.exp(-times / 20.0))).astype(np.complex128)
    result = T1Result(times, signals)
    original = result.signals.copy()

    fit = analyze_t1(result)

    assert fit.lifetime_us == pytest.approx(20.0, abs=0.01)
    assert np.isfinite(fit.error_us)
    assert fit.error_us >= 0.0
    assert fit.fitted.shape == result.signals.shape
    assert fit.quality.normalized_residual_rms is not None
    assert fit.quality.normalized_residual_rms < 0.01
    assert fit.quality.relative_parameter_errors == {}
    np.testing.assert_array_equal(result.signals, original)
