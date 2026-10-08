"""Workflow analysis contract with actual native projections and optimizers."""

import numpy as np
import pytest

from zcu_lab.v2.twotone.freq.core import FreqResult
from zcu_lab.v2.twotone.rabi.len_rabi.core import LenRabiResult
from zcu_lab.v2.twotone.time_domain.t1.core import T1Result
from zcu_lab.v2.twotone.time_domain.t2echo.core import T2EchoResult
from zcu_lab.workflows.analysis import (
    analyze_len_rabi,
    analyze_qubit_freq,
    analyze_t1,
    analyze_t2echo,
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


def test_echo_fit_excludes_first_sample_before_projection() -> None:
    times = np.linspace(0.0, 80.0, 41)
    signals = ((1.0 + 0.4j) * (0.1 + 0.9 * np.exp(-times / 20.0))).astype(np.complex128)
    signals[0] = 1000.0 - 500.0j
    result = T2EchoResult(times, signals, true_activate_detune=0.0)
    original = result.signals.copy()

    fit = analyze_t2echo(result)

    assert fit.lifetime_us == pytest.approx(20.0, abs=0.01)
    assert np.isfinite(fit.error_us)
    assert fit.error_us >= 0.0
    assert fit.observed.shape == times[1:].shape
    assert fit.fitted.shape == times[1:].shape
    assert fit.quality.normalized_residual_rms is not None
    assert fit.quality.normalized_residual_rms < 0.01
    assert fit.quality.relative_parameter_errors == {}
    np.testing.assert_allclose(fit.observed, fit.fitted, atol=1e-5)
    np.testing.assert_array_equal(result.signals, original)
    np.testing.assert_array_equal(result.times, times)


def test_echo_analysis_accepts_four_retained_samples_without_promising_accuracy() -> (
    None
):
    times = np.linspace(0.0, 80.0, 5)
    signals = np.exp(-times / 20.0).astype(np.complex128)

    fit = analyze_t2echo(T2EchoResult(times, signals, true_activate_detune=0.0))

    assert np.isfinite(fit.lifetime_us)
    assert fit.observed.shape == (4,)
    assert fit.fitted.shape == (4,)
    assert fit.quality.normalized_residual_rms is not None


@pytest.mark.parametrize("detune", [None, -0.1, 0.1, np.nan])
def test_echo_analysis_requires_explicit_zero_activated_detune(
    detune: float | None,
) -> None:
    times = np.linspace(0.0, 80.0, 41)
    signals = np.exp(-times / 20.0).astype(np.complex128)

    with pytest.raises(ValueError, match="explicit zero detune"):
        analyze_t2echo(T2EchoResult(times, signals, true_activate_detune=detune))


@pytest.mark.parametrize(
    "invalid", ["shape", "2d", "short", "nan_time", "nan_iq", "time_order"]
)
def test_echo_analysis_rejects_invalid_retained_trace(invalid: str) -> None:
    times = np.linspace(0.0, 80.0, 41)
    signals = np.exp(-times / 20.0).astype(np.complex128)
    if invalid == "shape":
        signals = signals[:-1]
    elif invalid == "2d":
        signals = signals.reshape(1, -1)
    elif invalid == "short":
        times, signals = times[:4], signals[:4]
    elif invalid == "nan_time":
        times[3] = np.nan
    elif invalid == "nan_iq":
        signals[3] = np.nan
    else:
        times[3] = times[2]

    with pytest.raises(ValueError, match="T2echo"):
        analyze_t2echo(T2EchoResult(times, signals, true_activate_detune=0.0))


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
