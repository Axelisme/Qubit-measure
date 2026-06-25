from __future__ import annotations

import numpy as np
from zcu_tools.analysis.fluxdep import (
    detect_peaks,
    max_dispersion_freq_index,
    onetone_peak_points,
    smoothed_slice,
)


def _crafted_spectrum() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    int,
]:
    devs = np.linspace(0.0, 1.0, 40).astype(np.float64)
    freqs = np.linspace(5.0, 6.0, 20).astype(np.float64)
    feature_row = 10
    fr = freqs[feature_row]
    freq_profile = np.exp(-((freqs - fr) ** 2) / (2 * 0.08**2))
    amp = np.ones((len(devs), len(freqs)), dtype=np.float64)
    depth = np.full(len(devs), 0.3)
    for center in (0.25, 0.75):
        depth += 0.5 * np.exp(-((devs - center) ** 2) / (2 * 0.03**2))
    amp -= depth[:, None] * freq_profile[None, :]
    return amp.astype(np.complex128), devs, freqs, feature_row


def test_max_dispersion_freq_index_finds_feature_row() -> None:
    signals, _devs, freqs, feature_row = _crafted_spectrum()
    idx = max_dispersion_freq_index(signals, freqs)
    assert abs(idx - feature_row) <= 1


def test_detect_peaks_finds_two_onetone_dips() -> None:
    signals, devs, freqs, _feature_row = _crafted_spectrum()
    idx = max_dispersion_freq_index(signals, freqs)
    smoothed = smoothed_slice(signals, idx)
    peaks = detect_peaks(smoothed, threshold=1.0)
    found = np.sort(devs[peaks])
    np.testing.assert_allclose(found, [0.25, 0.75], atol=0.05)


def test_onetone_peak_points_returns_selected_points_and_plot_context() -> None:
    signals, devs, freqs, _feature_row = _crafted_spectrum()
    s_dev, s_freq, idx, smoothed, peaks = onetone_peak_points(
        signals, devs, freqs, threshold=1.0
    )
    assert idx == max_dispersion_freq_index(signals, freqs)
    np.testing.assert_array_equal(s_dev, devs[peaks])
    np.testing.assert_array_equal(s_freq, np.full_like(s_dev, freqs[idx]))
    assert smoothed.shape == devs.shape


def test_flat_slice_returns_finite_zeros() -> None:
    signals = np.ones((4, 3), dtype=np.complex128)
    smoothed = smoothed_slice(signals, 1)
    np.testing.assert_array_equal(smoothed, np.zeros(4))
    assert detect_peaks(smoothed, threshold=0.0).size == 0
