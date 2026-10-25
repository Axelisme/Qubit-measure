"""Acceptance tests for the numeric peak-candidate interface."""

import numpy as np
import pytest
from zcu_tools.analysis.fitting.peaks import detect_peaks


def test_single_peak_reports_center_height_and_half_prominence_width():
    axis = np.linspace(-10, 10, 2001)
    signal = 3 + 2 * np.exp(-4 * np.log(2) * ((axis - 1.5) / 0.8) ** 2)

    peaks = detect_peaks(axis, signal, min_prominence=0.2, min_distance=0.1)

    assert len(peaks) == 1
    peak = peaks[0]
    assert peak.center == pytest.approx(1.5, abs=0.01)
    assert peak.height == pytest.approx(5)
    assert peak.width == pytest.approx(0.8, abs=0.01)
    assert peak.prominence == pytest.approx(2)
    assert not peak.truncated


def test_two_peaks_sort_and_filter_by_prominence_and_axis_distance():
    # Descending and nonuniform axes still use physical, not sample, distances.
    axis = np.linspace(-1, 1, 2001) ** 3 * 10
    signal = np.exp(-(((axis + 3) / 0.25) ** 2)) + 2 * np.exp(
        -(((axis - 2) / 0.4) ** 2)
    )
    peaks = detect_peaks(axis[::-1], signal[::-1], min_prominence=0.2)
    assert [peak.center for peak in peaks] == pytest.approx([2, -3], abs=0.02)
    assert all(peak.width > 0 for peak in peaks)
    assert peaks[0].prominence > peaks[1].prominence
    separated = detect_peaks(axis, signal, min_distance=6)
    assert len(separated) == 1
    assert separated[0].center == pytest.approx(2, abs=0.02)
    prominent = detect_peaks(axis, signal, min_prominence=0.75)
    assert len(prominent) == 1
    assert prominent[0].center == pytest.approx(2, abs=0.02)


@pytest.fixture
def noisy_spectrum():
    axis = np.linspace(-10, 10, 201)
    noise = np.random.default_rng(42).normal(0, 0.05, axis.size)
    return axis, noise


def test_pure_noise_has_no_candidates(noisy_spectrum):
    axis, noise = noisy_spectrum
    assert detect_peaks(axis, noise) == []


def test_real_peak_on_noise_reports_one_significant_candidate(noisy_spectrum):
    axis, noise = noisy_spectrum
    signal = noise + np.exp(-4 * np.log(2) * (axis / 2) ** 2)

    peaks = detect_peaks(axis, signal)

    assert len(peaks) == 1
    assert peaks[0].center == pytest.approx(0, abs=0.2)
    assert peaks[0].snr >= 5


def test_snr_measures_height_above_median_rather_than_prominence():
    axis = np.arange(100, dtype=np.float64)
    signal = np.tile(np.array([-1.0, 0.0, 1.0, 0.0]), 25)
    signal[48:52] = 10.0

    peaks = detect_peaks(axis, signal)

    assert len(peaks) == 1
    # Worked trace: median=0, first-difference MAD=1, peak height=10.
    assert peaks[0].snr == pytest.approx(9.53872552)
    assert peaks[0].prominence == pytest.approx(11)


def test_lowering_snr_threshold_reveals_low_contrast_peak(noisy_spectrum):
    axis, noise = noisy_spectrum
    signal = noise + 0.15 * np.exp(-4 * np.log(2) * (axis / 2) ** 2)

    assert detect_peaks(axis, signal) == []
    peaks = detect_peaks(axis, signal, min_snr=3.5)
    assert len(peaks) == 1
    assert peaks[0].center == pytest.approx(0, abs=0.2)
    assert 3.5 <= peaks[0].snr < 5


def test_single_sample_spike_is_not_a_peak():
    axis = np.linspace(-10, 10, 201)
    signal = np.zeros_like(axis)
    signal[100] = 10

    assert detect_peaks(axis, signal) == []
    assert len(detect_peaks(axis, signal, min_width_samples=0)) == 1


@pytest.mark.parametrize("option", ["min_snr", "min_width_samples"])
@pytest.mark.parametrize("value", [-1.0, np.inf, np.nan])
def test_invalid_noise_gates_raise_value_error(option, value):
    axis = np.linspace(-10, 10, 201)
    with pytest.raises(ValueError, match=option):
        detect_peaks(axis, np.zeros_like(axis), **{option: value})


def test_flat_signal_has_no_candidates():
    axis = np.linspace(-5, 5, 101)
    assert detect_peaks(axis, np.full_like(axis, 4.0)) == []


@pytest.mark.parametrize("center", [-5.0, 5.0])
def test_edge_peak_is_retained_with_truncated_observable_width(center):
    axis = np.linspace(-5, 5, 1001)
    signal = np.exp(-4 * np.log(2) * ((axis - center) / 0.8) ** 2)
    peaks = detect_peaks(axis, signal)
    assert len(peaks) == 1
    assert peaks[0].center == pytest.approx(center, abs=0.01)
    assert peaks[0].truncated
    assert peaks[0].width == pytest.approx(0.4, abs=0.01)
    assert peaks[0].prominence == pytest.approx(1)


@pytest.mark.parametrize(
    ("axis", "signal", "prominence", "distance"),
    [
        ([0, 1, 1], [0, 1, 0], 0.2, 0),
        ([0, 2, 1], [0, 1, 0], 0.2, 0),
        ([0, 1, 2], [0, np.nan, 0], 0.2, 0),
        ([0, 1, 2], [0, 1], 0.2, 0),
        ([0], [1], 0.2, 0),
        ([[0, 1, 2]], [[0, 1, 0]], 0.2, 0),
        ([0, 1, 2], [0, 1, 0], -0.1, 0),
        ([0, 1, 2], [0, 1, 0], 1.1, 0),
        ([0, 1, 2], [0, 1, 0], 0.2, -1),
        ([0, 1, 2], [0, 1, 0], 0.2, np.inf),
    ],
)
def test_invalid_inputs_raise_value_error(axis, signal, prominence, distance):
    with pytest.raises(ValueError, match=".+"):
        detect_peaks(
            np.asarray(axis, dtype=np.float64),
            np.asarray(signal, dtype=np.float64),
            min_prominence=prominence,
            min_distance=distance,
        )
