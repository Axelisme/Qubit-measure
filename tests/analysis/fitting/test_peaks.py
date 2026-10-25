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
