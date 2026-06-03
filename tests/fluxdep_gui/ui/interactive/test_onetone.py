"""Tests for OneToneWidget + its pure peak-detection core (headless)."""

from __future__ import annotations

import numpy as np
import pytest
from zcu_tools.fluxdep_gui.ui.interactive.onetone import (
    OneToneWidget,
    detect_peaks,
    max_dispersion_freq_index,
    smoothed_slice,
)


def _crafted_spectrum():
    """A spectrum whose max-dispersion frequency carries two device-value dips.

    dev axis 0..1 (40 pts), freq axis 5..6 GHz (20 pts). One frequency row sits at
    a sharp amplitude step versus its neighbours (so the relative gradient along
    frequency peaks there → it becomes the max-dispersion frequency), and on that
    row two narrow Gaussian dips at dev=0.25/0.75 give two detectable peaks in the
    inverted, smoothed device-value slice.
    """
    devs = np.linspace(0.0, 1.0, 40).astype(np.float64)
    freqs = np.linspace(5.0, 6.0, 20).astype(np.float64)
    feature_row = 10
    # A resonator dip with finite width in the FREQUENCY direction (Gaussian over
    # ~3 rows centred at feature_row), deepened at two device values. The finite
    # width keeps the gradient's argmax inside the dip band (a razor-thin single
    # row would put argmax on the leading edge, off by one), so the max-dispersion
    # frequency lands on the dip and its inverted slice carries two real peaks.
    fr = freqs[feature_row]
    freq_profile = np.exp(-((freqs - fr) ** 2) / (2 * 0.08**2))  # width ~3 rows
    amp = np.ones((len(devs), len(freqs)), dtype=np.float64)
    depth = np.full(len(devs), 0.3)
    for center in (0.25, 0.75):
        depth += 0.5 * np.exp(-((devs - center) ** 2) / (2 * 0.03**2))
    amp -= depth[:, None] * freq_profile[None, :]
    signals = amp.astype(np.complex128)
    return signals, devs, freqs, feature_row


# --- pure core -------------------------------------------------------------


def test_max_dispersion_freq_index_finds_feature_row():
    signals, devs, freqs, feature_row = _crafted_spectrum()
    idx = max_dispersion_freq_index(signals, freqs)
    # the dip row (or an immediate neighbour, due to the smoothing/gradient) wins
    assert abs(idx - feature_row) <= 1


def test_smoothed_slice_and_detect_peaks_find_two_dips():
    signals, devs, freqs, feature_row = _crafted_spectrum()
    idx = max_dispersion_freq_index(signals, freqs)
    smoothed = smoothed_slice(signals, idx)
    peaks = detect_peaks(smoothed, threshold=1.0)
    assert len(peaks) == 2
    # peaks near the two dip centres (0.25, 0.75)
    found = np.sort(devs[peaks])
    np.testing.assert_allclose(found, [0.25, 0.75], atol=0.05)


def test_detect_peaks_higher_threshold_fewer_peaks():
    signals, devs, freqs, _ = _crafted_spectrum()
    idx = max_dispersion_freq_index(signals, freqs)
    smoothed = smoothed_slice(signals, idx)
    low = detect_peaks(smoothed, threshold=0.1)
    high = detect_peaks(smoothed, threshold=10.0)
    assert len(high) <= len(low)


# --- widget (headless) -----------------------------------------------------


@pytest.fixture
def widget(qapp):
    signals, devs, freqs, _ = _crafted_spectrum()
    w = OneToneWidget(signals, devs, freqs, threshold=1.0)
    yield w
    w.deleteLater()


def test_widget_builds_and_detects(widget):
    devs, freqs = widget.get_result()
    assert len(devs) == 2
    # all selected points sit at the single max-dispersion frequency
    assert len(np.unique(freqs)) == 1


def test_widget_threshold_slider_updates_result(widget):
    # drive the threshold very high → fewer/zero peaks
    widget._threshold_slider.setValue(int(10.0 * 100))  # emits valueChanged
    devs_high, _ = widget.get_result()
    widget._threshold_slider.setValue(int(0.1 * 100))
    devs_low, _ = widget.get_result()
    assert len(devs_high) <= len(devs_low)


def test_widget_finished_signal(widget, qapp):
    fired = []
    widget.finished.connect(lambda: fired.append(True))
    # find the Finish button and click it
    from qtpy.QtWidgets import QPushButton

    buttons = widget.findChildren(QPushButton)
    finish = next(b for b in buttons if b.text() == "Finish")
    finish.click()
    assert fired == [True]
