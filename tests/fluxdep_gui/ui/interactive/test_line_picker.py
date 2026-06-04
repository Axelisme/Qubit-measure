"""Tests for LinePickerWidget + its pure helpers (headless)."""

from __future__ import annotations

import numpy as np
import pytest
from zcu_tools.gui.app.fluxdep.ui.interactive.line_picker import (
    LinePickerWidget,
    find_best_mirror_position,
    fold_initial_lines,
)


def _spectrum(n_dev=60, n_freq=30):
    devs = np.linspace(-5.0, 5.0, n_dev).astype(np.float64)
    freqs = np.linspace(4.0, 5.0, n_freq).astype(np.float64)
    # a symmetric feature about dev=0 so mirror-loss has a clear minimum there
    sig = np.zeros((n_dev, n_freq), dtype=np.complex128)
    sig += np.exp(-(devs[:, None] ** 2) / (2 * 1.0**2))
    return sig, devs, freqs


# --- pure helpers ----------------------------------------------------------


def test_fold_initial_lines_defaults():
    _sig, devs, _freqs = _spectrum()
    half, integer = fold_initial_lines(devs, None, None)
    # defaults: half at centre, int near the high end (per InteractiveLines)
    assert devs[0] <= half <= devs[-1]
    assert devs[0] <= integer <= devs[-1]


def test_fold_initial_lines_given_values_returns_floats():
    _sig, devs, _freqs = _spectrum()
    half, integer = fold_initial_lines(devs, 1.0, 2.0)
    assert isinstance(half, float) and isinstance(integer, float)


def test_find_best_mirror_position_returns_in_range():
    sig, devs, _freqs = _spectrum()
    real = np.abs(sig)
    pos = find_best_mirror_position(devs, real, current_pos=1.0, search_width=2.0)
    assert devs[0] <= pos <= devs[-1]


# --- widget (headless) -----------------------------------------------------


@pytest.fixture
def widget(qapp):
    sig, devs, freqs = _spectrum()
    w = LinePickerWidget(sig, devs, freqs)
    yield w
    w.deleteLater()


def test_widget_builds_and_get_result(widget):
    half, integer = widget.get_result()
    assert isinstance(half, float) and isinstance(integer, float)


def test_widget_finished_signal(widget):
    from qtpy.QtWidgets import QPushButton

    fired = []
    widget.finished.connect(lambda: fired.append(True))
    finish = next(b for b in widget.findChildren(QPushButton) if b.text() == "Finish")
    finish.click()
    assert fired == [True]


def test_force_magnitude_hides_checkbox(qapp):
    from qtpy.QtWidgets import QCheckBox

    sig, devs, freqs = _spectrum()
    w = LinePickerWidget(sig, devs, freqs, force_magnitude=True)
    labels = [c.text() for c in w.findChildren(QCheckBox)]
    assert "Magnitude Only" not in labels  # locked on, checkbox hidden
    assert w._only_use_magnitude is True
    w.deleteLater()


def test_no_force_keeps_magnitude_checkbox(qapp):
    from qtpy.QtWidgets import QCheckBox

    sig, devs, freqs = _spectrum()
    w = LinePickerWidget(sig, devs, freqs)
    labels = [c.text() for c in w.findChildren(QCheckBox)]
    assert "Magnitude Only" in labels
    w.deleteLater()
