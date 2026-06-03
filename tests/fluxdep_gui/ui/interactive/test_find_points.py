"""Tests for FindPointsWidget + toggle_near_mask (headless)."""

from __future__ import annotations

import numpy as np
import pytest
from zcu_tools.fluxdep_gui.ui.interactive.find_points import (
    FindPointsWidget,
    toggle_near_mask,
)


def _spectrum(n_dev=40, n_freq=30):
    devs = np.linspace(-5.0, 5.0, n_dev).astype(np.float64)
    freqs = np.linspace(4.0, 5.0, n_freq).astype(np.float64)
    rng = np.random.RandomState(0)
    sig = (rng.randn(n_dev, n_freq) + 1j * rng.randn(n_dev, n_freq)).astype(
        np.complex128
    )
    return sig, devs, freqs


def test_toggle_near_mask_select_then_erase():
    _sig, devs, freqs = _spectrum()
    mask = np.zeros((len(devs), len(freqs)), dtype=bool)
    toggle_near_mask(devs, freqs, mask, x=0.0, y=4.5, width=0.2, select=True)
    assert mask.any()  # some region selected near (0, 4.5)
    before = mask.sum()
    toggle_near_mask(devs, freqs, mask, x=0.0, y=4.5, width=0.2, select=False)
    assert mask.sum() < before  # erase removed the region


@pytest.fixture
def widget(qapp):
    sig, devs, freqs = _spectrum()
    w = FindPointsWidget(sig, devs, freqs, threshold=1.0, brush_width=0.05)
    yield w
    w.deleteLater()


def test_widget_builds(widget):
    devs, freqs = widget.get_result()
    assert devs.shape == freqs.shape  # sorted, paired


def test_perform_all_erase_clears_then_select_fills(widget):
    # erase everything → no points pass the mask
    widget._operation.setCurrentText("Erase")
    widget._on_perform_all()
    devs_erased, _ = widget.get_result()
    # select everything back
    widget._operation.setCurrentText("Select")
    widget._on_perform_all()
    devs_all, _ = widget.get_result()
    assert len(devs_erased) <= len(devs_all)


def test_result_sorted_by_dev(widget):
    widget._operation.setCurrentText("Select")
    widget._on_perform_all()
    devs, _ = widget.get_result()
    assert np.all(np.diff(devs) >= 0)
