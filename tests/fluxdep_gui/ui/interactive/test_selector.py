"""Tests for SelectorWidget (headless)."""

from __future__ import annotations

import numpy as np
import pytest
from zcu_tools.fluxdep_gui.ui.interactive.selector import SelectorWidget
from zcu_tools.notebook.persistance import (
    PointsData,
    SpectrumData,
    SpectrumResult,
)


def _spectrum_result(n_pts=5) -> SpectrumResult:
    fluxs = np.linspace(0.1, 0.4, n_pts).astype(np.float64)
    freqs = np.linspace(4.2, 4.6, n_pts).astype(np.float64)
    grid_flux = np.linspace(0.0, 0.5, 8).astype(np.float64)
    grid_freq = np.linspace(4.0, 5.0, 6).astype(np.float64)
    return SpectrumResult(
        flux_half=0.0,
        flux_int=0.25,
        flux_period=0.5,
        spectrum=SpectrumData(
            dev_values=grid_flux.copy(),
            fluxs=grid_flux,
            freqs=grid_freq,
            signals=np.ones((8, 6), dtype=np.complex128),
        ),
        points=PointsData(dev_values=fluxs.copy(), fluxs=fluxs, freqs=freqs),
    )


def _spectrums() -> dict[str, SpectrumResult]:
    return {"a": _spectrum_result(), "b": _spectrum_result()}


@pytest.fixture
def widget(qapp):
    w = SelectorWidget(_spectrums(), brush_width=0.05)
    yield w
    w.deleteLater()


def test_widget_builds_all_selected_by_default(widget):
    fluxs, freqs, selected = widget.get_result()
    # default: everything selected (10 points across two spectra)
    assert selected.sum() == 10
    assert fluxs.shape == freqs.shape == (10,)


def test_perform_all_erase_clears(widget):
    widget._operation.setCurrentText("Erase")
    widget._on_perform_all()
    _fluxs, _freqs, selected = widget.get_result()
    assert selected.sum() == 0


def test_perform_all_select_restores(widget):
    widget._operation.setCurrentText("Erase")
    widget._on_perform_all()
    widget._operation.setCurrentText("Select")
    widget._on_perform_all()
    _fluxs, _freqs, selected = widget.get_result()
    assert selected.sum() == 10


def test_finished_signal(widget):
    from qtpy.QtWidgets import QPushButton

    fired = []
    widget.finished.connect(lambda: fired.append(True))
    finish = next(b for b in widget.findChildren(QPushButton) if b.text() == "Finish")
    finish.click()
    assert fired == [True]


def test_apply_filter_is_debounced_async(widget):
    g0 = widget._generation
    widget.apply_filter()
    assert widget._generation == g0 + 1
    assert widget._debounce.isActive()


def test_stale_downsample_result_is_dropped(widget):
    widget._generation = 9
    before = widget._filter_mask.copy()
    n = widget._selected.sum()
    widget._on_worker_done(4, np.zeros(n, dtype=bool))  # stale generation
    np.testing.assert_array_equal(widget._filter_mask, before)


def test_fresh_downsample_result_applied(widget):
    widget._generation = 9
    n = int(widget._selected.sum())
    fresh = np.ones(n, dtype=bool)
    widget._on_worker_done(9, fresh)  # current generation
    np.testing.assert_array_equal(widget._filter_mask, fresh)


def test_starts_with_everything_selected(qapp):
    # the brush selection is NOT inherited — all points start selected
    w = SelectorWidget(_spectrums())
    assert w._selected.all()
    w.deleteLater()


def test_inherits_min_distance(qapp):
    w = SelectorWidget(_spectrums(), min_distance=0.05)
    assert abs(w._thresh_val() - 0.05) < 1e-9
    assert abs(w.min_distance() - 0.05) < 1e-9
    w.deleteLater()
