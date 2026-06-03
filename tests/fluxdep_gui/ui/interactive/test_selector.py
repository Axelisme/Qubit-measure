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
