"""Headless tests for AnalyzePanelWidget — tabs, bounds preset, guards.

The search worker / embedded-figure path is covered elsewhere; here we pin the
panel-local UI behaviour: the three tabs, a bounds preset filling the EJ/EC/EL
spin boxes, and Search being blocked when the database path is empty / missing.
"""

from __future__ import annotations

import pytest
from qtpy.QtWidgets import QTabWidget  # type: ignore[attr-defined]
from zcu_tools.fluxdep_gui.controller import Controller
from zcu_tools.fluxdep_gui.state import FluxDepState
from zcu_tools.fluxdep_gui.ui.analyze_panel import _BOUND_PRESETS, AnalyzePanelWidget


@pytest.fixture
def panel(qapp):
    ctrl = Controller(FluxDepState())
    w = AnalyzePanelWidget(ctrl)
    yield w, ctrl
    w.deleteLater()


def test_three_tabs_filter_search_show(panel):
    w, _ = panel
    tabs = w.findChildren(QTabWidget)
    assert len(tabs) == 1
    labels = [tabs[0].tabText(i) for i in range(tabs[0].count())]
    assert labels == ["Filter", "Search", "Show"]


def test_presets_cover_the_notebook_ranges():
    assert set(_BOUND_PRESETS) == {"general", "integer", "all"}
    for bounds in _BOUND_PRESETS.values():
        assert len(bounds) == 3
        assert all(lo < hi for lo, hi in bounds)


def test_preset_fills_bound_spinboxes(panel):
    w, _ = panel
    w._preset.setCurrentText("integer")
    w._on_preset_selected(0)
    ej, ec, el = _BOUND_PRESETS["integer"]
    assert (w._ej_lo.value(), w._ej_hi.value()) == ej
    assert (w._ec_lo.value(), w._ec_hi.value()) == ec
    assert (w._el_lo.value(), w._el_hi.value()) == el


def test_search_blocked_on_empty_database_path(panel):
    w, ctrl = panel
    w._db_edit.setText("")
    w._on_search()
    assert "database" in w._status.text().lower()
    assert ctrl.state.fit.database_path == ""


def test_search_blocked_on_missing_database_file(panel):
    w, _ = panel
    w._db_edit.setText("/nonexistent/path/to/db.h5")
    w._on_search()
    assert "not found" in w._status.text().lower()


def test_show_tab_has_display_tools(panel):
    w, _ = panel
    # x/y limit boxes + const-freq toggle + a transitions-to-show form exist
    assert hasattr(w, "_x_lo") and hasattr(w, "_y_hi")
    assert w._show_const_freq.isChecked()  # default on
    assert hasattr(w, "_transitions_show")


def test_auto_limits_match_visualizer(panel, qapp):
    import numpy as np
    from zcu_tools.fluxdep_gui.state import SpectrumEntry
    from zcu_tools.notebook.persistance import PointsData, SpectrumData

    w, ctrl = panel
    fx = np.linspace(0.0, 0.5, 6)
    fr = np.linspace(4.0, 6.0, 5)
    raw = SpectrumData(
        dev_values=fx, fluxs=fx, freqs=fr, signals=np.ones((6, 5), complex)
    )
    pts = PointsData(dev_values=fx, fluxs=fx, freqs=np.full(6, 5.0))
    ctrl.state.put_spectrum(
        SpectrumEntry(name="a", spec_type="TwoTone", raw=raw, points=pts, aligned=True)
    )
    w._apply_auto_limits()
    # y spans the spectrum's freq range [4, 6]
    assert w._y_lo.value() == pytest.approx(4.0)
    assert w._y_hi.value() == pytest.approx(6.0)
