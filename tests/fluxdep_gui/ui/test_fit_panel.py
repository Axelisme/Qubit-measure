"""Headless tests for FitPanelWidget — bounds preset + empty-path guard.

The search worker / embedded-figure path is covered elsewhere; here we pin the
panel-local UI behaviour: a bounds preset fills the EJ/EC/EL spin boxes, and
pressing Search with no database path is blocked before any worker runs.
"""

from __future__ import annotations

import pytest
from zcu_tools.fluxdep_gui.controller import Controller
from zcu_tools.fluxdep_gui.state import FluxDepState
from zcu_tools.fluxdep_gui.ui.fit_panel import _BOUND_PRESETS, FitPanelWidget


@pytest.fixture
def panel(qapp):
    ctrl = Controller(FluxDepState())
    w = FitPanelWidget(ctrl)
    yield w, ctrl
    w.deleteLater()


def test_presets_cover_the_notebook_ranges():
    assert set(_BOUND_PRESETS) == {"general", "integer", "all"}
    # each preset is three (min, max) pairs for EJ / EC / EL
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
    w._db_edit.setText("")  # no database
    w._on_search()
    assert "database" in w._status.text().lower()
    # State was not touched (no params committed, no result)
    assert ctrl.state.fit.database_path == ""


def test_search_blocked_on_missing_database_file(panel):
    w, _ = panel
    w._db_edit.setText("/nonexistent/path/to/db.h5")
    w._on_search()
    assert "not found" in w._status.text().lower()
