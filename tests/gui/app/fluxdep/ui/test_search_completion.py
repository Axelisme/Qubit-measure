"""Search button completion keeps numeric results independent of rendering."""

from __future__ import annotations

import numpy as np
import pytest
from matplotlib.figure import Figure
from qtpy.QtCore import QThread  # type: ignore[attr-defined]
from qtpy.QtWidgets import (  # type: ignore[attr-defined]
    QLabel,
    QMessageBox,
    QPushButton,
)
from zcu_tools.analysis.fluxdep.models import TransitionDict
from zcu_tools.analysis.fluxdep.search import DatabaseSearchResult, ParamBounds
from zcu_tools.gui.app.fluxdep.controller import Controller
from zcu_tools.gui.app.fluxdep.event_bus import FitChangedPayload
from zcu_tools.gui.app.fluxdep.state import FluxDepState
from zcu_tools.gui.app.fluxdep.ui.analyze_panel import AnalyzePanelWidget


@pytest.mark.parametrize("failure", [None, "builder", "show", "search"])
def test_search_button_commits_before_rendering(
    qapp, tmp_path, spectrum_hdf5, monkeypatch, failure
):
    import matplotlib.pyplot as plt
    from zcu_tools.gui.app.fluxdep.services import fit
    from zcu_tools.gui.app.fluxdep.ui import analyze_panel

    ctrl = Controller(FluxDepState())
    name = ctrl.load_spectrum(spectrum_hdf5[0], spec_type="TwoTone")
    ctrl.set_alignment(name, flux_half=0.0, flux_int=1.0)
    ctrl.set_points(name, np.array([0.0, 0.1]), np.array([5.0, 5.1]))
    db_path = tmp_path / "database.h5"
    db_path.touch()
    bounds = ParamBounds(EJ=(2.0, 15.0), EC=(0.2, 2.0), EL=(0.1, 2.0))
    ctrl.set_fit_params(
        str(db_path),
        bounds.EJ,
        bounds.EC,
        bounds.EL,
        TransitionDict({"transitions": [(0, 1)]}),
        None,
        None,
    )
    result = DatabaseSearchResult(
        params=(5.0, 1.0, 0.5),
        best_distance=0.1,
        best_scale=1.0,
        best_index=0,
        entry_results=np.array([[0.1, 1.0]]),
        entry_params=np.array([[5.0, 1.0, 0.5]]),
        fluxs=np.array([0.0, 0.1]),
        freqs=np.array([5.0, 5.1]),
        predicted_freqs=np.array([5.0, 5.1]),
        bounds=bounds,
    )
    facts: list[FitChangedPayload] = []
    ctrl.bus.subscribe(FitChangedPayload, facts.append)
    calls: list[str] = []
    warnings: list[tuple[str, str]] = []

    def search(*args):
        calls.append("search")
        assert QThread.currentThread() != qapp.thread()
        if failure == "search":
            raise RuntimeError("numeric search failed")
        return result

    def build(actual):
        calls.append("builder")
        assert QThread.currentThread() == qapp.thread()
        assert actual is result
        assert ctrl.state.fit.params == result.params
        assert [p.has_result for p in facts if p.has_result] == [True]
        if failure == "builder":
            raise RuntimeError("diagnostic builder failed")
        return Figure()

    def show():
        calls.append("show")
        assert ctrl.state.fit.params == result.params
        if failure == "show":
            raise RuntimeError("diagnostic show failed")

    monkeypatch.setattr(fit, "search_database", search)
    monkeypatch.setattr(analyze_panel, "make_search_diagnostic_figure", build)
    monkeypatch.setattr(plt, "show", show)
    monkeypatch.setattr(
        analyze_panel,
        "calculate_energy_vs_flux",
        lambda *args, **kwargs: (None, np.zeros((1000, 15))),
    )
    monkeypatch.setattr(
        analyze_panel, "render_fit_figure", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        lambda parent, title, message: warnings.append((title, message)),
    )
    panel = AnalyzePanelWidget(ctrl)
    buttons = {button.text(): button for button in panel.findChildren(QPushButton)}
    try:
        buttons["Search database"].click()
        panel.quiesce()
        assert buttons["Search database"].isEnabled()
        if failure == "search":
            assert calls == ["search"]
            assert ctrl.state.fit.params is None
            assert not any(p.has_result for p in facts)
            assert not buttons["Export params.json"].isEnabled()
            assert warnings[0][0] == "Search failed"
        else:
            assert ctrl.state.fit.params == result.params
            assert [p.has_result for p in facts if p.has_result] == [True]
            assert buttons["Export params.json"].isEnabled()
            assert any(
                "EJ=5.000" in label.text() for label in panel.findChildren(QLabel)
            )
            assert calls == (
                ["search", "builder"]
                if failure == "builder"
                else ["search", "builder", "show"]
            )
            if failure:
                assert len(warnings) == 1
                assert warnings[0][0] == "Diagnostic plot failed"
                assert "Search result retained" in warnings[0][1]
                assert f"diagnostic {failure} failed" in warnings[0][1]
            else:
                assert warnings == []
    finally:
        panel.quiesce()
        panel.deleteLater()
