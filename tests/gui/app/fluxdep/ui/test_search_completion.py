"""Search button completion keeps numeric results independent of rendering."""

from __future__ import annotations

from io import BytesIO

import numpy as np
import pytest
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


@pytest.fixture
def search_input(tmp_path, spectrum_hdf5):
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
    return ctrl, result


@pytest.fixture
def completed_search(qapp, search_input, monkeypatch, failure):
    from zcu_tools.gui.app.fluxdep.services import fit
    from zcu_tools.gui.app.fluxdep.ui import analyze_panel

    ctrl, result = search_input
    facts: list[FitChangedPayload] = []
    ctrl.bus.subscribe(FitChangedPayload, facts.append)
    calls: list[str] = []
    warnings: list[tuple[str, str]] = []
    make_figure = analyze_panel.make_search_diagnostic_figure
    present_figure = analyze_panel.QtPlotHost.present

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
        assert sum(p.has_result for p in facts) == calls.count("search")
        if failure == "builder":
            raise RuntimeError("diagnostic builder failed")
        return make_figure(actual)

    def present(host, figure):
        calls.append("present")
        assert QThread.currentThread() == qapp.thread()
        assert ctrl.state.fit.params == result.params
        if failure == "present":
            raise RuntimeError("diagnostic present failed")
        present_figure(host, figure)

    monkeypatch.setattr(fit, "search_database", search)
    monkeypatch.setattr(analyze_panel, "make_search_diagnostic_figure", build)
    monkeypatch.setattr(analyze_panel.QtPlotHost, "present", present)
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
        yield panel, ctrl, result, facts, calls, warnings, buttons
    finally:
        panel.quiesce()
        panel.release_figures()
        panel.deleteLater()


@pytest.mark.parametrize("failure", [None, "builder", "present", "search"])
def test_search_button_commits_before_rendering(completed_search, failure):
    panel, ctrl, result, facts, calls, warnings, buttons = completed_search
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
        assert any("EJ=5.000" in label.text() for label in panel.findChildren(QLabel))
        assert calls == (
            ["search", "builder"]
            if failure == "builder"
            else ["search", "builder", "present"]
        )
        if failure:
            assert len(warnings) == 1
            assert warnings[0][0] == "Diagnostic plot failed"
            assert "Search result retained" in warnings[0][1]
            assert f"diagnostic {failure} failed" in warnings[0][1]
        else:
            assert warnings == []


@pytest.mark.parametrize("failure", [None])
def test_replacing_and_releasing_diagnostics_keeps_retained_figures(
    completed_search, monkeypatch
):
    import matplotlib.pyplot as plt
    from zcu_tools.gui.app.fluxdep.services import fit

    panel, ctrl, _result, _facts, _calls, _warnings, buttons = completed_search
    retained = panel.figures
    first = retained["diagnostic"]
    before = BytesIO()
    first.savefig(before, format="png")
    unrelated = plt.figure()
    try:
        buttons["Search database"].click()
        panel.quiesce()
        second = panel.figures["diagnostic"]
        assert second is not first
        assert retained["diagnostic"] is first
        assert plt.fignum_exists(unrelated.number)
        after = BytesIO()
        first.savefig(after, format="png")
        assert before.getvalue() == after.getvalue()

        def fail_search(*args):
            raise RuntimeError("numeric search failed")

        monkeypatch.setattr(fit, "search_database", fail_search)
        buttons["Search database"].click()
        panel.quiesce()
        assert panel.figures["diagnostic"] is second
        assert ctrl.state.fit.params is None
        assert not buttons["Export params.json"].isEnabled()
        panel.release_figures()
        saved = BytesIO()
        second.savefig(saved, format="png")
        assert saved.getvalue().startswith(b"\x89PNG")
    finally:
        plt.close(unrelated)
