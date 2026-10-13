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
from zcu_tools.analysis.fluxdep.search import (
    ParamBounds,
    SearchExecution,
)
from zcu_tools.gui.app.fluxdep.controller import Controller
from zcu_tools.gui.app.fluxdep.event_bus import FitChangedPayload
from zcu_tools.gui.app.fluxdep.search import FluxDepSearchRuntime
from zcu_tools.gui.app.fluxdep.state import FluxDepState
from zcu_tools.gui.app.fluxdep.ui.analyze_panel import AnalyzePanelWidget
from zcu_tools.gui.app.fluxdep.ui.error_messages import friendly_fit_message
from zcu_tools.gui.session.adapters.qt_background import BackgroundRunner
from zcu_tools.gui.session.adapters.qt_owner_scheduler import QtOwnerScheduler
from zcu_tools.gui.session.adapters.qt_progress_transport import QtProgressTransport
from zcu_tools.gui.session.services.progress import ProgressService

from tests.gui.app.fluxdep._search_result import perfect_fit_result
from tests.gui.app.fluxdep._window import named


@pytest.fixture
def search_input(qapp, search_database_hdf5, spectrum_hdf5, failure, monkeypatch):
    runner = BackgroundRunner()
    transport = QtProgressTransport()
    ctrl = Controller(
        FluxDepState(),
        interactive_owner=QtOwnerScheduler(),
        search_runtime=FluxDepSearchRuntime(runner, ProgressService(transport)),
    )
    name = ctrl.load_spectrum(spectrum_hdf5[0], spec_type="TwoTone")
    ctrl.set_alignment(name, flux_half=0.0, flux_int=1.0)
    ctrl.set_points(name, np.array([0.0, 0.1]), np.array([5.0, 5.1]))
    bounds = ParamBounds(EJ=(2.0, 15.0), EC=(0.2, 2.0), EL=(0.1, 2.0))
    ctrl.set_fit_params(
        search_database_hdf5,
        bounds.EJ,
        bounds.EC,
        bounds.EL,
        TransitionDict({"transitions": [(0, 1)]}),
        None,
        None,
    )
    result = perfect_fit_result(np.array([0.0, 0.1]), np.array([5.0, 5.1]), bounds)
    if failure == "submit":

        def fail_submit(*args, **kwargs):
            raise RuntimeError("submit failed")

        monkeypatch.setattr(runner, "submit", fail_submit)
    yield ctrl, result, runner
    ctrl.search.begin_close()
    assert runner.quiesce()
    ctrl.interactive.dispose()


@pytest.fixture
def search_panel(qapp, search_input, monkeypatch, failure):
    from zcu_tools.gui.app.fluxdep.services import fit, viz
    from zcu_tools.gui.app.fluxdep.ui import analyze_panel

    ctrl, result, runner = search_input
    facts: list[FitChangedPayload] = []
    ctrl.bus.subscribe(FitChangedPayload, facts.append)
    calls: list[str] = []
    warnings: list[tuple[str, str]] = []
    make_figure = analyze_panel.make_search_diagnostic_figure
    present_figure = analyze_panel.QtPlotHost.present

    def search(*args, execution: SearchExecution | None = None):
        del args, execution
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
        viz,
        "calculate_energy_vs_flux",
        lambda *args, **kwargs: (None, np.zeros((1000, 15))),
    )
    monkeypatch.setattr(viz, "render_fit_figure", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        lambda parent, title, message: warnings.append((title, message)),
    )
    panel = AnalyzePanelWidget(ctrl)
    try:
        yield panel, ctrl, result, facts, calls, warnings, runner
    finally:
        assert runner.quiesce()
        panel.quiesce()
        panel.dispose()
        panel.deleteLater()


def _click_search(panel: AnalyzePanelWidget, runner: BackgroundRunner) -> None:
    """Click Search and drain the search worker and its owner delivery."""
    named(panel, QPushButton, "fluxdep.fit.search").click()
    assert runner.quiesce()


@pytest.mark.parametrize("failure", [None, "builder", "present", "search", "submit"])
def test_search_button_commits_before_rendering(search_panel, failure):
    panel, ctrl, result, facts, calls, warnings, runner = search_panel
    _click_search(panel, runner)
    assert named(panel, QPushButton, "fluxdep.fit.search").isEnabled()
    export = named(panel, QPushButton, "fluxdep.fit.export_params")
    if failure in ("search", "submit"):
        assert calls == (["search"] if failure == "search" else [])
        assert ctrl.state.fit.params is None
        assert not any(p.has_result for p in facts)
        assert not export.isEnabled()
        # A failed search is reported on the Search status line, not in a modal.
        assert warnings == []
        activity = ctrl.search.current
        assert activity is not None and activity.error is not None
        assert named(panel, QLabel, "fluxdep.fit.status").text() == (
            friendly_fit_message("Search", activity.error)
        )
    else:
        assert ctrl.state.fit.params == result.params
        assert [p.has_result for p in facts if p.has_result] == [True]
        assert export.isEnabled()
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
    search_panel, monkeypatch
):
    import matplotlib.pyplot as plt
    from zcu_tools.gui.app.fluxdep.services import fit

    panel, ctrl, _result, _facts, _calls, _warnings, runner = search_panel
    _click_search(panel, runner)
    retained = panel.figures
    first = retained["diagnostic"]
    before = BytesIO()
    first.savefig(before, format="png")
    unrelated = plt.figure()
    try:
        _click_search(panel, runner)
        second = panel.figures["diagnostic"]
        assert second is not first
        assert retained["diagnostic"] is first
        assert plt.fignum_exists(unrelated.number)
        after = BytesIO()
        first.savefig(after, format="png")
        assert before.getvalue() == after.getvalue()

        def fail_search(*args, execution: SearchExecution | None = None):
            del args, execution
            raise RuntimeError("numeric search failed")

        monkeypatch.setattr(fit, "search_database", fail_search)
        _click_search(panel, runner)
        assert panel.figures["diagnostic"] is second
        assert ctrl.state.fit.params is None
        assert not named(panel, QPushButton, "fluxdep.fit.export_params").isEnabled()
        panel.release_figures()
        saved = BytesIO()
        second.savefig(saved, format="png")
        assert saved.getvalue().startswith(b"\x89PNG")
    finally:
        plt.close(unrelated)
