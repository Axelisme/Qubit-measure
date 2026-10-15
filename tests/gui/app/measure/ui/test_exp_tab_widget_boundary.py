"""Behavior tests for the MainWindow-to-ExpTabWidget view boundary."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from matplotlib.figure import Figure
from qtpy.QtCore import QCoreApplication, QEvent
from qtpy.QtWidgets import QTabWidget, QWidget
from zcu_tools.gui.app.measure.controller import Controller
from zcu_tools.gui.app.measure.services import create_persistence_caretaker
from zcu_tools.gui.app.measure.ui.exp_tab_widget import ExpTabWidget
from zcu_tools.gui.app.measure.ui.interactive_frontend import InteractiveFrontend
from zcu_tools.gui.app.measure.ui.main_window import MainWindow
from zcu_tools.plotting.figures import FigureCollection
from zcu_tools.plotting.plots import NonPresentingHost, Plots

from tests.gui._dialog_fakes import RecordingDialogPresenter
from tests.gui.app.measure._native_save_support import (
    NativeSaveSession,
    PaneParams,
    PaneSummary,
)


def _tab(window: MainWindow, tab_id: str) -> ExpTabWidget:
    return next(
        tab for tab in window.findChildren(ExpTabWidget) if tab.tab_id == tab_id
    )


def test_result_focus_and_panel_width_are_owned_by_tab(
    live_ui: tuple[NativeSaveSession, Controller, MainWindow, str],
) -> None:
    _session, ctrl, window, tab_id = live_ui
    tab = _tab(window, tab_id)
    panels = tab.findChild(QTabWidget)
    assert panels is not None
    tab.select_pane("analysis")
    analysis_panel = panels.currentWidget()
    tab.select_pane("run")
    assert panels.currentWidget() is not analysis_panel

    tab.focus_result_panel()

    assert panels.currentWidget() is analysis_panel
    assert tab.left_panel_width() == ctrl.get_left_panel_width()
    assert window.current_left_panel_width() == tab.left_panel_width()


def test_prepare_run_container_clears_run_and_downstream_figures(
    live_ui: tuple[NativeSaveSession, Controller, MainWindow, str],
) -> None:
    _session, _ctrl, window, tab_id = live_ui
    tab = _tab(window, tab_id)
    analysis = FigureCollection()
    analysis.adopt("fit", Figure())
    analysis.seal()
    post = FigureCollection()
    post.adopt("diagnostic", Figure())
    post.seal()
    tab.show_run_figure(Figure())
    tab.show_analysis_figures(analysis)
    tab.show_post_analysis_figures(post)
    for pane in ("run", "analysis", "post_analysis"):
        assert tab.get_current_figure_for_pane(pane) is not None
    captured = tab.get_run_container()

    container = tab.prepare_run_container()

    assert container is captured
    assert container is tab.get_run_container()
    for pane in ("run", "analysis", "post_analysis"):
        assert tab.get_current_figure_for_pane(pane) is None


class _Interactive(InteractiveFrontend):
    def __init__(self) -> None:
        super().__init__()
        self.stopped = False
        self.read_only = False
        self._preview_active = True
        self._figure = Figure()

    @property
    def figure(self) -> Figure:
        return self._figure

    @property
    def preview_active(self) -> bool:
        return self._preview_active

    def set_read_only(self, *, read_only: bool) -> None:
        self.read_only = read_only

    def cancel_preview(self) -> None:
        self._preview_active = False

    def teardown(self) -> None:
        self.stopped = True


class _SelectedInteractive(_Interactive):
    """A frontend type selected for removal while other frontends remain."""


class _RecordingInteractive(_Interactive):
    """Record authority projections delivered to a mounted plugin view."""

    def __init__(self) -> None:
        super().__init__()
        self.read_only_changes: list[bool] = []

    def set_read_only(self, *, read_only: bool) -> None:
        self.read_only_changes.append(read_only)


def test_unmounted_interactive_frontend_stops_authority_projection(
    live_ui: tuple[NativeSaveSession, Controller, MainWindow, str],
) -> None:
    _session, ctrl, window, tab_id = live_ui
    widget = _RecordingInteractive()
    window.mount_interactive_analysis(tab_id, lambda _env: widget)
    assert widget.read_only_changes == [False]

    window.unmount_interactive_analysis(tab_id)
    ctrl.control_identity.connect("agent", "retired-view-check", "owner")

    assert widget.stopped is True
    assert widget.read_only_changes == [False]


@pytest.mark.parametrize(
    "retirement", ["close_tab", "window_close", "runtime_shutdown"]
)
def test_retired_view_stops_authority_projection(
    live_ui: tuple[NativeSaveSession, Controller, MainWindow, str],
    retirement: str,
) -> None:
    _session, ctrl, _fixture_window, _fixture_tab = live_ui
    ctrl.attach_caretaker(create_persistence_caretaker(ctrl))
    tab_id = ctrl.new_tab("saving")
    dialogs = RecordingDialogPresenter(destructive_answers=[True, True])
    window = MainWindow(ctrl, templates=lambda: {}, dialog_presenter=dialogs)
    window.add_tab_widget(tab_id, "saving")
    window.show()
    widget = _RecordingInteractive()
    window.mount_interactive_analysis(tab_id, lambda _env: widget)
    assert widget.read_only_changes == [False]
    try:
        if retirement == "close_tab":
            ctrl.tab_control.close_tab(tab_id)
        elif retirement == "window_close":
            window.close()
            QCoreApplication.processEvents()
            assert not window.isVisible()
        else:
            window.request_shutdown()
            QCoreApplication.processEvents()
            assert not window.isVisible()

        ctrl.control_identity.connect("agent", "retired-view-check", "owner")

        assert widget.read_only_changes == [False]
    finally:
        ctrl.abort_shutdown()
        if ctrl.has_tab(tab_id):
            ctrl.tab_control.close_tab(tab_id)
        window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


@pytest.mark.slow
def test_application_quit_stops_authority_projection(tmp_path: Path) -> None:
    """Exercise real Qt process exit without mutating the suite's QApplication."""
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "tests.gui.app.measure.ui._quit_authority",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "application quit retired authority projection" in completed.stdout


def test_interactive_widget_lifecycle_is_owned_by_tab(
    live_ui: tuple[NativeSaveSession, Controller, MainWindow, str],
) -> None:
    _session, _ctrl, window, tab_id = live_ui
    tab = _tab(window, tab_id)
    first = _SelectedInteractive()
    second = _SelectedInteractive()
    unrelated = _Interactive()
    for widget in (first, second, unrelated):
        tab.mount_interactive_widget(widget)
    assert tab.interactive_frontend() is unrelated

    tab.unmount_interactive_widgets(_SelectedInteractive)
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)

    assert tab.get_current_figure_for_pane("analysis") is None
    remaining = tab.findChildren(_Interactive)
    assert first not in remaining and second not in remaining
    assert unrelated in remaining
    tab.mount_interactive_widget(unrelated)
    assert tab.interactive_frontend() is unrelated
    tab.unmount_interactive_widgets(_Interactive)
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    assert not tab.findChildren(_Interactive)
    assert tab.get_current_figure_for_pane("analysis") is None


@pytest.mark.parametrize("failure_stage", ["factory", "mount"])
def test_interactive_setup_failure_clears_stale_figure_and_cleans_widget(
    live_ui: tuple[NativeSaveSession, Controller, MainWindow, str],
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
) -> None:
    _session, ctrl, window, tab_id = live_ui
    tab = _tab(window, tab_id)
    figures = FigureCollection()
    figures.adopt("stale", Figure())
    figures.seal()
    tab.show_analysis_figures(figures)
    assert tab.get_current_figure_for_pane("analysis") is figures["stale"]
    captured = tab.get_analysis_container()
    widget = _Interactive()
    if failure_stage == "mount":

        def fail_mount(_widget: QWidget) -> None:
            raise RuntimeError("mount failed")

        monkeypatch.setattr(tab, "mount_interactive_widget", fail_mount)

    def frontend_factory(env: Controller) -> InteractiveFrontend:
        assert env is ctrl
        if failure_stage == "factory":
            raise RuntimeError("factory failed")
        return widget

    with pytest.raises(RuntimeError, match=failure_stage):
        window.mount_interactive_analysis(tab_id, frontend_factory)

    assert tab.get_current_figure_for_pane("analysis") is None
    assert tab.get_analysis_container() is captured
    if failure_stage == "mount":
        assert widget.stopped is True


def test_interactive_mount_and_unmount_quiesce_frontend(
    live_ui: tuple[NativeSaveSession, Controller, MainWindow, str],
) -> None:
    _session, _ctrl, window, tab_id = live_ui
    tab = _tab(window, tab_id)
    captured = tab.get_analysis_container()
    widget = _Interactive()

    window.mount_interactive_analysis(tab_id, lambda _env: widget)
    assert tab.get_analysis_container() is captured
    assert widget in tab.findChildren(_Interactive)
    assert tab.interactive_frontend() is widget
    window.unmount_interactive_analysis(tab_id)
    assert widget.stopped is True
    assert tab.get_current_figure_for_pane("analysis") is None
    assert tab.get_analysis_container() is captured


def test_mounted_interactive_view_exposes_figure_and_discardable_preview(
    live_ui: tuple[NativeSaveSession, Controller, MainWindow, str],
) -> None:
    _session, _ctrl, window, tab_id = live_ui
    widget = _Interactive()
    window.mount_interactive_analysis(tab_id, lambda _env: widget)

    assert window.interactive_presentation(tab_id) == (widget.figure, True)
    window.discard_interactive_preview(tab_id)
    assert window.interactive_presentation(tab_id) == (widget.figure, False)
    window.unmount_interactive_analysis(tab_id)
    assert window.interactive_presentation(tab_id) is None


def test_interactive_success_restores_committed_figure_in_analysis_pane(
    live_ui: tuple[NativeSaveSession, Controller, MainWindow, str],
) -> None:
    session, ctrl, window, tab_id = live_ui
    tab = _tab(window, tab_id)
    widget = _Interactive()
    plots = Plots(NonPresentingHost())
    committed_figure, _axes = plots.subplots("fit")
    plots.finish(present=False)
    session.state.update_tab_analyze(
        tab_id, PaneSummary(), plots, analyze_params_instance=PaneParams()
    )
    window.refresh_tab_figure(tab_id, ctrl.get_tab_snapshot(tab_id))
    assert tab.get_current_figure_for_pane("analysis") is committed_figure
    captured = tab.get_analysis_container()

    window.mount_interactive_analysis(tab_id, lambda _env: widget)
    assert tab.get_current_figure_for_pane("analysis") is widget.figure
    window.unmount_interactive_analysis(tab_id, restore_result=True)

    assert widget.stopped is True
    assert tab.get_analysis_container() is captured
    assert tab.get_current_figure_for_pane("analysis") is committed_figure
    assert committed_figure is not widget.figure
