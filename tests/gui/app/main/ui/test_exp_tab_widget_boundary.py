"""Behavior tests for the MainWindow-to-ExpTabWidget view boundary."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from matplotlib.figure import Figure
from qtpy.QtWidgets import QWidget
from zcu_tools.gui.app.main.services import PersistedStartup
from zcu_tools.gui.app.main.ui.exp_tab_widget import ExpTabWidget
from zcu_tools.gui.app.main.ui.interactive_frontend import InteractiveFrontend
from zcu_tools.gui.event_bus import BaseEventBus as EventBus


def _tab() -> ExpTabWidget:
    from zcu_tools.gui.app.main.adapter import AdapterCapabilities, AnalysisMode

    ctrl = MagicMock()
    ctrl.get_persisted_startup.return_value = PersistedStartup(left_panel_width=500)
    caps = AdapterCapabilities(analysis=AnalysisMode.FIT, post_analysis=True)
    return ExpTabWidget("tab-1", ctrl, caps)


def test_result_focus_and_panel_width_are_owned_by_tab(qapp) -> None:
    tab = _tab()
    tab._left_tabs.setCurrentIndex(0)

    tab.focus_result_panel()

    assert tab._left_tabs.currentIndex() == 1
    assert tab.left_panel_width() == 500


def test_prepare_run_container_clears_run_and_downstream_figures(qapp) -> None:
    tab = _tab()
    tab.show_run_figure(Figure())
    tab.show_analysis_figure(Figure())
    tab.show_post_analysis_figure(Figure())

    container = tab.prepare_run_container()

    assert container is tab.get_run_container()
    assert tab.get_current_figure_for_pane("run") is None
    assert tab.get_current_figure_for_pane("analysis") is None
    assert tab.get_current_figure_for_pane("post_analysis") is None
    assert tab._run_stack.count() == 1
    assert tab._analysis_stack.count() == 1
    assert tab._post_stack.count() == 1


def test_interactive_widget_lifecycle_is_owned_by_tab(qapp) -> None:
    class _Interactive(QWidget):
        pass

    tab = _tab()
    first = _Interactive()
    second = _Interactive()
    unrelated = QWidget()
    tab.mount_interactive_widget(first)
    # Interactive mounts into analysis pane
    tab._analysis_stack.addWidget(second)  # type: ignore[attr-defined]
    tab._analysis_stack.addWidget(unrelated)  # type: ignore[attr-defined]

    tab.unmount_interactive_widgets(_Interactive)

    assert tab._analysis_stack.indexOf(first) == -1  # type: ignore[attr-defined]
    assert tab._analysis_stack.indexOf(second) == -1  # type: ignore[attr-defined]
    assert tab._analysis_stack.indexOf(unrelated) >= 0  # type: ignore[attr-defined]
    assert tab.get_current_figure_for_pane("analysis") is None


class _Interactive(InteractiveFrontend):
    def __init__(self) -> None:
        super().__init__()
        self.stopped = False
        self._figure = Figure()

    @property
    def figure(self) -> Figure:
        return self._figure

    @property
    def preview_active(self) -> bool:
        return False

    def teardown(self) -> None:
        self.stopped = True


def _window_with_tab():
    from zcu_tools.gui.app.main.ui.main_window import MainWindow

    ctrl = MagicMock()
    ctrl.get_bus.return_value = EventBus()
    ctrl.active_operation_count.return_value = 0
    ctrl.has_agent_connected.return_value = False
    window = MainWindow(ctrl)
    tab = _tab()
    window._tab_widgets["tab-1"] = tab  # type: ignore[reportPrivateUsage] - fixture injection
    return window, tab, ctrl


@pytest.mark.parametrize("failure_stage", ["factory", "mount"])
def test_interactive_setup_failure_clears_stale_figure_and_cleans_widget(
    qapp, monkeypatch, failure_stage: str
) -> None:
    window, tab, _ctrl = _window_with_tab()
    tab.show_analysis_figure(Figure())
    captured = tab.get_analysis_container()
    widget = _Interactive()
    if failure_stage == "mount":

        def fail_mount(_widget: QWidget) -> None:
            raise RuntimeError("mount failed")

        monkeypatch.setattr(tab, "mount_interactive_widget", fail_mount)

    def frontend_factory(_env):
        if failure_stage == "factory":
            raise RuntimeError("factory failed")
        return widget

    with pytest.raises(RuntimeError, match=failure_stage):
        window.mount_interactive_analysis("tab-1", frontend_factory)

    assert tab.get_current_figure_for_pane("analysis") is None
    assert tab.get_analysis_container() is captured
    if failure_stage == "mount":
        assert widget.stopped is True


def test_interactive_mount_and_unmount_quiesce_frontend(qapp) -> None:
    window, tab, _ctrl = _window_with_tab()
    captured = tab.get_analysis_container()
    widget = _Interactive()

    window.mount_interactive_analysis("tab-1", lambda _env: widget)
    assert tab.get_analysis_container() is captured
    assert widget in tab.findChildren(_Interactive)
    window.unmount_interactive_analysis("tab-1")
    assert widget.stopped is True
    assert tab.get_current_figure_for_pane("analysis") is None


def test_interactive_success_restores_committed_figure_in_analysis_pane(qapp) -> None:
    window, tab, ctrl = _window_with_tab()
    widget = _Interactive()
    committed_figure = widget.figure
    ctrl.get_tab_analyze_result.return_value = MagicMock(figure=committed_figure)

    window.mount_interactive_analysis("tab-1", lambda _env: widget)
    window.unmount_interactive_analysis("tab-1", restore_result=True)

    assert widget.stopped is True
    assert tab.get_current_figure_for_pane("analysis") is committed_figure
