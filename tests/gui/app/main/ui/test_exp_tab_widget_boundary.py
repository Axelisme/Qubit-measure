"""Behavior tests for the MainWindow-to-ExpTabWidget view boundary."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from matplotlib.figure import Figure
from qtpy.QtWidgets import QWidget
from zcu_tools.gui.app.main.services import PersistedStartup
from zcu_tools.gui.app.main.ui.exp_tab_widget import ExpTabWidget
from zcu_tools.gui.event_bus import BaseEventBus as EventBus


def _tab() -> ExpTabWidget:
    ctrl = MagicMock()
    ctrl.get_persisted_startup.return_value = PersistedStartup(left_panel_width=500)
    return ExpTabWidget("tab-1", ctrl)


def test_result_focus_and_panel_width_are_owned_by_tab(qapp) -> None:
    tab = _tab()
    tab._left_tabs.setCurrentIndex(0)

    tab.focus_result_panel()

    assert tab._left_tabs.currentIndex() == 1
    assert tab.left_panel_width() == 500


def test_prepare_live_container_clears_stale_figure(qapp) -> None:
    tab = _tab()
    tab.show_analysis_figure(Figure())

    container = tab.prepare_live_container()

    assert container is tab._figure_container
    assert tab.current_figure() is None
    assert tab._plot_stack.count() == 1


def test_interactive_widget_lifecycle_is_owned_by_tab(qapp) -> None:
    class _Interactive(QWidget):
        pass

    tab = _tab()
    reset_plot = MagicMock(wraps=tab.reset_plot)
    tab.reset_plot = reset_plot
    first = _Interactive()
    second = _Interactive()
    unrelated = QWidget()
    tab.mount_interactive_widget(first)
    tab._plot_stack.addWidget(second)
    tab._plot_stack.addWidget(unrelated)

    tab.unmount_interactive_widgets(_Interactive)

    assert tab._plot_stack.indexOf(first) == -1
    assert tab._plot_stack.indexOf(second) == -1
    assert tab._plot_stack.indexOf(unrelated) >= 0
    assert tab.current_figure() is None
    reset_plot.assert_not_called()


@pytest.mark.parametrize("failure_stage", ["session_factory", "bind"])
def test_interactive_setup_failure_clears_stale_figure_before_setup(
    qapp, monkeypatch, failure_stage: str
) -> None:
    from zcu_tools.gui.app.main.ui.main_window import MainWindow

    class _Interactive(QWidget):
        def bind(self, session: object, *, on_done: object) -> None:
            del session, on_done
            if failure_stage == "bind":
                raise RuntimeError("bind failed")

    ctrl = MagicMock()
    ctrl.get_bus.return_value = EventBus()
    ctrl.active_operation_count.return_value = 0
    ctrl.has_agent_connected.return_value = False
    window = MainWindow(ctrl)
    tab = _tab()
    tab.show_analysis_figure(Figure())
    reset_plot = MagicMock(wraps=tab.reset_plot)
    tab.reset_plot = reset_plot
    window._tab_widgets["tab-1"] = tab
    monkeypatch.setattr(
        "zcu_tools.gui.app.main.ui.interactive_analysis.InteractiveAnalysisWidget",
        lambda _ctrl: _Interactive(),
    )

    def session_factory(_widget: QWidget) -> object:
        if failure_stage == "session_factory":
            raise RuntimeError("session factory failed")
        return object()

    with pytest.raises(RuntimeError, match=failure_stage.replace("_", " ")):
        window.mount_interactive_analysis(
            "tab-1", session_factory, lambda _session: None
        )

    assert tab.current_figure() is None
    reset_plot.assert_called_once_with()


def test_interactive_mount_resets_plot_exactly_once(qapp, monkeypatch) -> None:
    from zcu_tools.gui.app.main.ui.main_window import MainWindow

    class _Interactive(QWidget):
        def bind(self, session: object, *, on_done: object) -> None:
            del session, on_done

    ctrl = MagicMock()
    ctrl.get_bus.return_value = EventBus()
    ctrl.active_operation_count.return_value = 0
    ctrl.has_agent_connected.return_value = False
    window = MainWindow(ctrl)
    tab = _tab()
    reset_plot = MagicMock(wraps=tab.reset_plot)
    tab.reset_plot = reset_plot
    window._tab_widgets["tab-1"] = tab
    monkeypatch.setattr(
        "zcu_tools.gui.app.main.ui.interactive_analysis.InteractiveAnalysisWidget",
        lambda _ctrl: _Interactive(),
    )

    window.mount_interactive_analysis("tab-1", lambda _widget: object(), lambda _: None)

    reset_plot.assert_called_once_with()


def test_current_figure_validates_visible_plot_content(qapp) -> None:
    tab = _tab()
    figure = Figure()
    tab.show_analysis_figure(figure)

    assert tab.current_figure() is figure

    invalid = QWidget()
    tab._plot_stack.addWidget(invalid)
    tab._plot_stack.setCurrentWidget(invalid)

    with pytest.raises(
        RuntimeError, match="tab 'tab-1' canvas has no matplotlib figure"
    ):
        tab.current_figure()
