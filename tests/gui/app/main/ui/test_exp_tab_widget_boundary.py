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


def test_prepare_run_container_clears_stale_run_figure(qapp) -> None:
    tab = _tab()
    tab.show_run_figure(Figure())

    container = tab.prepare_run_container()

    assert container is tab.get_run_container()
    assert tab.get_current_figure_for_pane("run") is None
    assert tab._run_stack.count() == 1


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
    # Capture analysis container for S2 stability check
    captured = tab.get_analysis_container()  # type: ignore[attr-defined]
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

    # Failure should have cleared analysis presentation but retained container identity
    assert tab.get_current_figure_for_pane("analysis") is None  # type: ignore[attr-defined]
    assert tab.get_analysis_container() is captured  # type: ignore[attr-defined]


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
    # Capture analysis container identity before mount
    captured = tab.get_analysis_container()  # type: ignore[attr-defined]
    window._tab_widgets["tab-1"] = tab
    monkeypatch.setattr(
        "zcu_tools.gui.app.main.ui.interactive_analysis.InteractiveAnalysisWidget",
        lambda _ctrl: _Interactive(),
    )

    window.mount_interactive_analysis("tab-1", lambda _widget: object(), lambda _: None)

    # Mount should have cleared analysis pane but kept container identity
    assert tab.get_analysis_container() is captured  # type: ignore[attr-defined]
    assert tab._analysis_stack.count() >= 2  # type: ignore[attr-defined] # placeholder + interactive widget


def test_current_figure_validates_visible_plot_content(qapp) -> None:
    tab = _tab()
    figure = Figure()
    tab.show_analysis_figure(figure)

    assert tab.get_current_figure_for_pane("analysis") is figure  # type: ignore[attr-defined]
    assert tab.current_figure() is figure

    invalid = QWidget()
    tab._analysis_stack.addWidget(invalid)  # type: ignore[attr-defined]
    tab._analysis_stack.setCurrentWidget(invalid)  # type: ignore[attr-defined]

    with pytest.raises(
        RuntimeError, match="tab 'tab-1' canvas has no matplotlib figure"
    ):
        tab.current_figure()
