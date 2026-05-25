"""UI structure tests for ExpTabWidget layout decisions."""

from __future__ import annotations

from unittest.mock import MagicMock

from qtpy.QtCore import Qt
from zcu_tools.gui.event_bus import EventBus
from zcu_tools.gui.state import TabInteractionState


def test_left_panel_toggle_is_attached_to_tab_bar(qapp):
    from qtpy.QtWidgets import QApplication
    from zcu_tools.gui.ui.main_window import ExpTabWidget

    tab = ExpTabWidget("tab-1", MagicMock())
    tab.show()
    QApplication.processEvents()

    corner = tab._left_tabs.cornerWidget(Qt.TopLeftCorner)  # type: ignore[attr-defined]
    assert corner is None
    assert tab._left_edge_handle.isVisible() is True


def test_left_panel_toggle_uses_collapsed_boundary_handle(qapp):
    from qtpy.QtWidgets import QApplication
    from zcu_tools.gui.ui.main_window import ExpTabWidget

    tab = ExpTabWidget("tab-1", MagicMock())
    tab.resize(1000, 700)
    tab.show()
    QApplication.processEvents()

    expanded_left = tab._splitter.sizes()[0]
    expanded_handle_x = tab._left_edge_handle.x()
    assert expanded_handle_x > 0
    assert tab._left_panel_collapsed is False

    tab._left_edge_handle.click()
    QApplication.processEvents()

    assert tab._splitter.sizes()[0] == 0
    assert tab._left_panel_collapsed is True
    assert tab._left_edge_handle.x() == 0

    tab._left_edge_handle.click()
    QApplication.processEvents()

    assert tab._left_panel_collapsed is False
    assert tab._left_edge_handle.x() > 0
    assert tab._splitter.sizes()[0] >= expanded_left


def test_left_panel_handle_tracks_splitter_boundary(qapp):
    from qtpy.QtWidgets import QApplication
    from zcu_tools.gui.ui.main_window import ExpTabWidget

    tab = ExpTabWidget("tab-1", MagicMock())
    tab.resize(1000, 700)
    tab.show()
    QApplication.processEvents()

    initial_x = tab._left_edge_handle.x()
    splitter_x = tab._splitter.geometry().x()
    left_boundary = splitter_x + tab._left_tabs.geometry().right() + 1
    assert abs(initial_x - (left_boundary - tab._left_edge_handle.width() // 2)) <= 2

    tab._splitter.setSizes([260, 740])
    tab._schedule_handle_layout()
    QApplication.processEvents()

    moved_x = tab._left_edge_handle.x()
    splitter_x = tab._splitter.geometry().x()
    left_boundary = splitter_x + tab._left_tabs.geometry().right() + 1
    assert abs(moved_x - (left_boundary - tab._left_edge_handle.width() // 2)) <= 2


def test_exp_tab_disables_local_buttons_while_analyzing(qapp):
    from zcu_tools.gui.ui.main_window import ExpTabWidget

    tab = ExpTabWidget("tab-1", MagicMock())
    tab.update_writeback_items([MagicMock(selected=True)])
    tab.update_interaction_state(
        TabInteractionState(
            global_run_active=False,
            is_running=False,
            is_analyzing=True,
            is_saving_data=False,
            has_context=True,
            has_soc=True,
            has_run_result=True,
            has_analyze_result=True,
            has_figure=True,
        )
    )

    assert tab.analyze_btn.isEnabled() is False
    assert tab.writeback_widget.isEnabled() is False
    assert tab.save_image_btn.isEnabled() is False  # disabled because is_analyzing


def test_exp_tab_keeps_analyze_enabled_while_other_tab_running(qapp):
    from zcu_tools.gui.ui.main_window import ExpTabWidget

    tab = ExpTabWidget("tab-1", MagicMock())
    tab.update_interaction_state(
        TabInteractionState(
            global_run_active=True,
            is_running=False,
            is_analyzing=False,
            is_saving_data=False,
            has_context=True,
            has_soc=True,
            has_run_result=True,
            has_analyze_result=True,
            has_figure=True,
        )
    )

    assert tab.run_btn.isEnabled() is False
    assert tab.analyze_btn.isEnabled() is True
    assert tab.save_data_btn.isEnabled() is True


def test_exp_tab_disables_save_buttons_while_saving_data(qapp):
    from zcu_tools.gui.ui.main_window import ExpTabWidget

    tab = ExpTabWidget("tab-1", MagicMock())
    tab.update_interaction_state(
        TabInteractionState(
            global_run_active=False,
            is_running=False,
            is_analyzing=False,
            is_saving_data=True,
            has_context=True,
            has_soc=True,
            has_run_result=True,
            has_analyze_result=True,
            has_figure=True,
        )
    )

    assert tab.save_data_btn.isEnabled() is False
    assert tab.save_both_btn.isEnabled() is False
    assert tab.run_btn.text() == "Run"


def test_main_window_run_lock_disables_only_new_tab_and_run(qapp):
    from zcu_tools.gui.ui.main_window import MainWindow

    ctrl = MagicMock()
    ctrl.get_bus.return_value = EventBus()
    ctrl.is_run_active.return_value = True
    ctrl.has_context.return_value = True
    ctrl.has_soc.return_value = True
    ctrl.has_tab.return_value = True
    ctrl.has_run_result.return_value = True
    ctrl.has_analyze_result.return_value = True
    ctrl.is_tab_running.side_effect = lambda tab_id: tab_id == "tab-1"
    ctrl.is_tab_analyzing.return_value = False
    ctrl.is_tab_saving_data.return_value = False

    window = MainWindow(ctrl)
    tab_one = MagicMock()
    tab_two = MagicMock()
    window._tab_widgets["tab-1"] = tab_one
    window._tab_widgets["tab-2"] = tab_two

    window.refresh_run_lock("tab-1")

    assert window._new_tab_btn.isEnabled() is False
    tab_one.update_interaction_state.assert_called_once()
    tab_two.update_interaction_state.assert_called_once()


def test_main_window_cancel_setup_before_closing(qapp, monkeypatch):
    from qtpy.QtGui import QCloseEvent
    from qtpy.QtWidgets import QMessageBox
    from zcu_tools.gui.ui.main_window import MainWindow

    ctrl = MagicMock()
    ctrl.get_bus.return_value = EventBus()
    ctrl.get_active_device_setup.return_value = object()
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *args, **kwargs: QMessageBox.StandardButton.Yes,
    )
    window = MainWindow(ctrl)
    event = QCloseEvent()

    window.closeEvent(event)

    assert event.isAccepted() is False
    assert window._shutdown_waiting_for_device_setup is True
    ctrl.cancel_device_setup.assert_called_once_with()


def test_main_window_persists_session_on_close_when_idle(qapp):
    from qtpy.QtGui import QCloseEvent
    from zcu_tools.gui.ui.main_window import MainWindow

    ctrl = MagicMock()
    ctrl.get_bus.return_value = EventBus()
    ctrl.get_active_device_setup.return_value = None
    window = MainWindow(ctrl)
    event = QCloseEvent()

    window.closeEvent(event)

    ctrl.persist_tabs_session.assert_called_once_with()


def test_show_analysis_figure_draws_canvas(qapp, monkeypatch):
    from matplotlib.figure import Figure
    from zcu_tools.gui.ui.main_window import ExpTabWidget

    del qapp
    tab = ExpTabWidget("tab-1", MagicMock())
    canvas = MagicMock()

    monkeypatch.setattr(
        "zcu_tools.gui.ui.main_window.attach_existing_figure_to_container",
        lambda fig, container: canvas,
    )

    tab.show_analysis_figure(Figure())

    canvas.draw.assert_called_once_with()
    assert tab._canvas_widget is canvas


def test_show_analysis_figure_keeps_new_canvas_current_when_replacing_old(qapp):
    from matplotlib.figure import Figure
    from qtpy.QtWidgets import QApplication
    from zcu_tools.gui.ui.main_window import ExpTabWidget

    del qapp
    tab = ExpTabWidget("tab-1", MagicMock())
    tab.show()
    QApplication.processEvents()

    fig1 = Figure()
    fig2 = Figure()

    tab.show_analysis_figure(fig1)
    first_canvas = tab._canvas_widget
    assert first_canvas is not None
    assert tab._figure_container._stack.currentWidget() is first_canvas

    tab.show_analysis_figure(fig2)
    second_canvas = tab._canvas_widget
    assert second_canvas is not None
    assert second_canvas is not first_canvas
    assert tab._figure_container._stack.currentWidget() is second_canvas
