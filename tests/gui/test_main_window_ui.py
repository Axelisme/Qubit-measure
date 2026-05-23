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

    collapsed_left = tab._splitter.sizes()[0]
    assert collapsed_left == 0
    assert tab._left_panel_collapsed is True
    assert tab._left_edge_handle.isVisible() is True
    assert tab._left_edge_handle.x() == 0

    tab._left_edge_handle.click()
    QApplication.processEvents()

    reopened_left = tab._splitter.sizes()[0]
    assert tab._left_panel_collapsed is False
    assert tab._left_edge_handle.isVisible() is True
    assert tab._left_edge_handle.x() > 0
    assert reopened_left >= expanded_left


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


def test_exp_tab_disables_analyze_and_writeback_while_analyzing(qapp):
    from zcu_tools.gui.ui.main_window import ExpTabWidget

    tab = ExpTabWidget("tab-1", MagicMock())
    tab.set_writeback_count(1)
    tab.update_interaction_state(
        TabInteractionState(
            is_running=False,
            is_analyzing=True,
            is_saving_data=False,
            has_context=True,
            has_soc=True,
            has_run_result=True,
            has_analyze_result=True,
        )
    )

    assert tab.analyze_btn.isEnabled() is False
    assert tab.writeback_btn.isEnabled() is False
    assert tab.save_image_btn.isEnabled() is False


def test_exp_tab_disables_save_buttons_while_saving_data(qapp):
    from zcu_tools.gui.ui.main_window import ExpTabWidget

    tab = ExpTabWidget("tab-1", MagicMock())
    tab.update_interaction_state(
        TabInteractionState(
            is_running=False,
            is_analyzing=False,
            is_saving_data=True,
            has_context=True,
            has_soc=True,
            has_run_result=True,
            has_analyze_result=True,
        )
    )

    assert tab.save_data_btn.isEnabled() is False
    assert tab.save_both_btn.isEnabled() is False
    assert tab.run_btn.text() == "Run"


def test_main_window_save_data_busy_does_not_reset_plot(qapp):
    from zcu_tools.gui.ui.main_window import MainWindow

    ctrl = MagicMock()
    ctrl.get_bus.return_value = EventBus()
    ctrl.is_run_active.return_value = False
    ctrl.is_running.return_value = True
    ctrl.is_analyzing.return_value = False
    ctrl.is_saving_data.return_value = True
    ctrl.has_context.return_value = True
    ctrl.has_soc.return_value = True
    ctrl.has_tab.return_value = True
    ctrl.has_run_result.return_value = True
    ctrl.has_analyze_result.return_value = True

    window = MainWindow(ctrl)
    mock_tab = MagicMock()
    window._tab_widgets["tab-1"] = mock_tab

    window.refresh_run_state(False)

    mock_tab.reset_plot.assert_not_called()
    mock_tab.update_interaction_state.assert_called_once()
    assert window._new_tab_btn.isEnabled() is False
