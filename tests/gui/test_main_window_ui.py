"""UI structure tests for ExpTabWidget layout decisions."""

from __future__ import annotations

from unittest.mock import MagicMock

from qtpy.QtCore import Qt


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
    assert (
        abs(initial_x - (left_boundary - tab._left_edge_handle.width() // 2))
        <= 2
    )

    tab._splitter.setSizes([260, 740])
    tab._schedule_handle_layout()
    QApplication.processEvents()

    moved_x = tab._left_edge_handle.x()
    splitter_x = tab._splitter.geometry().x()
    left_boundary = splitter_x + tab._left_tabs.geometry().right() + 1
    assert (
        abs(moved_x - (left_boundary - tab._left_edge_handle.width() // 2))
        <= 2
    )
