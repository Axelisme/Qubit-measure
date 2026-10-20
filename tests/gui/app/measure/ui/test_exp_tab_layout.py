"""Behavior tests for ExpTabWidget left-panel and splitter layout."""

from __future__ import annotations

from qtpy.QtCore import Qt
from qtpy.QtWidgets import QSplitter, QToolButton
from zcu_tools.gui.app.measure.adapter import AdapterCapabilities, AnalysisMode
from zcu_tools.gui.app.measure.ui.exp_tab_widget import ExpTabWidget

from tests.gui.app.measure.ui._main_window_fakes import mock_ctrl, tab_pane_tabs


def _left_panel_handle(tab: ExpTabWidget) -> QToolButton:
    """Return the tooltip-identified panel handle, failing unless it is unique."""
    handles = [
        button
        for button in tab.findChildren(QToolButton)
        if button.toolTip() in ("Collapse left panel", "Expand left panel")
    ]
    assert len(handles) == 1
    return handles[0]


def test_left_panel_toggle_is_attached_to_tab_bar(main_window_control_factory, qapp):
    from qtpy.QtWidgets import QApplication
    from zcu_tools.gui.app.measure.ui.main_window import ExpTabWidget

    tab = ExpTabWidget(
        "tab-1",
        mock_ctrl(main_window_control_factory),
        AdapterCapabilities(analysis=AnalysisMode.FIT, post_analysis=False),
    )
    tab.show()
    QApplication.processEvents()

    left_tabs = tab_pane_tabs(tab)
    corner = left_tabs.cornerWidget(Qt.Corner.TopLeftCorner)
    assert corner is None
    assert _left_panel_handle(tab).isVisible() is True


def test_left_panel_toggle_uses_collapsed_boundary_handle(
    main_window_control_factory, qapp
):
    from qtpy.QtWidgets import QApplication
    from zcu_tools.gui.app.measure.ui.main_window import ExpTabWidget

    tab = ExpTabWidget(
        "tab-1",
        mock_ctrl(main_window_control_factory),
        AdapterCapabilities(analysis=AnalysisMode.FIT, post_analysis=False),
    )
    tab.resize(1000, 700)
    tab.show()
    QApplication.processEvents()

    splitter = tab.findChild(QSplitter)
    assert splitter is not None
    handle = _left_panel_handle(tab)
    expanded_left = splitter.sizes()[0]
    expanded_handle_x = handle.x()
    assert expanded_handle_x > 0
    assert handle.toolTip() == "Collapse left panel"

    handle.click()
    QApplication.processEvents()

    assert splitter.sizes()[0] == 0
    assert handle.toolTip() == "Expand left panel"
    assert handle.x() == 0

    handle.click()
    QApplication.processEvents()

    assert handle.toolTip() == "Collapse left panel"
    assert handle.x() > 0
    assert splitter.sizes()[0] >= expanded_left


def test_left_panel_handle_tracks_splitter_boundary(main_window_control_factory, qapp):
    from qtpy.QtWidgets import QApplication
    from zcu_tools.gui.app.measure.ui.main_window import ExpTabWidget

    tab = ExpTabWidget(
        "tab-1",
        mock_ctrl(main_window_control_factory),
        AdapterCapabilities(analysis=AnalysisMode.FIT, post_analysis=False),
    )
    tab.resize(1000, 700)
    tab.show()
    QApplication.processEvents()

    splitter = tab.findChild(QSplitter)
    assert splitter is not None
    left_tabs = tab_pane_tabs(tab)
    handle = _left_panel_handle(tab)
    initial_x = handle.x()
    splitter_x = splitter.geometry().x()
    left_boundary = splitter_x + left_tabs.geometry().right() + 1
    assert abs(initial_x - (left_boundary - handle.width() // 2)) <= 2

    splitter.moveSplitter(260, 1)
    QApplication.processEvents()

    moved_x = handle.x()
    splitter_x = splitter.geometry().x()
    left_boundary = splitter_x + left_tabs.geometry().right() + 1
    assert abs(moved_x - (left_boundary - handle.width() // 2)) <= 2
