"""Navigate experiment tabs with native overflow arrows and an all-tabs menu."""

from __future__ import annotations

from qtpy.QtCore import Qt
from qtpy.QtGui import QAction
from qtpy.QtWidgets import QMenu, QTabWidget, QToolButton


class TabListButton(QToolButton):
    """Corner button whose menu lists every tab and selects the chosen one."""

    def __init__(self, tabs: QTabWidget) -> None:
        super().__init__(tabs)
        self._tabs = tabs
        self.setText("▾")
        self.setToolTip("All tabs")
        self.setAutoRaise(True)
        self.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        menu = QMenu(self)
        menu.aboutToShow.connect(self._rebuild_menu)
        menu.triggered.connect(self._select_tab)
        self.setMenu(menu)

    def _rebuild_menu(self) -> None:
        menu = self.menu()
        assert menu is not None
        menu.clear()
        current = self._tabs.currentIndex()
        for index in range(self._tabs.count()):
            action = menu.addAction(self._tabs.tabText(index))
            assert action is not None
            action.setCheckable(True)
            action.setChecked(index == current)
            action.setData(index)

    def _select_tab(self, action: QAction) -> None:
        self._tabs.setCurrentIndex(int(action.data()))


def configure_tab_bar(tabs: QTabWidget) -> TabListButton:
    """Configure ``tabs`` with close buttons, dragging and native overflow arrows.

    Install and return the top-right all-tabs menu button. Tab labels retain both
    ends when elided. This function does not change control authority; the window
    can hide close buttons while Local editing is unavailable. No I/O is performed.
    """
    tabs.setTabsClosable(True)
    tabs.setMovable(True)
    tabs.setUsesScrollButtons(True)
    # Adapter names share prefixes ("twotone/...") and differ at the end, so
    # keep both ends visible.
    tabs.setElideMode(Qt.TextElideMode.ElideMiddle)
    button = TabListButton(tabs)
    tabs.setCornerWidget(button, Qt.Corner.TopRightCorner)
    return button
