"""The tab bar shrinks tabs instead of scrolling and lists every tab."""

from __future__ import annotations

from qtpy.QtCore import Qt
from qtpy.QtWidgets import QTabWidget, QWidget
from zcu_tools.gui.app.measure.ui.main_window_tab_list import configure_tab_bar


def test_tab_list_menu_lists_tabs_and_selects_the_chosen_one(qapp) -> None:
    tabs = QTabWidget()
    try:
        button = configure_tab_bar(tabs)
        for name in ("twotone/rabi/amp_rabi", "twotone/zigzag", "onetone/freq"):
            tabs.addTab(QWidget(), name)
        tabs.setCurrentIndex(0)

        assert tabs.tabsClosable() and tabs.isMovable()
        assert not tabs.usesScrollButtons()
        assert tabs.elideMode() == Qt.TextElideMode.ElideMiddle
        assert tabs.cornerWidget(Qt.Corner.TopRightCorner) is button

        menu = button.menu()
        assert menu is not None
        menu.aboutToShow.emit()
        actions = menu.actions()
        assert [a.text() for a in actions] == [
            "twotone/rabi/amp_rabi",
            "twotone/zigzag",
            "onetone/freq",
        ]
        assert [a.isChecked() for a in actions] == [True, False, False]

        actions[2].trigger()
        assert tabs.currentIndex() == 2

        tabs.removeTab(0)
        menu.aboutToShow.emit()
        assert [a.text() for a in menu.actions()] == ["twotone/zigzag", "onetone/freq"]
    finally:
        tabs.deleteLater()
