"""Overflowed tabs remain reachable by native arrows and the corner menu."""

from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest
from qtpy.QtWidgets import QApplication, QTabBar, QTabWidget, QWidget
from zcu_tools.gui.app.measure.ui.main_window_tab_list import configure_tab_bar
from zcu_tools.gui.lifetime import LifetimeScope

from tests.gui._user_input import ComponentDriver


@dataclass(frozen=True)
class TabDisplay:
    """Rendered tab availability and the fully visible close button's hit point."""

    fully_visible: bool
    close_center: tuple[int, int] | None


class TabListDriver(ComponentDriver):
    """Observe Qt's public tab geometry without exposing application state."""

    def __init__(self, tabs: QTabWidget) -> None:
        super().__init__(tabs)
        self.host = tabs

    def tab_display(self, caption: str) -> TabDisplay:
        """Read caption's clipping and close affordance; raise for absent captions."""
        bar = self.host.tabBar()
        if bar is None:
            raise LookupError("No tab bar is mounted")
        index = next(i for i in range(bar.count()) if bar.tabText(i) == caption)
        rect = bar.tabRect(index)
        close = bar.tabButton(index, QTabBar.ButtonPosition.RightSide)
        if close is None:
            close = bar.tabButton(index, QTabBar.ButtonPosition.LeftSide)
        center = None
        if (
            close is not None
            and close.isVisible()
            and close.visibleRegion().contains(close.rect())
        ):
            point = close.mapToGlobal(close.rect().center())
            center = (point.x(), point.y())
        return TabDisplay(bar.rect().contains(rect), center)


@pytest.fixture
def tab_list(qapp: QApplication) -> Iterator[QTabWidget]:
    tabs = QTabWidget()
    configure_tab_bar(tabs)
    scope = LifetimeScope(tabs)
    scope.connect(tabs.tabCloseRequested, lambda index: tabs.removeTab(index))
    tabs.resize(1200, 250)
    try:
        yield tabs
    finally:
        scope.close()
        tabs.close()
        tabs.deleteLater()
        qapp.processEvents()


def evidence_dir(tmp_path: Path) -> Path:
    """Use caller-owned evidence when requested, otherwise pytest's temp directory."""
    return Path(os.environ.get("TAB_LIST_EVIDENCE", str(tmp_path)))


def test_tab_list_menu_lists_tabs_and_selects_the_chosen_one(
    tab_list: QTabWidget, qapp: QApplication, tmp_path: Path
) -> None:
    names = ("twotone/rabi/amp_rabi", "twotone/zigzag", "onetone/freq")
    for name in names:
        tab_list.addTab(QWidget(), name)
    tab_list.show()
    qapp.processEvents()
    driver = ComponentDriver(tab_list)
    evidence = evidence_dir(tmp_path)
    driver.screenshot(evidence / "menu-00-start.png")
    actions = driver.menu_actions("▾", evidence=evidence / "menu-01-expanded.png")
    assert tuple(a.text for a in actions) == names
    driver.select_menu("▾", (names[-1],))
    driver.screenshot(evidence / "menu-02-selected.png")
    assert driver.tabs(names[-1]).current == names[-1]


def test_overflow_arrows_reveal_last_tab_and_its_close_button(
    tab_list: QTabWidget, qapp: QApplication, tmp_path: Path
) -> None:
    names = tuple(f"synthetic/spectrum/scan-{i:02d}" for i in range(20))
    for name in names:
        tab_list.addTab(QWidget(), name)
    tab_list.show()
    qapp.processEvents()
    driver = TabListDriver(tab_list)
    evidence = evidence_dir(tmp_path)
    driver.screenshot(evidence / "overflow-00-start.png")
    assert not driver.tab_display(names[-1]).fully_visible
    assert driver.observe(object_name="ScrollLeftButton").visible
    assert driver.observe(object_name="ScrollRightButton").visible

    for step in range(1, len(names) + 1):
        if not driver.observe(object_name="ScrollRightButton").enabled:
            break
        driver.click(object_name="ScrollRightButton")
        driver.screenshot(evidence / f"overflow-{step:02d}-scroll.png")
    display = driver.tab_display(names[-1])
    assert display.fully_visible
    assert display.close_center is not None
    driver.select_tab(names[-1])
    driver.screenshot(evidence / "overflow-90-selected.png")
    assert driver.tabs(names[-1]).current == names[-1]
    driver.click_at(*display.close_center)
    driver.screenshot(evidence / "overflow-91-closed.png")
    assert names[-1] not in driver.tabs(names[0]).captions
    assert driver.observe("▾").visible
