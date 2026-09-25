"""Writeback presentation follows the shipped Apply-above ledger layout."""

from unittest.mock import MagicMock

import pytest
from qtpy.QtCore import QPoint
from qtpy.QtWidgets import (
    QApplication,
    QCheckBox,
    QFrame,
    QLabel,
    QPushButton,
    QScrollArea,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from zcu_tools.gui.app.main.adapter import (
    AdapterCapabilities,
    MetaDictWriteback,
    WritebackItem,
)
from zcu_tools.gui.app.main.services import PersistedStartup
from zcu_tools.gui.app.main.ui.exp_tab_widget import ExpTabWidget
from zcu_tools.gui.app.main.ui.writeback_widget import WritebackWidget


def items(count: int) -> list[WritebackItem]:
    result: list[WritebackItem] = []
    for index in range(count):
        item = MetaDictWriteback(
            target_name=f"p{index}",
            description="parameter",
            proposed_value=float(index),
        )
        item.session_id = f"md-{index}"
        result.append(item)
    return result


def settle(qapp: QApplication) -> None:
    for _ in range(3):
        qapp.processEvents()


def ledger(widget: WritebackWidget) -> tuple[QFrame, QPushButton, list[QWidget]]:
    panel = widget.findChild(QFrame, "writebackPanel")
    apply = widget.findChild(QPushButton, "writebackApply")
    legend = widget.findChild(QLabel, "writebackAppliedLegend")
    assert panel is not None and apply is not None and legend is not None
    layout = panel.layout()
    assert layout is not None
    rows: list[QWidget] = []
    for index in range(layout.count()):
        entry = layout.itemAt(index)
        assert entry is not None
        row = entry.widget()
        assert row is not None
        rows.append(row)
    assert rows
    assert widget.findChildren(QScrollArea) == []
    assert panel.isVisible() and apply.isVisible()
    assert panel.width() <= widget.width() + 2
    assert abs(panel.height() - (rows[-1].geometry().bottom() + 1)) <= 6
    root = widget.layout()
    assert root is not None
    assert abs(legend.y() - (apply.y() + apply.height() + root.spacing())) <= 6
    assert abs(panel.y() - (legend.y() + legend.height() + root.spacing())) <= 6
    return panel, apply, rows


def test_resize_reflows_rows_without_moving_apply_below_the_ledger(
    qapp: QApplication,
) -> None:
    widget = WritebackWidget(MagicMock(), tab_id="t", pane="analysis")
    widget.populate(items(3))
    widget.show()
    try:
        for width in [500, 400, 500]:
            widget.resize(width, 400)
            settle(qapp)
            _, _, rows = ledger(widget)
            check = rows[0].findChild(QCheckBox)
            current = rows[0].findChild(QLabel, "writebackCurrent")
            assert check is not None and current is not None
            if width == 400:
                assert check.y() < current.y() - 5
            else:
                assert abs(check.y() - current.y()) < 30
    finally:
        widget.close()
        widget.deleteLater()
        settle(qapp)


def assert_scroll_reaches_both_ends(
    outer: QScrollArea, widget: WritebackWidget, qapp: QApplication
) -> None:
    _, apply, rows = ledger(widget)
    viewport = outer.viewport()
    bar = outer.verticalScrollBar()
    inner = outer.widget()
    assert viewport is not None and bar is not None and inner is not None
    assert inner.height() > viewport.height()
    assert bar.maximum() > 0
    bar.setValue(bar.maximum())
    settle(qapp)
    bottom = rows[-1].mapTo(viewport, QPoint(0, rows[-1].height() - 1)).y()
    assert 0 <= bottom < viewport.height()
    bar.setValue(0)
    settle(qapp)
    top = apply.mapTo(viewport, QPoint(0, 0)).y()
    assert top >= 0 and top + apply.height() <= viewport.height()


@pytest.mark.parametrize("width", [500, 400])
def test_replacing_short_and_long_ledgers_grows_only_the_outer_scroll(
    qapp: QApplication, width: int
) -> None:
    outer = QScrollArea()
    outer.setWidgetResizable(True)
    outer.resize(width + 20, 300)
    inner = QWidget()
    layout = QVBoxLayout(inner)
    widget = WritebackWidget(MagicMock(), tab_id="t", pane="analysis")
    layout.addWidget(widget)
    layout.addStretch()
    outer.setWidget(inner)
    outer.show()
    try:
        heights: list[int] = []
        for count in [4, 30, 4]:
            widget.populate(items(count))
            settle(qapp)
            panel, _, _ = ledger(widget)
            heights.append(panel.height())
            if count == 30:
                assert_scroll_reaches_both_ends(outer, widget, qapp)
            else:
                assert widget.sizeHint().height() < 600
        assert heights[1] > heights[0] + 200
        assert heights[2] == heights[0]
    finally:
        outer.close()
        outer.deleteLater()
        settle(qapp)


def test_shipped_analysis_scroll_reaches_apply_and_the_last_ledger_row(
    qapp: QApplication,
) -> None:
    ctrl = MagicMock()
    ctrl.get_persisted_startup.return_value = PersistedStartup(left_panel_width=500)
    ctrl.progress_control.attach_progress.return_value = lambda: None
    ctrl.progress_control.progress_bars.return_value = []
    ctrl.get_tab_adapter_name.return_value = "fake"
    ctrl.get_adapter_guide.return_value = {}
    tab = ExpTabWidget("t", ctrl, AdapterCapabilities())
    tab.resize(600, 500)
    tab.show()
    try:
        tabs = next(
            tabs
            for tabs in tab.findChildren(QTabWidget)
            if "Analysis" in [tabs.tabText(i) for i in range(tabs.count())]
        )
        index = next(i for i in range(tabs.count()) if tabs.tabText(i) == "Analysis")
        tabs.setCurrentIndex(index)
        tab.update_writeback_items(items(30))
        settle(qapp)
        widget = tab.writeback_widget
        outer = next(
            scroll
            for scroll in tab.findChildren(QScrollArea)
            if (inner := scroll.widget()) is not None and inner.isAncestorOf(widget)
        )
        for width in [600, 500]:
            tab.resize(width, 500)
            settle(qapp)
            assert_scroll_reaches_both_ends(outer, widget, qapp)
        long_height = ledger(widget)[0].height()
        tab.update_writeback_items(items(2))
        settle(qapp)
        assert ledger(widget)[0].height() < long_height
    finally:
        tab.close()
        tab.deleteLater()
        settle(qapp)
