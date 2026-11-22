"""Widget-lifetime safety tests for ComponentDriver observation walks."""

from collections.abc import Generator

import pytest
from PyQt6 import sip
from PyQt6.QtCore import QCoreApplication, QEvent, QRect
from PyQt6.QtWidgets import (
    QComboBox,
    QLabel,
    QMainWindow,
    QProxyStyle,
    QStyle,
    QStyleOptionComplex,
    QWidget,
)

from tests.gui import _user_input


@pytest.fixture
def surface(qapp) -> Generator[QMainWindow]:
    window = QMainWindow()
    window.resize(640, 480)
    window.show()
    qapp.processEvents()
    yield window
    window.close()
    window.deleteLater()
    qapp.processEvents()


class _DeferredDeleteStyle(QProxyStyle):
    """Deliver a sibling widget's queued deletion during a combo style query."""

    def __init__(self, target: QWidget) -> None:
        super().__init__()
        self._target = target

    def subControlRect(
        self,
        cc: QStyle.ComplexControl,
        opt: QStyleOptionComplex | None,
        sc: QStyle.SubControl,
        widget: QWidget | None = None,
    ) -> QRect:
        """Deliver deletion before returning the normal combo edit rectangle."""
        if sc == QStyle.SubControl.SC_ComboBoxEditField and not sip.isdeleted(
            self._target
        ):
            QCoreApplication.sendPostedEvents(self._target, QEvent.Type.DeferredDelete)
        return super().subControlRect(cc, opt, sc, widget)


def test_visible_snapshot_ignores_child_deleted_during_observation(surface) -> None:
    combo = QComboBox(surface)
    combo.setGeometry(10, 10, 180, 25)
    combo.addItem("Current choice")
    combo.show()
    retired = QLabel("Retired control", surface)
    retired.setGeometry(10, 40, 180, 25)
    retired.show()

    style = _DeferredDeleteStyle(retired)
    combo.setStyle(style)
    retired.deleteLater()
    try:
        snapshot = _user_input.ComponentDriver(surface).visible()
    finally:
        combo.setStyle(None)

    assert sip.isdeleted(retired)
    captions = {widget.text for widget in snapshot.widgets}
    assert "Current choice" in captions
    assert "Retired control" not in captions


def test_visible_snapshot_ignores_untracked_child_deleted_during_observation(
    surface,
) -> None:
    trigger = QComboBox(surface)
    trigger.setGeometry(10, 10, 180, 25)
    trigger.addItem("Trigger choice")
    trigger.show()
    retired_combo = QComboBox(surface)
    retired_combo.setGeometry(10, 40, 180, 25)
    retired_combo.setEditable(True)
    retired_combo.show()
    retired_edit = retired_combo.lineEdit()
    assert retired_edit is not None
    retired_edit.setText("Untracked retired text")
    # Wildcard disconnect drops PyQt's destroyed hook on this C++-owned child,
    # leaving a wrapper that stays not-deleted in SIP after C++ frees it.
    retired_edit.disconnect()

    style = _DeferredDeleteStyle(retired_combo)
    trigger.setStyle(style)
    retired_combo.deleteLater()
    try:
        snapshot = _user_input.ComponentDriver(surface).visible()
        assert sip.isdeleted(retired_combo)
        assert not sip.isdeleted(retired_edit)
    finally:
        trigger.setStyle(None)
        if not sip.isdeleted(retired_edit):
            sip.setdeleted(retired_edit)

    captions = {widget.text for widget in snapshot.widgets}
    assert "Trigger choice" in captions
    assert "Untracked retired text" not in captions
