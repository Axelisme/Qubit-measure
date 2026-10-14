from __future__ import annotations

from collections.abc import Iterator

import pytest
from qtpy.QtCore import QCoreApplication, QEvent, QObject, Qt
from qtpy.QtWidgets import QApplication, QLabel, QMessageBox, QTextEdit, QWidget
from zcu_tools.gui.widgets.selectable_text import (
    install_selectable_message_boxes,
    make_label_selectable,
)

SELECTABLE = (
    Qt.TextInteractionFlag.TextSelectableByMouse
    | Qt.TextInteractionFlag.TextSelectableByKeyboard
)


def _is_selectable(flags: Qt.TextInteractionFlag) -> bool:
    return (flags & SELECTABLE) == SELECTABLE


def _label_with_text(parent: QWidget, text: str) -> QLabel:
    matches = [label for label in parent.findChildren(QLabel) if label.text() == text]
    assert len(matches) == 1
    return matches[0]


@pytest.fixture
def selector(qapp: QApplication) -> Iterator[QObject]:
    installed = install_selectable_message_boxes(qapp)
    yield installed
    installed.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


@pytest.fixture
def shown_widgets() -> Iterator[list[QWidget]]:
    widgets: list[QWidget] = []
    yield widgets
    for widget in widgets:
        widget.close()
        widget.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def test_shown_message_box_text_is_selectable(
    selector: QObject, shown_widgets: list[QWidget]
) -> None:
    del selector
    box = QMessageBox()
    box.setText("Main message")
    box.setInformativeText("Informative message")
    box.setDetailedText("Detailed message")
    shown_widgets.append(box)

    box.show()

    assert _is_selectable(_label_with_text(box, "Main message").textInteractionFlags())
    assert _is_selectable(
        _label_with_text(box, "Informative message").textInteractionFlags()
    )
    details = [
        edit
        for edit in box.findChildren(QTextEdit)
        if edit.toPlainText() == "Detailed message"
    ]
    assert len(details) == 1
    assert _is_selectable(details[0].textInteractionFlags())


def test_labels_outside_message_boxes_are_unchanged(
    selector: QObject, shown_widgets: list[QWidget]
) -> None:
    del selector
    label = QLabel("Plain label")
    shown_widgets.append(label)
    before = label.textInteractionFlags()

    label.show()

    assert label.textInteractionFlags() == before
    assert not _is_selectable(label.textInteractionFlags())


def test_repeated_install_returns_the_installed_selector(
    qapp: QApplication, selector: QObject
) -> None:
    assert install_selectable_message_boxes(qapp) is selector


def test_made_selectable_label_accepts_mouse_selection(qapp: QApplication) -> None:
    del qapp
    label = QLabel("192.168.0.10")

    make_label_selectable(label)

    flags = label.textInteractionFlags()
    assert flags & Qt.TextInteractionFlag.TextSelectableByMouse


def test_made_selectable_label_keeps_existing_link_access(qapp: QApplication) -> None:
    del qapp
    label = QLabel('<a href="https://example.com">link</a>')
    label.setTextInteractionFlags(Qt.TextInteractionFlag.LinksAccessibleByMouse)

    make_label_selectable(label)

    flags = label.textInteractionFlags()
    assert flags & Qt.TextInteractionFlag.TextSelectableByMouse
    assert flags & Qt.TextInteractionFlag.LinksAccessibleByMouse
