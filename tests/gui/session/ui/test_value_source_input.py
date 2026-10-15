from __future__ import annotations

from collections.abc import Sequence

from qtpy.QtCore import QEvent, Qt
from qtpy.QtGui import QKeyEvent
from qtpy.QtWidgets import QApplication, QLineEdit
from zcu_tools.gui.session.ui.value_source_input import ValueSourceInputController


class _Host:
    def __init__(self) -> None:
        self.resolved_keys: list[str] = []

    def list_value_source_keys(self) -> Sequence[str]:
        return (
            "context.chip_name",
            "device.flux.name",
            "device.flux.value",
            "device.readout.value",
            "predictor.flux_half",
        )

    def resolve_value_source(self, key: str) -> object:
        self.resolved_keys.append(key)
        if key == "device.flux.value":
            return 0.5
        if key == "device.flux.name":
            return "fake_flux"
        raise RuntimeError(f"unknown key {key}")

    def format_resolved_value(self, value: object) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        return str(value)


def test_tab_accepts_first_completion(qapp) -> None:
    edit = QLineEdit("@{dev")
    edit.setCursorPosition(len(edit.text()))
    host = _Host()
    ValueSourceInputController(edit, host)
    event = QKeyEvent(
        QEvent.Type.KeyPress,
        Qt.Key.Key_Tab,
        Qt.KeyboardModifier.NoModifier,
    )

    QApplication.sendEvent(edit, event)

    assert edit.text() == "@{device."
    assert edit.cursorPosition() == len("@{device.")
    assert host.resolved_keys == []
    assert event.isAccepted()


def test_tab_accepts_leaf_completion_without_drill_down(qapp) -> None:
    edit = QLineEdit("@{device.flux.v")
    edit.setCursorPosition(len(edit.text()))
    host = _Host()
    ValueSourceInputController(edit, host)
    event = QKeyEvent(
        QEvent.Type.KeyPress,
        Qt.Key.Key_Tab,
        Qt.KeyboardModifier.NoModifier,
    )

    QApplication.sendEvent(edit, event)

    assert edit.text() == "@{device.flux.value}"
    assert edit.cursorPosition() == len("@{device.flux.value}")
    assert host.resolved_keys == []
    assert event.isAccepted()


def test_space_after_complete_token_resolves_and_replaces_text(qapp) -> None:
    edit = QLineEdit("prefix @{device.flux.value}")
    edit.setCursorPosition(len(edit.text()))
    host = _Host()
    controller = ValueSourceInputController(edit, host)
    resolved: list[object] = []
    controller.resolved.connect(resolved.append)

    edit.insert(" ")

    assert edit.text() == "prefix 0.5"
    assert edit.cursorPosition() == len("prefix 0.5")
    assert host.resolved_keys == ["device.flux.value"]
    assert resolved == [0.5]


def test_failed_resolve_keeps_text_and_sets_tooltip(qapp) -> None:
    edit = QLineEdit("@{missing.key}")
    edit.setCursorPosition(len(edit.text()))
    host = _Host()
    controller = ValueSourceInputController(edit, host)
    failures: list[str] = []
    controller.resolve_failed.connect(failures.append)

    edit.insert(" ")

    assert edit.text() == "@{missing.key} "
    assert failures == ["unknown key missing.key"]
    assert edit.toolTip() == "unknown key missing.key"
