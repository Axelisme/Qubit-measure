from __future__ import annotations

from collections.abc import Sequence

from qtpy.QtCore import QEvent, Qt  # type: ignore[attr-defined]
from qtpy.QtGui import QKeyEvent  # type: ignore[attr-defined]
from qtpy.QtWidgets import QLineEdit  # type: ignore[attr-defined]
from zcu_tools.gui.session.ui.value_source_input import (
    ValueSourceInputController,
    _active_token,
    _committed_token,
    _completion_candidates,
)


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


def test_active_token_detects_current_prefix() -> None:
    token = _active_token("cos(@{device.active", len("cos(@{device.active"))
    assert token is not None
    assert token.start == 4
    assert token.end == len("cos(@{device.active")
    assert token.key == "device.active"


def test_active_token_ignores_closed_or_spaced_tokens() -> None:
    assert _active_token("@{device.flux.value}", len("@{device.flux.value}")) is None
    assert _active_token("@{device active", len("@{device active")) is None


def test_committed_token_requires_space_after_closing_brace() -> None:
    text = "prefix @{device.flux.value} "
    token = _committed_token(text, len(text))
    assert token is not None
    assert token.key == "device.flux.value"
    assert token.start == len("prefix ")
    assert token.end == len(text)


def test_completion_candidates_are_segment_scoped() -> None:
    keys = (
        "context.chip_name",
        "device.flux.name",
        "device.flux.value",
        "device.readout.value",
        "predictor.flux_half",
    )

    assert _completion_candidates(keys, "") == ["context", "device", "predictor"]
    assert _completion_candidates(keys, "dev") == ["device"]
    assert _completion_candidates(keys, "device") == []
    assert _completion_candidates(keys, "device.") == [
        "device.flux",
        "device.readout",
    ]
    assert _completion_candidates(keys, "device.fl") == ["device.flux"]
    assert _completion_candidates(keys, "device.flux.") == [
        "device.flux.name",
        "device.flux.value",
    ]


def test_completion_drills_into_namespace_without_resolving(qapp) -> None:
    edit = QLineEdit("@{device.fl")
    edit.setCursorPosition(len(edit.text()))
    host = _Host()
    controller = ValueSourceInputController(edit, host)

    controller._on_completion_activated("device.flux")

    assert edit.text() == "@{device.flux."
    assert controller._model.stringList() == [
        "device.flux.name",
        "device.flux.value",
    ]
    assert host.resolved_keys == []


def test_completion_activation_drills_into_top_level_namespace(qapp) -> None:
    edit = QLineEdit("@{dev")
    edit.setCursorPosition(len(edit.text()))
    host = _Host()
    controller = ValueSourceInputController(edit, host)

    controller._on_completion_activated("device")

    assert edit.text() == "@{device."
    assert controller._model.stringList() == ["device.flux", "device.readout"]
    assert host.resolved_keys == []


def test_completion_inserts_full_key_and_closes_token(qapp) -> None:
    edit = QLineEdit("@{device.flux.v")
    edit.setCursorPosition(len(edit.text()))
    host = _Host()
    controller = ValueSourceInputController(edit, host)

    controller._on_completion_activated("device.flux.value")

    assert edit.text() == "@{device.flux.value}"
    assert host.resolved_keys == []


def test_completion_popup_width_is_readable(qapp) -> None:
    edit = QLineEdit("@{device.flux.")
    edit.resize(80, edit.sizeHint().height())
    edit.setCursorPosition(len(edit.text()))
    controller = ValueSourceInputController(edit, _Host())

    controller._refresh_completion_popup()

    popup = controller._completer.popup()
    assert popup is not None
    longest = "device.flux.value"
    expected_text_width = edit.fontMetrics().horizontalAdvance(longest) + 36
    assert popup.minimumWidth() >= expected_text_width
    assert popup.minimumWidth() > edit.cursorRect().width()


def test_tab_accepts_first_completion(qapp) -> None:
    edit = QLineEdit("@{dev")
    edit.setCursorPosition(len(edit.text()))
    host = _Host()
    controller = ValueSourceInputController(edit, host)
    event = QKeyEvent(
        QEvent.Type.KeyPress,
        Qt.Key.Key_Tab,  # type: ignore[attr-defined]
        Qt.KeyboardModifier.NoModifier,  # type: ignore[attr-defined]
    )

    assert controller.eventFilter(edit, event)

    assert edit.text() == "@{device."
    assert edit.cursorPosition() == len("@{device.")
    assert controller._model.stringList() == ["device.flux", "device.readout"]
    assert host.resolved_keys == []
    assert event.isAccepted()


def test_tab_accepts_first_completion_from_popup(qapp) -> None:
    edit = QLineEdit("@{dev")
    edit.setCursorPosition(len(edit.text()))
    host = _Host()
    controller = ValueSourceInputController(edit, host)

    controller._refresh_completion_popup()
    popup = controller._completer.popup()
    assert popup is not None
    event = QKeyEvent(
        QEvent.Type.KeyPress,
        Qt.Key.Key_Tab,  # type: ignore[attr-defined]
        Qt.KeyboardModifier.NoModifier,  # type: ignore[attr-defined]
    )

    assert controller.eventFilter(popup, event)

    assert edit.text() == "@{device."
    assert edit.cursorPosition() == len("@{device.")
    assert controller._model.stringList() == ["device.flux", "device.readout"]
    assert host.resolved_keys == []
    assert event.isAccepted()


def test_tab_accepts_leaf_completion_without_drill_down(qapp) -> None:
    edit = QLineEdit("@{device.flux.v")
    edit.setCursorPosition(len(edit.text()))
    host = _Host()
    controller = ValueSourceInputController(edit, host)
    event = QKeyEvent(
        QEvent.Type.KeyPress,
        Qt.Key.Key_Tab,  # type: ignore[attr-defined]
        Qt.KeyboardModifier.NoModifier,  # type: ignore[attr-defined]
    )

    assert controller.eventFilter(edit, event)

    assert edit.text() == "@{device.flux.value}"
    assert edit.cursorPosition() == len("@{device.flux.value}")
    assert host.resolved_keys == []
    assert event.isAccepted()


def test_leaf_completion_hides_popup(qapp) -> None:
    edit = QLineEdit("@{device.flux.v")
    edit.setCursorPosition(len(edit.text()))
    controller = ValueSourceInputController(edit, _Host())
    popup = controller._completer.popup()
    assert popup is not None
    popup.show()
    qapp.processEvents()
    assert not popup.isHidden()

    controller._on_completion_activated("device.flux.value")

    assert edit.text() == "@{device.flux.value}"
    assert popup.isHidden()


def test_space_after_complete_token_resolves_and_replaces_text(qapp) -> None:
    edit = QLineEdit("prefix @{device.flux.value} ")
    edit.setCursorPosition(len(edit.text()))
    host = _Host()
    controller = ValueSourceInputController(edit, host)
    resolved: list[object] = []
    controller.resolved.connect(resolved.append)  # type: ignore[attr-defined]

    controller._on_text_edited(edit.text())

    assert edit.text() == "prefix 0.5"
    assert edit.cursorPosition() == len("prefix 0.5")
    assert host.resolved_keys == ["device.flux.value"]
    assert resolved == [0.5]


def test_space_after_complete_token_hides_popup(qapp) -> None:
    edit = QLineEdit("prefix @{device.flux.value} ")
    edit.setCursorPosition(len(edit.text()))
    controller = ValueSourceInputController(edit, _Host())
    popup = controller._completer.popup()
    assert popup is not None
    popup.show()
    qapp.processEvents()
    assert not popup.isHidden()

    controller._on_text_edited(edit.text())

    assert edit.text() == "prefix 0.5"
    assert popup.isHidden()


def test_failed_resolve_keeps_text_and_sets_tooltip(qapp) -> None:
    edit = QLineEdit("@{missing.key} ")
    edit.setCursorPosition(len(edit.text()))
    host = _Host()
    controller = ValueSourceInputController(edit, host)
    failures: list[str] = []
    controller.resolve_failed.connect(failures.append)  # type: ignore[attr-defined]
    popup = controller._completer.popup()
    assert popup is not None
    popup.show()
    qapp.processEvents()
    assert not popup.isHidden()

    controller._on_text_edited(edit.text())

    assert edit.text() == "@{missing.key} "
    assert failures == ["unknown key missing.key"]
    assert edit.toolTip() == "unknown key missing.key"
    assert popup.isHidden()
