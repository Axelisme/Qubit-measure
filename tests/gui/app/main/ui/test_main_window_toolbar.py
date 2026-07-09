"""Tests for the main-window toolbar coordinator."""

from __future__ import annotations

from qtpy.QtWidgets import QPushButton, QWidget  # type: ignore[attr-defined]
from zcu_tools.gui.app.main.services.remote.dialogs import DialogName
from zcu_tools.gui.app.main.ui.main_window_toolbar import (
    AdapterMenuItem,
    MainWindowToolbar,
    adapter_menu_items,
)


def test_adapter_menu_items_preserve_flat_adapter_names() -> None:
    assert adapter_menu_items(["lookback", "onetone"]) == (
        AdapterMenuItem("lookback", "lookback", ()),
        AdapterMenuItem("onetone", "onetone", ()),
    )


def test_adapter_menu_items_split_nested_adapter_names() -> None:
    assert adapter_menu_items(
        [
            "twotone/rabi/amp_rabi",
            "twotone/rabi/len_rabi",
            "time/t1",
        ]
    ) == (
        AdapterMenuItem("twotone/rabi/amp_rabi", "amp_rabi", ("twotone", "rabi")),
        AdapterMenuItem("twotone/rabi/len_rabi", "len_rabi", ("twotone", "rabi")),
        AdapterMenuItem("time/t1", "t1", ("time",)),
    )


def test_adapter_menu_items_preserve_input_order() -> None:
    items = adapter_menu_items(["b/second", "a/first", "root"])

    assert [item.adapter_name for item in items] == ["b/second", "a/first", "root"]


class _RecordingToolbarHost:
    def __init__(self) -> None:
        self.opened: list[DialogName] = []

    def list_adapter_names(self) -> list[str]:
        return []

    def create_tab(self, adapter_name: str) -> None:
        raise AssertionError(f"unexpected tab creation: {adapter_name!r}")

    def open_dialog(self, name: DialogName) -> None:
        self.opened.append(name)


def test_toolbar_dialog_buttons_open_expected_dialogs(qapp) -> None:  # noqa: ANN001
    parent = QWidget()
    host = _RecordingToolbarHost()
    toolbar = MainWindowToolbar(host, parent=parent)

    buttons: dict[str, QPushButton] = {}
    for index in range(toolbar.layout.count()):
        item = toolbar.layout.itemAt(index)
        if item is None:
            continue
        widget = item.widget()
        if isinstance(widget, QPushButton):
            buttons[widget.text()] = widget

    buttons["Setup…"].click()
    buttons["Devices…"].click()
    buttons["Predictor…"].click()
    buttons["Inspect…"].click()

    assert host.opened == [
        DialogName.SETUP,
        DialogName.DEVICE,
        DialogName.PREDICTOR,
        DialogName.INSPECT,
    ]
