"""Tests for the main-window toolbar coordinator."""

from __future__ import annotations

from zcu_tools.gui.app.main.ui.main_window_toolbar import (
    AdapterMenuItem,
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
