from __future__ import annotations

from collections.abc import Callable
from typing import cast

import pytest
from qtpy.QtWidgets import QDialog  # type: ignore[attr-defined]
from zcu_tools.gui.widgets.dialog_lifecycle import DialogRefStore


class _Signal:
    def __init__(self) -> None:
        self._callbacks: list[Callable[..., None]] = []

    def connect(self, callback: Callable[..., None]) -> None:
        self._callbacks.append(callback)

    def emit(self, *args: object) -> None:
        for callback in list(self._callbacks):
            callback(*args)


class _Dialog:
    def __init__(self) -> None:
        self.finished = _Signal()
        self.destroyed = _Signal()
        self.attributes: list[object] = []
        self.open_count = 0

    def setAttribute(self, attribute: object) -> None:
        self.attributes.append(attribute)

    def open(self) -> None:
        self.open_count += 1


def test_dialog_ref_store_cleans_up_on_destroyed_only() -> None:
    store = DialogRefStore()
    dialog = _Dialog()

    store.open_named("setup", cast(QDialog, dialog))

    assert store.get("setup") is dialog
    assert len(store) == 1
    assert dialog.open_count == 1

    dialog.destroyed.emit(object())

    assert store.get("setup") is None
    assert len(store) == 0


def test_dialog_ref_store_duplicate_key_fast_fails() -> None:
    store = DialogRefStore()
    first = _Dialog()
    second = _Dialog()

    store.open_named("setup", cast(QDialog, first))
    with pytest.raises(RuntimeError, match="dialog key is already retained"):
        store.open_named("setup", cast(QDialog, second))

    assert second.open_count == 0


def test_dialog_ref_store_retains_without_opening_and_reports_release() -> None:
    store = DialogRefStore()
    dialog = _Dialog()
    releases = 0

    def on_released() -> None:
        nonlocal releases
        releases += 1

    store.retain_named("startup", cast(QDialog, dialog), on_released=on_released)
    dialog.destroyed.emit(object())

    assert dialog.open_count == 0
    assert store.get("startup") is None
    assert releases == 1
