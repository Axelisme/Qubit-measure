"""Tests for the measure-gui named dialog registry."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast
from unittest.mock import MagicMock

import pytest
from qtpy.QtWidgets import QDialog, QWidget
from zcu_tools.gui.app.main.services.remote.dialogs import DialogName
from zcu_tools.gui.app.main.ui.main_dialog_registry import MainDialogRegistry


class _FactoryRegistry(MainDialogRegistry):
    def __init__(
        self, parent: QWidget, factory: Callable[[DialogName], QDialog]
    ) -> None:
        super().__init__(MagicMock(), parent=parent)
        self._factory = factory

    def _build_dialog(self, name: DialogName) -> QDialog:
        return self._factory(name)


class _DestroyedDialog:
    def raise_(self) -> None:
        raise RuntimeError("wrapped C/C++ object has been deleted")


def test_persistent_predictor_dialog_is_reused_when_hidden(qapp) -> None:
    parent = QWidget()
    created: list[QDialog] = []

    def build_dialog(name: DialogName) -> QDialog:
        assert name is DialogName.PREDICTOR
        dialog = QDialog(parent)
        created.append(dialog)
        return dialog

    registry = _FactoryRegistry(parent, build_dialog)

    registry.open(DialogName.PREDICTOR)
    qapp.processEvents()
    first = created[0]
    assert first.isVisible() is True

    registry.close(DialogName.PREDICTOR)
    qapp.processEvents()
    assert first.isVisible() is False
    assert registry.dialog(DialogName.PREDICTOR) is first
    assert registry.visible_names() == []

    registry.open(DialogName.PREDICTOR)
    qapp.processEvents()

    assert created == [first]
    assert first.isVisible() is True
    assert registry.visible_names() == [DialogName.PREDICTOR]


def test_nonpersistent_dialog_close_clears_registry(qapp) -> None:
    parent = QWidget()
    registry = _FactoryRegistry(parent, lambda _name: QDialog(parent))

    registry.open(DialogName.SETUP)
    qapp.processEvents()
    assert registry.visible_names() == [DialogName.SETUP]

    registry.close(DialogName.SETUP)
    qapp.processEvents()

    assert registry.dialog(DialogName.SETUP) is None
    assert registry.visible_names() == []


def test_register_dialog_tracks_visible_only_and_cleans_on_finish(qapp) -> None:
    parent = QWidget()
    registry = MainDialogRegistry(MagicMock(), parent=parent)
    visible = QDialog(parent)
    hidden = QDialog(parent)

    registry.register(DialogName.STARTUP, visible)
    registry.register(DialogName.SETUP, hidden)
    visible.open()
    qapp.processEvents()

    assert registry.visible_names() == [DialogName.STARTUP]

    visible.reject()
    qapp.processEvents()

    assert registry.dialog(DialogName.STARTUP) is None
    assert registry.dialog(DialogName.SETUP) is hidden
    assert registry.visible_names() == []


def test_open_rebuilds_stale_destroyed_dialog(qapp) -> None:
    parent = QWidget()
    created: list[QDialog] = []

    def build_dialog(_name: DialogName) -> QDialog:
        dialog = QDialog(parent)
        created.append(dialog)
        return dialog

    registry = _FactoryRegistry(parent, build_dialog)
    registry._dialogs[DialogName.SETUP] = cast(QDialog, _DestroyedDialog())

    registry.open(DialogName.SETUP)
    qapp.processEvents()

    assert registry.dialog(DialogName.SETUP) is created[0]
    assert created[0].isVisible() is True


def test_take_screenshot_requires_visible_dialog(qapp) -> None:
    parent = QWidget()
    registry = _FactoryRegistry(parent, lambda _name: QDialog(parent))

    with pytest.raises(RuntimeError, match="not currently open"):
        registry.take_screenshot(DialogName.SETUP)

    registry.open(DialogName.SETUP)
    qapp.processEvents()

    png = registry.take_screenshot(DialogName.SETUP)
    assert png.startswith(b"\x89PNG")
