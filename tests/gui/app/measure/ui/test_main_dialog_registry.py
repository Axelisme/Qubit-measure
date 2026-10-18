"""Tests for the measure-gui named dialog registry."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast
from unittest.mock import MagicMock

import pytest
from qtpy.QtWidgets import QDialog, QLineEdit, QPushButton, QWidget
from zcu_tools.gui.app.measure.remote.dialogs import DialogName
from zcu_tools.gui.app.measure.ui.main_dialog_registry import MainDialogRegistry
from zcu_tools.gui.session.services.arb_waveform import ArbWaveformService
from zcu_tools.gui.session.state import SessionState
from zcu_tools.gui.session.types import SessionEnv
from zcu_tools.gui.widgets import DialogRefStore

from tests.gui._dialog_fakes import RecordingDialogPresenter


class _FactoryRegistry(MainDialogRegistry):
    def __init__(
        self, parent: QWidget, factory: Callable[[DialogName], QDialog]
    ) -> None:
        self.refs = DialogRefStore()
        super().__init__(
            MagicMock(),
            parent=parent,
            dialog_refs=self.refs,
            templates=lambda: {},
            dialog_presenter=RecordingDialogPresenter(),
            alias_presets=None,
        )
        self._factory = factory

    def _build_dialog(self, name: DialogName, *, writable: bool) -> QDialog:
        return self._factory(name)


class _Signal:
    def __init__(self) -> None:
        self.callbacks: list[Callable[..., None]] = []

    def connect(self, callback: Callable[..., None]) -> None:
        self.callbacks.append(callback)

    def emit(self, *args: object) -> None:
        for callback in list(self.callbacks):
            callback(*args)


class _DestroyedDialog:
    def __init__(self) -> None:
        self.finished = _Signal()
        self.destroyed = _Signal()

    def raise_(self) -> None:
        raise RuntimeError("wrapped C/C++ object has been deleted")


class _PersistentDialog:
    def __init__(self) -> None:
        self.finished = _Signal()
        self.destroyed = _Signal()
        self.visible = False

    def open(self) -> None:
        self.visible = True

    def reject(self) -> None:
        self.visible = False

    def raise_(self) -> None:
        pass

    def activateWindow(self) -> None:
        pass

    def isVisible(self) -> bool:
        return self.visible

    def show(self) -> None:
        self.visible = True


class _TransientDialog(_PersistentDialog):
    def setAttribute(self, _attribute: object) -> None:
        pass

    def reject(self) -> None:
        self.visible = False
        self.finished.emit(0)


def test_arb_writability_is_applied_when_opened(qapp) -> None:
    parent = QWidget()
    ctrl = MagicMock()
    ctrl.arb_waveforms = ArbWaveformService(
        SessionState(SessionEnv(parameters=None, soc=None, soccfg=None))
    )
    registry = MainDialogRegistry(
        ctrl,
        parent=parent,
        dialog_refs=DialogRefStore(),
        templates=lambda: {},
        dialog_presenter=RecordingDialogPresenter(),
        alias_presets=None,
    )
    try:
        for writable in (False, True):
            registry.open(DialogName.ARB_WAVEFORM, writable=writable)
            dialog = registry.dialog(DialogName.ARB_WAVEFORM)
            assert dialog is not None
            key = dialog.findChild(QLineEdit)
            assert key is not None
            assert key.isReadOnly() is not writable
            save = next(
                b for b in dialog.findChildren(QPushButton) if b.text() == "Save"
            )
            assert save.isEnabled() is writable
            registry.close(DialogName.ARB_WAVEFORM)
            qapp.processEvents()
    finally:
        registry.close(DialogName.ARB_WAVEFORM)
        parent.deleteLater()
        qapp.processEvents()


def test_persistent_predictor_dialog_is_reused_when_hidden(qapp) -> None:
    parent = QWidget()
    created: list[_PersistentDialog] = []

    def build_dialog(name: DialogName) -> QDialog:
        assert name is DialogName.PREDICTOR
        dialog = _PersistentDialog()
        created.append(dialog)
        # tolerated(cast): Signal/visibility fake meets the Qt factory boundary.
        return cast(QDialog, dialog)

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
    first.destroyed.emit()


def test_nonpersistent_dialog_close_clears_registry(qapp) -> None:
    parent = QWidget()
    # tolerated(cast): Finished-signal fake drives close without Qt deletion.
    registry = _FactoryRegistry(parent, lambda _name: cast(QDialog, _TransientDialog()))

    registry.open(DialogName.SETUP)
    qapp.processEvents()
    assert registry.visible_names() == [DialogName.SETUP]

    registry.close(DialogName.SETUP)
    qapp.processEvents()

    assert registry.dialog(DialogName.SETUP) is None
    assert registry.visible_names() == []


def test_open_setup_again_raises_the_existing_dialog_until_it_closes(qapp) -> None:
    parent = QWidget()
    created: list[QDialog] = []

    def build_dialog(name: DialogName) -> QDialog:
        assert name is DialogName.SETUP
        # tolerated(cast): Fresh finished-signal fake exercises Setup reopen.
        dialog = cast(QDialog, _TransientDialog())
        created.append(dialog)
        return dialog

    registry = _FactoryRegistry(parent, build_dialog)

    registry.open(DialogName.SETUP)
    registry.open(DialogName.SETUP)
    qapp.processEvents()

    assert len(created) == 1
    assert registry.visible_names() == [DialogName.SETUP]

    registry.close(DialogName.SETUP)
    qapp.processEvents()
    registry.open(DialogName.SETUP)
    qapp.processEvents()

    assert len(created) == 2
    assert registry.dialog(DialogName.SETUP) is created[1]
    assert registry.visible_names() == [DialogName.SETUP]
    registry.close(DialogName.SETUP)
    qapp.processEvents()


def test_open_rebuilds_stale_destroyed_dialog(qapp) -> None:
    parent = QWidget()
    created: list[QDialog] = []

    def build_dialog(_name: DialogName) -> QDialog:
        # tolerated(cast): Visibility fake replaces the stale Qt registry entry.
        dialog = cast(QDialog, _TransientDialog())
        created.append(dialog)
        return dialog

    registry = _FactoryRegistry(parent, build_dialog)
    # tolerated(cast): Non-Qt fake deterministically raises the deleted-wrapper error.
    stale = cast(QDialog, _DestroyedDialog())
    registry.refs.retain_named(DialogName.SETUP, stale, on_released=lambda: None)

    registry.open(DialogName.SETUP)
    qapp.processEvents()

    assert registry.dialog(DialogName.SETUP) is created[0]
    assert created[0].isVisible() is True
    registry.close(DialogName.SETUP)
    qapp.processEvents()


def test_take_screenshot_requires_visible_dialog(qapp) -> None:
    parent = QWidget()
    registry = _FactoryRegistry(parent, lambda _name: QDialog(parent))

    with pytest.raises(RuntimeError, match="not currently open"):
        registry.take_screenshot(DialogName.SETUP)

    registry.open(DialogName.SETUP)
    qapp.processEvents()

    png = registry.take_screenshot(DialogName.SETUP)
    assert png.startswith(b"\x89PNG")
    registry.close(DialogName.SETUP)
    qapp.processEvents()
