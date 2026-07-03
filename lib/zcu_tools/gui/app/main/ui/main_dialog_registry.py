"""Named dialog registry for the measure-gui main window."""

from __future__ import annotations

from typing import TYPE_CHECKING

from qtpy.QtCore import QBuffer, QIODevice, Qt  # type: ignore[attr-defined]
from qtpy.QtWidgets import QDialog, QWidget  # type: ignore[attr-defined]

from zcu_tools.gui.app.main.services.remote.dialogs import DialogName

if TYPE_CHECKING:
    from zcu_tools.gui.app.main.controller import Controller


_PERSISTENT_DIALOGS: frozenset[DialogName] = frozenset({DialogName.PREDICTOR})


def _visible_dialog_names(dialogs: dict[DialogName, QDialog]) -> list[DialogName]:
    visible: list[DialogName] = []
    for name, dialog in dialogs.items():
        try:
            if dialog.isVisible():
                visible.append(name)
        except RuntimeError:
            continue
    return visible


class MainDialogRegistry:
    """Owns the named non-modal dialogs exposed through ``MainWindow``."""

    def __init__(self, ctrl: Controller, parent: QWidget) -> None:
        self._ctrl = ctrl
        self._parent = parent
        self._dialogs: dict[DialogName, QDialog] = {}

    def _build_dialog(self, name: DialogName) -> QDialog:
        """Construct a fresh QDialog for ``name``.

        Per-name imports stay lazy to avoid loading unused dialog modules during
        startup and lightweight tests.
        """
        if name is DialogName.SETUP:
            from zcu_tools.gui.session.ui.setup_dialog import SetupDialog

            return SetupDialog(self._ctrl.setup_control, parent=self._parent)
        if name is DialogName.DEVICE:
            from zcu_tools.gui.session.ui.device_dialog import DeviceDialog

            return DeviceDialog(
                self._ctrl.device_control,
                md_provider=self._ctrl.context_control.get_current_md,
                parent=self._parent,
            )
        if name is DialogName.PREDICTOR:
            from zcu_tools.gui.session.ui.predictor_dialog import PredictorDialog

            return PredictorDialog(
                self._ctrl.predictor_control,
                parent=self._parent,
                device=self._ctrl.device_control,
                persistent_on_close=True,
            )
        if name is DialogName.INSPECT:
            from .inspect_dialog import InspectDialog

            return InspectDialog(
                self._ctrl, bus=self._ctrl.get_bus(), parent=self._parent
            )
        if name is DialogName.ARB_WAVEFORM:
            from .arb_waveform_dialog import ArbWaveformDialog

            return ArbWaveformDialog(self._ctrl, parent=self._parent)
        if name is DialogName.STARTUP:
            # STARTUP dialogs need startup_mode=True and are usually opened by
            # application bootstrap, but the registry factory still supports a
            # fresh instance after the bootstrap dialog has closed.
            from zcu_tools.gui.session.ui.setup_dialog import SetupDialog

            return SetupDialog(
                self._ctrl.setup_control, parent=self._parent, startup_mode=True
            )
        raise ValueError(f"Unknown DialogName: {name!r}")  # pragma: no cover

    def open(self, name: DialogName) -> None:
        """Open a named dialog non-modally, or raise an existing instance."""
        existing = self._dialogs.get(name)
        if existing is not None:
            try:
                existing.raise_()
                existing.activateWindow()
                if not existing.isVisible():
                    existing.show()
                return
            except RuntimeError:
                self._dialogs.pop(name, None)

        dialog = self._build_dialog(name)
        if name not in _PERSISTENT_DIALOGS:
            dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self._dialogs[name] = dialog
        if name not in _PERSISTENT_DIALOGS:
            dialog.finished.connect(lambda _status, n=name: self._dialogs.pop(n, None))
        dialog.open()

    def close(self, name: DialogName) -> None:
        """Close or hide a named dialog if the registry currently owns it."""
        existing = self._dialogs.get(name)
        if existing is None:
            return
        try:
            existing.reject()
        except RuntimeError:
            self._dialogs.pop(name, None)

    def visible_names(self) -> list[DialogName]:
        """Return named dialogs that are currently visible on screen."""
        return _visible_dialog_names(self._dialogs)

    def dialog(self, name: DialogName) -> QDialog | None:
        """Return the registered dialog object, visible or hidden."""
        return self._dialogs.get(name)

    def register(self, name: DialogName, dialog: QDialog) -> None:
        """Register a dialog constructed outside the registry factory."""
        self._dialogs[name] = dialog
        dialog.finished.connect(lambda _status, n=name: self._dialogs.pop(n, None))

    def take_screenshot(self, dialog_name: DialogName) -> bytes:
        """Grab a currently-open dialog and return raw PNG bytes."""
        dialog = self._dialogs.get(dialog_name)
        if dialog is None or not dialog.isVisible():
            raise RuntimeError(f"dialog {dialog_name.value!r} is not currently open")
        pixmap = dialog.grab()
        buffer = QBuffer()
        buffer.open(QIODevice.OpenModeFlag.WriteOnly)
        if not pixmap.save(buffer, "PNG"):
            raise RuntimeError(
                f"Qt failed to encode {dialog_name.value!r} dialog as PNG"
            )
        return bytes(buffer.data().data())  # type: ignore[arg-type]


__all__ = ["MainDialogRegistry"]
