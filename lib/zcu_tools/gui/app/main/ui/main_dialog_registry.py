"""Named dialog registry for the measure-gui main window."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from qtpy.QtWidgets import QDialog, QWidget  # type: ignore[attr-defined]

from zcu_tools.gui.app.main.services.remote.dialogs import DialogName
from zcu_tools.gui.expected_error import FailedPreconditionError
from zcu_tools.gui.widgets import DialogRefStore, widget_to_png_bytes

if TYPE_CHECKING:
    from zcu_tools.gui.app.main.controller import Controller


_PERSISTENT_DIALOGS: frozenset[DialogName] = frozenset({DialogName.PREDICTOR})


def _visible_dialog_names(
    names: dict[DialogName, None], dialog_for: Callable[[DialogName], QDialog | None]
) -> list[DialogName]:
    visible: list[DialogName] = []
    for name in names:
        dialog = dialog_for(name)
        if dialog is None:
            continue
        try:
            if dialog.isVisible():
                visible.append(name)
        except RuntimeError:
            continue
    return visible


class MainDialogRegistry:
    """Owns the named non-modal dialogs exposed through ``MainWindow``."""

    def __init__(
        self, ctrl: Controller, parent: QWidget, *, dialog_refs: DialogRefStore
    ) -> None:
        self._ctrl = ctrl
        self._parent = parent
        self._dialog_refs = dialog_refs
        self._dialog_names: dict[DialogName, None] = {}

    def _dialog(self, name: DialogName) -> QDialog | None:
        return self._dialog_refs.get(name)

    def _forget(self, name: DialogName) -> None:
        self._dialog_refs.discard(name)
        self._dialog_names.pop(name, None)

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
                md_provider=self._ctrl.context_control.get_current_md,
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
        existing = self._dialog(name)
        if existing is not None:
            try:
                existing.raise_()
                existing.activateWindow()
                if not existing.isVisible():
                    existing.show()
                return
            except RuntimeError:
                self._forget(name)

        dialog = self._build_dialog(name)
        self._dialog_names[name] = None
        if name in _PERSISTENT_DIALOGS:
            self._dialog_refs.retain_named(
                name,
                dialog,
                on_released=lambda n=name: self._dialog_names.pop(n, None),
            )
            dialog.open()
        else:
            self._dialog_refs.open_named(
                name,
                dialog,
                on_released=lambda n=name: self._dialog_names.pop(n, None),
            )

    def close(self, name: DialogName) -> None:
        """Close or hide a named dialog if the registry currently owns it."""
        existing = self._dialog(name)
        if existing is None:
            return
        try:
            existing.reject()
        except RuntimeError:
            self._forget(name)

    def visible_names(self) -> list[DialogName]:
        """Return named dialogs that are currently visible on screen."""
        return _visible_dialog_names(self._dialog_names, self._dialog)

    def dialog(self, name: DialogName) -> QDialog | None:
        """Return the registered dialog object, visible or hidden."""
        return self._dialog(name)

    def register(self, name: DialogName, dialog: QDialog) -> None:
        """Register a dialog constructed outside the registry factory."""
        self._dialog_refs.discard(name)
        self._dialog_names.setdefault(name, None)
        self._dialog_refs.retain_named(
            name,
            dialog,
            on_released=lambda n=name: self._dialog_names.pop(n, None),
        )

    def take_screenshot(self, dialog_name: DialogName) -> bytes:
        """Grab a currently-open dialog and return raw PNG bytes."""
        dialog = self._dialog(dialog_name)
        if dialog is None or not dialog.isVisible():
            raise FailedPreconditionError(
                f"dialog {dialog_name.value!r} is not currently open"
            )
        return widget_to_png_bytes(dialog, subject=f"{dialog_name.value!r} dialog")


__all__ = ["MainDialogRegistry"]
