"""Named dialog registry for the measure-gui main window."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING

from qtpy.QtCore import Qt
from qtpy.QtWidgets import QDialog, QWidget

from zcu_tools.gui.app.measure.remote.dialogs import DialogName
from zcu_tools.gui.expected_error import FailedPreconditionError
from zcu_tools.gui.widgets import DialogRefStore, widget_to_png_bytes

if TYPE_CHECKING:
    from zcu_tools.cfg_model import TemplateDef
    from zcu_tools.gui.app.measure.controller import Controller
    from zcu_tools.gui.widgets import DialogPresenter


_PERSISTENT_DIALOGS: frozenset[DialogName] = frozenset({DialogName.PREDICTOR})
_RUN_LOCKED_DIALOGS: frozenset[DialogName] = frozenset(
    {DialogName.SETUP, DialogName.SOC}
)


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
    """Retain the named dialogs exposed through ``MainWindow``.

    Use on the Qt owner thread. Named windows are non-modal. The host pushes
    Local writability to new and retained Setup, SoC, Inspect, Device and ARB
    windows; dialogs never observe authority. Run separately locks Setup/SoC.
    """

    def __init__(
        self,
        ctrl: Controller,
        parent: QWidget,
        *,
        dialog_refs: DialogRefStore,
        templates: Callable[[], Mapping[str, TemplateDef]],
        dialog_presenter: DialogPresenter,
    ) -> None:
        """Hold the factories for every named dialog.

        templates returns the composition's current template catalog; it is
        read each time Inspect opens, so a reloaded catalog reaches the next
        Inspect. dialog_presenter shows Inspect's questions and failures and
        ARB's failure notifications.
        """
        self._ctrl = ctrl
        self._writable = True
        self._running = False
        self._parent = parent
        self._dialog_refs = dialog_refs
        self._templates = templates
        self._dialog_presenter = dialog_presenter
        self._dialog_names: dict[DialogName, None] = {}

    def _reclaim_for_stop(self) -> None:
        if self._ctrl.control_identity.status().holder != "gui":
            self._ctrl.control_identity.reclaim_local()

    def _dialog(self, name: DialogName) -> QDialog | None:
        return self._dialog_refs.get(name)

    def _forget(self, name: DialogName) -> None:
        self._dialog_refs.discard(name)
        self._dialog_names.pop(name, None)

    def _build_dialog(self, name: DialogName, *, writable: bool) -> QDialog:
        """Construct a fresh QDialog for ``name``.

        Per-name imports stay lazy to avoid loading unused dialog modules during
        startup and lightweight tests.
        """
        if name is DialogName.SETUP:
            from zcu_tools.gui.session.ui.setup_dialog import SetupDialog

            dialog = SetupDialog(
                self._ctrl.setup_control,
                parent=self._parent,
            )
            dialog.set_read_only(read_only=not writable)
            return dialog
        if name is DialogName.SOC:
            from zcu_tools.gui.session.ui.soc_dialog import SocDialog

            dialog = SocDialog(self._ctrl.soc_control, parent=self._parent)
            dialog.set_read_only(read_only=not writable)
            return dialog
        if name is DialogName.DEVICE:
            from zcu_tools.gui.session.ui.device_dialog import DeviceDialog

            dialog = DeviceDialog(
                self._ctrl.device_control,
                parameter_provider=self._ctrl.context_control.read_context,
                parent=self._parent,
                before_stop=self._reclaim_for_stop,
            )
            dialog.set_read_only(read_only=not writable)
            return dialog
        if name is DialogName.PREDICTOR:
            from zcu_tools.gui.session.ui.predictor_dialog import PredictorDialog

            dialog = PredictorDialog(
                self._ctrl.predictor_control,
                parent=self._parent,
                device=self._ctrl.device_control,
                parameter_provider=self._ctrl.context_control.read_context,
                persistent_on_close=True,
            )
            dialog.enable_parameter_cache()
            return dialog
        if name is DialogName.INSPECT:
            from zcu_tools.gui.session.ui.inspect_dialog import InspectDialog

            dialog = InspectDialog(
                self._ctrl.context_control,
                self._ctrl.get_bus(),
                templates=self._templates(),
                arb_waveforms=self._ctrl.arb_waveforms,
                # The named entry keeps the ARB dialog visible to screenshots.
                # Read current host writability, not the opening-time snapshot.
                open_arb_waveforms=lambda: self.open(
                    DialogName.ARB_WAVEFORM, writable=self._writable
                ),
                dialog_presenter=self._dialog_presenter,
                parent=self._parent,
            )
            dialog.set_read_only(read_only=not writable)
            return dialog
        if name is DialogName.ARB_WAVEFORM:
            from zcu_tools.gui.session.ui.arb_waveform_dialog import ArbWaveformDialog

            dialog = ArbWaveformDialog(
                self._ctrl.arb_waveforms,
                parent=self._parent,
                dialog_presenter=self._dialog_presenter,
            )
            dialog.set_read_only(read_only=not writable)
            return dialog
        raise ValueError(f"Unknown DialogName: {name!r}")  # pragma: no cover

    def set_writable(self, *, writable: bool) -> None:
        """Push Local authority to all editors, preserving their Run policy."""
        from zcu_tools.gui.session.ui.arb_waveform_dialog import ArbWaveformDialog
        from zcu_tools.gui.session.ui.device_dialog import DeviceDialog
        from zcu_tools.gui.session.ui.inspect_dialog import InspectDialog
        from zcu_tools.gui.session.ui.setup_dialog import SetupDialog
        from zcu_tools.gui.session.ui.soc_dialog import SocDialog

        self._writable = writable
        for name in tuple(self._dialog_names):
            dialog = self._dialog(name)
            if isinstance(
                dialog,
                (
                    SetupDialog,
                    SocDialog,
                    DeviceDialog,
                    InspectDialog,
                    ArbWaveformDialog,
                ),
            ):
                run_locked = self._running and name in _RUN_LOCKED_DIALOGS
                dialog.set_read_only(read_only=not writable or run_locked)

    def set_running(self, *, running: bool) -> None:
        """Lock only Setup/SoC submissions during Run, retaining Local authority."""
        self._running = running
        self.set_writable(writable=self._writable)

    def open(self, name: DialogName, *, writable: bool = True) -> None:
        """Show a non-modal named window, or focus its retained instance."""
        self.set_writable(writable=writable)
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

        run_locked = self._running and name in _RUN_LOCKED_DIALOGS
        dialog = self._build_dialog(name, writable=writable and not run_locked)
        dialog.setWindowModality(Qt.WindowModality.NonModal)
        self._dialog_names[name] = None
        if name in _PERSISTENT_DIALOGS:
            self._dialog_refs.retain_named(
                name,
                dialog,
                on_released=lambda n=name: self._dialog_names.pop(n, None),
            )
            dialog.show()
        else:
            self._dialog_refs.open_named(
                name,
                dialog,
                on_released=lambda n=name: self._dialog_names.pop(n, None),
                modal=False,
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

    def take_screenshot(self, dialog_name: DialogName) -> bytes:
        """Grab a currently-open dialog and return raw PNG bytes."""
        dialog = self._dialog(dialog_name)
        if dialog is not None and dialog.isVisible():
            return widget_to_png_bytes(dialog, subject=f"{dialog_name.value!r} dialog")
        raise FailedPreconditionError(
            f"dialog {dialog_name.value!r} is not currently open"
        )


__all__ = ["MainDialogRegistry"]
