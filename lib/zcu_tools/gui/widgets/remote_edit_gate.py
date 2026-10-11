"""Qt editing projection of the application's single authority owner."""

from __future__ import annotations

from collections.abc import Callable

from qtpy.QtCore import QObject
from qtpy.QtWidgets import QWidget

from zcu_tools.gui.control_identity import ControlIdentity, ControlStatus


class RemoteEditGate(QObject):
    """Apply Local edit availability to dedicated containers and Qt adapters.

    Guard containers, not children with domain-dependent enabled states. This
    preserves busy/valid/input-open child gates across authority transfers.
    Callbacks handle non-widget input (canvas gestures and tab admission).
    Watches run on the application owner thread. Dispose at view close; target
    destruction also detaches its watch while the view remains open.
    """

    def __init__(self, control: ControlIdentity, parent: QObject) -> None:
        super().__init__(parent)
        self._control = control
        self._bindings: list[
            tuple[QObject, Callable[[], None], Callable[[], None]]
        ] = []
        self.destroyed.connect(self.dispose)

    def bind(self, widget: QWidget) -> None:
        """Guard a dedicated editable container until that container is destroyed."""
        self.watch(widget.setEnabled, lifetime=widget)

    def watch(self, callback: Callable[[bool], None], *, lifetime: QObject) -> None:
        """Call immediately and on transitions, with True only for Local editing."""

        def update(status: ControlStatus) -> None:
            callback(status.holder == "gui")

        dispose = self._control.watch(update)

        def detach() -> None:
            dispose()
            self._bindings.remove(binding)

        binding = (lifetime, dispose, detach)
        self._bindings.append(binding)
        lifetime.destroyed.connect(detach)

    def dispose(self) -> None:
        """Detach signals as well as watches before Qt destroys the view tree.

        Leaving Python closures connected to children during parent destruction
        can reenter Python GC through PyQt's slot proxy. Disconnect them while
        all targets are alive, rather than relying on deferred child deletion.
        """
        for lifetime, dispose, detach in self._bindings:
            lifetime.destroyed.disconnect(detach)
            dispose()
        self._bindings.clear()
