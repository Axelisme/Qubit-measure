"""Always-available Qt projection of application control authority."""

from __future__ import annotations

from qtpy.QtWidgets import QPushButton, QWidget

from zcu_tools.gui.control_identity import ControlIdentity, ControlStatus


class ControlIndicator(QPushButton):
    """Click to reclaim Remote or toggle the Local lock; never cancel work."""

    def __init__(self, control: ControlIdentity, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("control_indicator")
        self.clicked.connect(control.toggle_local)
        self._dispose = control.watch(self._show_status)
        self.destroyed.connect(self._dispose)

    def _show_status(self, status: ControlStatus) -> None:
        if status.holder == "gui":
            self.setText("● Local (locked)" if status.locked else "● Local")
            self.setToolTip("Click to unlock" if status.locked else "Click to lock")
            color = "#b8860b" if status.locked else "#228b22"
        else:
            presence = "online" if status.online else "offline"
            self.setText(f"● Remote: {status.holder.removeprefix('rpc:')} ({presence})")
            self.setToolTip("Click to reclaim and lock Local control")
            color = "#c43b3b" if status.online else "#808080"
        self.setStyleSheet(f"QPushButton {{ color: {color}; }}")
