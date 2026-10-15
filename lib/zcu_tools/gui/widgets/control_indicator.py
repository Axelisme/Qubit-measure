"""Qt holder indicator; control transitions remain application-owned."""

from qtpy.QtWidgets import QPushButton, QWidget

from zcu_tools.gui.control_identity import ControlIdentity, ControlStatus
from zcu_tools.gui.widgets import (
    accent_colour,
    set_foreground,
    success_colour,
    warning_colour,
)


class ControlIndicator(QPushButton):
    """Always-available holder display; clicks delegate the transition to owner."""

    def __init__(
        self, authority: ControlIdentity, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.setObjectName("control_indicator")
        self.clicked.connect(authority.toggle_local)
        self.set_status(authority.status())

    def set_status(self, status: ControlStatus) -> None:
        """Display the main window's current holder projection."""
        if status.holder == "gui":
            self.setText("Local locked" if status.locked else "Local")
            set_foreground(
                self, warning_colour(self) if status.locked else success_colour(self)
            )
            self.setToolTip(
                "Unlock for agents" if status.locked else "Lock local control"
            )
        else:
            presence = "online" if status.online else "offline"
            self.setText(f"Remote · {status.holder.removeprefix('rpc:')} · {presence}")
            set_foreground(self, accent_colour(self))
            self.setToolTip("Take control back and lock Local")
