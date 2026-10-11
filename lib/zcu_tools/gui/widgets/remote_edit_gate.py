"""Qt write gates driven by the application's single control authority."""

from collections.abc import Callable

from qtpy.QtWidgets import QWidget

from zcu_tools.gui.control_identity import ControlIdentity, ControlStatus


def watch_control(
    widget: QWidget,
    authority: ControlIdentity,
    render: Callable[[ControlStatus], None],
) -> None:
    """Render immediately and dispose the owner-thread watch on Qt destruction."""
    dispose = authority.watch(render)
    widget.destroyed.connect(lambda *_args: dispose())


def bind_control_enabled(widget: QWidget, authority: ControlIdentity) -> None:
    """Gate a write-only widget, leaving its children's domain gates intact."""
    watch_control(
        widget, authority, lambda status: widget.setEnabled(status.holder == "gui")
    )
