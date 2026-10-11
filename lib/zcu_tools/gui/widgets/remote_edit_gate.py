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


def local_write_allowed(authority: ControlIdentity | None) -> bool:
    """Recheck GUI authority immediately before a write after a modal decision.

    None preserves the behavior of shared widgets in apps without control identity.
    This is only a projection of the application owner, not another holder state.
    """
    return authority is None or authority.status().holder == "gui"


def bind_control_enabled(widget: QWidget, authority: ControlIdentity | None) -> None:
    """Gate a write-only region, preserving child domain gates and dialog exit.

    Bind content containers and write controls, never a closable dialog root.
    None leaves shared widgets' existing behavior unchanged.
    """
    if authority is None:
        return
    watch_control(
        widget, authority, lambda status: widget.setEnabled(status.holder == "gui")
    )
