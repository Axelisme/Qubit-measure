"""Narrow Qt mounting contract for plugin-owned interactive frontends."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from matplotlib.figure import Figure
from qtpy.QtWidgets import QWidget

from zcu_tools.gui.control_identity import ControlIdentity, ControlStatus
from zcu_tools.gui.widgets.remote_edit_gate import watch_control


class InteractiveFrontendEnv(Protocol):
    def run_background(
        self,
        compute: Callable[[], object],
        on_done: Callable[[object], None],
        on_error: Callable[[Exception], None],
    ) -> None: ...


class InteractiveFrontend(QWidget):
    """Only the presentation/lifecycle capabilities the generic View may use."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._dispose_authority: Callable[[], None] | None = None

    def bind_control(self, authority: ControlIdentity) -> None:
        """Project one application's authority until owner-thread retirement.

        Rebinding first retires the previous projection. This does not start or
        stop the plugin's domain session, timers or background work.
        """
        self.dispose_control_watch()

        def render(status: ControlStatus) -> None:
            self.set_read_only(read_only=status.holder != "gui")

        self._dispose_authority = watch_control(self, authority, render)

    def dispose_control_watch(self) -> None:
        """Idempotently stop authority projection before detach or Qt deletion."""
        if self._dispose_authority is not None:
            self._dispose_authority()
            self._dispose_authority = None

    def set_read_only(self, *, read_only: bool) -> None:
        """Gate plugin edits without disabling the figure's viewing interactions."""
        raise NotImplementedError

    @property
    def figure(self) -> Figure:
        raise NotImplementedError

    @property
    def preview_active(self) -> bool:
        raise NotImplementedError

    def cancel_preview(self) -> None:
        """Drop local edits before the service finishes from committed state."""
        raise NotImplementedError

    def teardown(self) -> None:
        raise NotImplementedError
