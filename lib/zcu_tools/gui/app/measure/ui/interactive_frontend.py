"""Narrow Qt mounting contract for plugin-owned interactive frontends."""

from __future__ import annotations

from matplotlib.figure import Figure
from qtpy.QtWidgets import QWidget

from zcu_tools.gui.lifetime import LifetimeScope

from ..adapter.hooks import InteractiveFrontendEnv as InteractiveFrontendEnv


class InteractiveFrontend(QWidget):
    """Only the presentation/lifecycle capabilities the generic View may use."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._scope = LifetimeScope(self)

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
        """Retire registered callbacks and presentation borrows without closing the session.

        Subclasses register their subscriptions, connections, filters and local
        presentation cleanup in _scope as acquired. Cleanup failures propagate
        and may be retried before deleting this frontend.
        """
        self._scope.close()
