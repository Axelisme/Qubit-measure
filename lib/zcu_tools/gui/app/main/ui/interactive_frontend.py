"""Narrow Qt mounting contract for plugin-owned interactive frontends."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from matplotlib.figure import Figure
from qtpy.QtWidgets import QWidget  # type: ignore[attr-defined]


class InteractiveFrontendEnv(Protocol):
    def run_background(
        self,
        compute: Callable[[], object],
        on_done: Callable[[object], None],
        on_error: Callable[[Exception], None],
    ) -> None: ...


class InteractiveFrontend(QWidget):
    """Only the presentation/lifecycle capabilities the generic View may use."""

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
