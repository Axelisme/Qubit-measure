"""Liveplot rendering backend contract.

A backend owns how a plot frame is built, shown, refreshed and closed for one
frontend (Jupyter notebook, plain matplotlib window, GUI Qt container). The
liveplot core stays frontend-unaware: it asks the *active* backend (see
``backend/__init__.py``) to do these four things and never knows which one is
live. The GUI registers its own backend rather than being detected — keeping
the dependency direction gui → liveplot.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from matplotlib.axes import Axes
from matplotlib.figure import Figure


class LivePlotBackend(ABC):
    @abstractmethod
    def make_plot_frame(
        self, n_row: int, n_col: int, plot_instant: bool = False, **kwargs: Any
    ) -> tuple[Figure, list[list[Axes]]]:
        """Build an (n_row x n_col) grid of axes; optionally show it now."""

    @abstractmethod
    def instant_plot(self, fig: Figure) -> None:
        """Show a self-built figure immediately in this frontend."""

    @abstractmethod
    def refresh_figure(self, fig: Figure) -> None:
        """Repaint the figure to reflect the latest data."""

    @abstractmethod
    def close_figure(self, fig: Figure) -> None:
        """Dispose the figure (frontend decides whether that is a real close)."""
