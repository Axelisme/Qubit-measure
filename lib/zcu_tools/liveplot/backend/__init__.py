"""Liveplot rendering backends + active-backend selection.

Selection order (frontend-agnostic, decoupled from the matplotlib backend name):

1. A backend explicitly registered for the current task via
   ``set_liveplot_backend`` (ContextVar — the GUI run worker does this).
2. A process-wide default set via ``set_default_liveplot_backend``.
3. Fallback: pick by matplotlib backend name (nbagg → Jupyter, else plain).

The GUI is *registered* (1), never detected — so this package has zero GUI
import and the dependency direction stays gui → liveplot.
"""

from __future__ import annotations

import warnings
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator, Optional

import matplotlib as mpl
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .base import LivePlotBackend
from .fallback import FallbackBackend
from .jupyter import JupyterBackend

_default_backend: Optional[LivePlotBackend] = None
_backend: ContextVar[Optional[LivePlotBackend]] = ContextVar(
    "liveplot_backend", default=None
)


@contextmanager
def set_liveplot_backend(backend: LivePlotBackend) -> Iterator[None]:
    """Install a backend for the dynamic extent of the ``with`` block.

    Thread/task-safe (ContextVar). The GUI run worker uses this to register its
    Qt backend for the duration of a run. Mirrors ``progress_bar.use_pbar_factory``.
    """
    token = _backend.set(backend)
    try:
        yield
    finally:
        _backend.reset(token)


def set_default_liveplot_backend(backend: Optional[LivePlotBackend]) -> None:
    """Set the process-wide default backend (notebook/setup sets once)."""
    global _default_backend
    _default_backend = backend


def _select_by_mpl_name() -> LivePlotBackend:
    backend = mpl.get_backend().lower()
    if "nbagg" in backend:
        return JupyterBackend()
    if not any(name in backend for name in ["inline", "agg", "qtagg", "module://"]):
        warnings.warn(
            f"Auto-selected backend for matplotlib is '{backend}', "
            "which may not be fully supported."
        )
    return FallbackBackend()


def active_backend() -> LivePlotBackend:
    """The backend in effect now: registered (ContextVar → default) or name-fallback."""
    registered = _backend.get()
    if registered is not None:
        return registered
    if _default_backend is not None:
        return _default_backend
    return _select_by_mpl_name()


def make_plot_frame(
    n_row: int, n_col: int, plot_instant: bool = False, **kwargs
) -> tuple[Figure, list[list[Axes]]]:
    return active_backend().make_plot_frame(
        n_row, n_col, plot_instant=plot_instant, **kwargs
    )


def instant_plot(fig: Figure) -> None:
    """Show a self-built figure immediately via the active backend (e.g. for
    figures built outside ``make_plot_frame`` with a custom gridspec layout)."""
    active_backend().instant_plot(fig)


def refresh_figure(fig: Figure) -> None:
    active_backend().refresh_figure(fig)


def close_figure(fig: Figure) -> None:
    active_backend().close_figure(fig)


__all__ = [
    "LivePlotBackend",
    "FallbackBackend",
    "JupyterBackend",
    "active_backend",
    "set_liveplot_backend",
    "set_default_liveplot_backend",
    "make_plot_frame",
    "instant_plot",
    "refresh_figure",
    "close_figure",
]
