"""Qt liveplot backend — embeds matplotlib figures into a QStackedWidget container.

Usage pattern (main thread, before run):
    qt.register_pending_container(container)   # container = tab's _plot_stack
    # ... start run ...
    qt.clear_pending_container()               # after run finishes

Inside the run (worker thread), LivePlot calls make_plot_frame() which
automatically embeds the figure into the registered container via
BlockingQueuedConnection so the embedding happens synchronously on the main thread.

refresh_figure() calls canvas.draw_idle(), which is internally thread-safe in Qt.
close_figure() removes the canvas from its container and calls plt.close().
"""

from __future__ import annotations

import threading
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

# id(fig) → QStackedWidget container
_fig_container_registry: dict[int, Any] = {}

# thread_id → container; set by main thread before run, consumed by make_plot_frame
_pending_container: dict[int, Any] = {}

# Singleton QObject used as the target for cross-thread slot invocations
_embedder: Any = None


def _get_embedder() -> Any:
    global _embedder
    if _embedder is None:
        from qtpy.QtCore import QObject  # type: ignore[attr-defined]

        class _Embedder(QObject):
            def embed(self, canvas: Any, container: Any) -> None:
                container.addWidget(canvas)
                container.setCurrentWidget(canvas)

        _embedder = _Embedder()
    return _embedder


def register_pending_container(container: Any) -> None:
    """Called from the main thread before a run starts.

    The next make_plot_frame() on the worker thread will embed the figure here.
    """
    _pending_container[threading.get_ident()] = container


def clear_pending_container() -> None:
    """Called from the main thread after a run ends."""
    _pending_container.pop(threading.get_ident(), None)


def set_figure_container(fig: Figure, container: Any) -> None:
    """Embed fig's canvas into container (main-thread call only)."""
    from matplotlib.backends.backend_qtagg import (  # type: ignore[import-untyped]
        FigureCanvasQTAgg,
    )

    canvas = fig.canvas
    if not isinstance(canvas, FigureCanvasQTAgg):
        canvas = FigureCanvasQTAgg(fig)

    _fig_container_registry[id(fig)] = container
    container.addWidget(canvas)
    container.setCurrentWidget(canvas)


def make_plot_frame(
    n_row: int,
    n_col: int,
    plot_instant: bool = False,  # noqa: ARG001
    **kwargs: Any,
) -> tuple[Figure, list[list[Axes]]]:
    """Create a figure with FigureCanvasQTAgg.

    If register_pending_container() was called for the current thread, the
    figure is automatically embedded into that container via a
    BlockingQueuedConnection (runs on the main thread, synchronously).
    """
    from matplotlib.backends.backend_qtagg import (  # type: ignore[import-untyped]
        FigureCanvasQTAgg,
    )
    from qtpy.QtCore import QMetaObject, Qt  # type: ignore[attr-defined]

    kwargs.setdefault("squeeze", False)
    kwargs.setdefault("figsize", (6 * n_col, 4 * n_row))
    fig, axs = plt.subplots(n_row, n_col, **kwargs)
    canvas = FigureCanvasQTAgg(fig)

    container = _pending_container.get(threading.get_ident())
    if container is not None:
        _fig_container_registry[id(fig)] = container
        embedder = _get_embedder()
        QMetaObject.invokeMethod(  # type: ignore[call-overload]
            embedder,
            "embed",
            Qt.ConnectionType.BlockingQueuedConnection,  # type: ignore[attr-defined]
            canvas,
            container,
        )

    return fig, axs


def refresh_figure(fig: Figure) -> None:
    """Thread-safe: draw_idle() schedules a repaint on the main-thread event loop."""
    fig.canvas.draw_idle()


def close_figure(fig: Figure) -> None:
    """Remove canvas from its container and close the figure."""
    container = _fig_container_registry.pop(id(fig), None)
    if container is not None:
        canvas = fig.canvas
        container.removeWidget(canvas)  # type: ignore[union-attr]
        canvas.deleteLater()  # type: ignore[union-attr]
    plt.close(fig)
