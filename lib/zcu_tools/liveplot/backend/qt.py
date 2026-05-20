"""Qt liveplot backend — embeds matplotlib figures into a QStackedWidget container.

Usage pattern (main thread, before run):
    qt.register_pending_container(container)   # container = tab's _plot_stack
    # ... start run ...
    qt.clear_pending_container()               # after run finishes

Inside the run (worker thread), LivePlot calls make_plot_frame() which
automatically embeds the figure into the registered container.  The embedding
is dispatched to the main thread via a Qt signal and a threading.Event is used
to block the worker until the main-thread slot has finished, so make_plot_frame
returns only after the widget is live.

refresh_figure() calls canvas.draw_idle(), which is internally thread-safe in Qt.
close_figure() removes the canvas from its container and calls plt.close().
"""

from __future__ import annotations

import threading
from typing import Any, Optional

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

# id(fig) → QStackedWidget container
_fig_container_registry: dict[int, Any] = {}

# Single-slot container; set by main thread before run, consumed by make_plot_frame
# (only one run can be active at a time, so no per-thread keying needed)
_pending_container: Optional[Any] = None

# Singleton bridge living on the main thread
_embedder: Any = None


def _get_embedder() -> Any:
    global _embedder
    if _embedder is None:
        from qtpy.QtCore import QObject, Signal  # type: ignore[attr-defined]

        class _Embedder(QObject):
            # Signal carries a dict payload to avoid per-call dynamic signal types.
            # Keys: n_row, n_col, kwargs, container, result (list), done (Event)
            make_requested = Signal(object)

            def __init__(self) -> None:
                super().__init__()
                self.make_requested.connect(self._on_make)

            def _on_make(self, payload: Any) -> None:
                import matplotlib.pyplot as _plt
                from matplotlib.backends.backend_qtagg import (  # type: ignore[import-untyped]
                    FigureCanvasQTAgg,
                )

                n_row = payload["n_row"]
                n_col = payload["n_col"]
                kw = payload["kwargs"]
                container = payload["container"]
                result = payload["result"]
                done = payload["done"]

                fig, axs = _plt.subplots(n_row, n_col, **kw)
                canvas = FigureCanvasQTAgg(fig)

                if container is not None:
                    container.addWidget(canvas)
                    container.setCurrentWidget(canvas)

                result.append((fig, axs))
                done.set()

        _embedder = _Embedder()
    return _embedder


def register_pending_container(container: Any) -> None:
    """Called from the main thread before a run starts."""
    global _pending_container
    _pending_container = container
    _get_embedder()  # ensure embedder QObject is created on the main thread


def clear_pending_container() -> None:
    """Called from the main thread after a run ends."""
    global _pending_container
    _pending_container = None


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
    """Create a figure with FigureCanvasQTAgg and embed it into the pending container.

    Figure and canvas creation is dispatched to the main thread via a Signal so
    that Qt widgets are always constructed on the main thread.  The worker blocks
    on a threading.Event until the main-thread slot completes.
    """
    kwargs.setdefault("squeeze", False)
    kwargs.setdefault("figsize", (6 * n_col, 4 * n_row))

    container = _pending_container
    done = threading.Event()
    result: list[Any] = []

    _get_embedder().make_requested.emit(
        {
            "n_row": n_row,
            "n_col": n_col,
            "kwargs": kwargs,
            "container": container,
            "result": result,
            "done": done,
        }
    )
    done.wait(timeout=5.0)

    fig, axs = result[0]
    if container is not None:
        _fig_container_registry[id(fig)] = container
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
