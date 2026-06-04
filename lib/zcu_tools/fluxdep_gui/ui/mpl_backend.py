"""mpl_backend — client side of fluxdep's embedded-matplotlib substrate.

Registered as the active matplotlib backend (``module://...``) so that plain
``plt.figure()`` / ``plt.show()`` inside analysis code (notably the notebook's
``search_in_database(plot=True)``) are intercepted and routed into the GUI's
current ``FigureContainer`` (see ``plot_host``) instead of opening a detached
window or crashing on ``plt.show()`` off the main thread.

The figure may be created on a worker thread (the search runs off-main), so the
canvas marshals any worker-thread ``draw_idle`` to the main thread. This keeps
``fitting.py`` untouched: it just uses pyplot, and the backend handles embedding
and threading.

Process-wide selection — incompatible with a Jupyter backend in the same
process. The entry point must configure it BEFORE pyplot is first imported
(see ``mpl_backend_setup``).
"""

from __future__ import annotations

from typing import cast

from matplotlib.backend_bases import FigureCanvasBase, FigureManagerBase, _Backend
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from qtpy.QtWidgets import QWidget  # type: ignore[attr-defined]

from .plot_host import (
    activate_figure,
    attach_figure_to_current_container,
    get_figure_container,
    is_main_thread,
    refresh_figure_in_main_thread,
    remove_canvas,
)


class GuiFigureManager(FigureManagerBase):
    @classmethod
    def create_with_canvas(
        cls,
        canvas_class: type[FigureCanvasBase],
        figure: Figure,
        num: int | str,
    ) -> "GuiFigureManager":
        canvas = cast(
            GuiFigureCanvas,
            attach_figure_to_current_container(
                figure, cast(type[FigureCanvasQTAgg], canvas_class)
            ),
        )
        return cls(canvas, num)

    @classmethod
    def start_main_loop(cls) -> None:
        return None

    def show(self) -> None:
        # plt.show() → just make this figure current in its container (no window).
        if get_figure_container(self.canvas.figure) is None:
            raise RuntimeError("figure is not attached to any FigureContainer")
        activate_figure(self.canvas.figure)

    def destroy(self) -> None:
        remove_canvas(cast(QWidget, self.canvas))


class GuiFigureCanvas(FigureCanvasQTAgg):
    required_interactive_framework = None
    manager_class = GuiFigureManager

    def draw_idle(self, *args: object, **kwargs: object) -> None:
        """Thread-safe ``draw_idle``: marshal a worker-thread call to the main thread.

        Painting a Qt canvas off the main thread is undefined; the search worker
        creates the diagnostic figure, so its draws are marshalled fire-and-forget
        to the main thread. A main-thread call draws inline.
        """
        if is_main_thread():
            super().draw_idle(*args, **kwargs)
        else:
            refresh_figure_in_main_thread(self.figure)


FigureCanvas = GuiFigureCanvas
FigureManager = GuiFigureManager


# Registered into matplotlib's backend registry by @_Backend.export.
@_Backend.export
class _BackendGui(_Backend):  # noqa: F811  # pyright: ignore[reportUnusedClass]
    FigureCanvas = GuiFigureCanvas
    FigureManager = GuiFigureManager
    backend_version = "fluxdep-embedded"
