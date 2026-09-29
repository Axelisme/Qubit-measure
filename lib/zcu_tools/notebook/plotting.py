"""Explicit Notebook presentation without pyplot figure registration."""

from collections.abc import Callable
from typing import TYPE_CHECKING, TypeVar, cast

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from ipympl.backend_nbagg import Canvas, FigureManager, Toolbar

_T = TypeVar("_T")


class NotebookPlotHost:
    """Present native figures as widgets until explicitly released.

    Calls run synchronously on the caller's thread. Callers serialize operations;
    this host does not schedule work or detect frontend availability. Figures
    with an existing manager must be released by that owner before presentation.
    """

    def __init__(self) -> None:
        self._managers: dict[Figure, FigureManager] = {}

    def call(self, callback: Callable[[], _T]) -> _T:
        return callback()

    def present(self, figure: Figure) -> None:
        if figure in self._managers:
            return
        if figure.canvas.manager is not None:
            raise ValueError("Figure already has a presentation manager")

        from ipympl.backend_nbagg import Canvas, FigureManager
        from IPython.display import display

        # Construct directly: pyplot/Gcf registration would redisplay at cell end.
        canvas = Canvas(figure)
        manager = FigureManager(canvas, num=0)
        self._managers[figure] = manager
        # Retain the manager even if publishing fails, so release can clean up.
        display(canvas)

    def refresh(self, figure: Figure, *, final: bool = False) -> None:
        manager = self._managers.get(figure)
        if manager is None:
            raise ValueError("Figure is not presented by this host")
        if final:
            manager.canvas.draw()
        else:
            manager.canvas.draw_idle()

    def release(self, figure: Figure) -> None:
        manager = self._managers.get(figure)
        if manager is None:
            return
        # Matplotlib types only the bases of these concrete ipympl widgets.
        toolbar = cast("Toolbar", manager.toolbar)
        canvas = cast("Canvas", manager.canvas)
        toolbar.layout.close()
        toolbar.close()
        canvas.layout.close()
        manager.destroy()
        FigureCanvasAgg(figure)
        del self._managers[figure]
