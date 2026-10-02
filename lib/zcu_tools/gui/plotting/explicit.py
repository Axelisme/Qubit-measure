"""Qt presentation for explicit operation plots, without routing scopes."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from qtpy.QtWidgets import QWidget

from zcu_tools.gui.session.ports import OwnerScheduler

from . import host
from .container import FigureContainer

_T = TypeVar("_T")


class QtPlotHost:
    """Bind plots to one container and the application's existing owner scheduler."""

    def __init__(self, container: FigureContainer, owner: OwnerScheduler) -> None:
        self._container = container
        self._owner = owner

    def call(self, callback: Callable[[], _T]) -> _T:
        if self._owner.is_owner_thread():
            return callback()
        return self._owner.call(callback)

    def _require_owner(self) -> None:
        if not self._owner.is_owner_thread():
            raise RuntimeError("Plot presentation requires the owner thread")

    def present(self, figure: Figure) -> None:
        self._require_owner()
        previous = host.get_figure_container(figure)
        if previous is not None and previous is not self._container:
            raise ValueError("Figure is presented by another container")
        host.attach_existing_figure_to_container(figure, self._container)
        figure.canvas.draw_idle()

    def refresh(self, figure: Figure, *, final: bool = False) -> None:
        self._require_owner()
        if host.get_figure_container(figure) is not self._container:
            raise ValueError("Figure is not presented by this container")
        if final:
            figure.canvas.draw()
        else:
            figure.canvas.draw_idle()

    def release(self, figure: Figure) -> None:
        self._require_owner()
        previous = host.get_figure_container(figure)
        if previous is None:
            return
        if previous is not self._container:
            raise ValueError("Figure is presented by another container")
        canvas = figure.canvas
        if not isinstance(canvas, QWidget):
            raise RuntimeError("Presented Figure has no Qt canvas")
        host.remove_canvas(canvas)
        FigureCanvasAgg(figure)
