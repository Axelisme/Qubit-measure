"""Qt presentation for explicit operation plots, without routing scopes."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from qtpy.QtCore import QTimer
from qtpy.QtWidgets import QWidget

from zcu_tools.gui.session.ports import OwnerScheduler

from . import host
from .container import FigureContainer

_T = TypeVar("_T")


class QtPlotHost:
    """Present owner-thread artists, coalescing redraws at 50 ms intervals.

    Typed updates still mutate the current Figure synchronously through call.
    Only presentation is delayed; final refresh draws the latest data immediately.
    The caller stops its producer before final refresh/release, and releases every
    presented Figure before destroying its container or owner scheduler.
    """

    def __init__(self, container: FigureContainer, owner: OwnerScheduler) -> None:
        """Borrow the container and scheduler; construct on their Qt owner thread."""
        self._container = container
        self._owner = owner
        self._refresh_timers: dict[Figure, QTimer] = {}

    def call(self, callback: Callable[[], _T]) -> _T:
        """Execute callback synchronously on the owner, returning or propagating it."""
        if self._owner.is_owner_thread():
            return callback()
        return self._owner.call(callback)

    def _require_owner(self) -> None:
        if not self._owner.is_owner_thread():
            raise RuntimeError("Plot presentation requires the owner thread")

    def present(self, figure: Figure) -> None:
        """Attach the native Figure on the owner; reject another container's claim."""
        self._require_owner()
        previous = host.get_figure_container(figure)
        if previous is not None and previous is not self._container:
            raise ValueError("Figure is presented by another container")
        host.attach_existing_figure_to_container(figure, self._container)
        figure.canvas.draw_idle()

    def refresh(self, figure: Figure, *, final: bool = False) -> None:
        """Request a coalesced redraw, or synchronously draw the final current data.

        Call on the owner for a Figure presented by this container, otherwise
        raise RuntimeError/ValueError. Non-final requests share one 50 ms timer
        per Figure, without delaying its artist updates. final cancels that timer
        and draws immediately; native rendering errors propagate unchanged.
        """
        self._require_owner()
        if host.get_figure_container(figure) is not self._container:
            raise ValueError("Figure is not presented by this container")
        timer = self._refresh_timers.get(figure)
        if final:
            if timer is not None:
                timer.stop()
            figure.canvas.draw()
            return
        if timer is None:
            timer = QTimer()
            timer.setSingleShot(True)
            timer.setInterval(50)
            timer.timeout.connect(lambda: figure.canvas.draw_idle())
            self._refresh_timers[figure] = timer
        if not timer.isActive():
            timer.start()

    def release(self, figure: Figure) -> None:
        """Cancel pending presentation and detach to Agg without clearing artists.

        Call on the owner after the producer stops. A released Figure is a no-op;
        another container's Figure raises ValueError and stays attached.
        """
        self._require_owner()
        previous = host.get_figure_container(figure)
        if previous is not None and previous is not self._container:
            raise ValueError("Figure is presented by another container")
        timer = self._refresh_timers.pop(figure, None)
        if timer is not None:
            timer.stop()
            timer.deleteLater()
        if previous is None:
            return
        canvas = figure.canvas
        if not isinstance(canvas, QWidget):
            raise RuntimeError("Presented Figure has no Qt canvas")
        host.remove_canvas(canvas)
        FigureCanvasAgg(figure)
