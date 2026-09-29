"""Explicit operation plotting with typed updates and frontend-owned presentation."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, TypeVar

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray

from .figures import FigureCollection
from .liveplot.segments.plot1d import Plot1DSegment

_T = TypeVar("_T")


class PlotHost(Protocol):
    """Execute plotting work on one owner; never infer an ambient destination."""

    def call(self, callback: Callable[[], _T]) -> _T: ...

    def present(self, figure: Figure) -> None: ...

    def refresh(self, figure: Figure, *, final: bool = False) -> None: ...

    def release(self, figure: Figure) -> None: ...


class NonPresentingHost:
    """Keep native artists and savefig usable without rendering or opening a view."""

    def call(self, callback: Callable[[], _T]) -> _T:
        return callback()

    def present(self, figure: Figure) -> None:
        pass

    def refresh(self, figure: Figure, *, final: bool = False) -> None:
        pass

    def release(self, figure: Figure) -> None:
        pass


class LinePlot:
    """A typed update handle; active artists belong to the plots host's owner."""

    def __init__(
        self,
        host: PlotHost,
        figure: Figure,
        axes: Axes,
        segment: Plot1DSegment,
        ensure_active: Callable[[], None],
    ) -> None:
        self._host = host
        self._figure = figure
        self._axes = axes
        self._segment = segment
        self._ensure_active = ensure_active

    def update(
        self,
        xs: NDArray[np.float64],
        signals: NDArray[np.float64],
        title: str | None = None,
        *,
        refresh: bool = True,
    ) -> None:
        """Copy real data before dispatch; reject invalid shapes before mutation."""
        self._ensure_active()
        if np.iscomplexobj(xs) or np.iscomplexobj(signals):
            raise ValueError("Line plot updates require real-valued data")
        x_data = np.array(xs, dtype=np.float64, copy=True)
        y_data = np.array(signals, dtype=np.float64, copy=True)
        if y_data.ndim == 1:
            y_data = y_data[None, :]
        if (
            x_data.ndim != 1
            or y_data.ndim != 2
            or y_data.shape != (self._segment.num_line, x_data.size)
        ):
            raise ValueError("Line plot data must match the line count and x length")

        def apply() -> None:
            self._ensure_active()
            self._segment.update(self._axes, x_data, y_data, title)
            if refresh:
                self._host.refresh(self._figure)

        self._host.call(apply)

    def refresh(self) -> None:
        self._ensure_active()
        self._host.call(lambda: self._host.refresh(self._figure))


class Plots(FigureCollection):
    """One producer's plotting lifetime, shared by frontend adapters.

    Stop the producer before finish. Host calls are synchronous and preserve
    that producer's order. General figures stay unpresented until finish, while
    live figures present immediately. Finishing fixes membership and stops typed
    updates; it does not declare analysis success. Release presentation separately
    after finishing. Neither finish nor release destroys retained Figure objects.
    """

    def __init__(self, host: PlotHost) -> None:
        super().__init__()
        self._host = host
        self._live: list[Figure] = []
        self._finished = False
        self._released: set[Figure] = set()

    def _ensure_active(self) -> None:
        if self._finished:
            raise RuntimeError("Plot operation has finished")

    def liveplot_1d(
        self,
        name: str,
        xlabel: str,
        ylabel: str,
        *,
        title: str | None = None,
        num_lines: int = 1,
    ) -> LinePlot:
        self._ensure_active()
        if num_lines < 1:
            raise ValueError("Line count must be positive")

        def create() -> LinePlot:
            self._ensure_active()
            figure, axes = self.subplots(name)
            segment = Plot1DSegment(xlabel, ylabel, title=title, num_lines=num_lines)
            segment.init_ax(axes)
            viewer = LinePlot(self._host, figure, axes, segment, self._ensure_active)
            self._host.present(figure)
            self._live.append(figure)
            return viewer

        return self._host.call(create)

    def finish(self, *, present: bool = True) -> FigureCollection:
        if self._finished:
            return self
        self._finished = True
        self.seal()

        def complete() -> None:
            for figure in self._live:
                self._host.refresh(figure, final=True)
            if present:
                for figure in self.values():
                    if figure not in self._live:
                        self._host.present(figure)

        self._host.call(complete)
        return self

    def release(self) -> None:
        """Release every presentation, reporting failures after trying all figures."""
        if not self._finished:
            raise RuntimeError(
                "Finish the plot operation before releasing presentation"
            )

        def cleanup() -> None:
            errors: list[Exception] = []
            for figure in self.values():
                if figure in self._released:
                    continue
                try:
                    self._host.release(figure)
                except Exception as error:  # noqa: BLE001 - report all cleanup failures together
                    errors.append(error)
                else:
                    self._released.add(figure)
            if errors:
                raise ExceptionGroup("Failed to release plot presentation", errors)

        self._host.call(cleanup)
