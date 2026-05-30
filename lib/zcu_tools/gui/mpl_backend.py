"""GUI matplotlib backend — intercepts pyplot figure creation.

One of three plotting-substrate modules (see also ``plot_routing`` = task-local
routing, ``plot_host`` = Qt canvas lifecycle; full picture in AI_NOTE "Plotting
Substrate").

Behaviour guarantees this module provides:
- Registered as the active matplotlib backend so that ``plt.figure()`` /
  ``plt.subplots()`` inside experiment code are intercepted and routed to the
  current ``FigureContainer`` (from ``plot_routing``) instead of opening a
  detached window.
- This is a **process-wide** backend selection; it is incompatible with a
  Jupyter-notebook backend in the same process. The GUI configures the backend
  before any ``pyplot`` import (the "configure backend before pyplot" invariant
  — the reason ``state.py`` only TYPE_CHECKING-imports ``device.base``).
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
        if get_figure_container(self.canvas.figure) is None:
            raise RuntimeError("Figure is not attached to any FigureContainer")
        activate_figure(self.canvas.figure)

    def destroy(self) -> None:
        remove_canvas(cast(QWidget, self.canvas))


class GuiFigureCanvas(FigureCanvasQTAgg):
    required_interactive_framework = None
    manager_class = GuiFigureManager


FigureCanvas = GuiFigureCanvas
FigureManager = GuiFigureManager


@_Backend.export
class _BackendGui(_Backend):
    FigureCanvas = GuiFigureCanvas
    FigureManager = GuiFigureManager
    backend_version = "gui-embedded"
