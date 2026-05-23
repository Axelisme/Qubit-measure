from __future__ import annotations

from typing import cast

from matplotlib.backend_bases import FigureCanvasBase, FigureManagerBase, _Backend
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from qtpy.QtWidgets import QWidget  # type: ignore[attr-defined]

from .mpl_backend_setup import BACKEND_NAME
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
