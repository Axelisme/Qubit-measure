"""TuneCanvasWidget — a display-only Qt-embedded matplotlib canvas.

A plain ``FigureCanvasQTAgg`` (like fluxdep's Show tab), NOT the routed embedded
backend: every draw here happens on the Qt main thread (the slider tuning recomputes
synchronously via the LRU-cached predictor), so a plain ``draw_idle`` suffices — no
cross-thread marshalling. The widget owns one Figure and exposes it for the
VizService render functions to draw onto.
"""

from __future__ import annotations

from typing import Optional

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from qtpy.QtWidgets import QVBoxLayout, QWidget  # type: ignore[attr-defined]


class TuneCanvasWidget(QWidget):
    """A QWidget wrapping one matplotlib Figure + canvas (main-thread drawing)."""

    def __init__(
        self,
        figsize: tuple[float, float] = (6.0, 4.0),
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.figure = Figure(figsize=figsize)
        self.canvas = FigureCanvasQTAgg(self.figure)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)

    def redraw(self) -> None:
        self.canvas.draw_idle()
