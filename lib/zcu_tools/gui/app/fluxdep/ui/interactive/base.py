"""InteractiveMplWidget — shared base for the fluxdep interactive tools.

A QWidget that owns a matplotlib Figure + FigureCanvasQTAgg and connects the
canvas mouse events to overridable handlers (``on_press`` / ``on_move`` /
``on_release``, all no-ops by default). Subclasses add Qt controls to
``controls_layout`` and implement ``get_result``; a "Finish" button emits the
``finished`` signal so the host can collect the result and write it via the
Controller.

This mirrors the notebook tools' ``fig.canvas.mpl_connect`` interaction, but the
canvas is a Qt-embedded FigureCanvasQTAgg (not an inline/ipympl canvas) — all
painting happens on the Qt main thread, so a plain ``draw_idle`` is sufficient
(no cross-thread marshalling like measure-gui's liveplot).
"""

from __future__ import annotations

from typing import Optional, cast

from matplotlib.backend_bases import Event, MouseEvent
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from qtpy.QtCore import Signal  # type: ignore[attr-defined]
from qtpy.QtWidgets import (  # type: ignore[attr-defined]
    QHBoxLayout,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class InteractiveMplWidget(QWidget):
    """Base widget: a Qt-embedded mpl canvas + a controls column + Finish.

    Layout: canvas on the left (stretches), a controls column on the right. The
    ``finished`` signal fires when the user clicks Finish; the host then calls
    ``get_result()``.
    """

    finished = Signal()

    def __init__(
        self, parent: Optional[QWidget] = None, controls_side: str = "right"
    ) -> None:
        super().__init__(parent)
        self.figure = Figure(figsize=(8, 5))
        self.canvas = FigureCanvasQTAgg(self.figure)

        controls = QWidget()
        self.controls_layout = QVBoxLayout(controls)
        controls.setFixedWidth(220)

        # ``controls_side`` places the control column left or right of the canvas.
        # Default "right" suits the standalone editor widgets; the analysis tabs
        # put their controls on the LEFT, so the cross-spectrum filter uses "left"
        # to match Search / Show.
        self._root = QHBoxLayout(self)
        if controls_side == "left":
            self._root.addWidget(controls)
            self._root.addWidget(self.canvas, stretch=1)
        else:
            self._root.addWidget(self.canvas, stretch=1)
            self._root.addWidget(controls)

        # Subclasses populate controls_layout before/after super().__init__;
        # the Finish button is appended at the bottom by add_finish_button().

        self.canvas.mpl_connect("button_press_event", self._dispatch_press)
        self.canvas.mpl_connect("motion_notify_event", self._dispatch_move)
        self.canvas.mpl_connect("button_release_event", self._dispatch_release)

    def add_finish_button(self, label: str = "Finish") -> QPushButton:
        """Append the finish/apply button (call after adding subclass controls).

        ``label`` lets a subclass name it for its semantics (e.g. "Apply" for the
        non-terminal cross-spectrum filter). Returns the button so callers can
        give feedback on it.
        """
        self.controls_layout.addStretch(1)
        finish = QPushButton(label)
        finish.clicked.connect(self.finished.emit)
        self.controls_layout.addWidget(finish)
        return finish

    def redraw(self) -> None:
        self.canvas.draw_idle()

    # --- overridable mouse handlers (default no-op) ----------------------

    def on_press(self, event: MouseEvent) -> None:  # noqa: D401
        """Mouse button pressed inside an axes. Override in subclass."""

    def on_move(self, event: MouseEvent) -> None:
        """Mouse moved inside an axes. Override in subclass."""

    def on_release(self, event: MouseEvent) -> None:
        """Mouse button released inside an axes. Override in subclass."""

    # --- dispatch (filter to in-axes events) -----------------------------

    def _dispatch_press(self, event: Event) -> None:
        mouse = cast(MouseEvent, event)
        if mouse.inaxes is not None:
            self.on_press(mouse)

    def _dispatch_move(self, event: Event) -> None:
        mouse = cast(MouseEvent, event)
        if mouse.inaxes is not None:
            self.on_move(mouse)

    def _dispatch_release(self, event: Event) -> None:
        mouse = cast(MouseEvent, event)
        if mouse.inaxes is not None:
            self.on_release(mouse)
