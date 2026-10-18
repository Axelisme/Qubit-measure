"""InteractiveMplWidget — shared base for the fluxdep interactive tools.

A QWidget that owns a matplotlib Figure + FigureCanvasQTAgg and connects the
canvas mouse events to overridable handlers (``on_press`` / ``on_move`` /
``on_release``, all no-ops by default). Subclasses add Qt controls to
``controls_layout`` top-down: settings, edit actions (Undo / Clear), then the
status text. Finish stays pinned at the column bottom in ``bottom_layout``;
the window's stage bar switches stages. Controls carry ``fluxdep.picker.<action>`` objectNames. A "Finish"
button emits the ``finished`` signal so the host can collect the result and
write it via the Controller.

This mirrors the notebook tools' ``fig.canvas.mpl_connect`` interaction, but the
canvas is a Qt-embedded FigureCanvasQTAgg (not an inline/ipympl canvas) — all
painting happens on the Qt main thread, so a plain ``draw_idle`` is sufficient
(no cross-thread marshalling like measure-gui's liveplot).
"""

from __future__ import annotations

from matplotlib.backend_bases import Event, MouseEvent
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6.QtCore import pyqtSignal as Signal
from qtpy.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


def picker_object_name(action: str) -> str:
    """objectName of a picker control, ``fluxdep.picker.<action>``."""
    return f"fluxdep.picker.{action}"


class InteractiveMplWidget(QWidget):
    """Base widget: a Qt-embedded mpl canvas + a controls column + Finish.

    Layout: canvas on the left (stretches), a controls column on the right.
    The column holds ``controls_layout`` on top and ``bottom_layout`` pinned
    to the column bottom for Finish. ``addWidget`` on
    either layout appends below what is already there. The ``finished`` signal
    fires when the user clicks Finish; the host then calls ``get_result()``.
    """

    finished = Signal()

    def __init__(
        self, parent: QWidget | None = None, controls_side: str = "right"
    ) -> None:
        super().__init__(parent)
        self.figure = Figure(figsize=(8, 5))
        self.canvas = FigureCanvasQTAgg(self.figure)

        self._editable = True
        controls = QWidget()
        self._controls = controls
        column = QVBoxLayout(controls)
        self.controls_layout = QVBoxLayout()
        column.addLayout(self.controls_layout)
        column.addStretch(1)
        self.bottom_layout = QVBoxLayout()
        column.addLayout(self.bottom_layout)
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

        # Subclasses populate controls_layout after super().__init__;
        # add_finish_button() fills the pinned bottom.

        # Plot builders lay out once for the default figure size; re-fit on resize
        # so tick and axis labels stay inside the canvas Qt actually gives us.
        self.canvas.mpl_connect("resize_event", self._fit_layout)
        self.canvas.mpl_connect("button_press_event", self._dispatch_press)
        self.canvas.mpl_connect("motion_notify_event", self._dispatch_move)
        self.canvas.mpl_connect("button_release_event", self._dispatch_release)

    def set_editable(self, *, editable: bool) -> None:
        """Gate controls and domain gestures, leaving canvas viewing available."""
        self._editable = editable
        self._controls.setEnabled(editable)
        if not editable:
            self.cancel_pointer_preview()

    def cancel_pointer_preview(self) -> None:
        """Discard presentation-only input when authority changes mid-gesture."""

    def add_labelled(self, name: str, action: str, control: QWidget) -> None:
        """Append a visible label for control, then control, to controls_layout.

        The label names control as its buddy and as its accessibleName; action
        sets objectName ``fluxdep.picker.<action>``. Use it for every slider and
        drop-down so none appears without a name.
        """
        label = QLabel(name)
        label.setBuddy(control)
        control.setAccessibleName(name)
        control.setObjectName(picker_object_name(action))
        self.controls_layout.addWidget(label)
        self.controls_layout.addWidget(control)

    def add_finish_button(self) -> QPushButton:
        """Pin the finish button to the bottom of the control column.

        It emits ``finished`` and carries objectName ``fluxdep.picker.finish``.
        Returns the button so callers can gate or rewire it.
        """
        finish = QPushButton("Finish")
        finish.setObjectName(picker_object_name("finish"))
        finish.clicked.connect(self.finished.emit)
        self.bottom_layout.addWidget(finish)
        return finish

    def redraw(self) -> None:
        self.canvas.draw_idle()

    def _fit_layout(self, _event: Event) -> None:
        self.figure.tight_layout()

    # --- overridable mouse handlers (default no-op) ----------------------

    def on_press(self, event: MouseEvent) -> None:
        """Mouse button pressed inside an axes. Override in subclass."""

    def on_move(self, event: MouseEvent) -> None:
        """Mouse moved inside an axes. Override in subclass."""

    def on_release(self, event: MouseEvent) -> None:
        """Mouse released, including outside axes so a gesture can finish."""

    # --- dispatch (filter to in-axes events) -----------------------------

    def _dispatch_press(self, event: MouseEvent) -> None:
        if self._editable and event.inaxes is not None:
            self.on_press(event)

    def _dispatch_move(self, event: MouseEvent) -> None:
        if self._editable and event.inaxes is not None:
            self.on_move(event)

    def _dispatch_release(self, event: MouseEvent) -> None:
        if self._editable:
            self.on_release(event)
