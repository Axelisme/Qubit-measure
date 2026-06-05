"""TuneCanvasWidget — a Qt-embedded matplotlib canvas with draggable sample lines.

A ``FigureCanvasQTAgg`` (like fluxdep's Show tab), NOT the routed embedded backend:
every draw here happens on the Qt main thread (the slider tuning recomputes
synchronously via the LRU-cached predictor), so a plain ``draw_idle`` suffices — no
cross-thread marshalling. The widget owns one Figure and exposes it for the
VizService render functions to draw onto.

On top of display it provides **drag interaction for the sample-flux lines**: the
panel registers the live ``TuneArtists`` and a drag callback; pressing near a
sample line's vertical and dragging moves it (the panel recomputes that line's
ground/excited dots on each move). This is the slider-driven figure's one piece of
mouse interaction — distinct from fluxdep's point-picking, so it is NOT built on
fluxdep's InteractiveMplWidget (which carries a Finish button + controls column).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional, cast

from matplotlib.backend_bases import Event, MouseEvent
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from qtpy.QtWidgets import QVBoxLayout, QWidget  # type: ignore[attr-defined]

if TYPE_CHECKING:
    from zcu_tools.gui.app.dispersive.services.viz import SampleArtists, TuneArtists

# How close (in axes-width fraction) a click must be to a sample line to grab it.
_PICK_TOL_FRAC = 0.02


class TuneCanvasWidget(QWidget):
    """A QWidget wrapping one matplotlib Figure + canvas (main-thread drawing).

    Hosts draggable sample-flux lines: ``bind_drag`` installs the live ``TuneArtists``
    and a callback ``on_drag(sample, flux)`` invoked while a sample line is dragged.
    """

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

        self._artists: Optional["TuneArtists"] = None
        self._on_drag: Optional[Callable[["SampleArtists", float], None]] = None
        self._on_drop: Optional[Callable[["SampleArtists"], None]] = None
        self._dragging: Optional["SampleArtists"] = None
        self._dragged: bool = False  # whether the grabbed line actually moved

        self.canvas.mpl_connect("button_press_event", self._on_press)
        self.canvas.mpl_connect("motion_notify_event", self._on_move)
        self.canvas.mpl_connect("button_release_event", self._on_release)

    def redraw(self) -> None:
        self.canvas.draw_idle()

    def bind_drag(
        self,
        artists: "TuneArtists",
        on_drag: Callable[["SampleArtists", float], None],
        on_drop: Callable[["SampleArtists"], None],
    ) -> None:
        """Register the live artists + sample-line drag callbacks.

        Called whenever the tune figure is (re)rendered so dragging targets the
        current artists. ``on_drag(sample, flux)`` runs on each motion event while a
        sample line is grabbed (move the line VISUALLY — no compute); ``on_drop(sample)``
        runs once on release (recompute the dot only after the user stops moving it).
        """
        self._artists = artists
        self._on_drag = on_drag
        self._on_drop = on_drop
        self._dragging = None
        self._dragged = False

    # --- drag interaction ------------------------------------------------

    def _pick_sample(self, x_data: float) -> Optional["SampleArtists"]:
        """The sample line nearest ``x_data`` within the pick tolerance, else None."""
        if self._artists is None or not self._artists.samples:
            return None
        ax = self._artists.ax
        x_lo, x_hi = ax.get_xlim()
        tol = abs(x_hi - x_lo) * _PICK_TOL_FRAC
        best: Optional["SampleArtists"] = None
        best_d = tol
        for sample in self._artists.samples:
            d = abs(sample.flux - x_data)
            if d <= best_d:
                best_d = d
                best = sample
        return best

    def _on_press(self, event: Event) -> None:
        mouse = cast(MouseEvent, event)
        if mouse.inaxes is None or mouse.xdata is None:
            return
        sample = self._pick_sample(float(mouse.xdata))
        if sample is not None:
            self._dragging = sample
            self._dragged = False

    def _on_move(self, event: Event) -> None:
        mouse = cast(MouseEvent, event)
        if self._dragging is None or mouse.inaxes is None or mouse.xdata is None:
            return
        self._dragged = True
        if self._on_drag is not None:
            self._on_drag(self._dragging, float(mouse.xdata))

    def _on_release(self, event: Event) -> None:
        del event
        # Recompute the dot only on release (after the user stops moving the line),
        # and only if it actually moved.
        if self._dragging is not None and self._dragged and self._on_drop is not None:
            self._on_drop(self._dragging)
        self._dragging = None
        self._dragged = False
