"""PredictorCurveCanvas — interactive matplotlib canvas for the PredictorDialog.

Draws f_ij transition-frequency curves vs device value (A) with a draggable
flux-position marker. Designed for main-thread synchronous drawing only
(FigureCanvasQTAgg + draw_idle), so no cross-thread marshalling is needed.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from matplotlib.backend_bases import Event, MouseEvent
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from qtpy.QtWidgets import QVBoxLayout, QWidget  # type: ignore[attr-defined]

if TYPE_CHECKING:
    from zcu_tools.gui.session.services.connection import PredictCurveResult

# How close (in axes-width fraction) a click must be to the marker line to grab it.
_PICK_TOL_FRAC = 0.02

# Highlight style for the selected transition curve.
_HIGHLIGHT_LW = 2.5
_NORMAL_LW = 1.0
_HIGHLIGHT_ALPHA = 1.0
_NORMAL_ALPHA = 0.6


def _compute_xlim(
    flux_window: tuple[float, float],
    marker_value: float,
    flux_to_value: Callable[[float], float],
) -> tuple[float, float]:
    """Return the (xlo, xhi) in device-value space that covers flux_window and marker.

    Window width is preserved.  If marker_value falls outside the translated
    window, the window is shifted (clamped to include marker) while keeping its
    width constant.
    """
    v_lo = flux_to_value(flux_window[0])
    v_hi = flux_to_value(flux_window[1])
    # Ensure v_lo < v_hi regardless of monotonicity direction.
    if v_lo > v_hi:
        v_lo, v_hi = v_hi, v_lo
    width = v_hi - v_lo

    if marker_value < v_lo:
        # Shift window left so marker sits at the left edge.
        v_lo = marker_value
        v_hi = v_lo + width
    elif marker_value > v_hi:
        # Shift window right so marker sits at the right edge.
        v_hi = marker_value
        v_lo = v_hi - width

    return v_lo, v_hi


class PredictorCurveCanvas(QWidget):
    """QWidget wrapping a matplotlib Figure that shows f_ij curves + marker line.

    Usage:
    1. Call ``render_curves(...)`` once after a new PredictCurveResult is ready.
    2. Call ``set_marker(value)`` to reposition the marker without recomputing curves.
    3. Call ``clear()`` to blank the canvas (predictor cleared).

    Drag callbacks (injected by the dialog):
    - ``on_drag(value)``  — called on each motion event while dragging the marker
                            (use to update a spinbox without triggering a recompute).
    - ``on_drop(value)``  — called once on button-release after an actual drag
                            (use to trigger the single-point prediction label).
    """

    def __init__(
        self,
        figsize: tuple[float, float] = (6.0, 4.0),
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.figure = Figure(figsize=figsize, tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)

        # Drag-state fields.
        self._dragging: bool = False
        self._dragged: bool = False  # did the mouse actually move while held?

        # Injected drag callbacks; set to no-ops so we never check for None.
        self._on_drag: Callable[[float], None] = lambda _v: None
        self._on_drop: Callable[[float], None] = lambda _v: None

        # Live rendering state (populated by render_curves / set_marker).
        self._marker_line: Line2D | None = None
        self._marker_value: float | None = None
        self._flux_to_value: Callable[[float], float] | None = None
        self._value_to_flux: Callable[[float], float] | None = None
        self._flux_window: tuple[float, float] = (0.4, 1.1)
        # Curve artists keyed by (from, to) transition; populated by render_curves.
        self._curve_lines: dict[tuple[int, int], Line2D] = {}
        self._current_highlight: tuple[int, int] | None = None

        self.canvas.mpl_connect("button_press_event", self._on_press)
        self.canvas.mpl_connect("motion_notify_event", self._on_move)
        self.canvas.mpl_connect("button_release_event", self._on_release)

    def bind_callbacks(
        self,
        on_drag: Callable[[float], None],
        on_drop: Callable[[float], None],
    ) -> None:
        """Inject the dialog's drag/drop handlers."""
        self._on_drag = on_drag
        self._on_drop = on_drop

    # ------------------------------------------------------------------
    # Public rendering API
    # ------------------------------------------------------------------

    def render_curves(
        self,
        result: PredictCurveResult,
        *,
        highlight: tuple[int, int],
        marker_value: float,
        flux_window: tuple[float, float] = (0.4, 1.1),
        value_to_flux: Callable[[float], float],
        flux_to_value: Callable[[float], float],
    ) -> None:
        """Redraw all transition curves onto the figure.

        The primary x-axis is device value (A).  A secondary top x-axis shows
        flux (Φ/Φ₀) using the affine callables ``value_to_flux`` / ``flux_to_value``.
        """
        self._flux_window = flux_window
        self._value_to_flux = value_to_flux
        self._flux_to_value = flux_to_value
        self._marker_value = marker_value

        self.figure.clear()
        self._curve_lines = {}
        self._current_highlight = highlight
        ax = self.figure.add_subplot(1, 1, 1)

        # Draw each transition curve; highlight the selected one.
        # Store each artist in _curve_lines so set_highlight can restyle without recompute.
        highlight_label = f"{highlight[0]}→{highlight[1]}"
        for i, label in enumerate(result.labels):
            is_hi = label == highlight_label
            # Parse "frm→to" back to a key tuple for the artist registry.
            parts = label.split("→")
            key: tuple[int, int] = (int(parts[0]), int(parts[1]))
            (line,) = ax.plot(
                result.values,
                result.freqs_mhz[i],
                label=label,
                linewidth=_HIGHLIGHT_LW if is_hi else _NORMAL_LW,
                alpha=_HIGHLIGHT_ALPHA if is_hi else _NORMAL_ALPHA,
                zorder=3 if is_hi else 2,
            )
            self._curve_lines[key] = line

        ax.set_xlabel("Device value (A)")
        ax.set_ylabel("Frequency (MHz)")
        ax.legend(fontsize=8, loc="best")

        # Secondary top x-axis: flux Φ/Φ₀ (monotone affine, so both callables
        # are safe for secondary_xaxis which expects a pair of inverse functions).
        # The matplotlib stubs type functions as (ArrayLike)->ArrayLike, but our
        # scalar callables also satisfy that at runtime (numpy passes floats and
        # numpy arrays alike). Suppress the stub mismatch here.
        ax.secondary_xaxis(
            "top",
            functions=(value_to_flux, flux_to_value),  # type: ignore[arg-type]
        ).set_xlabel("Flux (Φ/Φ₀)")

        # Set x display window (shift if marker is outside).
        x_lo, x_hi = _compute_xlim(flux_window, marker_value, flux_to_value)
        ax.set_xlim(x_lo, x_hi)

        # Draw the marker vertical line.
        self._marker_line = ax.axvline(
            marker_value,
            color="red",
            linewidth=1.5,
            linestyle="--",
            zorder=4,
            label="_marker",
        )

        self.canvas.draw_idle()

    def set_marker(self, value: float) -> None:
        """Move the marker line to ``value`` (device A) and pan the window if needed.

        Does NOT trigger a curve recompute.
        """
        self._marker_value = value

        ax = self._get_ax()
        if ax is None or self._flux_to_value is None:
            return

        # Pan window if needed.
        x_lo, x_hi = _compute_xlim(self._flux_window, value, self._flux_to_value)
        ax.set_xlim(x_lo, x_hi)

        if self._marker_line is not None:
            self._marker_line.set_xdata([value, value])

        self.canvas.draw_idle()

    def clear(self) -> None:
        """Blank the canvas (called when the predictor is cleared)."""
        self.figure.clear()
        self._marker_line = None
        self._marker_value = None
        self._flux_to_value = None
        self._value_to_flux = None
        self._curve_lines = {}
        self._current_highlight = None
        self.canvas.draw_idle()

    def set_highlight(self, transition: tuple[int, int]) -> None:
        """Restyle curve artists to highlight ``transition`` without recomputing data.

        If ``transition`` is not among the rendered curves (e.g. an arbitrary
        (from, to) pair not in _DEFAULT_TRANSITIONS), all curves revert to the
        normal (non-highlighted) style instead of raising.
        """
        if not self._curve_lines:
            return
        self._current_highlight = transition
        for key, line in self._curve_lines.items():
            is_hi = key == transition
            line.set_linewidth(_HIGHLIGHT_LW if is_hi else _NORMAL_LW)
            line.set_alpha(_HIGHLIGHT_ALPHA if is_hi else _NORMAL_ALPHA)
            line.set_zorder(3 if is_hi else 2)
        self.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_ax(self):  # type: ignore[return]
        """Return the primary axes if the figure has one, else None."""
        axes = self.figure.get_axes()
        return axes[0] if axes else None

    def _pick_marker(self, x_data: float) -> bool:
        """True if ``x_data`` is within the drag-tolerance of the current marker."""
        if self._marker_value is None:
            return False
        ax = self._get_ax()
        if ax is None:
            return False
        x_lo, x_hi = ax.get_xlim()
        tol = abs(x_hi - x_lo) * _PICK_TOL_FRAC
        return abs(x_data - self._marker_value) <= tol

    def _on_press(self, event: Event) -> None:
        mouse = event  # type: MouseEvent  # type: ignore[assignment]
        if not hasattr(mouse, "inaxes") or mouse.inaxes is None or mouse.xdata is None:
            return
        if self._pick_marker(float(mouse.xdata)):
            self._dragging = True
            self._dragged = False

    def _on_move(self, event: Event) -> None:
        mouse = event  # type: MouseEvent  # type: ignore[assignment]
        if (
            not self._dragging
            or not hasattr(mouse, "inaxes")
            or mouse.inaxes is None
            or mouse.xdata is None
        ):
            return
        self._dragged = True
        value = float(mouse.xdata)
        # Move the visual marker immediately (no recompute).
        self.set_marker(value)
        self._on_drag(value)

    def _on_release(self, event: Event) -> None:
        del event
        if self._dragging and self._dragged and self._marker_value is not None:
            self._on_drop(self._marker_value)
        self._dragging = False
        self._dragged = False
