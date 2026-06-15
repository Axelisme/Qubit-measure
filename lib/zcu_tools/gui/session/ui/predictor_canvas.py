"""PredictorCurveCanvas — interactive matplotlib canvas for the PredictorDialog.

Draws f_ij transition-frequency curves vs device value with a click-follow-click
marker. Designed for main-thread synchronous drawing only (FigureCanvasQTAgg +
draw_idle), so no cross-thread marshalling is needed.

Marker interaction is click-follow-click (not press-drag-release): a single click
near the marker engages "follow" mode in which the marker tracks the cursor's
x-position on motion WITHOUT any button held; a second click anywhere in the axes
disengages and locks the marker at its current position.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from matplotlib.backend_bases import Event
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from numpy.typing import NDArray
from qtpy.QtWidgets import QVBoxLayout, QWidget  # type: ignore[attr-defined]

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

    Follow callbacks (injected by the dialog):
    - ``on_follow(value)`` — called on each motion event while in follow mode
                             (use to update a spinbox / schedule a debounced recompute).
    - ``on_lock(value)``   — called once when follow mode disengages (second click)
                             (use to trigger a final immediate recompute).
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

        # Follow-state field: True between the engaging click and the locking click.
        self._following: bool = False

        # Injected follow callbacks; set to no-ops so we never check for None.
        self._on_follow: Callable[[float], None] = lambda _v: None
        self._on_lock: Callable[[float], None] = lambda _v: None

        # Live rendering state (populated by render_curves / set_marker).
        self._marker_line: Line2D | None = None
        self._marker_value: float | None = None
        self._flux_to_value: Callable[[float], float] | None = None
        self._value_to_flux: Callable[[float], float] | None = None
        self._flux_window: tuple[float, float] = (0.4, 1.1)
        # Curve artists keyed by (from, to) transition; populated by render_curves.
        self._curve_lines: dict[tuple[int, int], Line2D] = {}
        self._current_highlight: tuple[int, int] | None = None

        # Click-follow-click only needs press (toggle) + motion (follow); the
        # button is never held, so no release handler is wired.
        # axes_leave_event is the canonical mpl signal for cursor leaving axes bounds;
        # _on_move also covers the backstop case where inaxes/xdata is None.
        self.canvas.mpl_connect("button_press_event", self._on_press)
        self.canvas.mpl_connect("motion_notify_event", self._on_move)
        self.canvas.mpl_connect("axes_leave_event", self._on_axes_leave)
        self.canvas.mpl_connect("figure_leave_event", self._on_axes_leave)

    def bind_callbacks(
        self,
        on_follow: Callable[[float], None],
        on_lock: Callable[[float], None],
    ) -> None:
        """Inject the dialog's follow/lock handlers."""
        self._on_follow = on_follow
        self._on_lock = on_lock

    # ------------------------------------------------------------------
    # Public rendering API
    # ------------------------------------------------------------------

    def render_curves(
        self,
        *,
        values: NDArray[np.float64],
        labels: tuple[str, ...],
        series: NDArray[np.float64],
        ylabel: str,
        highlight: tuple[int, int] | None = None,
        marker_value: float,
        flux_window: tuple[float, float] = (0.4, 1.1),
        value_to_flux: Callable[[float], float],
        flux_to_value: Callable[[float], float],
    ) -> None:
        """Redraw all transition curves onto the figure.

        Decoupled from any specific result type: ``values`` is the x-axis array,
        ``labels`` are the per-curve legend labels (e.g. "0→1"), ``series`` is
        shape (n_transitions, n_values), and ``ylabel`` is the y-axis label.

        The primary x-axis is device value (no unit).  A secondary top x-axis shows
        flux (Φ/Φ₀) using the affine callables ``value_to_flux`` / ``flux_to_value``.
        ``highlight=None`` renders all curves in normal style (no curve highlighted).
        """
        self._flux_window = flux_window
        self._value_to_flux = value_to_flux
        self._flux_to_value = flux_to_value
        self._marker_value = marker_value

        self.figure.clear()
        self._curve_lines = {}
        self._current_highlight = highlight
        ax = self.figure.add_subplot(1, 1, 1)

        # Draw each transition curve; highlight the selected one (if any).
        # Store each artist in _curve_lines so set_highlight can restyle without recompute.
        highlight_label = (
            f"{highlight[0]}→{highlight[1]}" if highlight is not None else None
        )
        for i, label in enumerate(labels):
            is_hi = highlight_label is not None and label == highlight_label
            # Parse "frm→to" back to a key tuple for the artist registry.
            parts = label.split("→")
            key: tuple[int, int] = (int(parts[0]), int(parts[1]))
            (line,) = ax.plot(
                values,
                series[i],
                label=label,
                linewidth=_HIGHLIGHT_LW if is_hi else _NORMAL_LW,
                alpha=_HIGHLIGHT_ALPHA if is_hi else _NORMAL_ALPHA,
                zorder=3 if is_hi else 2,
            )
            self._curve_lines[key] = line

        # No unit on the device-value axis: the bound device is not necessarily a
        # current source. The secondary top axis below carries the reduced flux
        # Φ/Φ₀ (a genuinely different physical quantity, not a relabel).
        ax.set_xlabel("Device value")
        ax.set_ylabel(ylabel)
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
        """Move the marker line to ``value`` (device value) and pan the window if needed.

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

    def set_highlight(self, transition: tuple[int, int] | None) -> None:
        """Restyle curve artists to highlight ``transition`` without recomputing data.

        Pass ``None`` (or a transition not in the rendered set) to revert all curves
        to normal style.  Graceful: never raises if the transition is absent.
        """
        if not self._curve_lines:
            return
        self._current_highlight = transition
        for key, line in self._curve_lines.items():
            is_hi = transition is not None and key == transition
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

    @staticmethod
    def _event_xdata(event: Event) -> float | None:
        """xdata if the event is an in-axes location event, else None.

        Duck-typed (getattr) rather than isinstance so headless tests can drive
        the handlers with a lightweight fake event; matplotlib only ever passes a
        real MouseEvent here at runtime.
        """
        inaxes = getattr(event, "inaxes", None)
        xdata = getattr(event, "xdata", None)
        if inaxes is None or xdata is None:
            return None
        return float(xdata)

    def _disengage(self) -> None:
        """Disengage follow mode and fire on_lock at the last valid marker position.

        Shared by the second-click path and the auto-untrack path (cursor leaves axes).
        The marker stays at its last in-range value; no visual jump occurs.
        """
        self._following = False
        if self._marker_value is not None:
            self._on_lock(self._marker_value)

    def _on_press(self, event: Event) -> None:
        """Click toggles follow mode.

        First click (near the marker) engages follow; the next click anywhere in
        the axes disengages and locks the marker at its current position, firing
        on_lock so the dialog can do a final immediate recompute.
        """
        x = self._event_xdata(event)
        if x is None:
            return
        if self._following:
            # Second click → lock at the current marker value and disengage.
            self._disengage()
        elif self._pick_marker(x):
            # First click on the marker → engage follow mode.
            self._following = True

    def _on_move(self, event: Event) -> None:
        if not self._following:
            return
        x = self._event_xdata(event)
        if x is None:
            # Backstop: motion event with no in-axes position while following means
            # the cursor has left the display area.  Auto-untrack at the last valid
            # position so the user does not need to click again.
            self._disengage()
            return
        # Track the cursor: move the visual marker immediately (no recompute);
        # the dialog debounces the actual recompute behind on_follow.
        self.set_marker(x)
        self._on_follow(x)

    def _on_axes_leave(self, event: Event) -> None:  # noqa: ARG002
        """Auto-untrack when the cursor leaves the axes (axes_leave_event / figure_leave_event).

        This is the canonical mpl path; _on_move's inaxes-None backstop handles
        the case where mpl fires a motion event outside the axes instead of (or
        before) the leave event.
        """
        if not self._following:
            return
        self._disengage()
