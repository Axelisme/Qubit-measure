"""TwoLinePicker — UI-toolkit-agnostic interactive half/int flux line picker.

The matplotlib core extracted from fluxdep's Qt line picker. Given a ``Figure``,
the 2D spectrum, and the device / frequency axes, it draws the spectrum plus a
mirror-loss view, two draggable vertical lines (half=red, int=blue), and owns the
drag / swap / auto-align / magnitude-toggle state.

It is toolkit-agnostic (matplotlib only — never imports Qt or ipywidgets). The
caller (fluxdep-gui's Qt ``LinePickerWidget`` or measure-gui's interactive
analysis session) owns the canvas *widget* + the surrounding controls (info
label, buttons) and drives the picker with host-fed x/y coordinates and toolbar
actions. Host-driven mutations (drag / swap / magnitude toggle) only change state
+ artists; the host repaints (it also refreshes its own info label).

The one thing the picker owns itself is the **live mirror-loss refresh**: a
backend-agnostic ``canvas.new_timer`` (the toolkit-agnostic equivalent of the
original ``InteractiveLines`` ``FuncAnimation``) throttles the ``diff_mirror``
recompute during a drag and repaints the figure via ``draw_idle`` — it fires on
the GUI main thread and the host has no hook for it, so the picker drives it.
Heavy *off-main* work (auto-align) stays a pure ``compute_*`` + a main-thread
``apply_*`` so the caller may run it off the main thread; the picker never spawns
threads itself. The notebook ``InteractiveLines`` (FuncAnimation/ipywidgets)
keeps its own separate shell.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from numpy.typing import NDArray

from zcu_tools.notebook.analysis.fluxdep.processing import (
    cast2real_and_norm,
    diff_mirror,
)

# Throttle interval (ms) for the live mirror-loss refresh during a drag: the
# timer fires at most this often at the latest line position, so diff_mirror is
# not recomputed on every motion event but the view still follows the line.
_LOSS_REFRESH_MS = 120


def fold_initial_lines(
    dev_values: NDArray[np.float64],
    flux_half: float | None,
    flux_int: float | None,
) -> tuple[float, float]:
    """Initial (flux_half, flux_int) positions, folded near the spectrum center.

    Defaults: half -> spectrum center, int -> dev_values[-5]. When both are
    supplied, each is folded by an integer number of periods (``2 * |int - half|``)
    to the copy nearest the spectrum center.
    """
    flux_center = (dev_values[0] + dev_values[-1]) / 2
    half = flux_center if flux_half is None else flux_half
    intg = dev_values[-5] if flux_int is None else flux_int

    if flux_half is not None and flux_int is not None:
        fix_period = 2 * abs(intg - half)
        if fix_period != 0.0:
            half = half - round((half - flux_center) / fix_period) * fix_period
            intg = intg - round((intg - flux_center) / fix_period) * fix_period

    return float(half), float(intg)


def find_best_mirror_position(
    dev_values: NDArray[np.float64],
    real_signals: NDArray[np.float64],
    current_pos: float,
    search_width: float,
) -> float:
    """Position with minimal mean mirror loss within ``current_pos ± width/2``.

    Candidates are spaced at a half-grid precision; for each, the mean of the
    non-zero ``diff_mirror`` amplitudes is the loss, and the candidate with the
    smallest finite loss wins. Falls back to ``current_pos`` when no valid
    candidate / loss exists. Pure (numpy only) — safe to run off the main thread.
    """
    lo = float(dev_values.min())
    hi = float(dev_values.max())
    precision = 0.25 * (hi - lo) / len(dev_values)
    if precision <= 0.0:
        return current_pos

    left_bound = max(lo, current_pos - search_width / 2)
    right_bound = min(hi, current_pos + search_width / 2)
    left_steps = int(np.floor((left_bound - lo) / precision))
    right_steps = int(np.ceil((right_bound - lo) / precision))

    candidates = [
        lo + i * precision
        for i in range(left_steps, right_steps + 1)
        if lo <= lo + i * precision <= hi
    ]
    if not candidates:
        return current_pos

    best_pos = current_pos
    min_loss = float("inf")
    for candidate in candidates:
        diff_amps = diff_mirror(dev_values, real_signals, candidate)
        valid_amps = diff_amps[diff_amps != 0.0]
        if len(valid_amps) == 0:
            continue
        loss = float(np.mean(valid_amps))
        if not np.isnan(loss) and loss < min_loss:
            min_loss = loss
            best_pos = candidate

    return best_pos


class TwoLinePicker:
    """Draw + drive two draggable half/int flux lines on a caller-provided Figure.

    The caller owns the ``Figure`` widget; this core lays out a main spectrum
    subplot + a mirror-loss subplot on it and manages the line state. Host-driven
    methods (drag / swap / magnitude / apply) only **mutate** state + artists and
    leave the repaint to the caller; the picker self-repaints only for its own
    throttled live mirror-loss timer (``_on_loss_timer`` → ``draw_idle``). Heavy
    work (auto-align) is a pure ``compute_*`` + a main-thread ``apply_*`` so the
    caller may run the compute off the main thread.
    """

    def __init__(
        self,
        figure: Figure,
        signals: NDArray[np.complex128],
        dev_values: NDArray[np.float64],
        freqs: NDArray[np.float64],
        *,
        flux_half: float | None = None,
        flux_int: float | None = None,
        force_magnitude: bool = False,
    ) -> None:
        self._figure = figure
        self._signals = signals
        self._dev_values = dev_values
        self._freqs = freqs
        # OneTone spectra are locked to magnitude-only (phase is uninformative).
        self._force_magnitude = force_magnitude

        self.flux_half, self.flux_int = fold_initial_lines(
            dev_values, flux_half, flux_int
        )
        self._only_use_magnitude = force_magnitude
        self._conjugate = False
        self._real_signals = cast2real_and_norm(
            signals, use_phase=not self._only_use_magnitude
        )
        # Minimum allowed gap between the two lines, and the pick threshold.
        self._min_flux_dist = 0.01 * abs(dev_values[-1] - dev_values[0])
        # Which line is currently being dragged: None / the half or int Line2D.
        self._picked: Line2D | None = None

        self._init_plots()

        # Live mirror-loss refresh timer (toolkit-agnostic; the picker owns it
        # because it fires on the GUI loop and the host has no hook). ``None`` on
        # a headless / non-interactive backend (unit tests) — then the loss view
        # simply refreshes on drag-release only, as before.
        self._loss_refresh_pending = False
        self._loss_timer = self._make_loss_timer()

    # --- queries ---------------------------------------------------------

    @property
    def magnitude_only(self) -> bool:
        return self._only_use_magnitude

    def positions(self) -> tuple[float, float]:
        """Selected (flux_half, flux_int)."""
        return float(self.flux_half), float(self.flux_int)

    def period(self) -> float:
        return 2 * abs(self.flux_int - self.flux_half)

    def info_text(self) -> str:
        return (
            f"half flux: {self.flux_half:.2e}\n"
            f"integer flux: {self.flux_int:.2e}\n"
            f"flux period: {self.period():.2e}"
        )

    # --- plotting (state only; caller repaints) --------------------------

    def _init_plots(self) -> None:
        # add_subplot (vs figure.subplots) gives a precise Axes type to pyright.
        # Stacked vertically: the main spectrum on top, the mirror-loss view below
        # (so both share the device-value x-axis at a glance).
        self._ax_main = self._figure.add_subplot(2, 1, 1)
        self._ax_loss = self._figure.add_subplot(2, 1, 2)

        self._main_im = self._ax_main.imshow(
            self._real_signals.T,
            aspect="auto",
            origin="lower",
            interpolation="none",
            extent=self._extent(),
        )
        self._ax_main.set_xlim(self._dev_values[0], self._dev_values[-1])
        self._ax_main.set_ylim(self._freqs[0], self._freqs[-1])
        self._ax_main.set_xlabel("Device value")
        self._ax_main.set_ylabel("Frequency (GHz)")

        self._half_line = self._ax_main.axvline(
            x=self.flux_half, color="r", linestyle="--"
        )
        self._int_line = self._ax_main.axvline(
            x=self.flux_int, color="b", linestyle="--"
        )

        # Loss subplot: zoomed mirror-loss view around the active line, refreshed
        # on drag-release and on auto-align.
        self._loss_im = self._ax_loss.imshow(
            self._real_signals.T,
            aspect="auto",
            origin="lower",
            interpolation="none",
            extent=self._extent(),
        )
        self._ax_loss.set_xticks([])
        self._ax_loss.set_yticks([])
        self._ax_loss.set_title("mirror loss: -")
        center_y = 0.5 * (self._freqs[0] + self._freqs[-1])
        self._loss_dot = self._ax_loss.plot([self.flux_half], [center_y], "ro")[0]

        self._figure.tight_layout()

    def _extent(self) -> tuple[float, float, float, float]:
        dev = self._dev_values
        freqs = self._freqs
        dx = (dev[-1] - dev[0]) / (len(dev) - 1)
        dy = (freqs[-1] - freqs[0]) / (len(freqs) - 1)
        return (
            float(dev[0] - dx / 2),
            float(dev[-1] + dx / 2),
            float(freqs[0] - dy / 2),
            float(freqs[-1] + dy / 2),
        )

    def _apply_line_positions(self) -> None:
        """Push self.flux_half/int onto the Line2D artists."""
        self._half_line.set_xdata([self.flux_half])
        self._int_line.set_xdata([self.flux_int])

    def _update_loss_view(self, x: float, y: float) -> None:
        """Zoom the loss subplot around (x, y) and recolor the marker."""
        loss = diff_mirror(self._dev_values, self._real_signals, x)
        self._loss_im.set_data(loss.T)
        self._loss_im.autoscale()
        valid = loss[loss != 0.0]
        mean_loss = float(np.mean(valid)) if len(valid) > 0 else float("nan")

        dev = self._dev_values
        freqs = self._freqs
        dx = 0.3 * abs(dev[-1] - dev[0])
        dy = 0.3 * abs(freqs[-1] - freqs[0])
        self._ax_loss.set_xlim(x - dx, x + dx)
        self._ax_loss.set_ylim(y - dy, y + dy)
        self._ax_loss.set_title(f"mirror loss: {mean_loss:.4f}")
        self._loss_dot.set_xdata([x])
        self._loss_dot.set_ydata([y])
        self._loss_dot.set_color("r" if self._picked is self._half_line else "b")

    # --- live mirror-loss refresh (picker-owned, backend-agnostic timer) --

    def _make_loss_timer(self) -> Any:
        """A single-shot backend-agnostic timer for the throttled loss refresh,
        or ``None`` if the figure has no interactive canvas (headless tests)."""
        canvas = getattr(self._figure, "canvas", None)
        if canvas is None or not hasattr(canvas, "new_timer"):
            return None
        try:
            timer = canvas.new_timer(interval=_LOSS_REFRESH_MS)
        except (NotImplementedError, AttributeError):
            return None
        timer.single_shot = True
        timer.add_callback(self._on_loss_timer)
        return timer

    def _schedule_loss_refresh(self) -> None:
        """Throttle: arm the timer on the first drag motion; it fires once at the
        latest line position, then re-arms on the next motion (so a continuous
        drag updates at most every ``_LOSS_REFRESH_MS``). No-op when no line is
        picked or the backend has no timer."""
        if (
            self._picked is None
            or self._loss_timer is None
            or self._loss_refresh_pending
        ):
            return
        self._loss_refresh_pending = True
        self._loss_timer.start()

    def _on_loss_timer(self) -> None:
        self._loss_refresh_pending = False
        if self._picked is None:
            return
        # Recompute diff_mirror at the picked line's CURRENT position (always the
        # latest); keep the loss view's vertical zoom centre where it is.
        x = self.flux_half if self._picked is self._half_line else self.flux_int
        y0, y1 = self._ax_loss.get_ylim()
        self._update_loss_view(x, 0.5 * (y0 + y1))
        self._figure.canvas.draw_idle()

    # --- mouse interaction (caller feeds coordinates from its canvas) -----

    def on_press(self, xdata: float | None) -> None:
        if xdata is None:
            return
        # Already dragging -> drop the line.
        if self._picked is not None:
            self._picked = None
            return

        half_dist = abs(xdata - self.flux_half)
        int_dist = abs(xdata - self.flux_int)
        thresh = 3 * self._min_flux_dist
        if half_dist < int_dist and half_dist < thresh:
            self._picked = self._half_line
        elif int_dist <= half_dist and int_dist < thresh:
            self._picked = self._int_line

    def on_move(self, xdata: float | None) -> None:
        if self._picked is None or xdata is None:
            return
        x = float(xdata)

        picked_is_half = self._picked is self._half_line
        picked_x = self.flux_half if picked_is_half else self.flux_int
        other_x = self.flux_int if picked_is_half else self.flux_half
        # Keep the lines at least min_flux_dist apart.
        if x > other_x and x - other_x < self._min_flux_dist:
            x = other_x + self._min_flux_dist
        elif x < other_x and other_x - x < self._min_flux_dist:
            x = other_x - self._min_flux_dist

        if self._conjugate:
            dx = x - picked_x
            self.flux_half += dx
            self.flux_int += dx
        elif picked_is_half:
            self.flux_half = x
        else:
            self.flux_int = x

        self._apply_line_positions()
        # Keep the mirror-loss view following the dragged line (throttled).
        self._schedule_loss_refresh()

    def on_release(self, xdata: float | None, ydata: float | None) -> None:
        if self._picked is None or xdata is None or ydata is None:
            return
        x = self.flux_half if self._picked is self._half_line else self.flux_int
        self._update_loss_view(x, float(ydata))

    # --- toolbar actions (caller wires its buttons/checkboxes to these) ---

    def set_conjugate(self, on: bool) -> None:
        self._conjugate = bool(on)

    def set_magnitude_only(self, on: bool) -> None:
        self._picked = None
        self._only_use_magnitude = bool(on)
        self._real_signals = cast2real_and_norm(
            self._signals, use_phase=not self._only_use_magnitude
        )
        self._main_im.set_data(self._real_signals.T)
        self._main_im.autoscale()

    def swap(self) -> None:
        self._picked = None
        self.flux_half, self.flux_int = self.flux_int, self.flux_half
        self._apply_line_positions()

    def compute_aligned_positions(self) -> tuple[float, float]:
        """The mirror-loss-minimising (half, int) positions. HEAVY but PURE — it
        only reads the data, so the caller may run it off the main thread."""
        total_width = abs(self._dev_values[-1] - self._dev_values[0])
        search_width = total_width / 20
        half = find_best_mirror_position(
            self._dev_values, self._real_signals, self.flux_half, search_width
        )
        intg = find_best_mirror_position(
            self._dev_values, self._real_signals, self.flux_int, search_width
        )
        return half, intg

    def apply_positions(self, flux_half: float, flux_int: float) -> None:
        """Set both lines (+ loss view) to the given positions. MAIN-THREAD —
        mutates matplotlib artists; the caller repaints afterwards."""
        self._picked = None
        self.flux_half, self.flux_int = float(flux_half), float(flux_int)
        self._apply_line_positions()
        center_y = 0.5 * (self._freqs[0] + self._freqs[-1])
        self._update_loss_view(self.flux_half, center_y)

    def auto_align(self) -> None:
        """Synchronous convenience: compute + apply on the calling thread."""
        self.apply_positions(*self.compute_aligned_positions())
