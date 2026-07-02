"""Two-line interaction state for half/int flux selection."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from numpy.typing import NDArray

from zcu_tools.analysis.fluxdep.processing import (
    cast2real_and_norm,
    diff_mirror,
)

_LOSS_REFRESH_MS = 120


def _mirror_inbounds_mask(
    dev_values: NDArray[np.float64], center: float
) -> NDArray[np.bool_]:
    """Rows whose mirror counterpart stays inside the device-value axis."""

    n = len(dev_values)
    c_idx = (n - 1) * (center - dev_values[0]) / (dev_values[-1] - dev_values[0])
    idxs = np.arange(n)
    mirror_idxs = np.round(2 * c_idx - idxs).astype(int)
    return (mirror_idxs >= 0) & (mirror_idxs < n)


def fold_initial_lines(
    dev_values: NDArray[np.float64],
    flux_half: float | None,
    flux_int: float | None,
) -> tuple[float, float]:
    """Initial ``(flux_half, flux_int)`` folded near the spectrum center."""

    if dev_values.ndim != 1:
        raise ValueError("dev_values must be a 1D axis")
    if dev_values.size < 5:
        raise ValueError("dev_values must contain at least five values")

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
    """Position with minimal mean mirror loss within ``current_pos +/- width/2``."""

    if dev_values.ndim != 1:
        raise ValueError("dev_values must be a 1D axis")
    if real_signals.shape[0] != dev_values.size:
        raise ValueError("real_signals first dimension must match len(dev_values)")
    if search_width < 0:
        raise ValueError("search_width must be non-negative")

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
        # ADR-0028: this pure loss rule is shared by notebook and GUI adapters.
        # In-bounds rows distinguish real zero loss from out-of-bounds zero fill.
        inbounds = _mirror_inbounds_mask(dev_values, candidate)
        if not inbounds.any():
            continue
        loss = float(np.mean(diff_amps[inbounds]))
        if not np.isnan(loss) and loss < min_loss:
            min_loss = loss
            best_pos = candidate

    return best_pos


class TwoLinePicker:
    """Matplotlib-backed state machine for half/int flux line selection."""

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
        self._force_magnitude = force_magnitude

        self.flux_half, self.flux_int = fold_initial_lines(
            dev_values, flux_half, flux_int
        )
        self._only_use_magnitude = force_magnitude
        self._conjugate = False
        self._real_signals = cast2real_and_norm(
            signals, use_phase=not self._only_use_magnitude
        )
        self._min_flux_dist = 0.01 * abs(dev_values[-1] - dev_values[0])
        self._picked: Line2D | None = None

        self._init_plots()

        self._loss_refresh_pending = False
        self._loss_timer = self._make_loss_timer()

    @property
    def magnitude_only(self) -> bool:
        return self._only_use_magnitude

    @property
    def selected_role(self) -> Literal["half", "integer"] | None:
        if self._picked is self._half_line:
            return "half"
        if self._picked is self._int_line:
            return "integer"
        return None

    def is_main_axes(self, axes: Any) -> bool:
        return axes is self._ax_main

    def positions(self) -> tuple[float, float]:
        """Selected ``(flux_half, flux_int)``."""

        return float(self.flux_half), float(self.flux_int)

    def period(self) -> float:
        return 2 * abs(self.flux_int - self.flux_half)

    def info_text(self) -> str:
        return (
            f"half flux: {self.flux_half:.2e}\n"
            f"integer flux: {self.flux_int:.2e}\n"
            f"flux period: {self.period():.2e}"
        )

    def _init_plots(self) -> None:
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
        self._half_line.set_xdata([self.flux_half])
        self._int_line.set_xdata([self.flux_int])

    def _update_loss_view(self, x: float, y: float) -> None:
        loss = diff_mirror(self._dev_values, self._real_signals, x)
        self._loss_im.set_data(loss.T)
        self._loss_im.autoscale()
        inbounds = _mirror_inbounds_mask(self._dev_values, x)
        mean_loss = float(np.mean(loss[inbounds])) if inbounds.any() else float("nan")

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

    def _make_loss_timer(self) -> Any:
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
        x = self.flux_half if self._picked is self._half_line else self.flux_int
        y0, y1 = self._ax_loss.get_ylim()
        self._update_loss_view(x, 0.5 * (y0 + y1))
        self._figure.canvas.draw_idle()

    def on_press(self, xdata: float | None) -> None:
        if xdata is None:
            return
        if self._picked is not None:
            self.clear_selection()
            return

        half_dist = abs(xdata - self.flux_half)
        int_dist = abs(xdata - self.flux_int)
        thresh = 3 * self._min_flux_dist
        if half_dist < int_dist and half_dist < thresh:
            self.pick_half()
        elif int_dist <= half_dist and int_dist < thresh:
            self.pick_integer()

    def on_move(self, xdata: float | None) -> None:
        if self._picked is None or xdata is None:
            return
        x = float(xdata)

        picked_is_half = self._picked is self._half_line
        picked_x = self.flux_half if picked_is_half else self.flux_int
        other_x = self.flux_int if picked_is_half else self.flux_half
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
        self._schedule_loss_refresh()

    def on_release(self, xdata: float | None, ydata: float | None) -> None:
        if self._picked is None or xdata is None or ydata is None:
            return
        x = self.flux_half if self._picked is self._half_line else self.flux_int
        self._update_loss_view(x, float(ydata))

    def set_conjugate(self, on: bool) -> None:
        self._conjugate = bool(on)

    def pick_half(self) -> None:
        self._picked = self._half_line

    def pick_integer(self) -> None:
        self._picked = self._int_line

    def clear_selection(self) -> None:
        self._picked = None

    def set_magnitude_only(self, on: bool) -> None:
        self.clear_selection()
        self._only_use_magnitude = bool(on)
        self._real_signals = cast2real_and_norm(
            self._signals, use_phase=not self._only_use_magnitude
        )
        self._main_im.set_data(self._real_signals.T)
        self._main_im.autoscale()

    def swap(self) -> None:
        self.clear_selection()
        self.flux_half, self.flux_int = self.flux_int, self.flux_half
        self._apply_line_positions()

    def compute_aligned_positions(self) -> tuple[float, float]:
        """Mirror-loss-minimising positions without mutating artists."""

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
        self.clear_selection()
        self.flux_half, self.flux_int = float(flux_half), float(flux_int)
        self._apply_line_positions()
        center_y = 0.5 * (self._freqs[0] + self._freqs[-1])
        self._update_loss_view(self.flux_half, center_y)

    def auto_align(self) -> None:
        self.apply_positions(*self.compute_aligned_positions())
