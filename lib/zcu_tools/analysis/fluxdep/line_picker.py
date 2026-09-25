"""Two-line interaction state for half/int flux selection."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Literal

import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from numpy.typing import NDArray

from zcu_tools.analysis.fluxdep.line_state import (
    FluxLineRole,
    FluxPickInputs,
    FluxPickState,
    align_lines,
    fold_initial_lines,
    mirror_loss_at,
    move_line,
    swap_lines,
)
from zcu_tools.analysis.fluxdep.line_state import (
    find_best_mirror_position as find_best_mirror_position,
)
from zcu_tools.analysis.fluxdep.processing import cast2real_and_norm

_LOSS_REFRESH_MS = 120


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
        self._inputs = FluxPickInputs(signals, dev_values, freqs)
        self._signals = self._inputs.signals
        self._dev_values = self._inputs.dev_values
        self._freqs = self._inputs.freqs
        half, integer = fold_initial_lines(self._dev_values, flux_half, flux_int)
        self._state = FluxPickState(
            flux_half=half, flux_int=integer, magnitude_only=force_magnitude
        )
        self._real_signals = cast2real_and_norm(
            self._signals, use_phase=not self._state.magnitude_only
        )
        self._min_flux_dist = self._inputs.min_distance
        self._picked: Line2D | None = None

        self._init_plots()

        self._loss_refresh_pending = False
        self._loss_timer = self._make_loss_timer()

    @property
    def flux_half(self) -> float:
        return self._state.flux_half

    @property
    def flux_int(self) -> float:
        return self._state.flux_int

    @property
    def magnitude_only(self) -> bool:
        return self._state.magnitude_only

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
        loss, mean_loss = mirror_loss_at(self._dev_values, self._real_signals, x)
        self._loss_im.set_data(loss.T)
        self._loss_im.autoscale()

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
        role = self.selected_role
        if role is None or xdata is None:
            return
        self._state = move_line(
            self._state, role, float(xdata), min_distance=self._min_flux_dist
        )

        self._apply_line_positions()
        self._schedule_loss_refresh()

    def on_release(self, xdata: float | None, ydata: float | None) -> None:
        if self._picked is None or xdata is None or ydata is None:
            return
        x = self.flux_half if self._picked is self._half_line else self.flux_int
        self._update_loss_view(x, float(ydata))

    def show_loss(self, role: FluxLineRole, x: float, y: float) -> None:
        """Refresh the selected line's loss after a framework commit."""
        if role not in ("half", "integer"):
            raise ValueError(f"unknown flux-line role: {role!r}")
        previous = self._picked
        self._picked = self._half_line if role == "half" else self._int_line
        try:
            self._update_loss_view(x, y)
        finally:
            self._picked = previous

    def set_conjugate(self, on: bool) -> None:
        self._state = replace(self._state, conjugate=bool(on))

    def pick_half(self) -> None:
        self._picked = self._half_line

    def pick_integer(self) -> None:
        self._picked = self._int_line

    def clear_selection(self) -> None:
        self._picked = None

    def show_state(self, state: FluxPickState) -> None:
        """Reconcile artists with committed state, abandoning any local preview."""
        if self._loss_timer is not None:
            self._loss_timer.stop()
        self._loss_refresh_pending = False
        self.clear_selection()
        if state.magnitude_only != self._state.magnitude_only:
            self._real_signals = cast2real_and_norm(
                self._signals, use_phase=not state.magnitude_only
            )
            self._main_im.set_data(self._real_signals.T)
            self._main_im.autoscale()
        self._state = state
        self._apply_line_positions()
        center_y = 0.5 * (self._freqs[0] + self._freqs[-1])
        self.show_loss("half", self.flux_half, center_y)

    def set_magnitude_only(self, on: bool) -> None:
        self.clear_selection()
        projection = cast2real_and_norm(self._signals, use_phase=not bool(on))
        self._state = replace(self._state, magnitude_only=bool(on))
        self._real_signals = projection
        self._main_im.set_data(self._real_signals.T)
        self._main_im.autoscale()

    def swap(self) -> None:
        self.clear_selection()
        self._state = swap_lines(self._state)
        self._apply_line_positions()

    def compute_aligned_positions(self) -> tuple[float, float]:
        """Mirror-loss-minimising positions without mutating artists."""

        candidate = align_lines(self._state, self._dev_values, self._real_signals)
        return candidate.flux_half, candidate.flux_int

    def apply_positions(self, flux_half: float, flux_int: float) -> None:
        self.clear_selection()
        self._state = replace(
            self._state, flux_half=float(flux_half), flux_int=float(flux_int)
        )
        self._apply_line_positions()
        center_y = 0.5 * (self._freqs[0] + self._freqs[-1])
        self._update_loss_view(self.flux_half, center_y)

    def auto_align(self) -> None:
        self.apply_positions(*self.compute_aligned_positions())
