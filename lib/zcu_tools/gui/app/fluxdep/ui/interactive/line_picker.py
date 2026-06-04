"""LinePickerWidget — pick the half-flux and integer-flux lines on a 2D spectrum.

Qt port of the notebook's ``InteractiveLines``: imshow the (optionally
phase-folded) real signal, draw two draggable vertical lines (half=red,
int=blue), and let the user drag them — optionally in lockstep (conjugate) —
swap them, or auto-align each to the position that minimises the mirror loss.

The numerical core (``cast2real_and_norm`` / ``diff_mirror``) is reused verbatim
from ``zcu_tools.notebook.analysis.fluxdep.processing``; only the ipywidgets +
FuncAnimation shell becomes Qt. Because the canvas is a Qt-embedded
FigureCanvasQTAgg painted on the main thread, dragging just updates the line
position and calls ``redraw()`` directly — no FuncAnimation/QTimer is needed
(see InteractiveMplWidget's note on plain ``draw_idle``).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from matplotlib.lines import Line2D
from numpy.typing import NDArray
from qtpy.QtWidgets import (  # type: ignore[attr-defined]
    QCheckBox,
    QLabel,
    QPushButton,
    QWidget,
)

from zcu_tools.notebook.analysis.fluxdep.processing import (
    cast2real_and_norm,
    diff_mirror,
)

from .base import InteractiveMplWidget


def fold_initial_lines(
    dev_values: NDArray[np.float64],
    flux_half: Optional[float],
    flux_int: Optional[float],
) -> tuple[float, float]:
    """Initial (flux_half, flux_int) positions, folded near the spectrum center.

    Pure port of InteractiveLines.__init__:40-56. Defaults: half -> spectrum
    center, int -> dev_values[-5]. When both are supplied, each is folded by an
    integer number of periods (``2 * |int - half|``) to the copy nearest the
    spectrum center.
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

    Pure port of InteractiveLines._find_best_position: candidates are spaced at
    a half-grid precision; for each, the mean of the non-zero ``diff_mirror``
    amplitudes is the loss, and the candidate with the smallest finite loss wins.
    Falls back to ``current_pos`` when no valid candidate/loss exists.
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


class LinePickerWidget(InteractiveMplWidget):
    """Drag/align the half-flux (red) and integer-flux (blue) vertical lines."""

    def __init__(
        self,
        signals: NDArray[np.complex128],
        dev_values: NDArray[np.float64],
        freqs: NDArray[np.float64],
        flux_half: Optional[float] = None,
        flux_int: Optional[float] = None,
        force_magnitude: bool = False,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._signals = signals
        self._dev_values = dev_values
        self._freqs = freqs
        # OneTone spectra are locked to magnitude-only (phase is uninformative);
        # the magnitude checkbox is hidden in that case.
        self._force_magnitude = force_magnitude

        self.flux_half, self.flux_int = fold_initial_lines(
            dev_values, flux_half, flux_int
        )

        self._only_use_magnitude = force_magnitude
        self._real_signals = cast2real_and_norm(
            signals, use_phase=not self._only_use_magnitude
        )

        # Minimum allowed gap between the two lines, and the pick threshold.
        self._min_flux_dist = 0.01 * abs(dev_values[-1] - dev_values[0])
        # Which line is currently being dragged: None / the half or int Line2D.
        self._picked: Optional[Line2D] = None

        self._build_controls()
        self._init_plots()

    # --- controls --------------------------------------------------------

    def _build_controls(self) -> None:
        self._conjugate_checkbox = QCheckBox("Conjugate Line")
        self.controls_layout.addWidget(self._conjugate_checkbox)

        # Magnitude toggle only when not forced (OneTone locks it on).
        if not self._force_magnitude:
            self._magnitude_checkbox = QCheckBox("Magnitude Only")
            self._magnitude_checkbox.toggled.connect(self._on_toggle_magnitude)
            self.controls_layout.addWidget(self._magnitude_checkbox)

        swap_button = QPushButton("Swap Lines")
        swap_button.clicked.connect(self._on_swap)
        self.controls_layout.addWidget(swap_button)

        align_button = QPushButton("Auto Align")
        align_button.clicked.connect(self._on_auto_align)
        self.controls_layout.addWidget(align_button)

        self._info = QLabel(self._info_text())
        self._info.setWordWrap(True)
        self.controls_layout.addWidget(self._info)

        self.add_finish_button()

    def _info_text(self) -> str:
        period = 2 * abs(self.flux_int - self.flux_half)
        return (
            f"half flux: {self.flux_half:.2e}\n"
            f"integer flux: {self.flux_int:.2e}\n"
            f"flux period: {period:.2e}"
        )

    def _refresh_info(self) -> None:
        self._info.setText(self._info_text())

    # --- plotting --------------------------------------------------------

    def _init_plots(self) -> None:
        # add_subplot (vs figure.subplots) gives a precise Axes type to pyright,
        # avoiding the subplots() overload's "not iterable" unpacking error.
        # Stacked vertically: the main spectrum on top, the mirror-loss view below
        # (so both share the device-value x-axis at a glance).
        self._ax_main = self.figure.add_subplot(2, 1, 1)
        self._ax_loss = self.figure.add_subplot(2, 1, 2)

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

        # Loss subplot: zoomed mirror-loss view around the active line. It is
        # refreshed on drag-release and on auto-align (not via a 500ms timer as
        # the notebook did — Qt repaint on the main thread is cheap enough).
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

        self.figure.tight_layout()
        self.redraw()

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

    def _sync_lines(self) -> None:
        """Push self.flux_half/int onto the Line2D artists + refresh info."""
        self._half_line.set_xdata([self.flux_half])
        self._int_line.set_xdata([self.flux_int])
        self._refresh_info()

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

    # --- mouse interaction ----------------------------------------------

    def on_press(self, event) -> None:  # noqa: ANN001 - MouseEvent via base
        if event.xdata is None:
            return
        # Already dragging -> drop the line.
        if self._picked is not None:
            self._picked = None
            return

        half_dist = abs(event.xdata - self.flux_half)
        int_dist = abs(event.xdata - self.flux_int)
        thresh = 3 * self._min_flux_dist
        if half_dist < int_dist and half_dist < thresh:
            self._picked = self._half_line
        elif int_dist <= half_dist and int_dist < thresh:
            self._picked = self._int_line

    def on_move(self, event) -> None:  # noqa: ANN001 - MouseEvent via base
        if self._picked is None or event.xdata is None:
            return
        x = float(event.xdata)

        picked_is_half = self._picked is self._half_line
        picked_x = self.flux_half if picked_is_half else self.flux_int
        other_x = self.flux_int if picked_is_half else self.flux_half
        # Keep the lines at least min_flux_dist apart.
        if x > other_x and x - other_x < self._min_flux_dist:
            x = other_x + self._min_flux_dist
        elif x < other_x and other_x - x < self._min_flux_dist:
            x = other_x - self._min_flux_dist

        if self._conjugate_checkbox.isChecked():
            dx = x - picked_x
            self.flux_half += dx
            self.flux_int += dx
        elif picked_is_half:
            self.flux_half = x
        else:
            self.flux_int = x

        self._sync_lines()
        self.redraw()

    def on_release(self, event) -> None:  # noqa: ANN001 - MouseEvent via base
        if self._picked is None or event.xdata is None or event.ydata is None:
            return
        x = self.flux_half if self._picked is self._half_line else self.flux_int
        self._update_loss_view(x, float(event.ydata))
        self.redraw()

    # --- button actions --------------------------------------------------

    def _on_swap(self) -> None:
        self._picked = None
        self.flux_half, self.flux_int = self.flux_int, self.flux_half
        self._sync_lines()
        self.redraw()

    def _on_auto_align(self) -> None:
        self._picked = None
        total_width = abs(self._dev_values[-1] - self._dev_values[0])
        search_width = total_width / 20
        self.flux_half = find_best_mirror_position(
            self._dev_values, self._real_signals, self.flux_half, search_width
        )
        self.flux_int = find_best_mirror_position(
            self._dev_values, self._real_signals, self.flux_int, search_width
        )
        self._sync_lines()
        center_y = 0.5 * (self._freqs[0] + self._freqs[-1])
        self._update_loss_view(self.flux_half, center_y)
        self.redraw()

    def _on_toggle_magnitude(self, checked: bool) -> None:
        self._picked = None
        self._only_use_magnitude = checked
        self._real_signals = cast2real_and_norm(
            self._signals, use_phase=not self._only_use_magnitude
        )
        self._main_im.set_data(self._real_signals.T)
        self._main_im.autoscale()
        self.redraw()

    # --- result ----------------------------------------------------------

    def get_result(self) -> tuple[float, float]:
        """Selected (flux_half, flux_int)."""
        return float(self.flux_half), float(self.flux_int)
