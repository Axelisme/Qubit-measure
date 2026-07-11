"""FindPointsWidget — two-tone point selection via a hand-painted brush mask.

Port of the notebook's InteractiveFindPoints: an imshow spectrum + a translucent
selection-mask overlay + scatter of auto-detected peaks. Clicking paints/erases a
circular brush region of the mask; ``spectrum2d_findpoint`` (reused verbatim)
re-detects peaks within the mask on every change. Threshold / brush-width /
smoothing are Qt sliders; Select/Erase is a combo box.
"""

from __future__ import annotations

import logging

import numpy as np
from matplotlib.backend_bases import MouseEvent
from numpy.typing import NDArray
from qtpy.QtCore import (  # type: ignore[attr-defined]
    Qt,
    QTimer,
)
from qtpy.QtWidgets import (  # type: ignore[attr-defined]
    QCheckBox,
    QComboBox,
    QLabel,
    QPushButton,
    QSlider,
    QWidget,
)

from zcu_tools.analysis.fluxdep import (
    cast2real_and_norm,
    spectrum2d_findpoint,
    toggle_near_mask,
)
from zcu_tools.gui.session.adapters.qt_background import BackgroundRunner

from .base import InteractiveMplWidget
from .display import contrast_limits

logger = logging.getLogger(__name__)

_SCALE = 1000  # int-QSlider scale for float sliders


class FindPointsWidget(InteractiveMplWidget):
    """Brush-mask point selection for a two-tone flux spectrum."""

    def __init__(
        self,
        signals: NDArray[np.complex128],
        dev_values: NDArray[np.float64],
        freqs: NDArray[np.float64],
        threshold: float = 1.0,
        brush_width: float = 0.05,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._signals = signals
        self._dev_values = dev_values
        self._freqs = freqs
        self._mask = np.ones((len(dev_values), len(freqs)), dtype=bool)
        self._s_dev_values: NDArray[np.float64] = np.empty(0, dtype=np.float64)
        self._s_freqs: NDArray[np.float64] = np.empty(0, dtype=np.float64)

        # Off-main-thread point detection with instant-cancel by generation:
        # a parameter change bumps the generation and (after a short debounce)
        # launches a worker; results from an older generation are dropped.
        self._runner = BackgroundRunner(self)
        self._generation = 0
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(80)  # ms — coalesce rapid slider/brush changes
        self._debounce.timeout.connect(self._launch_worker)

        self._build_controls(threshold, brush_width)
        self._init_plots()
        self.update_points()

    # --- controls --------------------------------------------------------

    def _slider(self, lo: float, hi: float, val: float) -> QSlider:
        s = QSlider(Qt.Orientation.Horizontal)
        s.setMinimum(int(lo * _SCALE))
        s.setMaximum(int(hi * _SCALE))
        s.setValue(int(val * _SCALE))
        return s

    def _build_controls(self, threshold: float, brush_width: float) -> None:
        self.controls_layout.addWidget(QLabel("Threshold"))
        self._threshold = self._slider(1.0, 20.0, threshold)
        self._threshold.valueChanged.connect(lambda _v: self.update_points())
        self.controls_layout.addWidget(self._threshold)

        self.controls_layout.addWidget(QLabel("Brush width"))
        self._width = self._slider(0.01, 0.1, brush_width)
        self.controls_layout.addWidget(self._width)

        self.controls_layout.addWidget(QLabel("Smooth"))
        self._smooth = self._slider(0.0, 5.0, 1.0)
        self._smooth.valueChanged.connect(lambda _v: self.update_points())
        self.controls_layout.addWidget(self._smooth)

        self._operation = QComboBox()
        self._operation.addItems(["Select", "Erase"])
        self.controls_layout.addWidget(self._operation)

        self._show_mask = QCheckBox("Show mask")
        self._show_mask.stateChanged.connect(lambda _s: self._refresh_mask_overlay())
        self.controls_layout.addWidget(self._show_mask)

        self._show_origin = QCheckBox("Show origin")
        self._show_origin.setChecked(True)
        self._show_origin.stateChanged.connect(lambda _s: self.update_points())
        self.controls_layout.addWidget(self._show_origin)

        perform_all = QPushButton("Perform on all")
        perform_all.clicked.connect(self._on_perform_all)
        self.controls_layout.addWidget(perform_all)

        self.add_finish_button()

    def _threshold_val(self) -> float:
        return self._threshold.value() / _SCALE

    def _width_val(self) -> float:
        return self._width.value() / _SCALE

    def _smooth_val(self) -> float:
        return self._smooth.value() / _SCALE

    def _operation_select(self) -> bool:
        return self._operation.currentText() == "Select"

    # --- plotting --------------------------------------------------------

    def _init_plots(self) -> None:
        self._ax = self.figure.add_subplot(1, 1, 1)
        amps = cast2real_and_norm(self._signals, sigma=self._smooth_val())
        dx = (self._dev_values[-1] - self._dev_values[0]) / (len(self._dev_values) - 1)
        dy = (self._freqs[-1] - self._freqs[0]) / (len(self._freqs) - 1)
        extent = (
            self._dev_values[0] - dx / 2,
            self._dev_values[-1] + dx / 2,
            self._freqs[0] - dy / 2,
            self._freqs[-1] + dy / 2,
        )
        vmin, vmax = contrast_limits(amps)
        # gray_r: low values white, high values (the resonance line) black — so a
        # red scatter point sitting on the line has strong colour + value contrast
        # (the points typically land on the high-value feature).
        self._img = self._ax.imshow(
            amps.T,
            aspect="auto",
            origin="lower",
            interpolation="none",
            extent=extent,
            cmap="gray_r",
            vmin=vmin,
            vmax=vmax,
        )
        self._mask_img = self._ax.imshow(
            self._mask.T,
            aspect="auto",
            origin="lower",
            interpolation="none",
            extent=(
                self._dev_values[0],
                self._dev_values[-1],
                self._freqs[0],
                self._freqs[-1],
            ),
            alpha=0.0,
            cmap="gray",
            vmin=0,
            vmax=1,
        )
        self._scatter = self._ax.scatter(
            [], [], color="r", s=18, edgecolors="white", linewidths=0.3
        )
        self._ax.set_xlabel("Device value")
        self._ax.set_ylabel("Frequency (GHz)")

    def update_points(self) -> None:
        """Schedule a (debounced, off-main-thread) point re-detection.

        Bumps the generation so any in-flight worker's result is discarded, then
        (re)starts the debounce timer; the actual compute runs in _launch_worker.
        """
        self._generation += 1
        self._debounce.start()

    def _launch_worker(self) -> None:
        # Capture the generation + an immutable parameter snapshot in the closure;
        # the staleness check (cancellation by discarding stale-generation results)
        # stays in this widget, not the runner.
        generation = self._generation
        signals = self._signals
        dev_values = self._dev_values
        freqs = self._freqs
        threshold = self._threshold_val()
        smooth = self._smooth_val()
        mask = self._mask.copy()  # snapshot so later edits don't race the worker

        def _compute() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            real_signals = cast2real_and_norm(signals, sigma=smooth)
            s_dev, s_freq = spectrum2d_findpoint(
                dev_values, freqs, real_signals, threshold, weight=mask
            )
            return real_signals, s_dev, s_freq

        self._runner.submit(
            _compute,
            on_done=lambda r, g=generation: self._on_worker_done(g, *r),
            on_error=self._on_worker_error,
            run_in_pool=True,
        )

    def _on_worker_error(self, exc: Exception) -> None:
        # A point re-detection failing is non-fatal (a transient parameter combo);
        # log and leave the prior result on screen rather than crashing the picker.
        logger.exception("find-points worker failed", exc_info=exc)

    def _on_worker_done(
        self,
        generation: int,
        real_signals: np.ndarray,
        s_dev: np.ndarray,
        s_freq: np.ndarray,
    ) -> None:
        if generation != self._generation:
            return  # a newer parameter change superseded this result — drop it
        self._s_dev_values, self._s_freqs = s_dev, s_freq
        self._scatter.set_offsets(
            np.column_stack((s_dev, s_freq)) if s_dev.size else np.empty((0, 2))
        )
        shown = (
            real_signals
            if self._show_origin.isChecked()
            else (self._mask * real_signals)
        )
        self._img.set_data(shown.T)
        vmin, vmax = contrast_limits(shown)
        self._img.set_clim(vmin, vmax)
        self.redraw()

    def _refresh_mask_overlay(self) -> None:
        self._mask_img.set_data(self._mask.T)
        self._mask_img.set_alpha(0.2 if self._show_mask.isChecked() else 0.0)
        self.redraw()

    # --- interaction -----------------------------------------------------

    def on_press(self, event: MouseEvent) -> None:
        if event.xdata is None or event.ydata is None:
            return
        toggle_near_mask(
            self._dev_values,
            self._freqs,
            self._mask,
            float(event.xdata),
            float(event.ydata),
            self._width_val(),
            self._operation_select(),
        )
        self._refresh_mask_overlay()
        self.update_points()

    def _on_perform_all(self) -> None:
        self._mask = (
            np.ones_like(self._mask)
            if self._operation_select()
            else np.zeros_like(self._mask)
        )
        self._refresh_mask_overlay()
        self.update_points()

    # --- result ----------------------------------------------------------

    def quiesce(self) -> None:
        """Stop the debounce timer and join any in-flight pool worker.

        Call this before ``deleteLater()`` (e.g. from the host's ``_clear_editor``
        or ``closeEvent``) to prevent a pending ``QMetaCallEvent`` from being
        dispatched onto a freed C++ object after the widget is destroyed.
        """
        self._debounce.stop()
        self._runner.quiesce()

    def get_result(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        # Finish is the terminal action: compute the final points synchronously
        # from the current parameters (a pending worker may not have run yet), so
        # the returned result always matches what the user sees on commit.
        real_signals = cast2real_and_norm(self._signals, sigma=self._smooth_val())
        s_dev, s_freq = spectrum2d_findpoint(
            self._dev_values,
            self._freqs,
            real_signals,
            self._threshold_val(),
            weight=self._mask,
        )
        order = np.argsort(s_dev)
        return s_dev[order], s_freq[order]
