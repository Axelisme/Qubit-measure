"""OneToneWidget — one-tone point selection by threshold (no mouse interaction).

The simplest interactive tool, used to validate the InteractiveMplWidget base:
a threshold slider drives automatic peak detection on the most-dispersive
frequency slice. The numerical core is reused verbatim from the notebook's
InteractiveOneTone (gradient → max-dispersion frequency → smoothed slice →
scipy find_peaks); only the ipywidgets shell becomes Qt.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray
from qtpy.QtCore import Qt  # type: ignore[attr-defined]
from qtpy.QtWidgets import QLabel, QSlider, QWidget  # type: ignore[attr-defined]
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from zcu_tools.fluxdep_gui.ui.interactive.base import InteractiveMplWidget

# FloatSlider(0..5, step 0.01) emulated on an int QSlider scaled by 100.
_THRESHOLD_SCALE = 100
_THRESHOLD_MAX = 5.0


def max_dispersion_freq_index(
    signals: NDArray[np.complex128], freqs: NDArray[np.float64]
) -> int:
    """Index of the frequency with the largest mean relative dispersion.

    Pure port of InteractiveOneTone.init_plots: the relative gradient of the
    complex signal along frequency, gaussian-smoothed, averaged over device
    values, argmax'd.
    """
    abs_grad = (
        np.abs(signals[:, 1:] - signals[:, :-1]) / ((freqs[1:] - freqs[:-1])[None])
    )
    rel_grad = abs_grad / np.clip(np.abs(signals[:, 1:] + signals[:, :-1]), 1e-12, None)
    rel_grad = gaussian_filter1d(rel_grad, sigma=1, axis=1)
    return int(np.argmax(np.mean(rel_grad, axis=0)))


def smoothed_slice(
    signals: NDArray[np.complex128], freq_idx: int
) -> NDArray[np.float64]:
    """The normalised, inverted, smoothed amplitude slice at ``freq_idx``."""
    real_slice = np.abs(signals)[:, freq_idx]
    smoothed = gaussian_filter1d(np.max(real_slice) - real_slice, sigma=1)
    std = np.std(smoothed)
    if std == 0.0:
        # A perfectly flat slice (no dispersion at this frequency) — normalising
        # would divide by zero; return the flat (all-zero) slice so peak finding
        # simply yields nothing rather than NaNs.
        return np.zeros_like(smoothed)
    return smoothed / std


def detect_peaks(smoothed: NDArray[np.float64], threshold: float) -> NDArray[np.intp]:
    """Peak indices of ``smoothed`` with prominence ≥ ``threshold``."""
    peaks, _ = find_peaks(smoothed, prominence=threshold)
    return peaks


class OneToneWidget(InteractiveMplWidget):
    """Threshold-driven peak picking on a one-tone flux spectrum."""

    def __init__(
        self,
        signals: NDArray[np.complex128],
        dev_values: NDArray[np.float64],
        freqs: NDArray[np.float64],
        threshold: float = 1.0,
        flux_half: Optional[float] = None,
        flux_int: Optional[float] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._signals = signals
        self._dev_values = dev_values
        self._freqs = freqs
        self._flux_half = flux_half
        self._flux_int = flux_int

        self._max_freq_idx = max_dispersion_freq_index(signals, freqs)
        self._smoothed = smoothed_slice(signals, self._max_freq_idx)
        self._s_dev_values: NDArray[np.float64] = np.empty(0, dtype=np.float64)
        self._s_freqs: NDArray[np.float64] = np.empty(0, dtype=np.float64)

        self._build_controls(threshold)
        self._init_plots()
        self.update_peaks(threshold)

    # --- controls --------------------------------------------------------

    def _build_controls(self, threshold: float) -> None:
        self.controls_layout.addWidget(QLabel("Threshold"))
        self._threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self._threshold_slider.setMinimum(0)
        self._threshold_slider.setMaximum(int(_THRESHOLD_MAX * _THRESHOLD_SCALE))
        self._threshold_slider.setValue(int(threshold * _THRESHOLD_SCALE))
        self._threshold_slider.valueChanged.connect(self._on_threshold_change)
        self.controls_layout.addWidget(self._threshold_slider)
        self.add_finish_button()

    def _threshold(self) -> float:
        return self._threshold_slider.value() / _THRESHOLD_SCALE

    # --- plotting --------------------------------------------------------

    def _init_plots(self) -> None:
        # add_subplot (vs figure.subplots) gives a precise Axes type to pyright,
        # avoiding the subplots() overload's "not iterable" unpacking error.
        self._ax_img = self.figure.add_subplot(2, 1, 1)
        self._ax_curve = self.figure.add_subplot(2, 1, 2)
        real_signals = np.abs(self._signals)
        self._ax_img.imshow(
            real_signals.T,
            aspect="auto",
            origin="lower",
            extent=(
                self._dev_values[0],
                self._dev_values[-1],
                self._freqs[0],
                self._freqs[-1],
            ),
        )
        self._ax_img.axhline(
            self._freqs[self._max_freq_idx], color="red", label="max freqs"
        )
        # Half-flux (red) / integer-flux (blue) vertical markers from the
        # line-picker, on both panels (read-only reference for point selection).
        for ax in (self._ax_img, self._ax_curve):
            if self._flux_half is not None:
                ax.axvline(self._flux_half, color="red", linestyle="--", linewidth=1)
            if self._flux_int is not None:
                ax.axvline(self._flux_int, color="blue", linestyle="--", linewidth=1)
        self._ax_curve.plot(self._dev_values, self._smoothed)
        self._ax_curve.set_xlim(self._dev_values[0], self._dev_values[-1])
        # A horizontal marker on the amplitude curve showing the current
        # threshold (peak prominence cutoff); it tracks the slider.
        self._threshold_line = self._ax_curve.axhline(
            self._threshold(), color="green", linestyle="--", linewidth=1
        )
        self._ax_img.set_ylabel("Frequency (GHz)")
        self._ax_curve.set_xlabel("Device value")
        self._ax_curve.set_ylabel("Normalized Amplitude")
        self._scatter_img = None
        self._scatter_curve = None

    def update_peaks(self, threshold: float) -> None:
        if hasattr(self, "_threshold_line"):
            self._threshold_line.set_ydata([threshold, threshold])
        peaks = detect_peaks(self._smoothed, threshold)
        self._s_dev_values = self._dev_values[peaks]
        self._s_freqs = np.full_like(
            self._s_dev_values, self._freqs[self._max_freq_idx]
        )

        if self._scatter_img is not None:
            self._scatter_img.remove()
        if self._scatter_curve is not None:
            self._scatter_curve.remove()
        self._scatter_img = self._ax_img.scatter(
            self._s_dev_values, self._s_freqs, color="red", s=30, zorder=5
        )
        self._scatter_curve = self._ax_curve.scatter(
            self._s_dev_values, self._smoothed[peaks], color="red", s=30, zorder=5
        )
        self.redraw()

    def _on_threshold_change(self, _value: int) -> None:
        self.update_peaks(self._threshold())

    # --- result ----------------------------------------------------------

    def get_result(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Selected (dev_values, freqs) — all at the max-dispersion frequency."""
        return self._s_dev_values, self._s_freqs
