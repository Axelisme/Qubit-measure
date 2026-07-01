from __future__ import annotations

from typing import Any

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output, display
from numpy.typing import NDArray

from zcu_tools.analysis.fluxdep import (
    detect_peaks,
    max_dispersion_freq_index,
    smoothed_slice,
)


class InteractiveOneTone:
    def __init__(
        self,
        signals: NDArray[np.complex128],
        dev_values: NDArray[np.float64],
        freqs: NDArray[np.float64],
        threshold: float = 1.0,
    ) -> None:
        self.signals = signals
        self.dev_values = dev_values
        self.freqs = freqs
        self.is_finished = False

        plt.ioff()  # to avoid showing the plot immediately
        self.fig, self.axes = plt.subplots(2, 1, figsize=(8, 5))
        self.fig.tight_layout()
        plt.ion()

        # Initialize the spectrum and peak markers.
        self.init_plots(threshold)

        # Build controls after the plot artists exist.
        self.threshold_slider = widgets.FloatSlider(
            value=threshold, min=0.0, max=5.0, step=0.01, description="Threshold:"
        )
        self.threshold_slider.observe(self.on_threshold_change, names="value")

        self.finish_button = widgets.Button(
            description="Finish", button_style="success"
        )
        self.finish_button.on_click(self.on_finish)

        # Display the plot and controls together in the notebook.
        display(
            widgets.HBox(
                [
                    self.fig.canvas,
                    widgets.VBox(
                        [
                            self.threshold_slider,
                            self.finish_button,
                        ]
                    ),
                ]
            )
        )

    def init_plots(self, threshold: float) -> None:
        """Initialize the spectrum and selected one-tone slice."""
        # Show the 2D spectrum.
        self.real_signals = np.abs(self.signals)  # (mAs, freqs)

        self.max_freq_idx = max_dispersion_freq_index(self.signals, self.freqs)

        self.img = self.axes[0].imshow(
            self.real_signals.T,
            aspect="auto",
            origin="lower",
            extent=(
                self.dev_values[0],
                self.dev_values[-1],
                self.freqs[0],
                self.freqs[-1],
            ),
        )
        self.line = self.axes[0].axhline(
            self.freqs[self.max_freq_idx],
            color="red",
            label="max freqs",
        )

        # Show the 1D slice at the maximum-dispersion frequency.
        self.real_signals_slice = self.real_signals[:, self.max_freq_idx]  # (mAs,)
        self.smoothed_real_signals = smoothed_slice(self.signals, self.max_freq_idx)

        (self.curve,) = self.axes[1].plot(self.dev_values, self.smoothed_real_signals)
        self.axes[1].set_xlim(self.dev_values[0], self.dev_values[-1])

        # Draw initial peak markers.
        self.update_peaks(threshold)

        # Set axis labels.
        self.axes[0].set_ylabel("Frequency (GHz)")
        self.axes[1].set_xlabel("Current (mA)")
        self.axes[1].set_ylabel("Normalized Amplitude")

    def update_peaks(self, threshold: float) -> None:
        """Refresh peak markers for the current threshold."""

        # Detect peaks on the smoothed one-tone slice.
        peaks = detect_peaks(self.smoothed_real_signals, threshold)

        # Store selected device values and their common frequency.
        self.s_dev_values = self.dev_values[peaks]
        self.s_freqs = np.full_like(self.s_dev_values, self.freqs[self.max_freq_idx])

        # Replace existing peak markers.
        if hasattr(self, "scatter1"):
            self.scatter1.remove()
        if hasattr(self, "scatter2"):
            self.scatter2.remove()

        # Mark selected points on the 2D spectrum.
        self.scatter1 = self.axes[0].scatter(
            self.s_dev_values, self.s_freqs, color="red", s=30, zorder=5
        )

        # Mark peaks on the 1D slice.
        self.scatter2 = self.axes[1].scatter(
            self.s_dev_values,
            self.smoothed_real_signals[peaks],
            color="red",
            s=30,
            zorder=5,
        )

        # Redraw after marker replacement.
        self.fig.canvas.draw_idle()

    def on_threshold_change(self, change: Any) -> None:
        """Handle threshold-slider changes."""
        if self.is_finished:
            return

        self.update_peaks(change.new)

    def on_finish(self, _: Any) -> None:
        """Handle the Finish button."""
        plt.close(self.fig)
        self.is_finished = True

        # Clear the widget output after closing the figure.
        clear_output(wait=False)

    def get_positions(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return selected ``(dev_values, freqs)`` points."""
        if not self.is_finished:
            self.on_finish(None)
        return self.s_dev_values, self.s_freqs
