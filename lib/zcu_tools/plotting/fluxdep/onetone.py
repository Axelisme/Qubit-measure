"""Native one-tone peak-selection rendering, without GUI state ownership."""

from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from zcu_tools.analysis.fluxdep.onetone import (
    OneToneInputs,
    OneTonePickState,
    analyze_onetone_pick,
)


class OneTonePickPlot:
    """Own image/curve artists and project committed peak indices.

    Uses Device value/GHz axes and normalized-amplitude curve, ticks and grids.
    Does not own analysis state, Qt canvas, presentation, or a pyplot manager.
    """

    def __init__(
        self,
        figure: Figure,
        inputs: OneToneInputs,
        *,
        flux_half: float | None = None,
        flux_int: float | None = None,
    ) -> None:
        """Build two panels in the supplied empty figure without switching canvas.

        inputs provides read-only spectrum and curve preprocessing.
        Optional finite flux_half/int are native device-position reference lines.
        ValueError rejects nonfinite reference positions or nonempty figure.
        """
        if figure.axes:
            raise ValueError("one-tone plotting requires an empty figure")
        for position in (flux_half, flux_int):
            if position is not None and not np.isfinite(position):
                raise ValueError("calibration reference positions must be finite")
        self._inputs = inputs
        spectrum = inputs.spectrum
        ax_image = figure.add_subplot(2, 1, 1)
        ax_curve = figure.add_subplot(2, 1, 2)
        ax_image.imshow(
            np.abs(spectrum.signals).T,
            aspect="auto",
            origin="lower",
            extent=(
                float(spectrum.dev_values[0]),
                float(spectrum.dev_values[-1]),
                float(spectrum.freqs[0]),
                float(spectrum.freqs[-1]),
            ),
            cmap="gray_r",
        )
        ax_image.axhline(
            spectrum.freqs[inputs.max_freq_index], color="red", linewidth=1
        )
        ax_curve.plot(spectrum.dev_values, inputs.smoothed)
        for axes in (ax_image, ax_curve):
            axes.set_xlabel("Device value")
            axes.set_xlim(spectrum.dev_values[0], spectrum.dev_values[-1])
            axes.minorticks_on()
            axes.grid(True, which="major", alpha=0.35)
            axes.grid(True, which="minor", alpha=0.15)
            if flux_half is not None:
                axes.axvline(flux_half, color="red", linestyle="--", linewidth=1)
            if flux_int is not None:
                axes.axvline(flux_int, color="blue", linestyle="--", linewidth=1)
        ax_image.set_ylabel("Frequency (GHz)")
        ax_curve.set_ylabel("Normalized amplitude")
        self._scatter_image = ax_image.scatter([], [], color="red", s=30, zorder=5)
        self._scatter_curve = ax_curve.scatter([], [], color="red", s=30, zorder=5)
        figure.tight_layout()

    def show_state(self, state: OneTonePickState) -> None:
        """Update existing peak artists from committed state without analysis writes.

        ValueError rejects out-of-bounds indices. Caller chooses when to redraw.
        """
        result = analyze_onetone_pick(self._inputs, state)
        indices = np.asarray(state.peak_indices, dtype=np.intp)
        self._scatter_image.set_offsets(
            np.column_stack((result.dev_values, result.freqs))
        )
        self._scatter_curve.set_offsets(
            np.column_stack((result.dev_values, self._inputs.smoothed[indices]))
        )


def make_onetone_pick_figure(
    inputs: OneToneInputs,
    state: OneTonePickState,
    *,
    flux_half: float | None = None,
    flux_int: float | None = None,
) -> Figure:
    """Return a separate Agg Figure projecting native device/GHz peak selection.

    No pyplot registration, presentation or ownership transfer. Reference line
    and index validation follows OneTonePickPlot; caller owns figure lifetime.
    """
    figure = Figure(figsize=(8, 7))
    FigureCanvasAgg(figure)
    plot = OneTonePickPlot(figure, inputs, flux_half=flux_half, flux_int=flux_int)
    plot.show_state(state)
    return figure
