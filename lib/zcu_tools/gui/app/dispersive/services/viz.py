"""VizService — matplotlib render of the dispersive tuning figure.

The interactive ``search_proper_g`` figure body (notebook) becomes
``render_tune_figure`` + an in-place ``update_tune_lines``: the g/r_f lines over the
norm-phase image. The user's manual tuning IS the final fit, so this single figure
is both the tuning view and the result.

These are pure functions that draw onto a caller-supplied ``Figure`` and return the
artists the tuning canvas updates. They do not touch State, pyplot, or the notebook
modules — the numerical inputs are computed by PredictService and passed in.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from matplotlib.lines import Line2D
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass
class TuneArtists:
    """The mutable artists of the live g/r_f tuning figure (for in-place updates)."""

    figure: Figure
    ax: Axes
    line_ground: Line2D
    line_excited: Line2D
    line_bare: Line2D
    image: AxesImage


def render_tune_figure(
    figure: Figure,
    sp_fluxs: NDArray[np.float64],
    sp_freqs: NDArray[np.float64],
    norm_phases: NDArray[np.float64],
    t_fluxs: NDArray[np.float64],
    rf_0: NDArray[np.float64],
    rf_1: NDArray[np.float64],
    g: float,
    bare_rf: float,
) -> TuneArtists:
    """Draw the live g/r_f tuning figure and return its mutable artists.

    The norm-phase image is the static background; the ground/excited lines and the
    bare-resonator dashed line are updated in place via ``update_tune_lines`` on each
    slider move (the heatmap never changes). Frequencies displayed in MHz (×1e3).
    """
    figure.clear()
    ax = figure.add_subplot(1, 1, 1)
    image = ax.imshow(
        np.abs(norm_phases).T,
        aspect="auto",
        origin="lower",
        interpolation="none",
        extent=(
            float(sp_fluxs[0]),
            float(sp_fluxs[-1]),
            1e3 * float(sp_freqs[0]),
            1e3 * float(sp_freqs[-1]),
        ),
        cmap="viridis",
    )
    (line_ground,) = ax.plot(t_fluxs, 1e3 * rf_0, "b-", label="Ground state")
    (line_excited,) = ax.plot(t_fluxs, 1e3 * rf_1, "r-", label="Excited state")
    line_bare = ax.axhline(
        y=1e3 * bare_rf, color="k", linestyle="--", label="Bare resonator"
    )
    ax.set_ylim(1e3 * float(sp_freqs.min()), 1e3 * float(sp_freqs.max()))
    ax.set_xlabel(r"Flux $\Phi_{ext}/\Phi_0$")
    ax.set_ylabel("Frequency (MHz)")
    ax.set_title(f"g = {1e3 * g:.1f} MHz, r_f = {1e3 * bare_rf:.1f} MHz")
    ax.legend(loc="upper right")
    return TuneArtists(figure, ax, line_ground, line_excited, line_bare, image)


def update_tune_lines(
    artists: TuneArtists,
    t_fluxs: NDArray[np.float64],
    rf_0: NDArray[np.float64],
    rf_1: NDArray[np.float64],
    g: float,
    bare_rf: float,
) -> None:
    """Update the tuning figure's lines + title in place (no heatmap redraw)."""
    artists.line_ground.set_data(t_fluxs, 1e3 * rf_0)
    artists.line_excited.set_data(t_fluxs, 1e3 * rf_1)
    artists.line_bare.set_ydata([1e3 * bare_rf])
    artists.ax.set_title(f"g = {1e3 * g:.1f} MHz, r_f = {1e3 * bare_rf:.1f} MHz")
