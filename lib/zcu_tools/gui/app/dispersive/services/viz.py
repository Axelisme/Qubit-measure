"""VizService — matplotlib render of the dispersive tuning figure.

The interactive ``search_proper_g`` figure body (notebook) becomes three pieces:
``render_tune_figure`` draws the static parts (norm-phase background + the r_f line)
right after preprocessing; ``update_bare_line`` moves the r_f line live as the slider
drags; ``set_dispersion_lines`` adds the predicted ground/excited lines once "Use
these g/r_f" runs the predictor. The user's manual tuning IS the final fit, so this
single figure is both the tuning view and the result.

On top of that, the user can drop **sample-flux lines** (draggable vertical lines):
for each one the ground/excited resonator frequency at that single flux is computed
live (a fast single-point dispersive call) as the user drags r_f / changes g, shown
as a red (ground) / blue (excited) dot on the line. This gives instant feedback at a
few fluxes without the full all-flux recompute (which only runs on "Use these g/r_f").

These are pure functions that draw onto a caller-supplied ``Figure`` and mutate the
returned ``TuneArtists`` in place. They do not touch State, pyplot, or the notebook
modules — the numerical inputs are computed by the predictor and passed in.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from matplotlib.lines import Line2D
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass
class SampleArtists:
    """The artists of one draggable sample-flux line: the vertical line + its dots.

    ``flux`` is the line's current x position; ``dot_ground`` / ``dot_excited`` are
    the red/blue markers showing the predicted ground/excited resonator frequency at
    that flux (None until first computed).
    """

    flux: float
    line: Line2D
    dot_ground: Optional[Line2D] = None
    dot_excited: Optional[Line2D] = None


@dataclass
class TuneArtists:
    """The mutable artists of the live g/r_f tuning figure (for in-place updates).

    ``line_bare`` (the r_f dashed line) and ``image`` (the norm-phase background)
    always exist; ``line_ground`` / ``line_excited`` (the predicted dispersion
    lines) are None until a prediction has been computed via ``set_dispersion_lines``.
    ``samples`` holds the draggable sample-flux lines (each with its own dots).
    """

    figure: Figure
    ax: Axes
    line_bare: Line2D
    image: AxesImage
    line_ground: Optional[Line2D] = None
    line_excited: Optional[Line2D] = None
    g: Optional[float] = None  # GHz, the g of the last prediction (None = not yet)
    samples: list[SampleArtists] = field(default_factory=list)


def render_tune_figure(
    figure: Figure,
    sp_fluxs: NDArray[np.float64],
    sp_freqs: NDArray[np.float64],
    norm_phases: NDArray[np.float64],
    bare_rf: float,
) -> TuneArtists:
    """Draw the tuning figure's static parts: the norm-phase background + r_f line.

    The dispersion (ground/excited) lines are NOT drawn here — they are added by
    ``set_dispersion_lines`` once a prediction has been computed. The r_f dashed line
    moves in place via ``update_bare_line`` on each slider step (no heatmap redraw).
    Frequencies displayed in MHz (×1e3).
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
    line_bare = ax.axhline(y=1e3 * bare_rf, color="k", linestyle="--", label="r_f")
    ax.set_ylim(1e3 * float(sp_freqs.min()), 1e3 * float(sp_freqs.max()))
    ax.set_xlabel(r"Flux $\Phi_{ext}/\Phi_0$")
    ax.set_ylabel("Frequency (MHz)")
    ax.set_title(f"r_f = {1e3 * bare_rf:.1f} MHz")
    ax.legend(loc="upper right")
    return TuneArtists(figure, ax, line_bare, image)


def update_bare_line(artists: TuneArtists, bare_rf: float) -> None:
    """Move the r_f dashed line in place (slider drag — no heatmap redraw)."""
    artists.line_bare.set_ydata([1e3 * bare_rf])
    g_part = f"g = {1e3 * artists.g:.1f} MHz, " if artists.g is not None else ""
    artists.ax.set_title(f"{g_part}r_f = {1e3 * bare_rf:.1f} MHz")


def set_dispersion_lines(
    artists: TuneArtists,
    t_fluxs: NDArray[np.float64],
    rf_0: NDArray[np.float64],
    rf_1: NDArray[np.float64],
    g: float,
    bare_rf: float,
) -> None:
    """Draw/update the predicted ground/excited dispersion lines (after a predict).

    Creates the two lines on first call, then updates them in place. Also records
    ``g`` so the title shows it on subsequent r_f slider moves.
    """
    artists.g = g
    if artists.line_ground is None:
        (artists.line_ground,) = artists.ax.plot(
            t_fluxs, 1e3 * rf_0, "b-", label="Ground state"
        )
        (artists.line_excited,) = artists.ax.plot(
            t_fluxs, 1e3 * rf_1, "r-", label="Excited state"
        )
        artists.ax.legend(loc="upper right")
    else:
        artists.line_ground.set_data(t_fluxs, 1e3 * rf_0)
        assert artists.line_excited is not None
        artists.line_excited.set_data(t_fluxs, 1e3 * rf_1)
    artists.line_bare.set_ydata([1e3 * bare_rf])
    artists.ax.set_title(f"g = {1e3 * g:.1f} MHz, r_f = {1e3 * bare_rf:.1f} MHz")


# --- sample-flux lines (draggable, live single-point dots) -------------------


def add_sample_line(artists: TuneArtists, flux: float) -> SampleArtists:
    """Add a draggable vertical sample-flux line at ``flux`` (no dots yet).

    The dots (ground/excited markers) are filled in by ``update_sample_dots`` once
    the single-point prediction has been computed. Returns the new SampleArtists,
    also appended to ``artists.samples``.
    """
    line = artists.ax.axvline(x=flux, color="magenta", linestyle="-", linewidth=1.2)
    sample = SampleArtists(flux=flux, line=line)
    artists.samples.append(sample)
    return sample


def move_sample_line(sample: SampleArtists, flux: float) -> None:
    """Move a sample line to a new flux in place (its dots are refreshed separately)."""
    sample.flux = flux
    sample.line.set_xdata([flux, flux])


def remove_sample_line(artists: TuneArtists, sample: SampleArtists) -> None:
    """Remove a sample line and its dots from the figure and the artists list."""
    sample.line.remove()
    if sample.dot_ground is not None:
        sample.dot_ground.remove()
    if sample.dot_excited is not None:
        sample.dot_excited.remove()
    artists.samples.remove(sample)


def update_sample_dots(
    artists: TuneArtists,
    sample: SampleArtists,
    rf_ground: float,
    rf_excited: float,
) -> None:
    """Set the red (ground) / blue (excited) dots on a sample line (MHz on the y-axis).

    Creates the two dot artists on first call, then moves them in place. The x stays
    at the line's current flux; the y is the predicted resonator frequency in MHz.
    """
    x = sample.flux
    if sample.dot_ground is None:
        (sample.dot_ground,) = artists.ax.plot(
            [x], [1e3 * rf_ground], "o", color="red", markersize=7, zorder=5
        )
        (sample.dot_excited,) = artists.ax.plot(
            [x], [1e3 * rf_excited], "o", color="blue", markersize=7, zorder=5
        )
    else:
        sample.dot_ground.set_data([x], [1e3 * rf_ground])
        assert sample.dot_excited is not None
        sample.dot_excited.set_data([x], [1e3 * rf_excited])
