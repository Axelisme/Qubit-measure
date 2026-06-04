"""VizService — matplotlib renders of the dispersive figures (plotly→mpl).

The dispersive notebook draws its product figures with plotly
(``plot_dispersive_with_onetone`` / ``plot_dispersive_shift``); the GUI standardises
on matplotlib (so they share the embedded backend and a Qt-embedded canvas), so
they are rewritten here. The interactive ``search_proper_g`` figure body becomes
``render_tune_figure`` + an in-place ``update_tune_lines`` for live slider response.

These are pure functions that draw onto a caller-supplied ``Figure`` (the fluxdep
viz.py contract) and return the artists the live tuning canvas updates. They do not
touch State, pyplot, or the notebook modules — the numerical inputs are computed by
PredictService / the controller and passed in.
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

from zcu_tools.simulate.fluxonium import calculate_chi_vs_flux

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


def _heatmap_extent(
    sp_fluxs: NDArray[np.float64], sp_freqs: NDArray[np.float64]
) -> tuple[float, float, float, float]:
    return (
        float(sp_fluxs[0]),
        float(sp_fluxs[-1]),
        float(sp_freqs[0]),
        float(sp_freqs[-1]),
    )


def _remove_level_jumps(
    rf: NDArray[np.float64], freq_span: float
) -> NDArray[np.float64]:
    """NaN out single-point sign-flip jumps (avoid visual breaks at level crossings).

    Port of the notebook's de-jump loop (dispersive.py:322-332): a point whose
    forward difference flips sign relative to both neighbours and exceeds 1% of the
    frequency span is a spurious crossing artifact, set to NaN.
    """
    rf = rf.copy()
    diff = np.diff(rf, prepend=rf[0])
    for j in range(1, len(diff) - 1):
        sj, sp, sn = np.sign(diff[j]), np.sign(diff[j - 1]), np.sign(diff[j + 1])
        if sj != sp and sj != sn and abs(diff[j]) > 0.01 * freq_span:
            rf[j] = np.nan
    return rf


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


def render_dispersive_with_onetone(
    figure: Figure,
    bare_rf: float,
    g: float,
    t_fluxs: NDArray[np.float64],
    plot_rfs: tuple[NDArray[np.float64], ...],
    sp_fluxs: NDArray[np.float64],
    sp_freqs: NDArray[np.float64],
    norm_phases: NDArray[np.float64],
) -> None:
    """matplotlib port of ``plot_dispersive_with_onetone`` (product figure 1, GHz)."""
    figure.clear()
    ax = figure.add_subplot(1, 1, 1)
    ax.imshow(
        np.abs(norm_phases).T,
        aspect="auto",
        origin="lower",
        interpolation="none",
        extent=_heatmap_extent(sp_fluxs, sp_freqs),
        cmap="viridis",
    )
    freq_span = float(np.ptp(sp_freqs))
    colors = ["blue", "red", "green", "purple", "orange", "brown", "pink", "gray"]
    for i, rf in enumerate(plot_rfs):
        color = colors[i] if i < len(colors) else None
        ax.plot(
            t_fluxs, _remove_level_jumps(rf, freq_span), color=color, label=f"state {i}"
        )
    ax.axhline(y=bare_rf, color="black", linestyle="--", label="bare resonator")
    ax.set_xlabel(r"Flux $\Phi/\Phi_0$")
    ax.set_ylabel("Frequency (GHz)")
    ax.set_ylim(float(sp_freqs.min()), float(sp_freqs.max()))
    ax.set_title(f"g = {g:.3f} GHz, r_f = {bare_rf:.3f} GHz")
    ax.legend(loc="upper right", fontsize="small")


def compute_chi_vs_flux(
    params: tuple[float, float, float],
    t_fluxs: NDArray[np.float64],
    bare_rf: float,
    g: float,
    upto: int = 5,
) -> NDArray[np.float64]:
    """The chi-vs-flux array for the dispersive-shift figure (scqubits — heavy).

    Pure and off-main-safe: this is the expensive part of the result figure, split
    out so it can run on a worker thread while ``render_dispersive_shift`` (drawing
    only) stays on the Qt main thread.
    """
    return calculate_chi_vs_flux(
        params, t_fluxs, bare_rf, g, progress=False, res_dim=upto + 2
    )


def render_dispersive_shift(
    figure: Figure,
    params: tuple[float, float, float],
    t_fluxs: NDArray[np.float64],
    chi: NDArray[np.float64],
    upto: int = 5,
) -> None:
    """matplotlib port of ``plot_dispersive_shift`` (product figure 2, chi in MHz).

    ``chi`` is precomputed by ``compute_chi_vs_flux`` (off-main); this only draws the
    per-level differences in MHz, so it is cheap and stays on the main thread.
    """
    figure.clear()
    ax = figure.add_subplot(1, 1, 1)
    ax.axhline(y=0.0, color="black", linewidth=2, linestyle="--")
    abs_mean = 0.0
    for i in range(upto):
        diff_chi = chi[:, i + 1] - chi[:, i]
        ax.plot(t_fluxs, diff_chi * 1e3, label=f"chi_n{i}")
        abs_mean += float(np.mean(np.abs(diff_chi)))
    abs_mean /= upto
    ax.set_xlabel(r"$\phi_{ext}/\phi_0$")
    ax.set_ylabel("Chi (MHz)")
    ax.set_title(f"EJ/EC/EL = {params[0]:.3f}/{params[1]:.3f}/{params[2]:.3f}")
    if abs_mean > 0.0:
        ax.set_ylim(-abs_mean * 5e3, abs_mean * 5e3)
    ax.legend(loc="upper right", fontsize="small")
