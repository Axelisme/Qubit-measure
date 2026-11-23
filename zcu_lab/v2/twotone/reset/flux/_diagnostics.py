"""Analyze-only contrast ratio and residual plots shared by flux reset sweeps."""

import numpy as np
from matplotlib.axes import Axes
from numpy.typing import NDArray
from zcu_tools.analysis.shot_uncertainty import propagated_iq_shot_sem


def flux_ratio_sem(
    signals: NDArray[np.complex128],
    moments: NDArray[np.float64],
    ratio: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Propagate shot SEM through the four-branch flux reset contrast ratio.

    Signals and moments have shapes (2, 2, points) and (2, 2, points, 6)
    in before/after π order. Ratio is the already displayed Re(d₁ conj(d₀)) /
    |d₀|², with NaN at invalid points; dₐ is S[a, 0] − S[a, 1]. Each branch
    uses independent shots. Return one SEM per point, NaN at invalid points or
    where a required branch SEM is unavailable. Invalid moment data raise.
    """
    valid = np.isfinite(ratio)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        d0 = np.where(valid, signals[0, 0] - signals[0, 1], 1 + 0j)
        d1 = np.where(valid, signals[1, 0] - signals[1, 1], 0j)
        norm = np.abs(d0)
        unit = d0 / norm
        # Quotient rule: gradients w.r.t. the two complex differences.
        scaled_unit = np.where(valid, ratio, 0.0) * unit
        g0 = (d1 / norm - scaled_unit - scaled_unit) / norm
        g1 = unit / norm
    gradients = np.stack((g0, -g0, g1, -g1), axis=-1, dtype=np.complex128)
    branch_moments = moments.transpose(2, 0, 1, 3).reshape(ratio.size, 4, 6)
    sem = propagated_iq_shot_sem(branch_moments, gradients=gradients)
    return np.where(valid, sem, np.nan)


def plot_flux_ratio_residual(
    axes: tuple[Axes, Axes],
    scan: NDArray[np.float64],
    ratio: NDArray[np.float64],
    residual: NDArray[np.float64],
    ratio_sem: NDArray[np.float64] | None,
    selected_index: int | None,
) -> None:
    """Draw ratio and (1−ratio)/2 against the same scan in the given axes.

    Ratio, residual and optional ratio SEM align elementwise with scan. A
    NaN SEM or no SEM omits the corresponding bar. If selected_index is given,
    highlight that sample with an open marker and the same SEM as its trace.
    Axis labels, selection title and reference lines belong to the caller.
    """
    residual_sem = None if ratio_sem is None else ratio_sem / 2
    for ax, values, sem in zip(
        axes, (ratio, residual), (ratio_sem, residual_sem), strict=True
    ):
        series = ax.errorbar(scan, values, yerr=sem, fmt=".-")
        if selected_index is None:
            continue
        marker = slice(selected_index, selected_index + 1)
        ax.errorbar(
            scan[marker],
            values[marker],
            yerr=None if sem is None else sem[marker],
            fmt="o",
            color=series.lines[0].get_color(),
            markerfacecolor="none",
            markersize=7,
            zorder=3,
        )
