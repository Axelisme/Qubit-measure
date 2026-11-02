"""Live self-calibration shared by the five single-qubit heralding experiments."""

from collections.abc import Callable

import numpy as np
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from numpy.typing import NDArray
from zcu_tools.analysis.failure import AnalysisFailed
from zcu_tools.analysis.herald_transfer import TransferResult, selfcal_transfer
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2.support import live_line


class TransferProjector:
    """Project cumulative (N, 6, 6) sums to two-state C for live display or SNR.

    Each instance retains its last successful calibration as a warm start.
    AnalysisFailed returns an all-NaN (N,) curve and keeps that calibration.
    Input-contract errors propagate. Use separate instances for the plot owner
    and acquisition thread; an instance is not shared between threads.
    """

    def __init__(self) -> None:
        self._previous: TransferResult | None = None

    def __call__(self, sums: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return C, or NaNs on identification failure without ending acquisition."""
        try:
            calibrated = selfcal_transfer(sums, previous=self._previous)
        except AnalysisFailed:
            return np.full(len(sums), np.nan)
        self._previous = calibrated
        return calibrated.change_probability


def live_transfer(
    plots: Plots, ax: Axes, xs: NDArray[np.float64]
) -> Callable[[NDArray[np.float64]], tuple[Artist, ...]]:
    """Attach a live C line and return its plot-owner update callback.

    Before calibration succeeds show Calibrating without C points. Failed
    updates preserve the previous figure. The callback owns its projector;
    an early-stop checker must use a separate TransferProjector instance.
    """
    projector = TransferProjector()
    viewer = live_line(ax)
    ax.set_ylabel("Change probability C")
    status = ax.text(
        0.5, 0.95, "Calibrating", ha="center", va="top", transform=ax.transAxes
    )

    @plots.on_plot_thread
    def update(data: NDArray[np.float64]) -> tuple[Artist, ...]:
        change = projector(data)
        if np.isnan(change).all():
            return ()
        status.set_visible(False)
        return (*viewer.set(xs, change), status)

    return update
