"""Live two-state projection of T1 COV moment sums."""

from collections.abc import Callable

from matplotlib.artist import Artist
from matplotlib.axes import Axes
from numpy import float64
from numpy.typing import NDArray
from zcu_tools.analysis.cov_transfer import TransferResult, selfcal_transfer
from zcu_tools.analysis.failure import AnalysisFailed
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2.support import live_line


def live_transfer(
    plots: Plots, ax: Axes, times: NDArray[float64]
) -> Callable[[NDArray[float64]], None]:
    """Attach a live C line for the delay grid in us and return its update.

    Pass cumulative float64 (N, 6, 6) sums to the callback. Before the first
    successful two-state calibration, display Calibrating without C points.
    Reuse the last calibration as a seed. Identification failures leave the
    figure unchanged; malformed input errors propagate.
    """
    previous: TransferResult | None = None
    viewer = live_line(ax)
    status = ax.text(
        0.5, 0.95, "Calibrating", ha="center", va="top", transform=ax.transAxes
    )

    @plots.on_plot_thread
    def update(sums: NDArray[float64]) -> tuple[Artist, ...]:
        nonlocal previous
        try:
            calibrated = selfcal_transfer(sums, previous=previous)
        except AnalysisFailed:
            return ()
        previous = calibrated
        status.set_text(f"{sums[:, 0, 0].max():.0f} retained pairs/point")
        return (*viewer.set(times, calibrated.change_probability), status)

    return update
