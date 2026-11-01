"""Mean-IQ display mechanics shared by reset calibration scans."""

from collections.abc import Callable
from math import prod

import numpy as np
from matplotlib.artist import Artist
from matplotlib.ticker import MaxNLocator
from numpy.typing import NDArray
from zcu_tools.experiment.v2.runtime.schedule import SignalBuffer
from zcu_tools.plotting.plots import Plots
from zcu_tools.utils.process import rotate2real

from zcu_lab.v2.support.live import live_line


def reset_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    """Project complex mean IQ onto its response axis, without population calibration."""
    return rotate2real(signals).real


def reset_scan_buffer(
    plots: Plots,
    coordinates: NDArray[np.float64],
    *,
    xlabel: str,
    branch_shape: tuple[int, ...] = (),
    signal2real: Callable[
        [NDArray[np.complex128]], NDArray[np.float64]
    ] = reset_signal2real,
) -> SignalBuffer:
    """Retain complex IQ scans and publish caller-selected real display curves.

    coordinates contains hardware-rounded scan values; xlabel names their unit.
    branch_shape gives leading IQ dimensions, defaulting to one unbranched scan.
    signal2real maps that buffer to (prod(branch_shape), len(coordinates)), or a
    1D curve for one line. The default is an uncalibrated response-axis projection.
    Unmeasured points stay NaN. Display conversion never selects a reset optimum.
    """
    fig, ax = plots.subplots(num="measurement")
    ax.set(xlabel=xlabel, ylabel="Amplitude")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.grid(False)
    viewer = live_line(ax, num_lines=prod(branch_shape))

    @plots.on_plot_thread
    def update(data: NDArray[np.complex128]) -> tuple[Artist, ...]:
        return viewer.set(coordinates, signal2real(data))

    fig.show()
    return SignalBuffer(
        (*branch_shape, len(coordinates)),
        on_update=update,
    )
