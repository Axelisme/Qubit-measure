"""Combined live mux presentation with independent per-line coordinates."""

from collections.abc import Sequence

import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from zcu_tools.plotting.plots import Plots


def draw_mux_magnitudes(
    plots: Plots,
    coordinates: NDArray[np.float64],
    signals: NDArray[np.complex128],
    labels: Sequence[str],
    coordinate_label: str,
) -> None:
    """Plot mean IQ magnitude for each participating tone on its own scan grid.

    coordinates and signals are matching (K,P) arrays in participant/data-row
    order. labels names each line with its original index and source cfg name;
    GE callers additionally identify the preparation. NaN partial IQ stays NaN.
    coordinate_label names the horizontal axis, including its units.
    Existing measurement_i figures remain untouched. Invalid shapes raise
    ValueError; drawing failures propagate. Copy the arrays before drawing on
    the plots owner's thread; no active artists leave the callback.
    """
    # Copy this update so the figure uses one consistent acquisition snapshot.
    xs = np.array(coordinates, dtype=np.float64, copy=True)
    ys = np.abs(np.array(signals, dtype=np.complex128, copy=True))
    captions = tuple(labels)
    if xs.ndim != 2 or xs.shape != ys.shape or len(captions) != xs.shape[0]:
        raise ValueError("Mux live coordinates/signals must be (K,P) with K labels")

    # Keep artist creation and autoscaling on the plot owner's thread.
    if "measurement" not in plots:
        plots.subplots("measurement")

    def draw(figure: Figure) -> None:
        figure.clear()
        axes = figure.subplots()
        for x, y, label in zip(xs, ys, captions, strict=True):
            axes.plot(x, y, ".-", label=label)
        axes.set_xlabel(coordinate_label)
        axes.set_ylabel("Magnitude (a.u.)")
        axes.grid()
        axes.legend()
        axes.relim()
        axes.autoscale_view()

    plots.draw("measurement", draw)
