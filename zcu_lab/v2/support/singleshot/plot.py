"""Paired-preparation IQ presentation shared by single and mux GE experiments."""

import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from zcu_tools.plotting.plots import Plots


def draw_ge_scatter(
    plots: Plots,
    name: str,
    signals: NDArray[np.complex128],
    *,
    title: str = "GE IQ",
    labels: tuple[str, str] = ("g preparation (probe off)", "e preparation (probe on)"),
) -> None:
    """Draw two preparation clouds in one IQ plane, blue for g and red for e.

    signals contains normalized ADC IQ shaped (2, shots), in label order.
    Partial NaN shots remain missing. Copy the snapshot before dispatching
    drawing to the plot owner. Create name on first use and redraw it on later
    updates. Invalid shapes and drawing failures propagate to the caller.
    """
    clouds = np.array(signals, dtype=np.complex128, copy=True)
    if clouds.ndim != 2 or clouds.shape[0] != 2:
        raise ValueError("GE scatter requires two preparation rows (2, shots)")
    if name not in plots:
        plots.subplots(name)

    def draw(figure: Figure) -> None:
        figure.clear()
        axes = figure.subplots()
        for cloud, label, color in zip(
            clouds, labels, ("tab:blue", "tab:red"), strict=True
        ):
            axes.scatter(
                cloud.real, cloud.imag, s=8, alpha=0.5, color=color, label=label
            )
        axes.set_xlabel("I (a.u.)")
        axes.set_ylabel("Q (a.u.)")
        axes.set_title(title)
        axes.set_aspect("equal", adjustable="datalim")
        axes.grid()
        axes.legend()

    plots.draw(name, draw)
