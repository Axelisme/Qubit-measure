from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from matplotlib.axes import Axes
from matplotlib.patches import Circle


def calc_populations(signals: NDArray[np.float64]) -> NDArray[np.float64]:
    g_pop, e_pop = signals[..., 0], signals[..., 1]
    return np.stack([g_pop, e_pop, 1 - g_pop - e_pop], axis=-1)


def plot_classified_result(
    ax: Axes,
    signals: NDArray[np.complex128],
    g_center: complex,
    e_center: complex,
    radius: float,
) -> Tuple[float, float, float]:
    # Classify shots
    dists_g = np.abs(signals - g_center)
    dists_e = np.abs(signals - e_center)
    mask_g = dists_g < radius
    mask_e = dists_e < radius
    mask_o = ~(mask_g | mask_e)

    colors = np.full(signals.shape, "gray", dtype=object)
    colors[mask_g] = "blue"
    colors[mask_e] = "red"
    colors[mask_o] = "green"

    # plot shots
    num_sample = signals.shape[0]
    downsample_num = min(5000, num_sample)
    downsample_mask = np.arange(0, num_sample, max(1, num_sample // downsample_num))
    downsample_signals = signals[downsample_mask]
    ax.scatter(
        downsample_signals.real,
        downsample_signals.imag,
        c=colors[downsample_mask].tolist(),
        s=1,
        alpha=0.5,
    )

    # plot centers with circle
    plt_params = dict(linestyle=":", color="k", marker="o", markersize=5)
    ax.plot(
        g_center.real,
        g_center.imag,
        markerfacecolor="b",
        label="Ground",
        **plt_params,  # type: ignore
    )
    ax.plot(
        e_center.real,
        e_center.imag,
        markerfacecolor="r",
        label="Excited",
        **plt_params,  # type: ignore
    )
    ax.add_patch(
        Circle(
            (g_center.real, g_center.imag),
            radius,
            color="b",
            fill=False,
            linestyle="--",
        )
    )
    ax.add_patch(
        Circle(
            (e_center.real, e_center.imag),
            radius,
            color="r",
            fill=False,
            linestyle="--",
        )
    )

    ax.set_aspect("equal")
    ax.legend()
    ax.set_xlabel("I value (a.u.)")
    ax.set_ylabel("Q value (a.u.)")

    ng = np.sum(mask_g).item() / signals.shape[0]
    ne = np.sum(mask_e).item() / signals.shape[0]
    no = np.sum(mask_o).item() / signals.shape[0]

    return ng, ne, no
