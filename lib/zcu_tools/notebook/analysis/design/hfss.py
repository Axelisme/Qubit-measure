from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from zcu_tools.utils.fitting import fit_anticross


def analyze_1d_sweep(
    result_path: str, ref_freq: float, param_name: str
) -> Tuple[plt.Figure, plt.Axes, float]:
    data = pd.read_csv(result_path)

    params = data[param_name].values
    fpts1 = 1e-9 * data["re(Mode(1)) []"].values
    fpts2 = 1e-9 * data["re(Mode(2)) []"].values

    dist1 = np.abs(fpts1 - ref_freq)
    dist2 = np.abs(fpts2 - ref_freq)
    dist = np.where(dist1 < dist2, dist1, dist2)

    max_idx = np.argmax(dist)
    max_param = params[max_idx]

    fig, ax = plt.subplots()
    ax.plot(params, 1e3 * dist, marker=".")
    ax.scatter(
        max_param,
        1e3 * dist[max_idx],
        c="k",
        marker="x",
        zorder=10,
        label=f"({max_param:.4g}, {1e3 * dist[max_idx]:.2f} MHz)",
    )
    ax.set_xlabel(param_name)
    ax.set_ylabel("Distance to reference [MHz]")
    ax.grid()
    ax.legend()
    ax.set_title(f"Reference frequency: {ref_freq:.5g} GHz")

    return fig, ax, max_param


def analyze_xy_sweep(
    result_path: str, ref_freq: float
) -> Tuple[plt.Figure, plt.Axes, float, float]:
    data = pd.read_csv(result_path)

    Xs = data["Arm_X [mm]"].values
    Ys = data["Pad_Y [um]"].values
    fpts1 = 1e-9 * data["re(Mode(1)) []"].values
    fpts2 = 1e-9 * data["re(Mode(2)) []"].values

    # reorganize data to 2D array
    xs = np.unique(Xs)
    ys = np.unique(Ys)
    len_x = len(xs)
    len_y = len(ys)

    # force order: (x, y)
    if Xs[0] != Xs[1]:
        if Ys[0] != Ys[1]:
            raise ValueError("Unrecognized data order")
        else:
            print("force order: (x, y)")
            fpts1 = fpts1.reshape(len_y, len_x).T.flatten()
            fpts2 = fpts2.reshape(len_y, len_x).T.flatten()
    fpts1 = fpts1.reshape(len_x, len_y)
    fpts2 = fpts2.reshape(len_x, len_y)

    # calculate min distance to reference frequency
    dist1 = np.abs(fpts1 - ref_freq)
    dist2 = np.abs(fpts2 - ref_freq)
    dist = np.where(dist1 < dist2, dist1, dist2)

    # find points which are the maximum distance in both x and y direction
    peak_in_x = set([(i, np.argmax(dist[i, :])) for i in range(len_x)])
    peak_in_y = set([(np.argmax(dist[:, j]), j) for j in range(len_y)])
    peak_idxs = list(peak_in_x.intersection(peak_in_y))
    peak_idxs = sorted(peak_idxs, key=lambda x: x[0])
    peak_points = np.array([(xs[i], ys[j]) for i, j in peak_idxs])

    max_idx = np.unravel_index(np.argmax(dist), dist.shape)
    max_x, max_y = xs[max_idx[0]], ys[max_idx[1]]

    # plot
    fig, ax = plt.subplots()
    im = ax.imshow(
        dist.T,
        aspect="auto",
        interpolation="none",
        extent=[xs[0], xs[-1], ys[0], ys[-1]],
        origin="lower",
    )
    ax.plot(peak_points[:, 0], peak_points[:, 1], "r", marker="o")
    ax.scatter(
        max_x,
        max_y,
        c="k",
        marker="x",
        label=f"Arm_X = {max_x:.4g} mm, Pad_Y = {max_y:.4g} um",
        zorder=10,
    )

    ax.set_xlabel("Arm_X [mm]")
    ax.set_ylabel("Pad_Y [um]")
    ax.legend()
    fig.colorbar(im, ax=ax)

    return fig, ax, max_x, max_y


def fit_hfss_anticross(
    result_path: str,
) -> Tuple[plt.Figure, plt.Axes, float, float, float]:
    """
    Fit the anticrossing of the HFSS simulation
    """

    data = pd.read_csv(result_path)

    Ljs = data["Lj [nH]"].values
    fpts1 = 1e-9 * data["re(Mode(1)) []"].values
    fpts2 = 1e-9 * data["re(Mode(2)) []"].values

    aqf = 1 / np.sqrt(Ljs)
    cx, cy, width, m1, m2, fit_fpts1, fit_fpts2, _ = fit_anticross(
        aqf, fpts1, fpts2, horizontal_line=True
    )
    c_freq = cy
    c_Lj = 1 / cx**2

    fig, ax = plt.subplots()
    ax.plot(Ljs, fpts1, marker=".")
    ax.plot(Ljs, fpts2, marker=".")
    ax.plot(Ljs, fit_fpts1, label="fitting")
    ax.plot(Ljs, fit_fpts2)
    ax.plot(Ljs, cy + m1 * (aqf - cx), "r--")
    ax.plot(Ljs, cy + m2 * (aqf - cx), "b--")
    ax.plot(
        [c_Lj, c_Lj],
        [cy - width, cy + width],
        "g-",
        label=f"width = {1e3 * width:.0f} MHz",
    )
    ax.plot(c_Lj, cy, "o", color="black", label=f"({c_Lj:.1f} nH, {cy:.1f} GHz)")
    ax.set_xlabel("Lj [nH]")
    ax.set_ylabel("Frequency [GHz]")
    ax.legend()
    ax.grid()

    return fig, ax, c_Lj, c_freq, width
