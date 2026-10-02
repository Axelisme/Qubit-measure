from __future__ import annotations

from typing import Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray

from zcu_tools.plotting.plots import Plots

from .base import GE_FitResult, fidelity_func
from .center import fit_ge_by_center
from .manual import fit_ge_manual
from .pca import fit_ge_by_pca

NUM_BINS = 201


def singleshot_visualize(
    signals: NDArray[np.complex128], plot_center: bool = True
) -> tuple[Figure, Axes]:
    """
    Visualize single-shot measurements in IQ plane.

    Creates a scatter plot of I/Q signal points, optionally marking the center (mean) of each set of signals.

    Parameters
    ----------
    signals : NDArray
        Complex array of measurement signals. If 1D, it will be reshaped to 2D.
        For 2D arrays, first dimension represents different measurement sets (e.g., ground and excited states).
    plot_center : bool, default=True
        If True, calculate and plot the center (mean) point for each set of signals.

    Returns
    -------
    tuple[plt.Figure, plt.Axes]
        Figure and axes objects of the created plot.
    """
    fig, ax = plt.subplots()

    if signals.ndim == 1:
        signals = signals.reshape(1, -1)

    for i in range(signals.shape[0]):
        Is, Qs = signals[i].real, signals[i].imag

        MAX_POINTS = 1e5
        if len(Is) > MAX_POINTS:
            Is = Is[:: len(Is) // int(MAX_POINTS)]
            Qs = Qs[:: len(Qs) // int(MAX_POINTS)]

        ax.scatter(
            Is,
            Qs,
            marker=".",
            edgecolor="None",
            alpha=0.1,
            label=f"shot {i}",
            c=f"C{i}",
        )

    if plot_center:
        for i in range(signals.shape[0]):
            Ic = np.mean(signals[i].real)
            Qc = np.mean(signals[i].imag)
            ax.plot(
                Ic,
                Qc,
                linestyle=":",
                marker="o",
                markersize=5,
                c=f"C{i}",
                markeredgecolor="k",
            )

    ax.set_title(f"{signals.shape[1]} Shots")
    ax.set_xlabel("I [ADC levels]")
    ax.set_ylabel("Q [ADC levels]")
    if signals.shape[0] > 1:
        ax.legend(loc="upper right")
    ax.axis("equal")

    return fig, ax


def singleshot_ge_analysis(
    signals: NDArray[np.complex128],
    angle: float | None = None,
    backend: Literal["center", "pca"] = "pca",
    *,
    plots: Plots,
    **kwargs,
) -> tuple[float, NDArray[np.float64], GE_FitResult]:
    """
    Analyze ground and excited state signals to determine classification parameters.

    Performs analysis on IQ measurement data to determine optimal measurement axis,
    threshold for state discrimination, and calculates the resulting fidelity.

    Parameters
    ----------
    signals : NDArray
        Complex array of shape (2, N) containing measurement signals.
        First row should contain ground state signals, second row excited state signals.
    angle : float, default=None
        if given, use this angle for rotation, ignore backend
    backend : Literal["center", "pca"], default="pca"
        Method used for determining the rotation angle when angle is absent.
    plots : Plots
        Operation that owns the named fit figure.

    Returns
    -------
    tuple[float, NDArray, GE_FitResult]
        Assignment fidelity, refined preparation populations and calibration.
    """
    if angle is not None:
        return fit_ge_manual(signals, angle, plots=plots, **kwargs)

    if backend == "center":
        return fit_ge_by_center(signals, plots=plots, **kwargs)
    if backend == "pca":
        return fit_ge_by_pca(signals, plots=plots, **kwargs)

    raise ValueError(f"Unknown backend: {backend}")


__all__ = [
    # base
    "GE_FitResult",
    "fidelity_func",
    # center
    "fit_ge_by_center",
    # manual
    "fit_ge_manual",
    # pca
    "fit_ge_by_pca",
    # singleshot
    "singleshot_ge_analysis",
    "singleshot_visualize",
]
