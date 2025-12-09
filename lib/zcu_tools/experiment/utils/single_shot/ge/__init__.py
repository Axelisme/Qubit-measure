from typing import Literal, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .base import fidelity_func
from .center import fit_ge_by_center
from .manual import fit_ge_manual
from .pca import fit_ge_by_pca

NUM_BINS = 201


def singleshot_visualize(
    signals: np.ndarray, plot_center: bool = True
) -> Tuple[Figure, Axes]:
    """
    Visualize single-shot measurements in IQ plane.

    Creates a scatter plot of I/Q signal points, optionally marking the center (mean) of each set of signals.

    Parameters
    ----------
    signals : np.ndarray
        Complex array of measurement signals. If 1D, it will be reshaped to 2D.
        For 2D arrays, first dimension represents different measurement sets (e.g., ground and excited states).
    plot_center : bool, default=True
        If True, calculate and plot the center (mean) point for each set of signals.

    Returns
    -------
    Tuple[plt.Figure, plt.Axes]
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
    signals: np.ndarray,
    angle: Optional[float] = None,
    backend: Literal["center", "regression", "pca"] = "pca",
    **kwargs,
) -> Tuple[float, float, float, np.ndarray, dict, Figure]:
    """
    Analyze ground and excited state signals to determine classification parameters.

    Performs analysis on IQ measurement data to determine optimal measurement axis,
    threshold for state discrimination, and calculates the resulting fidelity.

    Parameters
    ----------
    signals : np.ndarray
        Complex array of shape (2, N) containing measurement signals.
        First row should contain ground state signals, second row excited state signals.
    angle : float, default=None
        if given, use this angle for rotation, ignore backend
    backend : Literal["center", "regression", "pca"], default="pca"
        Method used for determining optimal rotation angle:
        - "center": Uses median of ground and excited signal clusters to determine rotation.
        - "regression": Uses logistic regression to find optimal decision boundary.
        - "pca": Uses PCA to find optimal rotation angle.

    Returns
    -------
    Tuple[float, float, float, np.ndarray]
        A tuple containing:
        - fidelity: The assignment fidelity between ground and excited states (0.5-1.0)
        - threshold: The optimal threshold value for state discrimination
        - theta_deg: The optimal rotation angle in degrees
        - populations: The populations of ground and excited states
    """
    if angle is not None:
        return fit_ge_manual(signals, angle, **kwargs)

    if backend == "center":
        return fit_ge_by_center(signals, **kwargs)
    if backend == "pca":
        return fit_ge_by_pca(signals, **kwargs)

    raise ValueError(f"Unknown backend: {backend}")


__all__ = [
    "fidelity_func",
    "singleshot_ge_analysis",
    "singleshot_visualize",
]
