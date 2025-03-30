from typing import Literal, Tuple

import matplotlib.pyplot as plt
import numpy as np

from .base import fidelity_func
from .center import fit_ge_by_center
from .regression import fit_ge_by_regression


def singleshot_visualize(
    signals: np.ndarray, plot_center: bool = True
) -> Tuple[plt.Figure, plt.Axes]:
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

        ax.scatter(
            Is,
            Qs,
            marker=".",
            edgecolor="None",
            alpha=0.5,
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
    plot: bool = True,
    backend: Literal["center", "regression"] = "regression",
) -> Tuple[float, float, float]:
    """
    Analyze ground and excited state signals to determine classification parameters.

    Performs analysis on IQ measurement data to determine optimal measurement axis,
    threshold for state discrimination, and calculates the resulting fidelity.

    Parameters
    ----------
    signals : np.ndarray
        Complex array of shape (2, N) containing measurement signals.
        First row should contain ground state signals, second row excited state signals.
    plot : bool, default=True
        If True, generate visualization plots of the analysis results.
    backend : Literal["center", "regression"], default="regression"
        Method used for determining optimal rotation angle:
        - "center": Uses median of ground and excited signal clusters to determine rotation.
        - "regression": Uses logistic regression to find optimal decision boundary.

    Returns
    -------
    Tuple[float, float, float]
        A tuple containing:
        - fidelity: The assignment fidelity between ground and excited states (0.5-1.0)
        - threshold: The optimal threshold value for state discrimination
        - theta_deg: The optimal rotation angle in degrees
    """
    if backend == "center":
        return fit_ge_by_center(signals, plot=plot)
    elif backend == "regression":
        return fit_ge_by_regression(signals, plot=plot)


__all__ = ["singleshot_visualize", "singleshot_ge_analysis", "fidelity_func"]
