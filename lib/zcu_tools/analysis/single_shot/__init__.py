from typing import Literal

import matplotlib.pyplot as plt
import numpy as np

from .base import fidelity_func
from .center import fit_ge_by_center
from .regression import fit_ge_by_regression


def singleshot_visualize(signals, plot_center=True):
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
    signals, plot=True, backend: Literal["center", "regression"] = "regression"
):
    if backend == "center":
        return fit_ge_by_center(signals, plot=plot)
    elif backend == "regression":
        return fit_ge_by_regression(signals, plot=plot)


__all__ = ["singleshot_visualize", "singleshot_ge_analysis", "fidelity_func"]
