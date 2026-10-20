from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray

from zcu_tools.analysis.fitting.singleshot import transition_state_bin_probabilities
from zcu_tools.analysis.fitting.singleshot_rabi import RabiJointFitResult
from zcu_tools.analysis.singleshot import classify_rabi_iq
from zcu_tools.plotting.plots import Plots


def _effective_t1_us(ro_length_us: float | None, length_ratio: float) -> float | None:
    if ro_length_us is None or not np.isfinite(length_ratio) or length_ratio <= 0.0:
        return None
    value = ro_length_us / length_ratio
    return value if np.isfinite(value) and value > 0.0 else None


def _plot_initial_histogram(
    ax: Axes,
    fit: RabiJointFitResult,
    effective_t1_us: float | None,
    classified_distribution: NDArray[np.float64] | None,
) -> None:
    counts = fit.projection.counts[0]
    edges = fit.projection.bin_edges
    ax.stairs(counts, edges, label="Observed counts", color="black")

    fitted_values = np.array(
        [
            fit.projected_g_center,
            fit.projected_e_center,
            fit.sigma,
            fit.p_avg,
            fit.length_ratio,
            fit.fitted_populations[0, 0],
            fit.fitted_populations[0, 1],
        ]
    )
    if fit.backend.valid and np.isfinite(fitted_values).all():
        qg, qe = transition_state_bin_probabilities(
            edges,
            fit.projected_g_center,
            fit.projected_e_center,
            fit.sigma,
            fit.p_avg,
            fit.length_ratio,
        )
        p_g, p_e = fit.fitted_populations[0, :2]
        shots = float(counts.sum())
        g_counts = shots * p_g * qg
        e_counts = shots * p_e * qe
        ax.stairs(g_counts + e_counts, edges, label="Fitted total", color="purple")
        ax.stairs(
            g_counts,
            edges,
            label="Ground contribution",
            color="blue",
            linestyle="--",
        )
        ax.stairs(
            e_counts,
            edges,
            label="Excited contribution",
            color="red",
            linestyle="--",
        )
        title = "Initial-point projected IQ histogram"
        if effective_t1_us is not None:
            title += f"\nEffective $T_1$ = {effective_t1_us:.3g} μs"
        ax.set_title(title, fontsize=10)
        if (
            classified_distribution is not None
            and classified_distribution.shape == (3,)
            and np.isfinite(classified_distribution).all()
        ):
            g_population, e_population, l_population = classified_distribution
            ax.text(
                0.98,
                0.96,
                "Classified G/E/L\n"
                f"G {g_population:.1%}\n"
                f"E {e_population:.1%}\n"
                f"L {l_population:.1%}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                bbox={"facecolor": "white", "edgecolor": "#c9cdd3", "alpha": 0.9},
            )
    else:
        ax.set_title(
            "Initial-point projected IQ histogram\nJoint fit invalid",
            fontsize=10,
        )
    ax.set_xlabel("Pooled PCA projection (a.u.)")
    ax.set_ylabel("Counts")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(True)


def _plot_confusion_matrix(ax: Axes, fit: RabiJointFitResult) -> None:
    labels = ("g", "e", "other")
    matrix = fit.confusion_matrix
    if fit.backend.valid and np.isfinite(matrix).all():
        ax.imshow(matrix, cmap="Blues", vmin=0.0, vmax=1.0)
        for row in range(3):
            for column in range(3):
                value = matrix[row, column]
                ax.text(
                    column,
                    row,
                    f"{value:.3f}",
                    ha="center",
                    va="center",
                    color="white" if value > 0.5 else "black",
                    fontsize=8,
                )
        ax.set_title("Fitted confusion matrix\nOther row fixed by model", fontsize=10)
    else:
        ax.set_xlim(-0.5, 2.5)
        ax.set_ylim(2.5, -0.5)
        ax.text(1.0, 1.0, "Joint fit invalid", ha="center", va="center")
        ax.set_title("Confusion matrix unavailable", fontsize=10)
    ax.set_xticks(range(3), labels)
    ax.set_yticks(range(3), labels)
    ax.set_xlabel("Classified state")
    ax.set_ylabel("True state")


@dataclass(frozen=True)
class RabiJointPlotOptions:
    """Caller-owned display settings for a joint Rabi chart.

    ro_length_us is the readout duration in microseconds, or None to omit the
    effective-T1 annotation. Nonpositive/nonfinite values also omit it.
    sweep selects pulse-length (microseconds) or gain coordinates and labels.
    figsize is the base (width, height) in inches; the chart scales height by 1.6.
    """

    ro_length_us: float | None
    sweep: Literal["length", "gain"]
    figsize: tuple[float, float]


def plot_rabi_joint(
    xs: NDArray[np.float64],
    signals: NDArray[np.complex128],
    fit: RabiJointFitResult,
    options: RabiJointPlotOptions,
    *,
    plots: Plots,
) -> None:
    """Adopt a joint Rabi population/histogram/confusion chart as plots["fit"].

    xs is the sweep axis matching fit populations; signals is raw complex IQ
    (sweep, shot). fit is the numerical RabiJointFitResult. options supplies
    readout duration, sweep coordinates and base figure size; a nonpositive or
    nonfinite fit length ratio also omits the effective-T1 annotation.
    plots owns the adopted figure, not this function. Invalid fit
    states retain measured counts and show invalid-fit annotations. Shape,
    classification, figure-name conflicts and Matplotlib failures propagate.
    No cfg, program or experiment defaults are read."""
    omega_unit = "rad/μs" if options.sweep == "length" else "rad/gain"
    width, height = options.figsize
    fig = Figure(figsize=(width, height * 1.6), layout="constrained")
    plots.adopt("fit", fig)
    grid = fig.add_gridspec(
        2,
        2,
        height_ratios=(2.0, 1.2),
        hspace=0.16,
        wspace=0.12,
    )
    ax = fig.add_subplot(grid[0, :])
    histogram_ax = fig.add_subplot(grid[1, 0])
    confusion_ax = fig.add_subplot(grid[1, 1])

    colors = ("blue", "red", "green")
    labels = ("$|0\\rangle$", "$|1\\rangle$", "$|L\\rangle$")
    for index, (color, label) in enumerate(zip(colors, labels, strict=True)):
        ax.plot(
            xs,
            fit.measured_populations[:, index],
            color=color,
            ls="none",
            marker="o",
            markersize=3,
            label=label,
        )
        ax.plot(
            xs,
            fit.fitted_populations[:, index],
            color=color,
            ls="-",
        )
    if fit.backend.valid:
        ax.set_title(
            f"$\\Omega$={fit.omega:.3g} {omega_unit}, "
            f"phase={np.degrees(fit.phase):.2f}°, cond={fit.condition_number:.3g}"
        )
    else:
        ax.set_title("Rabi joint fit invalid")
    ax.set_xlabel(
        "Pulse length (μs)" if options.sweep == "length" else "Pulse gain (a.u.)"
    )
    ax.set_ylabel("Population")
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc=4)
    ax.grid(True)
    effective_t1_us = (
        _effective_t1_us(options.ro_length_us, fit.length_ratio)
        if fit.backend.valid
        else None
    )
    classified_distribution = None
    if fit.backend.valid:
        classified_ge = classify_rabi_iq(
            signals[:1],
            fit.g_center,
            fit.e_center,
            fit.radius,
        )[0]
        classified_distribution = np.array(
            (*classified_ge, 1.0 - classified_ge.sum()), dtype=np.float64
        )
    _plot_initial_histogram(
        histogram_ax,
        fit,
        effective_t1_us,
        classified_distribution,
    )
    _plot_confusion_matrix(confusion_ax, fit)
