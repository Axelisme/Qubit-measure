"""GE radius selection, preparation correction and confusion diagnostics."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Polygon
from numpy.typing import NDArray
from zcu_tools.analysis.singleshot import GEConfusionResult, classify_result
from zcu_tools.plotting.plots import Plots

if TYPE_CHECKING:
    from .core import GEAnalysis


def _plot_classified_iq(
    ax: Axes,
    signals: NDArray[np.complex128],
    g_center: complex,
    e_center: complex,
    radius: float,
    max_point: int = 5000,
) -> None:
    # Classify shots
    """Draw classified IQ points, centers and clipped circle boundaries on ax.

    signals is a nonempty 1D complex shot vector. g_center/e_center and radius
    follow analysis.singleshot.classify_result; max_point limits downsampling
    and must be positive. Mutates ax, adds labels/legend/equal aspect, and
    returns None. Invalid classification geometry raises ValueError; Matplotlib
    errors propagate. The caller owns the axes and figure lifecycle."""
    mask_g, mask_e, mask_o = classify_result(signals, g_center, e_center, radius)

    colors = np.full(signals.shape, "gray", dtype=object)
    colors[mask_g] = "blue"
    colors[mask_e] = "red"
    colors[mask_o] = "green"

    # plot shots
    num_sample = signals.shape[0]
    downsample_num = min(max_point, num_sample)
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
    ax.plot(
        g_center.real,
        g_center.imag,
        markerfacecolor="b",
        label="Ground",
        linestyle=":",
        color="k",
        marker="o",
        markersize=5,
    )
    ax.plot(
        e_center.real,
        e_center.imag,
        markerfacecolor="r",
        label="Excited",
        linestyle=":",
        color="k",
        marker="o",
        markersize=5,
    )
    separation = abs(e_center - g_center)
    axis = (e_center - g_center) / separation
    cut_angle = np.arccos(separation / (2 * radius)) if radius > separation / 2 else 0.0
    angles = np.linspace(cut_angle, 2 * np.pi - cut_angle, 361)
    for center, direction, color in ((g_center, axis, "b"), (e_center, -axis, "r")):
        boundary = center + direction * radius * np.exp(1j * angles)
        ax.add_patch(
            Polygon(
                np.column_stack((boundary.real, boundary.imag)),
                closed=True,
                color=color,
                fill=False,
                linestyle="--",
            )
        )

    ax.set_aspect("equal")
    ax.legend()
    ax.set_xlabel("I value (a.u.)")
    ax.set_ylabel("Q value (a.u.)")


def plot_confusion_matrix(
    confusion: GEConfusionResult,
    signals: tuple[NDArray[np.complex128], NDArray[np.complex128]],
    primary: GEAnalysis,
    plots: Plots,
    figure_name: str,
) -> None:
    """Publish IQ classifications and preparation/confusion matrices to plots.

    signals contains physical g/e IQ arrays; primary supplies their centers.
    figure_name must be unused. Plot ownership and duplicate names follow Plots.
    """
    g_signals, e_signals = signals
    g_center, e_center = primary.g_center, primary.e_center
    fig, _ = plots.subplots(
        num=figure_name, nrows=2, ncols=2, figsize=(8, 8), layout="constrained"
    )
    ax1, ax4, ax2, ax3 = fig.axes

    g_label = r"$|0\rangle$"
    e_label = r"$|1\rangle$"
    l_label = r"$|L\rangle$"
    _plot_classified_iq(ax1, g_signals, g_center, e_center, confusion.radius)
    _plot_classified_iq(ax2, e_signals, g_center, e_center, confusion.radius)

    n_gg, n_ge, n_go = confusion.g_classification
    n_eg, n_ee, n_eo = confusion.e_classification
    ax1.set_title(
        f"{g_label}: {n_gg:.1%}, {e_label}: {n_ge:.1%}, {l_label}: {n_go:.1%}"
    )
    ax2.set_title(
        f"{g_label}: {n_eg:.1%}, {e_label}: {n_ee:.1%}, {l_label}: {n_eo:.1%}"
    )
    ax1.set_xlabel("")

    im = ax4.imshow(confusion.init_matrix, cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, cax=ax4.inset_axes((1.04, 0, 0.06, 1)))
    for i in range(confusion.init_matrix.shape[0]):
        for j in range(confusion.init_matrix.shape[1]):
            val = confusion.init_matrix[i, j]
            ax4.text(
                j,
                i,
                f"{val:.2%}",
                fontsize=8,
                ha="center",
                va="center",
                color="white" if val > 0.5 else "black",
            )

    ax4.set_xticks([0, 1, 2])
    ax4.set_yticks([0, 1, 2])
    ax4.set_xticklabels([g_label, e_label, l_label])
    ax4.set_yticklabels([g_label, e_label, l_label])
    ax4.set_xlabel("Actual State")
    ax4.set_ylabel("Prepared State")
    ax4.set_title("Initial Populations")

    im = ax3.imshow(confusion.matrix, cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, cax=ax3.inset_axes((1.04, 0, 0.06, 1)))
    for i in range(confusion.matrix.shape[0]):
        for j in range(confusion.matrix.shape[1]):
            val = confusion.matrix[i, j]
            ax3.text(
                j,
                i,
                f"{val:.2%}",
                fontsize=8,
                ha="center",
                va="center",
                color="white" if val > 0.5 else "black",
            )

    ax3.set_xticks([0, 1, 2])
    ax3.set_yticks([0, 1, 2])
    ax3.set_xticklabels([g_label, e_label, l_label])
    ax3.set_yticklabels([g_label, e_label, l_label])
    ax3.set_xlabel("Measured State")
    ax3.set_ylabel("Actual State")
    ax3.set_title(f"Confusion Matrix (cond: {confusion.condition_number:.1f})")
