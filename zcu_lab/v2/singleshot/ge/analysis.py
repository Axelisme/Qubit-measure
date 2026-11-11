"""GE radius selection, preparation correction and confusion diagnostics."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from zcu_tools.analysis.singleshot import GEConfusionResult
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2.singleshot.plot import plot_with_classified

if TYPE_CHECKING:
    from .core import GEAnalysis


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
    plot_with_classified(ax1, g_signals, g_center, e_center, confusion.radius)
    plot_with_classified(ax2, e_signals, g_center, e_center, confusion.radius)

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
