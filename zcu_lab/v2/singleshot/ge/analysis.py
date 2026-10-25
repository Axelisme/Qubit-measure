"""GE radius selection, preparation correction and confusion diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
from numpy.typing import NDArray
from zcu_tools.analysis.singleshot import classify_result
from zcu_tools.plotting.plots import Plots
from zcu_tools.utils.shot_classification import gaussian_region_probability

from zcu_lab.v2.singleshot.plot import plot_with_classified

if TYPE_CHECKING:
    from .core import GEAnalysis, GEPostAnalyzeOptions


def make_init_matrix(init_pops: NDArray[np.float64]) -> NDArray[np.float64]:
    """Extend (2,2) g/e preparation populations to a 3x3 g/e/other matrix.

    The other preparation is isolated. Invalid probabilities raise ValueError.
    """
    # Each acquisition has its own refined preparation populations. Retaining
    # both rows makes the diagnostic equivariant under g/e relabeling.
    if (
        init_pops.shape != (2, 2)
        or not np.isfinite(init_pops).all()
        or np.any(init_pops < 0)
        or np.any(init_pops.sum(axis=1) > 1 + 1e-12)
    ):
        raise ValueError("GE initial populations must be a valid 2x2 probability array")
    p_gg_init, p_ge_init = init_pops[0]
    p_eg_init, p_ee_init = init_pops[1]
    p_go_init = max(0.0, 1.0 - p_gg_init - p_ge_init)
    p_eo_init = max(0.0, 1.0 - p_eg_init - p_ee_init)

    return np.array(
        [
            [p_gg_init, p_ge_init, p_go_init],
            [p_eg_init, p_ee_init, p_eo_init],
            [0.0, 0.0, 1.0],  # assume all initial to O
        ]
    )


class GEClassification(NamedTuple):
    """One prepared state's fractions classified as ground, excited and other."""

    ground: float
    excited: float
    other: float


ISOLATED_OTHER_CLASSIFICATION = GEClassification(0.0, 0.0, 1.0)


def make_result_matrix(
    ground: GEClassification,
    excited: GEClassification,
    other: GEClassification = ISOLATED_OTHER_CLASSIFICATION,
) -> NDArray[np.float64]:
    """Stack ground/excited/other preparation rows into a 3x3 result matrix.

    Columns follow GEClassification order. The default other-state row models
    perfectly isolated classification. Values are preserved without normalization.
    """
    return np.array([ground, excited, other])


def solve_confusion_matrix(
    A_init: NDArray[np.float64],
    Q: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Solve preparation mixing A_init @ confusion = Q for 3x3 matrices.

    Clip negative entries and normalize rows. Singular A_init raises LinAlgError.
    """
    confusion_matrix = np.linalg.solve(A_init, Q)
    confusion_matrix = np.clip(confusion_matrix, 0.0, None)
    confusion_matrix /= confusion_matrix.sum(axis=1, keepdims=True)

    return confusion_matrix


# tolerated: One caller supplies shot arrays, fit calibration and the other-state penalty switch.
def optimize_ge_radius(  # noqa: PLR0913
    g_signals: NDArray[np.complex128],
    e_signals: NDArray[np.complex128],
    g_center: complex,
    e_center: complex,
    init_pops: NDArray[np.float64],
    sigma: float,
    *,
    consider_other: bool = True,
) -> float:
    """Select a radius in normalized IQ units minimizing matrix condition number.

    Use fitted centers, width sigma and (2,2) preparation populations. When
    consider_other is true, penalize midpoint other-state leakage during radius
    selection only. Invalid geometry and unsuccessful optimization raise.
    """
    from scipy.optimize import OptimizeResult, minimize_scalar

    # Validate geometry even before the optimizer evaluates its first candidate.
    classify_result(np.empty(0, dtype=np.complex128), g_center, e_center, 0.0)
    ge_dist = abs(g_center - e_center)
    A_init = make_init_matrix(init_pops)

    def loss_fn(radius: float) -> float:

        gg_mask, ge_mask, go_mask = classify_result(
            g_signals, g_center, e_center, radius
        )
        n_gg = gg_mask.sum() / gg_mask.shape[0]
        n_ge = ge_mask.sum() / ge_mask.shape[0]
        n_go = go_mask.sum() / go_mask.shape[0]

        eg_mask, ee_mask, eo_mask = classify_result(
            e_signals, g_center, e_center, radius
        )
        n_eg = eg_mask.sum() / eg_mask.shape[0]
        n_ee = ee_mask.sum() / ee_mask.shape[0]
        n_eo = eo_mask.sum() / eo_mask.shape[0]

        # Radius-selection penalty only: model other at the midpoint.
        # The returned calibration retains the isolated-other row.
        n_og = (
            float(
                gaussian_region_probability(
                    np.array([ge_dist / 2]), sigma, radius, ge_dist
                )[0]
            )
            if consider_other
            else 0.0
        )
        n_oe = n_og
        n_oo = 1.0 - n_og - n_oe

        Q = make_result_matrix(
            GEClassification(n_gg, n_ge, n_go),
            GEClassification(n_eg, n_ee, n_eo),
            GEClassification(n_og, n_oe, n_oo),
        )
        confusion_matrix = solve_confusion_matrix(A_init, Q)

        # calculate condision number of confusion matrix as loss
        condition = float(np.linalg.cond(confusion_matrix))
        return condition if np.isfinite(condition) else 1e12

    result = minimize_scalar(loss_fn, bounds=(0.0, ge_dist), method="bounded")
    if not isinstance(result, OptimizeResult):
        raise TypeError("classification-radius optimizer must return OptimizeResult")
    if not result.success or not np.isfinite(result.x):
        raise RuntimeError("classification-radius optimization failed")
    # The optimum can be the upper bound; bounded minimization excludes endpoints.
    return min((float(result.x), ge_dist), key=loss_fn)


@dataclass(frozen=True)
class GEConfusionResult:
    """Radius in normalized IQ units and population-corrected g/e/other matrix.

    matrix and init_matrix are (3,3), with g/e/other rows and columns.
    g_classification/e_classification are observed fractions before correction.
    condition_number describes matrix invertibility, not calibration approval.
    """

    radius: float
    matrix: NDArray[np.float64]
    init_matrix: NDArray[np.float64]
    g_classification: GEClassification
    e_classification: GEClassification
    condition_number: float


def calc_confusion_matrix(
    signals: tuple[NDArray[np.complex128], NDArray[np.complex128]],
    primary: GEAnalysis,
    options: GEPostAnalyzeOptions,
) -> GEConfusionResult:
    """Classify physical g/e shot arrays and correct fitted preparation mixing.

    primary supplies fitted centers, width and populations; options selects an
    explicit radius or automatic optimization. Numerical failures propagate.
    """
    g_signals, e_signals = signals
    g_center, e_center = primary.g_center, primary.e_center
    init_matrix = make_init_matrix(primary.init_pops)

    radius = options.radius
    if radius is None:
        radius = optimize_ge_radius(
            g_signals,
            e_signals,
            g_center,
            e_center,
            primary.init_pops,
            primary.ge_s,
            consider_other=options.consider_other,
        )

    gg_mask, ge_mask, go_mask = classify_result(g_signals, g_center, e_center, radius)
    g_classification = GEClassification(
        float(gg_mask.mean()),
        float(ge_mask.mean()),
        float(go_mask.mean()),
    )

    eg_mask, ee_mask, eo_mask = classify_result(e_signals, g_center, e_center, radius)
    e_classification = GEClassification(
        float(eg_mask.mean()),
        float(ee_mask.mean()),
        float(eo_mask.mean()),
    )

    matrix = solve_confusion_matrix(
        init_matrix,
        make_result_matrix(g_classification, e_classification),
    )
    return GEConfusionResult(
        radius=radius,
        matrix=matrix,
        init_matrix=init_matrix,
        g_classification=g_classification,
        e_classification=e_classification,
        condition_number=float(np.linalg.cond(matrix)),
    )


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
        figure_name, nrows=2, ncols=2, figsize=(8, 8), layout="constrained"
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
