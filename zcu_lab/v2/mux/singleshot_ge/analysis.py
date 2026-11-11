"""Per-original-tone GE fits and plots using shared numerical tools."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Polygon
from numpy.typing import NDArray
from pydantic import Field
from pydantic.dataclasses import dataclass as pydantic_dataclass
from zcu_tools.analysis.failure import AnalysisFailed
from zcu_tools.analysis.fitting import FitQuality
from zcu_tools.analysis.singleshot import (
    GECalibration,
    GEConfusionResult,
    calc_confusion_matrix,
    classify_result,
)
from zcu_tools.experiment import RunRecord
from zcu_tools.experiment.utils.single_shot.ge import (
    GEFitOptions,
    GENumericalError,
    singleshot_ge_analysis,
)
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2.support.mux import MuxFitFailure
from zcu_lab.v2.support.singleshot.util import ge_signals_by_state

if TYPE_CHECKING:
    from .core import MuxGECfg, MuxGEResult


@pydantic_dataclass(frozen=True, config={"arbitrary_types_allowed": True})
class GEAnalyzeOptions:
    """Fit controls for one tone's raw probe-off/on IQ clouds.

    initial_state maps probe-off to predominantly ground or excited. backend
    selects PCA or center rotation; angle overrides the rotation in degrees.
    logscale selects histogram display scale. align_t1 enables T1 alignment;
    length_ratio is readout length / T1, or None to estimate it.
    """

    initial_state: Literal["ground", "excited"] = "ground"
    backend: Literal["pca", "center"] = "pca"
    logscale: bool = False
    align_t1: bool = True
    length_ratio: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    angle: float | None = Field(default=None, allow_inf_nan=False)


@dataclass(frozen=True)
class GEAnalysis:
    """Prepared-label assignment and model-based GE calibration estimates.

    initial_state maps probe off/on to predominantly g/e preparation rows, not
    pure states. fidelity is the empirical equal-weight prepared-label assignment
    score at the contrast-maximizing histogram edge, not latent pure-state readout
    fidelity. theta rotates IQ by exp(1j*theta), in radians; threshold is the
    selected edge in rotated ADC units. ge_s and g_center/e_center are fitted
    width and original-frame complex centers in normalized ADC units.
    init_pops is the fitted (2,2) g/e population matrix for the preparation rows.
    length_ratio is readout length / T1 (dimensionless), or None when the data
    cannot identify it.
    fit_quality describes those histogram fits, or None when unavailable.
    Poor histogram residuals make populations and the population-corrected post
    confusion matrix unreliable even when empirical assignment is high.
    """

    initial_state: Literal["ground", "excited"]
    fidelity: float
    theta: float
    threshold: float
    ge_s: float
    g_center: complex
    e_center: complex
    init_pops: NDArray[np.float64]
    length_ratio: float | None
    fit_quality: Mapping[str, FitQuality] | None = None

    def validate_calibration(self) -> None:
        """Raise ValueError when calibration fields violate these conditions.

        Fidelity, theta, threshold, ge_s and both centers must be finite, with
        fidelity in [0,1], ge_s positive and centers distinct under np.isclose.
        init_pops must be finite, nonnegative and (2,2), with row sums <= 1+1e-12.
        """
        populations = np.asarray(self.init_pops, dtype=np.float64)
        scalars = [
            self.fidelity,
            self.theta,
            self.threshold,
            self.ge_s,
            self.g_center,
            self.e_center,
        ]
        if (
            not np.isfinite(scalars).all()
            or not 0.0 <= self.fidelity <= 1.0
            or self.ge_s <= 0
            or np.isclose(self.g_center, self.e_center)
            or populations.shape != (2, 2)
            or not np.isfinite(populations).all()
            or np.any(populations < 0)
            or np.any(populations.sum(axis=1) > 1 + 1e-12)
        ):
            raise ValueError(
                "Invalid GE calibration: check centers, width and populations"
            )


@pydantic_dataclass(frozen=True, config={"arbitrary_types_allowed": True})
class GEPostAnalyzeOptions:
    """One tone's radius in normalized ADC units (None: automatic selection).

    consider_other adds the midpoint leakage penalty during selection only.
    """

    radius: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    consider_other: bool = False


@dataclass(frozen=True)
class GEPostAnalysis:
    """Single-qubit radius and preparation-corrected g/e/other diagnostic."""

    confusion: GEConfusionResult


@pydantic_dataclass(frozen=True, config={"arbitrary_types_allowed": True})
class MuxGEAnalyzeOptions:
    """Existing GE fit options, common default plus original-index overrides.

    default applies to every active tone unless overrides contains its original
    index. Each GEAnalyzeOptions supplies analysis-only ground/excited initial
    state, PCA/center backend, degree angle, T1 alignment, dimensionless length
    ratio and histogram log scale. Unknown/nonactive override indices raise.
    """

    default: GEAnalyzeOptions = field(default_factory=GEAnalyzeOptions)
    overrides: Mapping[int, GEAnalyzeOptions] = field(default_factory=dict)


@pydantic_dataclass(frozen=True, config={"arbitrary_types_allowed": True})
class MuxGEPostOptions:
    """Existing GE radius policy, common default plus original-index overrides.

    radius is None for automatic selection or nonnegative finite normalized ADC
    units. consider_other changes only automatic-radius penalty, not the isolated
    other row of the 3x3 diagnostic. Unknown/nonactive override indices raise.
    """

    default: GEPostAnalyzeOptions = field(default_factory=GEPostAnalyzeOptions)
    overrides: Mapping[int, GEPostAnalyzeOptions] = field(default_factory=dict)


@dataclass(frozen=True)
class MuxGEAnalysis:
    """Original tone index and adopted physical g/e primary values/fit quality."""

    index: int
    primary: GEAnalysis


@dataclass(frozen=True)
class MuxGEPostAnalysis:
    """Original tone index and single-qubit 3x3 confusion diagnostic, not joint."""

    index: int
    post: GEPostAnalysis


def _failure_plot(plots: Plots, name: str, reason: str) -> None:
    if name not in plots:
        _, ax = plots.subplots(num=name)
        ax.text(0.02, 0.98, reason, transform=ax.transAxes, va="top", wrap=True)
    else:
        figure = plots[name]

        @plots.on_plot_thread
        def update(message: str):
            return (figure.suptitle(message, wrap=True),)

        update(reason)


def analyze_ge(
    source: RunRecord[MuxGECfg, MuxGEResult],
    options: MuxGEAnalyzeOptions,
    *,
    plots: Plots,
) -> tuple[MuxGEAnalysis | MuxFitFailure, ...]:
    """Fit finite paired preparation rows without filtering or reordering shots.

    source is a validated explicit (K,2,S) raw IQ result in normalized ADC units.
    cfg=None is supported. Validate option keys before fitting; invalid options
    raise. Each original index gets fit_i and either existing GE calibration /
    quality or a numerical/missing-preparation failure reason. No calibration
    is invented for partial data. GENumericalError and AnalysisFailed are isolated;
    plotting and other errors propagate unchanged. Caller owns plots finish/release.
    No hardware, current cfg, binding or proposal access occurs.
    """
    indices = [int(i) for i in source.result.tone_indices]
    if any(type(i) is not int or i not in indices for i in options.overrides):
        raise ValueError("GE overrides must name active original tone indices")
    if any(f"fit_{i}" in plots for i in indices):
        raise ValueError("fit_i figures already exist in this operation")
    selected = [options.overrides.get(i, options.default) for i in indices]
    outcomes: list[MuxGEAnalysis | MuxFitFailure] = []
    for row, (index, option) in enumerate(zip(indices, selected, strict=True)):
        name = f"fit_{index}"
        signals = ge_signals_by_state(source.result.signals[row], option.initial_state)
        _, axes = plots.subplots(num=name)
        for cloud, label, color in zip(
            signals,
            ("g preparation", "e preparation"),
            ("tab:blue", "tab:red"),
            strict=True,
        ):
            axes.scatter(
                cloud.real, cloud.imag, s=8, alpha=0.5, color=color, label=label
            )
        axes.set(xlabel="I (a.u.)", ylabel="Q (a.u.)", title="GE IQ")
        axes.set_aspect("equal", adjustable="datalim")
        axes.grid()
        axes.legend()
        if not np.isfinite(signals).all():
            reason = "GE fit requires both complete probe-off/on preparations"
            _failure_plot(plots, name, reason)
            outcomes.append(MuxFitFailure(index, reason))
            continue
        try:
            fidelity, init_pops, fit = singleshot_ge_analysis(
                signals,
                angle=option.angle,
                backend=option.backend,
                options=GEFitOptions(
                    logscale=option.logscale,
                    align_t1=option.align_t1,
                    length_ratio=option.length_ratio,
                ),
                plots=None,
            )
            primary = GEAnalysis(
                initial_state=option.initial_state,
                fidelity=fidelity,
                theta=fit["theta"],
                threshold=fit["threshold"],
                ge_s=fit["s"],
                g_center=fit["g_center"],
                e_center=fit["e_center"],
                init_pops=init_pops,
                length_ratio=fit["length_ratio"],
                fit_quality=fit["fit_quality"],
            )
            primary.validate_calibration()
            axes.set_title(
                "GE IQ (length ratio: not identifiable)"
                if primary.length_ratio is None
                else f"GE IQ (length ratio = {primary.length_ratio:.2f})"
            )
            outcomes.append(MuxGEAnalysis(index, primary))
        except (GENumericalError, AnalysisFailed) as exc:
            _failure_plot(plots, name, str(exc))
            outcomes.append(MuxFitFailure(index, str(exc)))
    return tuple(outcomes)


def post_analyze_ge(
    source: RunRecord[MuxGECfg, MuxGEResult],
    primary: tuple[MuxGEAnalysis | MuxFitFailure, ...],
    options: MuxGEPostOptions,
    *,
    plots: Plots,
) -> tuple[MuxGEPostAnalysis | MuxFitFailure, ...]:
    """Classify explicit paired IQ with the adopted same-source primary, no refit.

    primary must contain exactly one outcome per active original index in source
    order. Caller must retain its source association. cfg=None is supported.
    Invalid keys/radii or index mismatch raise before diagnostics. Publish post_i;
    retain primary failures and per-tone numerical failures without fabricating a
    calibration. Values are existing GE 3x3 diagnostics, not four-state assignment
    matrices. GENumericalError and AnalysisFailed are isolated; plotting and other
    errors propagate unchanged. Caller finishes/releases plots; no hardware or live
    source is read.
    """
    indices = [int(i) for i in source.result.tone_indices]
    if [item.index for item in primary] != indices:
        raise ValueError("primary indices must match source active original order")
    if any(type(i) is not int or i not in indices for i in options.overrides):
        raise ValueError("post overrides must name active original tone indices")
    if any(f"post_{i}" in plots for i in indices):
        raise ValueError("post_i figures already exist in this operation")
    selected = [options.overrides.get(i, options.default) for i in indices]
    outcomes: list[MuxGEPostAnalysis | MuxFitFailure] = []
    for row, (item, option) in enumerate(zip(primary, selected, strict=True)):
        name = f"post_{item.index}"
        if isinstance(item, MuxFitFailure):
            _failure_plot(plots, name, item.reason)
            outcomes.append(item)
            continue
        if not np.isfinite(source.result.signals[row]).all():
            reason = "GE post requires both complete probe-off/on preparations"
            _failure_plot(plots, name, reason)
            outcomes.append(MuxFitFailure(item.index, reason))
            continue
        item.primary.validate_calibration()
        g_signals, e_signals = ge_signals_by_state(
            source.result.signals[row], item.primary.initial_state
        )
        signals = (g_signals, e_signals)
        try:
            confusion = calc_confusion_matrix(
                signals,
                GECalibration(
                    g_center=item.primary.g_center,
                    e_center=item.primary.e_center,
                    init_pops=item.primary.init_pops,
                    sigma=item.primary.ge_s,
                ),
                radius=option.radius,
                consider_other=option.consider_other,
            )
        except (
            ValueError,
            RuntimeError,
            ArithmeticError,
            np.linalg.LinAlgError,
        ) as exc:
            _failure_plot(plots, name, str(exc))
            outcomes.append(MuxFitFailure(item.index, str(exc)))
            continue
        _plot_confusion_matrix(confusion, signals, item.primary, plots, name)
        outcomes.append(MuxGEPostAnalysis(item.index, GEPostAnalysis(confusion)))
    return tuple(outcomes)


def _plot_confusion_matrix(
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
    fig, _ = plots.subplots(
        num=figure_name, nrows=2, ncols=2, figsize=(8, 8), layout="constrained"
    )
    g_iq_ax, initial_populations_ax, e_iq_ax, confusion_matrix_ax = fig.axes

    g_label = r"$|0\rangle$"
    e_label = r"$|1\rangle$"
    l_label = r"$|L\rangle$"
    _plot_classified(g_iq_ax, g_signals, primary, confusion.radius)
    _plot_classified(e_iq_ax, e_signals, primary, confusion.radius)

    n_gg, n_ge, n_go = confusion.g_classification
    n_eg, n_ee, n_eo = confusion.e_classification
    g_iq_ax.set_title(
        f"{g_label}: {n_gg:.1%}, {e_label}: {n_ge:.1%}, {l_label}: {n_go:.1%}"
    )
    e_iq_ax.set_title(
        f"{g_label}: {n_eg:.1%}, {e_label}: {n_ee:.1%}, {l_label}: {n_eo:.1%}"
    )
    g_iq_ax.set_xlabel("")

    for matrix_ax, matrix in (
        (initial_populations_ax, confusion.init_matrix),
        (confusion_matrix_ax, confusion.matrix),
    ):
        im = matrix_ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1)
        fig.colorbar(im, cax=matrix_ax.inset_axes((1.04, 0, 0.06, 1)))
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                val = matrix[i, j]
                matrix_ax.text(
                    j,
                    i,
                    f"{val:.2%}",
                    fontsize=8,
                    ha="center",
                    va="center",
                    color="white" if val > 0.5 else "black",
                )

    initial_populations_ax.set_xticks([0, 1, 2])
    initial_populations_ax.set_yticks([0, 1, 2])
    initial_populations_ax.set_xticklabels([g_label, e_label, l_label])
    initial_populations_ax.set_yticklabels([g_label, e_label, l_label])
    initial_populations_ax.set_xlabel("Actual State")
    initial_populations_ax.set_ylabel("Prepared State")
    initial_populations_ax.set_title("Initial Populations")

    confusion_matrix_ax.set_xticks([0, 1, 2])
    confusion_matrix_ax.set_yticks([0, 1, 2])
    confusion_matrix_ax.set_xticklabels([g_label, e_label, l_label])
    confusion_matrix_ax.set_yticklabels([g_label, e_label, l_label])
    confusion_matrix_ax.set_xlabel("Measured State")
    confusion_matrix_ax.set_ylabel("Actual State")
    confusion_matrix_ax.set_title(
        f"Confusion Matrix (cond: {confusion.condition_number:.1f})"
    )


def _plot_classified(
    ax: Axes,
    signals: NDArray[np.complex128],
    primary: GEAnalysis,
    radius: float,
) -> None:
    """Draw one tone's classified shots and nearest-center radius boundaries."""
    g_center, e_center = primary.g_center, primary.e_center
    max_point = 5000
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
