from __future__ import annotations

import math
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated, ClassVar, Literal

import numpy as np
from matplotlib.axes import Axes
from matplotlib.legend import Legend
from matplotlib.patches import Polygon
from matplotlib.ticker import MaxNLocator
from numpy.typing import NDArray
from pydantic import Field
from pydantic.dataclasses import dataclass as pydantic_dataclass
from zcu_tools.analysis.fitting import FitQuality
from zcu_tools.analysis.singleshot import (
    GECalibration,
    GEConfusionResult,
    calc_confusion_matrix,
    classify_result,
)
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.experiment import AxesSpec, Experiment, RunRecord
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.utils.single_shot.ge import (
    GEFitOptions,
    GENumericalError,
    singleshot_ge_analysis,
)
from zcu_tools.experiment.v2.runtime import RunOptions
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import Pulse, PulseReadout, Reset, TimelineBuilder

from zcu_lab.v2.support.singleshot.util import ge_signals_by_state, raw_shots_to_signal


@dataclass(frozen=True)
class GE_Result:
    """Raw IQ rows in fixed probe-off/on acquisition order."""

    signals: NDArray[np.complex128]
    shot_indices: NDArray[np.int64]


@pydantic_dataclass(frozen=True, config={"arbitrary_types_allowed": True})
class GEAnalyzeOptions:
    initial_state: Literal["ground", "excited"] = "ground"
    backend: Literal["pca", "center"] = "pca"
    logscale: bool = False
    align_t1: bool = True
    length_ratio: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    angle: float | None = Field(default=None, allow_inf_nan=False)


@pydantic_dataclass(frozen=True, config={"arbitrary_types_allowed": True})
class GEPostAnalyzeOptions:
    radius: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    consider_other: bool = False


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
        """Reject invalid physical GE calibration geometry or preparation populations."""
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


@dataclass(frozen=True)
class GEPostAnalysis:
    confusion: GEConfusionResult


def ge_calibration_review_note(primary: GEAnalysis) -> str | None:
    """Explain manual selection for single/mux GE model-derived calibrations.

    Joint histogram r2 must be >=0.9, normalized RMS <=0.1 and relative width
    error <=0.5. These dimensionless gates also guard empirical assignment and
    centers because a poor model is not calibration evidence. Required missing /
    nonfinite metrics or their invalid issues prevent preselection. Unrelated
    fixed-zero parameter issues do not. None means these review gates passed,
    not that preparation or pure-state readout fidelity has been established.
    """
    quality = None if primary.fit_quality is None else primary.fit_quality.get("joint")
    if quality is None:
        return "Review required: joint histogram quality unavailable"
    reasons = [
        f"joint {issue.path}: {issue.reason}"
        for issue in quality.invalid
        if issue.path
        in {"r2", "normalized_residual_rms", "relative_parameter_errors.s"}
    ]
    if quality.r2 is None or not math.isfinite(quality.r2):
        reasons.append("joint r² unavailable")
    elif quality.r2 < 0.9:
        reasons.append(f"joint r²={quality.r2:.4g} <0.9")
    rms = quality.normalized_residual_rms
    if rms is None or not math.isfinite(rms):
        reasons.append("joint normalized RMS unavailable")
    elif rms > 0.1:
        reasons.append(f"joint normalized RMS={rms:.4g} >0.1")
    width_error = quality.relative_parameter_errors.get("s")
    if width_error is None or not math.isfinite(width_error):
        reasons.append("relative width error unavailable")
    elif width_error > 0.5:
        reasons.append(f"relative width error={width_error:.4g} >0.5")
    return "Review required: " + "; ".join(reasons) if reasons else None


def ge_post_review_note(primary: GEAnalysis, condition_number: float) -> str | None:
    """Guard radius/confusion by primary GE quality and finite condition <=10.

    The condition is dimensionless. Manual draft selection remains possible;
    None means the review gates passed, not that the correction is reliable.
    """
    primary_note = ge_calibration_review_note(primary)
    reasons = [] if primary_note is None else [primary_note]
    if not math.isfinite(condition_number):
        reasons.append("post condition number unavailable")
    elif condition_number > 10:
        reasons.append(f"post condition number={condition_number:.4g} >10")
    return "; ".join(reasons) if reasons else None


def _annotate_primary(ax: Axes, analysis: GEAnalysis) -> None:
    """Overlay the fitted decision boundary and existing calibration diagnostics."""
    direction = np.exp(-1j * analysis.theta)
    anchor = analysis.threshold * direction
    tangent = anchor + 1j * direction
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    ax.axline(
        (float(anchor.real), float(anchor.imag)),
        (float(tangent.real), float(tangent.imag)),
        color="black",
        linestyle="--",
        label="Rotated I threshold",
    )
    ax.set(xlim=xlim, ylim=ylim)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
    lines = [
        f"Prepared-label assignment fidelity = {analysis.fidelity:.1%}",
        f"IQ rotation angle = {analysis.theta:.3g} rad · Rotated I threshold = {analysis.threshold:.3g} a.u.",
    ]
    for name in ("joint", "ground", "excited"):
        label = f"{name.capitalize()} histogram fit"
        quality = (
            None if analysis.fit_quality is None else analysis.fit_quality.get(name)
        )
        if quality is None:
            lines.append(f"{label} quality unavailable")
            continue
        r2 = "unavailable" if quality.r2 is None else f"{quality.r2:.3g}"
        rms = quality.normalized_residual_rms
        rms_label = "unavailable" if rms is None else f"{rms:.3g}"
        line = f"{label}: R² {r2} · normalized RMS {rms_label}"
        if name == "joint":
            width = quality.relative_parameter_errors.get("s")
            width_label = "unavailable" if width is None else f"{width:.3g}"
            line += f" · relative width error {width_label}"
        lines.append(line)
    if note := ge_calibration_review_note(analysis):
        lines.extend(note.split("; "))
    ax.set_title("\n".join(lines), fontsize=10)


def _plot_projection(
    plots: Plots,
    signals: NDArray[np.complex128],
    analysis: GEAnalysis,
    *,
    logarithmic: bool,
) -> None:
    """Compare the measured rotated-I distributions on common bins."""
    rotated = (signals * np.exp(1j * analysis.theta)).real
    bins = np.histogram_bin_edges(rotated[np.isfinite(rotated)], bins=50)
    _, ax = plots.subplots(num="projection", layout="constrained")
    for row, label, color in zip(
        rotated,
        ("g preparation", "e preparation"),
        ("tab:blue", "tab:orange"),
        strict=True,
    ):
        ax.hist(
            row[np.isfinite(row)],
            bins=bins.tolist(),
            histtype="step",
            linewidth=1.5,
            color=color,
            label=label,
        )
    ax.axvline(
        analysis.threshold, color="black", linestyle="--", label="Rotated I threshold"
    )
    ax.set(
        title="Single-shot GE · qubit readout · measured IQ projection",
        xlabel="Rotated IQ in-phase I (a.u.)",
        ylabel="Shots per bin (count)",
        yscale="log" if logarithmic else "linear",
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)


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
    and must be positive. Mutates ax, adds labels and equal aspect, and
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
        label="Ground circle center",
        linestyle=":",
        color="k",
        marker="o",
        markersize=5,
    )
    ax.plot(
        e_center.real,
        e_center.imag,
        markerfacecolor="r",
        label="Excited circle center",
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
    # Use the available I-axis width and prefer coarse steps on tall IQ panels.
    ax.xaxis.set_major_locator(MaxNLocator(nbins="auto", steps=[1, 2, 4, 5, 10]))
    ax.set_xlabel("IQ in-phase I (a.u.)")
    ax.set_ylabel("IQ quadrature Q (a.u.)")


class GE_Cfg(ConfigBase):
    reset: Annotated[
        Reset.Cfg | None,
        FieldMeta(
            label="Reset",
            description="Select the reset module applied before preparation, or leave empty to skip reset.",
        ),
    ] = None
    init_pulse: Annotated[
        Pulse.Cfg | None,
        FieldMeta(
            label="Initial preparation pulse",
            description="Select the pulse applied before the probe, or leave empty to omit it.",
        ),
    ] = None
    probe_pulse: Annotated[
        Pulse.Cfg,
        FieldMeta(
            label="π probe pulse",
            description="Enter the calibrated g↔e π pulse applied only in the probe-on preparation.",
        ),
    ]
    readout: Annotated[
        PulseReadout.Cfg,
        FieldMeta(
            label="Readout",
            description="Enter the readout module applied at the end of the shot.",
        ),
    ]
    shots: Annotated[
        int,
        FieldMeta(
            label="Shots per preparation",
            unit="count",
            description="Enter the number of IQ readouts for each of the π probe-off and probe-on preparations.",
        ),
    ]
    initial_delay: Annotated[
        float,
        FieldMeta(
            label="Initial delay", unit="µs", description="Delay before the first shot."
        ),
    ] = 1.0
    relax_delay: Annotated[
        float,
        FieldMeta(
            label="Relax delay",
            unit="µs",
            decimals=3,
            description="Delay after each shot.",
        ),
    ]


class GE_Exp(Experiment[GE_Result, GE_Cfg]):
    Options: ClassVar[type[GEAnalyzeOptions]] = GEAnalyzeOptions
    PostOptions: ClassVar[type[GEPostAnalyzeOptions]] = GEPostAnalyzeOptions

    AXES_SPEC: ClassVar[AxesSpec[GE_Result, GE_Cfg]] = AxesSpec(
        axes=(
            AxesSpec.Axis(
                "shot_indices", label="Shot Index", unit="None", dtype=np.int64
            ),
            AxesSpec.Axis(
                values=(0, 1), label="Prepared State", unit="None", dtype=np.int64
            ),
        ),
        z=AxesSpec.Z("signals", label="Signal", unit="a.u.", dtype=np.complex128),
        result_type=GE_Result,
        cfg_type=GE_Cfg,
        cfg_schema_version="3.0",
        tag="singleshot/ge",
    )

    def run(self, config: GE_Cfg, *, context: RunContext) -> GE_Result:
        """Collect probe-off/on IQ clouds for ground/excited readout calibration.

        Each shot applies optional reset and init, omits or applies the probe
        pulse, then reads out. Return complex IQ shaped (2, cfg.shots), shot
        indices, and prepared_states=[0, 1] denoting probe off/on, not guaranteed
        pure ground/excited states. Primary analysis maps these rows to physical
        G/E using its initial_state option; post-analysis derives the radius
        and confusion matrix from that fit.

        Use one round and cfg.shots reps on a copied cfg. Raw-shot conversion
        requires exactly one readout channel.
        """

        # 1. Configure raw-shot acquisition rather than averaging the IQ clouds.
        soc, soccfg = context.soc, context.soccfg
        cfg = deepcopy(config)

        figure, axes = context.plots.subplots(num="measurement", layout="constrained")
        legend: Legend | None = None

        @context.plots.on_plot_thread
        def update(clouds: NDArray[np.complex128]):
            """Draw acquired raw complex IQ on the measurement axes; return changed artists.

            Read scan-shaped samples without changing them. Update the live
            artists over the prepared coordinates; plot errors propagate.
            """
            nonlocal legend
            axes.clear()
            for cloud, label, color in zip(
                clouds,
                ("π probe off", "π probe on"),
                ("tab:blue", "tab:orange"),
                strict=True,
            ):
                axes.scatter(
                    cloud.real, cloud.imag, s=8, alpha=0.5, color=color, label=label
                )

            off, on = np.count_nonzero(np.isfinite(clouds), axis=1)
            axes.set(
                xlabel="IQ in-phase I (a.u.)",
                ylabel="IQ quadrature Q (a.u.)",
                title="Single-shot GE · qubit readout · IQ clouds",
            )
            axes.set_aspect("equal", adjustable="datalim")
            if legend is None:
                legend = figure.legend(loc="outside lower center", ncols=2)
            legend.set_title(f"Accumulated shots: {off} off, {on} on (count)")
            return (figure,)

        figure.show()

        signals_buffer = SignalBuffer((2, cfg.shots), on_update=update)
        with Schedule(signals_buffer, scope=context.scope) as sched:
            for with_probe, step in sched.scan("w/o probe pulse", [False, True]):
                probe_cfg = cfg.probe_pulse if with_probe else None

                # Compare identical preparations with the probe omitted/applied.
                builder = TimelineBuilder(
                    soccfg,
                    reps=cfg.shots,
                    initial_delay=cfg.initial_delay,
                    relax_delay=cfg.relax_delay,
                )
                if cfg.reset is not None:
                    builder.emit(Reset("reset", cfg.reset))
                builder.emit(Pulse("init_pulse", cfg.init_pulse))
                builder.emit(Pulse("probe_pulse", probe_cfg))
                builder.emit(PulseReadout("readout", cfg.readout))
                program = builder.build()

                # Store each cloud independently, preserving every complex shot.
                shots = step.run_acquire_shots(
                    program, soc, options=RunOptions(progress=True)
                )
                if shots is None:
                    break
                signals_buffer[step].set(raw_shots_to_signal(shots, program)[0])
        signals = signals_buffer.array

        # 2. Keep acquisition order; physical G/E mapping belongs to analysis.
        return GE_Result(
            signals=signals,
            shot_indices=np.arange(signals.shape[1], dtype=np.int64),
        )

    def analyze(
        self,
        source: RunRecord[GE_Cfg, GE_Result],
        options: GEAnalyzeOptions,
        *,
        plots: Plots,
    ) -> GEAnalysis:
        """Fit readout separation and preparation populations from raw IQ clouds.

        source contains probe-off/on IQ shaped (2, N), in normalized ADC units.
        options.initial_state maps the rows to physical G/E. Return fitted
        centers, width, model populations and empirical prepared-label assignment
        fidelity, which includes preparation mixing and is not pure-state readout
        fidelity. Poor histogram fit quality makes model populations and post
        population correction unreliable. Publish raw g/e IQ scatter to plots
        as "fit", including on numerical failure. Publish the measured 1D
        histogram as "projection" after a valid fit. cfg=None is supported.

        Rotation/classification, histogram fitting, fitted population/center
        validation and fidelity calculations wrap ValueError, RuntimeError,
        ArithmeticError and numpy.linalg.LinAlgError in GENumericalError with
        the original cause. Plotting, invalid backend/initial_state and final
        calibration validation errors propagate unchanged.
        """
        # 1. Acquisition order is always probe off/on; fit in physical g/e order.
        signals = ge_signals_by_state(source.result.signals, options.initial_state)

        if "fit" in plots:
            raise ValueError("Figure already exists: fit")
        figure, axes = plots.subplots(num="fit", figsize=(8, 6), layout="constrained")
        ground_prepared = options.initial_state == "ground"
        for cloud, label, color in zip(
            source.result.signals,
            (
                "π probe off · g preparation"
                if ground_prepared
                else "π probe off · e preparation",
                "π probe on · e preparation"
                if ground_prepared
                else "π probe on · g preparation",
            ),
            ("tab:blue", "tab:orange"),
            strict=True,
        ):
            axes.scatter(
                cloud.real, cloud.imag, s=8, alpha=0.5, color=color, label=label
            )
        figure.suptitle("Single-shot GE · qubit readout · IQ discrimination")
        axes.set(xlabel="IQ in-phase I (a.u.)", ylabel="IQ quadrature Q (a.u.)")
        axes.set_aspect("equal", adjustable="datalim")
        # 2. Fit readout separation and preparation mixing in physical g/e order.
        fidelity, init_pops, fit = singleshot_ge_analysis(
            signals,
            angle=options.angle,
            backend=options.backend,
            options=GEFitOptions(
                logscale=options.logscale,
                align_t1=options.align_t1,
                length_ratio=options.length_ratio,
            ),
            plots=None,
        )
        # 3. Validate the calibration and label an unidentifiable length ratio.
        analysis = GEAnalysis(
            initial_state=options.initial_state,
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
        analysis.validate_calibration()
        _annotate_primary(axes, analysis)
        _plot_projection(plots, signals, analysis, logarithmic=options.logscale)
        return analysis

    def post_analyze(
        self,
        source: RunRecord[GE_Cfg, GE_Result],
        primary: GEAnalysis,
        options: GEPostAnalyzeOptions,
        *,
        plots: Plots,
    ) -> GEPostAnalysis:
        """Derive classification radius and confusion matrix without refitting IQ.

        source contains probe-off/on IQ shaped (2, N), in normalized ADC units.
        primary supplies physical G/E mapping and fitted populations/centers.
        options sets the radius policy, in the same IQ units. Return the
        classification diagnostic and publish it as "post" in plots.
        Classification uses nearest-center regions and radius, not the primary
        histogram threshold. Solving the fitted preparation matrix removes its
        inferred mixing; poor primary histogram fits make this correction
        unreliable. It is not the empirical prepared-label assignment score.
        cfg=None is supported.

        Radius optimization, classification and confusion-matrix solving wrap
        ValueError, RuntimeError, ArithmeticError and numpy.linalg.LinAlgError
        in GENumericalError with the original cause. Primary validation, row
        mapping and plotting errors, including name conflicts, propagate unchanged.
        """
        # 1. Select the measured data and analysis controls.
        primary.validate_calibration()
        g_signals, e_signals = ge_signals_by_state(
            source.result.signals, primary.initial_state
        )
        signals = (g_signals, e_signals)
        try:
            confusion = calc_confusion_matrix(
                signals,
                GECalibration(
                    g_center=primary.g_center,
                    e_center=primary.e_center,
                    init_pops=primary.init_pops,
                    sigma=primary.ge_s,
                ),
                radius=options.radius,
                consider_other=options.consider_other,
            )
        except (
            ValueError,
            RuntimeError,
            ArithmeticError,
            np.linalg.LinAlgError,
        ) as exc:
            raise GENumericalError(str(exc)) from exc
        # 2. Display preparation mixing alongside classified IQ and confusion.
        g_center, e_center = primary.g_center, primary.e_center
        fig, _ = plots.subplots(num="post", nrows=2, ncols=2, figsize=(8, 8))
        fig.set_layout_engine("constrained", h_pad=0.10)
        ax1, ax4, ax2, ax3 = fig.axes

        heading = [
            "Single-shot GE · qubit readout · circle classification",
            f"Classification radius = {confusion.radius:.3g} a.u.",
        ]
        if note := ge_post_review_note(primary, confusion.condition_number):
            heading.extend(note.split("; "))
        fig.suptitle("\n".join(heading))
        g_label, e_label, l_label = "g", "e", "other"
        model_rows = [g_label, e_label, "other\n(model only)"]
        _plot_classified_iq(ax1, g_signals, g_center, e_center, confusion.radius)
        _plot_classified_iq(ax2, e_signals, g_center, e_center, confusion.radius)
        handles, labels = ax1.get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncols=2)

        n_gg, n_ge, n_go = confusion.g_classification
        n_eg, n_ee, n_eo = confusion.e_classification
        ax1.set_title(
            f"g preparation\nP(g) {n_gg:.1%} · P(e) {n_ge:.1%}\nP(other) {n_go:.1%}"
        )
        ax2.set_title(
            f"e preparation\nP(g) {n_eg:.1%} · P(e) {n_ee:.1%}\nP(other) {n_eo:.1%}"
        )

        im = ax4.imshow(confusion.init_matrix, cmap="Blues", vmin=0, vmax=1)
        ax4.grid(False)
        fig.colorbar(im, ax=ax4, label="Probability (0–1)", shrink=0.8)
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
        ax4.set_yticklabels(model_rows)
        ax4.set_xlabel("Model state")
        ax4.set_ylabel("Preparation label")
        ax4.set_title("Fitted preparation populations")

        im = ax3.imshow(confusion.matrix, cmap="Blues", vmin=0, vmax=1)
        ax3.grid(False)
        fig.colorbar(im, ax=ax3, label="Probability (0–1)", shrink=0.8)
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
        ax3.set_yticklabels(model_rows)
        ax3.set_xlabel("Assigned outcome")
        ax3.set_ylabel("Model state")
        ax3.set_title(
            f"Corrected confusion\nCondition number {confusion.condition_number:.2g}"
        )
        return GEPostAnalysis(confusion=confusion)
