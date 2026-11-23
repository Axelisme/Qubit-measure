from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from numbers import Real
from typing import Annotated, ClassVar, Literal

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray
from pydantic.dataclasses import dataclass as pydantic_dataclass
from zcu_tools.analysis.fitting.singleshot import transition_state_bin_probabilities
from zcu_tools.analysis.fitting.singleshot_rabi import (
    RabiJointFitResult,
    RabiJointFitSummary,
    RabiPhysicalParams,
    fit_rabi_joint,
    rabi_excited_population,
    summarize_rabi_joint_fit,
)
from zcu_tools.analysis.shot_uncertainty import class_probability_covariance
from zcu_tools.analysis.singleshot import classify_rabi_iq, classify_result
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.experiment import AxesSpec, Experiment
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.runtime import RunOptions
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import Pulse, Readout, Reset, SweepCfg, TimelineBuilder

from zcu_lab.v2.support import live_errorbar
from zcu_lab.v2.support.singleshot.util import (
    rabi_preview_classifier,
    raw_shots_to_signal,
)


@dataclass(frozen=True)
class LenRabiResult:
    lengths: NDArray[np.float64]
    shot_indices: NDArray[np.int64]
    signals: NDArray[np.complex128]


@pydantic_dataclass(frozen=True, config={"arbitrary_types_allowed": True})
class LenRabiAnalyzeOptions:
    decay: bool = True
    fit_phase: bool = False
    initial_state: Literal["ground", "excited"] = "ground"
    max_calls: int | None = None


def _plot_initial_histogram(
    ax: Axes,
    fit: RabiJointFitResult,
    effective_t1_us: float | None,
    classified_distribution: NDArray[np.float64] | None,
) -> None:
    """Draw the initial-point IQ histogram and fitted state contributions on ax.

    Read the joint fit without changing it. Annotate effective T1 in us and
    the optional G/E/Other classified fractions when available. Return None;
    only the caller-owned axes change, and plotting errors propagate.
    """
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
            title += f"\nEffective $T_1$ = {effective_t1_us:.3g} µs"
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
    ax.set_ylabel("Shots (count)")
    ax.legend(fontsize=10, loc="best")


def _plot_confusion_matrix(ax: Axes, fit: RabiJointFitResult) -> None:
    """Draw the fitted three-state confusion matrix on the caller-owned ax.

    Read the joint fit without changing it; show an unavailable label for an
    invalid fit. Return None after updating the image, labels and ticks.
    Plotting errors propagate.
    """
    labels = ("g", "e", "other")
    matrix = fit.confusion_matrix
    if fit.backend.valid and np.isfinite(matrix).all():
        image = ax.imshow(matrix, cmap="Blues")
        shades = np.asarray(image.norm(matrix), dtype=np.float64)
        for row in range(3):
            for column in range(3):
                value = matrix[row, column]
                ax.text(
                    column,
                    row,
                    f"{value:.3f}",
                    ha="center",
                    va="center",
                    color="white" if shades[row, column] > 0.5 else "black",
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
    ax.grid(False)
    ax.set_xlabel("Assigned outcome")
    ax.set_ylabel("Model state")


def _plot_rabi_populations(
    axes: NDArray[np.object_],
    coordinates: NDArray[np.float64],
    signals: NDArray[np.complex128],
    fit: RabiJointFitResult,
    initial_state: str,
) -> None:
    """Display retained-shot fractions and the corresponding model predictions."""
    panels: list[Axes] = list(axes)
    fig = panels[0].get_figure()
    if fig is not None:
        title = f"Single-shot length Rabi · qubit · classified outcomes ({initial_state} start)"
        if fit.backend.valid:
            pi = np.pi / fit.omega
            index = fit.backend.parameter_names.index("log_omega")
            pi_error = pi * np.sqrt(
                max(float(fit.backend.covariance[index, index]), 0.0)
            )
            title += (
                f"\nπ length = {pi:.3g} ± {pi_error:.2g} µs · "
                f"Fit NLL = {fit.backend.nll:.3g} · Readout condition = {fit.condition_number:.3g}"
            )
        else:
            title += "\nJoint fit invalid; fitted classification unavailable"
        fig.suptitle(title, fontsize=10)
    for ax, state in zip(panels, ("g", "e", "other"), strict=True):
        ax.set(xlabel="Drive envelope length (µs)", ylabel=f"P({state})")
    if not fit.backend.valid:
        return

    retained = np.isfinite(signals)
    masks = classify_result(signals, fit.g_center, fit.e_center, fit.radius)
    counts = np.stack([(mask & retained).sum(axis=1) for mask in masks], axis=1)
    with np.errstate(invalid="ignore"):
        fractions = counts / counts.sum(axis=-1, keepdims=True)
    covariance = class_probability_covariance(counts)
    errors = np.sqrt(np.diagonal(covariance, axis1=-2, axis2=-1))
    grid = np.linspace(float(coordinates.min()), float(coordinates.max()), 250)
    excited = rabi_excited_population(
        grid,
        RabiPhysicalParams(
            p_e0=float(fit.initial_populations[1]),
            p_inf=fit.p_inf,
            center_g=fit.projected_g_center,
            center_e=fit.projected_e_center,
            sigma=fit.sigma,
            p_avg=fit.p_avg,
            length_ratio=fit.length_ratio,
            t_r=fit.t_r,
            omega=fit.omega,
            phase=fit.phase,
        ),
    )
    predicted = (
        np.column_stack((1.0 - excited, excited, np.zeros_like(excited)))
        @ fit.confusion_matrix
    )
    for i, (ax, color) in enumerate(
        zip(panels, ("tab:blue", "tab:red", "tab:green"), strict=True)
    ):
        ax.errorbar(
            coordinates,
            fractions[:, i],
            yerr=errors[:, i],
            fmt="o",
            markersize=3,
            capsize=2,
            color=color,
            label="Data ± shot SEM",
        )
        ax.plot(grid, predicted[:, i], color=color, label="Joint fit prediction")
    pi = np.pi / fit.omega
    if coordinates.min() <= pi <= coordinates.max():
        panels[1].axvline(pi, linestyle=":", color="0.35", label="π length")
    panels[1].legend(loc="best", fontsize=10)


class LenRabiCfg(ConfigBase):
    reset: Annotated[
        Reset.Cfg | None,
        FieldMeta(
            label="Reset",
            description="Select the reset module applied before preparation, or leave empty to skip reset.",
        ),
    ] = None
    qub_pulse: Annotated[
        Pulse.Cfg,
        FieldMeta(
            label="Qubit drive pulse",
            description="Enter the resonant qubit pulse whose envelope length is swept.",
        ),
    ]
    readout: Annotated[
        Readout.Cfg,
        FieldMeta(
            label="Readout",
            description="Enter the readout module applied at the end of the shot.",
        ),
    ]
    length: Annotated[
        SweepCfg,
        FieldMeta(
            label="Drive envelope length",
            unit="µs",
            description="Set the qubit drive envelope duration sweep in µs.",
        ),
    ]
    shots: Annotated[
        int,
        FieldMeta(
            label="Shots per length",
            unit="count",
            description="Enter the number of IQ readouts at each drive length.",
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


class LenRabiExp(Experiment[LenRabiResult, LenRabiCfg]):
    Options: ClassVar[type[LenRabiAnalyzeOptions]] = LenRabiAnalyzeOptions

    AXES_SPEC: ClassVar[AxesSpec[LenRabiResult, LenRabiCfg]] = AxesSpec(
        axes=(
            AxesSpec.Axis(
                "shot_indices", label="Shot Index", unit="None", dtype=np.int64
            ),
            AxesSpec.Axis(
                "lengths", label="Length", unit="s", scale=1e-6, dtype=np.float64
            ),
        ),
        z=AxesSpec.Z("signals", label="Signal", unit="a.u.", dtype=np.complex128),
        result_type=LenRabiResult,
        cfg_type=LenRabiCfg,
        cfg_schema_version="3.0",
        tag="singleshot/len_rabi",
    )

    def run(
        self,
        cfg: LenRabiCfg,
        *,
        context: RunContext,
    ) -> LenRabiResult:
        """Measure Rabi oscillations versus pulse length with raw single-shot IQ.

        At each host-side length, apply optional reset, the qubit pulse, then
        readout for cfg.shots repetitions in one round. Return rounded lengths
        in us and complex IQ shaped (Nlength, shots), not averaged populations.
        Later analysis jointly fits Rabi dynamics and readout distributions.

        Const, flat_top and sampled waveforms (cosine, gauss, DRAG) use the
        same host scan, compiling one program per length. Native arb waveforms
        raise ValueError before scanning because their length is asset-derived.
        cfg.readout supplies an optional circle and radius for live preview.
        Without a classifier, raw acquisition and joint fitting still work.
        Acquisition requires exactly one readout channel and raises on an
        unexpected shot-buffer shape.
        """

        # 1. Set up one raw-shot cloud per length without changing the caller cfg.
        soc, soccfg = context.soc, context.soccfg
        cfg = deepcopy(cfg)
        classifier = rabi_preview_classifier(cfg.readout)

        if cfg.qub_pulse.waveform.style == "arb":
            raise ValueError("Arb waveform length is asset-derived and cannot be swept")
        lengths = sweep2array(
            cfg.length,
            "time",
            {"soccfg": soccfg, "gen_ch": cfg.qub_pulse.ch},
        )
        expected_shape = (len(lengths), cfg.shots)

        # 2. Preview classified populations while keeping raw IQ as the result.

        viewer = None
        progress = None
        if classifier is not None:
            figure, ax = context.plots.subplots(num="measurement")
            ax.set(xlabel="Drive envelope length (µs)", ylabel="Outcome probability")
            progress = figure.suptitle(
                "Single-shot length Rabi · qubit · classified populations"
            )
            viewer = live_errorbar(ax, num_lines=3)
            for line, label in zip(ax.lines, ("P(g)", "P(e)", "P(other)"), strict=True):
                line.set_label(label)
            ax.legend()
            figure.show()

        @context.plots.on_plot_thread
        def update_view(raw_iq: NDArray[np.complex128]):
            """Draw acquired raw complex IQ on the measurement axes; return changed artists.

            Read scan-shaped samples without changing them. Update the live
            artists over the prepared coordinates; plot errors propagate.
            """
            if viewer is None or classifier is None or classifier.radius is None:
                return ()
            acquired_rows = np.all(np.isfinite(raw_iq), axis=1)
            masks = classify_result(
                raw_iq, classifier.g_center, classifier.e_center, classifier.radius
            )
            counts = np.stack([mask.sum(axis=1) for mask in masks], axis=1).astype(
                np.float64
            )
            counts[~acquired_rows] = np.nan
            populations = counts[:, :2] / raw_iq.shape[1]
            other = 1.0 - populations.sum(axis=1)
            covariance = class_probability_covariance(counts)
            sem = np.sqrt(np.diagonal(covariance, axis1=-2, axis2=-1))
            artists = viewer.set(
                lengths, np.column_stack((populations, other)).T, sem.T
            )
            if progress is None:
                return artists
            progress.set_text(
                "Single-shot length Rabi · qubit · classified populations\n"
                f"{np.count_nonzero(acquired_rows)}/{acquired_rows.size} lengths acquired"
            )
            return (progress, *artists)

        buffer = SignalBuffer(
            expected_shape,
            dtype=np.complex128,
            on_update=update_view,
        )
        with Schedule(buffer, scope=context.scope) as sched:
            for length, step in sched.scan("length", lengths.tolist()):
                # Acquire one length at a time to bound the board raw-buffer size.
                builder = TimelineBuilder(
                    soccfg,
                    reps=cfg.shots,
                    initial_delay=cfg.initial_delay,
                    relax_delay=cfg.relax_delay,
                )
                if cfg.reset is not None:
                    builder.emit(Reset("reset", cfg.reset))
                builder.emit(Pulse("qub_pulse", cfg.qub_pulse, length=length))
                builder.emit(Readout("readout", cfg.readout))
                program = builder.build()
                shots = step.run_acquire_shots(
                    program, soc, options=RunOptions(progress=True)
                )
                if shots is None:
                    break

                # Write a full shot cloud only after this point finishes.
                raw_iq = raw_shots_to_signal(shots, program)[0]
                expected_point_shape = (cfg.shots,)
                if raw_iq.shape != expected_point_shape:
                    raise ValueError(
                        "Len Rabi raw IQ shape mismatch: "
                        f"expected {expected_point_shape}, got {raw_iq.shape}"
                    )
                buffer[step].set(raw_iq)

        # 3. Completed rows survive cancellation; unacquired lengths remain NaN.
        signals = buffer.array

        return LenRabiResult(
            lengths=lengths,
            shot_indices=np.arange(cfg.shots, dtype=np.int64),
            signals=signals,
        )

    def analyze(
        self,
        source: RunRecord[LenRabiCfg, LenRabiResult],
        options: LenRabiAnalyzeOptions,
        *,
        plots: Plots,
    ) -> RabiJointFitSummary:
        # 1. Fit the raw IQ and interpret the readout calibration.
        """Fit raw IQ jointly, return pulse-length calibrations, and draw populations and readout diagnostics."""
        result = source.result

        fit = fit_rabi_joint(
            result.lengths,
            result.signals,
            decay=options.decay,
            fit_phase=options.fit_phase,
            max_calls=options.max_calls,
            initial_state=options.initial_state,
        )
        # 2. Compare fitted populations and readout diagnostics.
        readout = source.cfg.readout if source.cfg is not None else None
        ro_length = readout.ro_length if readout is not None else None
        ro_length_us = float(ro_length) if isinstance(ro_length, Real) else None
        fig = Figure(figsize=(9, 4), layout="constrained")
        plots.adopt("fit", fig)
        axes = fig.subplots(ncols=3)
        _plot_rabi_populations(
            axes, result.lengths, result.signals, fit, options.initial_state
        )
        diagnostics = Figure(figsize=(9, 4.5), layout="constrained")
        plots.adopt("readout", diagnostics)
        histogram_ax, confusion_ax = diagnostics.subplots(ncols=2)
        diagnostics.suptitle("Single-shot length Rabi · qubit · readout diagnostics")
        # 3. Show readout diagnostics, deriving T1 and initial fractions only if valid.
        effective_t1_us = None
        if (
            fit.backend.valid
            and ro_length_us is not None
            and np.isfinite(fit.length_ratio)
            and fit.length_ratio > 0.0
        ):
            value = ro_length_us / fit.length_ratio
            if np.isfinite(value) and value > 0.0:
                effective_t1_us = value
        classified_distribution = None
        if fit.backend.valid:
            classified_ge = classify_rabi_iq(
                result.signals[:1],
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
        return summarize_rabi_joint_fit(fit)
