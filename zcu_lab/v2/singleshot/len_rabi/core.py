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
    fit_rabi_joint,
)
from zcu_tools.analysis.singleshot import classify_rabi_iq
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.experiment import AxesSpec, Experiment, config
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.runtime import RunOptions
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import Pulse, Readout, Reset, SweepCfg, TimelineBuilder

from zcu_lab.v2.support import live_line
from zcu_lab.v2.support.singleshot.util import (
    rabi_preview_classifier,
    raw_shots_to_signal,
)


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
class LenRabiResult:
    lengths: NDArray[np.float64]
    shot_indices: NDArray[np.int64]
    signals: NDArray[np.complex128]


class LenRabiCfg(ConfigBase):
    reset: Annotated[Reset.Cfg | None, FieldMeta(label="Reset")] = None
    qub_pulse: Annotated[Pulse.Cfg, FieldMeta(label="Qub Pulse")]
    readout: Annotated[Readout.Cfg, FieldMeta(label="Readout")]
    length: Annotated[SweepCfg, FieldMeta(label="Length", unit="us")]
    shots: Annotated[int, FieldMeta(label="Shots")]
    initial_delay: Annotated[
        float,
        FieldMeta(
            label="Initial delay", unit="us", description="Delay before the first shot."
        ),
    ] = 1.0
    relax_delay: Annotated[
        float,
        FieldMeta(
            label="Relax delay",
            unit="us",
            decimals=3,
            description="Delay after each shot.",
        ),
    ] = 1.0


@pydantic_dataclass(frozen=True, config={"arbitrary_types_allowed": True})
class LenRabiAnalyzeOptions:
    decay: bool = True
    fit_phase: bool = False
    initial_state: Literal["ground", "excited"] = "ground"
    max_calls: int | None = None


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

        # Set up one raw-shot cloud per length without changing the caller cfg.
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

        # Preview classified populations while keeping raw IQ as the result.
        def configure_axes(ax: Axes) -> None:
            ax.set_ylim(0.0, 1.0)
            for line, label in zip(
                ax.lines, ("Ground", "Excited", "Other"), strict=True
            ):
                line.set_label(label)
            ax.legend()

        viewer = None
        if classifier is not None:
            figure, ax = context.plots.subplots(num="measurement")
            ax.set(xlabel="Length (us)", ylabel="Signal")
            viewer = live_line(ax, num_lines=3)
            configure_axes(ax)
            figure.show()

        @context.plots.on_plot_thread
        def update_view(raw_iq: NDArray[np.complex128]):
            if viewer is None or classifier is None or classifier.radius is None:
                return ()
            acquired_rows = np.all(np.isfinite(raw_iq), axis=1)
            populations = classify_rabi_iq(
                raw_iq, classifier.g_center, classifier.e_center, classifier.radius
            )
            populations[~acquired_rows] = np.nan
            other = 1.0 - populations.sum(axis=1)
            return viewer.set(lengths, np.column_stack((populations, other)).T)

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
                raw_iq = raw_shots_to_signal(program)
                expected_point_shape = (cfg.shots,)
                if raw_iq.shape != expected_point_shape:
                    raise ValueError(
                        "Len Rabi raw IQ shape mismatch: "
                        f"expected {expected_point_shape}, got {raw_iq.shape}"
                    )
                buffer[step].set(raw_iq)

        # Completed rows survive cancellation; unacquired lengths remain NaN.
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
    ) -> RabiJointFitResult:
        result = source.result

        fit = fit_rabi_joint(
            result.lengths,
            result.signals,
            decay=options.decay,
            fit_phase=options.fit_phase,
            max_calls=options.max_calls,
            initial_state=options.initial_state,
        )
        readout = source.cfg.readout if source.cfg is not None else None
        ro_length = readout.ro_length if readout is not None else None
        ro_length_us = float(ro_length) if isinstance(ro_length, Real) else None
        width, height = config.figsize
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
                result.lengths,
                fit.measured_populations[:, index],
                color=color,
                ls="none",
                marker="o",
                markersize=3,
                label=label,
            )
            ax.plot(
                result.lengths,
                fit.fitted_populations[:, index],
                color=color,
                ls="-",
            )
        if fit.backend.valid:
            ax.set_title(
                f"$\\Omega$={fit.omega:.3g} rad/μs, "
                f"phase={np.degrees(fit.phase):.2f}°, cond={fit.condition_number:.3g}"
            )
        else:
            ax.set_title("Rabi joint fit invalid")
        ax.set_xlabel("Pulse length (μs)")
        ax.set_ylabel("Population")
        ax.set_ylim(0.0, 1.0)
        ax.legend(loc=4)
        ax.grid(True)
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
        return fit
