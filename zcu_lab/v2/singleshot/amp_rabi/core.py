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
class AmpRabiResult:
    gains: NDArray[np.float64]
    shot_indices: NDArray[np.int64]
    signals: NDArray[np.complex128]


@dataclass(frozen=True)
class AmpRabiFit:
    pi_gain: float
    pi_gain_error: float
    pi2_gain: float
    pi2_gain_error: float
    frequency: float
    amplitude: float
    joint_fit: RabiJointFitResult


class AmpRabiCfg(ConfigBase):
    reset: Annotated[Reset.Cfg | None, FieldMeta(label="Reset")] = None
    qub_pulse: Annotated[Pulse.Cfg, FieldMeta(label="Qub Pulse")]
    readout: Annotated[Readout.Cfg, FieldMeta(label="Readout")]
    gain: Annotated[SweepCfg, FieldMeta(label="Gain", unit="a.u.")]
    reps: Annotated[
        int, FieldMeta(label="Reps", description="Number of repetitions per round.")
    ] = 1
    rounds: Annotated[
        int,
        FieldMeta(label="Rounds", description="Number of software acquisition rounds."),
    ] = 1
    relax_delay: Annotated[
        float,
        FieldMeta(
            label="Relax delay",
            unit="us",
            decimals=3,
            description="Relaxation delay after each repetition.",
        ),
    ] = 1.0


def _flatten_round_shots(raw: NDArray[np.complex128]) -> NDArray[np.complex128]:
    """(round, gain, shot) -> (gain, round * shot), preserving every IQ sample."""
    return raw.transpose(1, 0, 2).reshape(raw.shape[1], -1)


def _gain_calibration(fit: RabiJointFitResult) -> AmpRabiFit:
    if not fit.backend.valid:
        return AmpRabiFit(np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, fit)
    pi_gain = np.pi / fit.omega
    omega_index = fit.backend.parameter_names.index("log_omega")
    # d(pi / omega) / d(log omega) = -pi / omega.
    variance = fit.backend.covariance[omega_index, omega_index]
    pi_error = pi_gain * np.sqrt(max(float(variance), 0.0))
    return AmpRabiFit(
        pi_gain,
        float(pi_error),
        pi_gain / 2,
        float(pi_error / 2),
        fit.omega / (2 * np.pi),
        abs(fit.initial_populations[1] - fit.p_inf),
        fit,
    )


@pydantic_dataclass(frozen=True, config={"arbitrary_types_allowed": True})
class AmpRabiAnalyzeOptions:
    initial_state: Literal["ground", "excited"] = "ground"
    max_calls: int | None = None


class AmpRabiExp(Experiment[AmpRabiResult, AmpRabiCfg]):
    Options: ClassVar[type[AmpRabiAnalyzeOptions]] = AmpRabiAnalyzeOptions

    AXES_SPEC: ClassVar[AxesSpec[AmpRabiResult, AmpRabiCfg]] = AxesSpec(
        axes=(
            AxesSpec.Axis(
                "shot_indices", label="Shot Index", unit="None", dtype=np.int64
            ),
            AxesSpec.Axis("gains", label="Gain", unit="a.u.", dtype=np.float64),
        ),
        z=AxesSpec.Z("signals", label="Signal", unit="a.u.", dtype=np.complex128),
        result_type=AmpRabiResult,
        cfg_type=AmpRabiCfg,
        cfg_schema_version="3.0",
        tag="singleshot/amp_rabi",
    )

    def run(
        self,
        cfg: AmpRabiCfg,
        *,
        context: RunContext,
    ) -> AmpRabiResult:
        """Measure Rabi oscillations versus drive gain while retaining raw IQ.

        Apply optional reset, the gain-swept qubit pulse, then readout.
        Each host round acquires cfg.reps shots at every hardware-swept gain.
        Return gains (a.u.) and complex IQ shaped (Ngain, reps * rounds);
        rounds are concatenated, never averaged into one IQ point.

        The saved readout's circle/radius supplies live G/E/Other classification.
        Without a classifier, skip preview but retain raw acquisition and fit.
        These live fractions are not the saved data. The later raw-IQ joint fit estimates
        the oscillation period, pi/pi2 gains and readout calibration.
        Acquisition requires one readout channel and raises on an unexpected
        raw-shot buffer shape.
        """

        # Validate live classification before acquiring any shots.
        soc, soccfg = context.soc, context.soccfg
        classifier = rabi_preview_classifier(cfg.readout)
        cfg = deepcopy(cfg)
        rounds, cfg.rounds = cfg.rounds, 1
        gains = sweep2array(
            cfg.gain,
            "gain",
            {"soccfg": soccfg, "gen_ch": cfg.qub_pulse.ch},
        )

        # Preview populations from complete shot columns without losing raw IQ.
        def configure_axes(ax: Axes) -> None:
            ax.set_ylim(-0.02, 1.02)
            for line, kwargs in zip(
                ax.lines,
                [
                    {"label": label, "color": color}
                    for label, color in (
                        ("Ground", "blue"),
                        ("Excited", "red"),
                        ("Other", "green"),
                    )
                ],
                strict=True,
            ):
                line.set(**kwargs)
            ax.legend()

        viewer = None
        if classifier is not None:
            figure, ax = context.plots.subplots(num="measurement")
            ax.set(xlabel="Pulse gain (a.u.)", ylabel="Classified population")
            viewer = live_line(ax, num_lines=3)
            configure_axes(ax)
            figure.show()

        @context.plots.on_plot_thread
        def update_view(raw: NDArray[np.complex128]):
            if viewer is None or classifier is None or classifier.radius is None:
                return ()
            shots = _flatten_round_shots(raw)
            complete = np.all(np.isfinite(shots), axis=0)
            populations = np.full((gains.size, 2), np.nan)
            if np.any(complete):
                populations = classify_rabi_iq(
                    shots[:, complete],
                    classifier.g_center,
                    classifier.e_center,
                    classifier.radius,
                )
            return viewer.set(
                gains, np.column_stack((populations, 1 - populations.sum(axis=1))).T
            )

        buffer = SignalBuffer(
            (rounds, gains.size, cfg.reps),
            dtype=np.complex128,
            on_update=update_view,
        )
        with Schedule(buffer, scope=context.scope) as sched:
            # The board sweeps gain within each reset/drive/readout acquisition.
            builder = TimelineBuilder(
                soccfg, reps=cfg.reps, relax_delay=cfg.relax_delay
            )
            gain = builder.add_sweep("gain", cfg.gain)
            if cfg.reset is not None:
                builder.emit(Reset("reset", cfg.reset))
            builder.emit(Pulse("qub_pulse", cfg.qub_pulse, gain=gain))
            builder.emit(Readout("readout", cfg.readout))
            program = builder.build()

            # Store each completed round separately so every shot survives.
            for _, step in sched.repeat("round", rounds):
                shots = step.run_acquire_shots(
                    program, soc, options=RunOptions(progress=False)
                )
                if shots is None:
                    break
                raw = raw_shots_to_signal(program)
                if raw.shape != (cfg.reps, gains.size):
                    raise ValueError(
                        f"Amp Rabi raw IQ shape mismatch: expected {(cfg.reps, gains.size)}, got {raw.shape}"
                    )
                buffer[step].set(raw.T)

        # Join round/shot axes only; unacquired rounds remain NaN.
        return AmpRabiResult(
            gains,
            np.arange(rounds * cfg.reps, dtype=np.int64),
            _flatten_round_shots(buffer.array),
        )

    def analyze(
        self,
        source: RunRecord[AmpRabiCfg, AmpRabiResult],
        options: AmpRabiAnalyzeOptions,
        *,
        plots: Plots,
    ) -> AmpRabiFit:
        result = source.result
        if result.signals.shape != (
            result.gains.size,
            result.shot_indices.size,
        ) or not np.iscomplexobj(result.signals):
            raise ValueError("Amp Rabi requires raw IQ with shape (gain, shot)")
        # Cancelled rounds occupy entirely missing shot columns. Do not turn
        # partially missing sweeps into unequal per-gain sample counts.
        acquired = ~np.all(np.isnan(result.signals), axis=0)
        order = np.argsort(result.gains)
        gains, signals = result.gains[order], result.signals[order][:, acquired]
        joint = fit_rabi_joint(
            gains,
            signals,
            decay=False,
            fit_phase=False,
            initial_state=options.initial_state,
            max_calls=options.max_calls,
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
                gains,
                joint.measured_populations[:, index],
                color=color,
                ls="none",
                marker="o",
                markersize=3,
                label=label,
            )
            ax.plot(
                gains,
                joint.fitted_populations[:, index],
                color=color,
                ls="-",
            )
        if joint.backend.valid:
            ax.set_title(
                f"$\\Omega$={joint.omega:.3g} rad/gain, "
                f"phase={np.degrees(joint.phase):.2f}°, cond={joint.condition_number:.3g}"
            )
        else:
            ax.set_title("Rabi joint fit invalid")
        ax.set_xlabel("Pulse gain (a.u.)")
        ax.set_ylabel("Population")
        ax.set_ylim(0.0, 1.0)
        ax.legend(loc=4)
        ax.grid(True)
        effective_t1_us = None
        if (
            joint.backend.valid
            and ro_length_us is not None
            and np.isfinite(joint.length_ratio)
            and joint.length_ratio > 0.0
        ):
            value = ro_length_us / joint.length_ratio
            if np.isfinite(value) and value > 0.0:
                effective_t1_us = value
        classified_distribution = None
        if joint.backend.valid:
            classified_ge = classify_rabi_iq(
                signals[:1],
                joint.g_center,
                joint.e_center,
                joint.radius,
            )[0]
            classified_distribution = np.array(
                (*classified_ge, 1.0 - classified_ge.sum()), dtype=np.float64
            )
        _plot_initial_histogram(
            histogram_ax,
            joint,
            effective_t1_us,
            classified_distribution,
        )
        _plot_confusion_matrix(confusion_ax, joint)
        return _gain_calibration(joint)
