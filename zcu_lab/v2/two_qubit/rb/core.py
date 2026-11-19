"""Paired two-qubit RB with acquisition-side classification and host rounds."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Annotated, ClassVar, Self

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from numpy.typing import NDArray
from pydantic import Field, model_validator
from zcu_tools.analysis.failure import AnalysisFailed
from zcu_tools.analysis.fitting import (
    FitQuality,
    IRBAnalysis,
    analyze_irb,
    compute_fit_quality,
)
from zcu_tools.analysis.fitting.shared import SharedFitResult
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.experiment import (
    Experiment,
    GroupedAxesSpec,
    GroupedLoadData,
    VariableAxisSpec,
    VariableSpec,
    VariableZSpec,
)
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.runtime import RunOptions, Schedule, SignalBuffer
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import (
    Classify,
    Pulse,
    Reset,
    TimelineBuilder,
    shot_sums,
)
from zcu_tools.program.v2.modules import MuxReadout

from zcu_lab.v2.support import live_line

from .._drive import DualLineCfg
from .._readout import JointRoles
from .._sums import (
    joint_probabilities,
    validate_classifiers,
)
from .program import CZPhasePair
from .scheduling import emit_sequence
from .sequence import make_paired_sequences


@dataclass(frozen=True)
class RBResult:
    """Float64 class counts with axes (depth,seed,arm,round,F1,F2).

    depths counts random Cliffords; seeds retains exact random.Random identities.
    The fixed arm axis [0,1] means reference/CZ-interleaved. rounds=[0..rounds-1] is a host
    acquisition axis. sums ends in (3,3), ordered g/e/other on each role.
    Counts are unnormalized; a wholly unacquired round stays NaN.
    """

    depths: NDArray[np.int64]
    seeds: NDArray[np.int64]

    rounds: NDArray[np.int64]
    sums: NDArray[np.float64]


class RBOptions(ConfigBase):
    """Analysis-only reproducible paired-seed bootstrap settings.

    bootstrap_samples is at least 100 whole-seed resamples, not independent
    shots. bootstrap_seed is the nonnegative RNG seed for this uncertainty
    calculation, separate from the acquired sequence identities.
    """

    bootstrap_samples: int = Field(default=1000, ge=100)
    bootstrap_seed: int = Field(default=42, ge=0)


@dataclass(frozen=True)
class RBEstimate:
    """Unitless d=4 RB estimates, conditional on the exponential decay model.

    p_reference/p_interleaved are per random Clifford (plus CZ for interleaved).
    reference_epc=3/4*(1-p_reference). gate_error=3/4*(1-p_i/p_r), not clipped.
    error_ci_low/error_ci_high is the 95% paired-seed percentile interval.
    n_paired_seeds/n_paired_rounds count complete curves/round-seed pairs.
    bootstrap_successes counts valid resampled fits; warning describes ratio
    and model limitations. No field asserts calibrated physical fidelity.
    """

    p_reference: float
    p_interleaved: float
    reference_epc: float
    gate_error: float
    error_ci_low: float
    error_ci_high: float
    n_paired_seeds: int
    n_paired_rounds: int
    bootstrap_successes: int
    warning: str


@dataclass(frozen=True)
class RBAnalysis:
    """Fixed-size d=4 paired RB estimate and reference/interleaved fit quality.

    estimate is None with reason when no fit is identifiable. fit_quality holds
    residual diagnostics for each arm, or None without a fit. Survival arrays,
    round completeness and model curves are used only inside analyze.
    """

    estimate: RBEstimate | None
    fit_quality: tuple[FitQuality, FitQuality] | None
    reason: str | None


class RBCfg(ConfigBase):
    """Optional reset → paired random Clifford + own recovery → terminal read.

    roles maps Q0 to F1 and Q1 to F2 at their original full mux tone positions.
    phase_pair is required, ordered and component-named. Delays are us, pulse
    frequency MHz and phase degrees. No DC device is driven. reset=None emits
    no reset. F1/F2 need circle classifiers with radius from GE Post before Run.
    depth counts random Cliffords, not primitive gates; nondecreasing
    nonnegative integers. seed contains distinct integer random.Random identities
    shared across depths/arms/rounds. The 31-bit bound preserves exact axes.
    reps is shots per host round; rounds remains a separate saved axis.
    """

    reset: Annotated[
        Reset.Cfg | None,
        FieldMeta(
            label="Reset", description="Optional reset at the start of every shot."
        ),
    ] = None
    x90_f1: Annotated[
        Pulse.Cfg,
        FieldMeta(
            label="F1 Xπ/2",
            description="Calibrated F1 X90 pulse, frequency MHz, gain [-1, 1] and duration us.",
        ),
    ]
    x90_f2: Annotated[
        Pulse.Cfg,
        FieldMeta(
            label="F2 Xπ/2",
            description="Calibrated F2 X90 pulse, frequency MHz, gain [-1, 1] and duration us.",
        ),
    ]
    drive: Annotated[
        DualLineCfg,
        FieldMeta(
            label="Dual-line drive",
            description="F1/F2 charge-line CZ pulses from the calibrated coupler slots.",
        ),
    ]
    readout: Annotated[
        MuxReadout.Cfg,
        FieldMeta(
            label="Mux readout",
            description="Full-group mux readout with calibrated frequencies, gains and timing.",
        ),
    ]
    roles: JointRoles
    phase_pair: CZPhasePair
    depth: list[Annotated[int, Field(strict=True, ge=0)]] = Field(
        default_factory=lambda: [0, 1, 2, 3], min_length=1
    )
    seed: list[Annotated[int, Field(strict=True, ge=0, le=2**31 - 1)]] = Field(
        default_factory=lambda: [0, 1], min_length=1
    )
    reps: Annotated[
        int, FieldMeta(label="Reps", description="Number of repetitions per round.")
    ] = Field(gt=0)
    rounds: Annotated[
        int,
        FieldMeta(label="Rounds", description="Number of software acquisition rounds."),
    ] = Field(default=1, gt=0)
    initial_delay: Annotated[
        float,
        FieldMeta(
            label="Initial delay",
            unit="us",
            description="Delay before the first repetition.",
        ),
    ] = 1.0
    relax_delay: Annotated[
        float,
        FieldMeta(
            label="Relax delay",
            unit="us",
            decimals=3,
            description="Relaxation delay after each repetition.",
        ),
    ]

    @model_validator(mode="after")
    def validate_sequence_inputs(self) -> Self:
        """Reject swapped phases, unordered depths, repeated seeds or wrong generators."""
        validate_classifiers(self.readout, self.roles)
        self.phase_pair.validate_roles(self.roles)
        if self.depth != sorted(self.depth):
            raise ValueError("RB depths must be increasing")
        if len(set(self.seed)) != len(self.seed):
            raise ValueError("RB seeds must be distinct complete integer identities")
        for drive, x90 in (
            (self.drive.f1, self.x90_f1),
            (self.drive.f2, self.x90_f2),
        ):
            if drive.ch != x90.ch:
                raise ValueError("CZ and X90 must use the same captured role generator")
            if not isinstance(x90.phase, (int, float)) or not np.isfinite(x90.phase):
                raise ValueError("X90 phase must be a finite scalar")
        return self


def estimate_d4(fit: IRBAnalysis) -> RBEstimate:
    """Convert numerical paired-IRB estimates and interval from d=2 to d=4.

    fit supplies reference/interleaved decays, d=2 confidence limits, paired counts
    and bootstrap diagnostics. Return RBEstimate with unchanged decay/diagnostic
    fields and d=4 error quantities. Leave fit unchanged; do not acquire or draw.
    """
    # The shared owner fits and resamples paired curves; only d=4 interpretation
    # belongs here. Its d=2 error is affine in the same decay ratio.
    return RBEstimate(
        p_reference=fit.p_reference,
        p_interleaved=fit.p_interleaved,
        reference_epc=0.75 * (1 - fit.p_reference),
        gate_error=0.75 * (1 - fit.p_interleaved / fit.p_reference),
        error_ci_low=1.5 * (1 - fit.fidelity_ci_high),
        error_ci_high=1.5 * (1 - fit.fidelity_ci_low),
        n_paired_seeds=fit.n_paired_seeds,
        n_paired_rounds=fit.n_paired_rounds,
        bootstrap_successes=fit.bootstrap_successes,
        warning=fit.warning,
    )


def estimate_joint_d4(
    fit: SharedFitResult,
    *,
    ci_low: float,
    ci_high: float,
    bootstrap_successes: int,
    paired_seeds: int,
    paired_rounds: int,
) -> RBEstimate:
    """Convert joint arm decays and a seed-bootstrap interval to d=4 estimates.

    fit supplies reference p_0 and interleaved p_1. ci_low/ci_high bound the
    statistical gate-error interval; bootstrap_successes counts accepted fits.
    paired_seeds and paired_rounds count the available identity union. Return
    RBEstimate with unclipped error and a statistical-only warning. Mutate no
    inputs, draw nothing and propagate missing decay keys or arithmetic errors.
    """
    pr, pi = fit.values["p_0"], fit.values["p_1"]
    error = 0.75 * (1 - pi / pr)
    warning = (
        "Statistical CI only; gate-dependent/coherent noise can bias the IRB ratio."
    )
    if error < 0:
        warning += " Interleaved decay exceeds reference: negative error estimate retained, not clipped."
    return RBEstimate(
        pr,
        pi,
        0.75 * (1 - pr),
        error,
        ci_low,
        ci_high,
        paired_seeds,
        paired_rounds,
        bootstrap_successes,
        warning,
    )


def count_completion(
    counts: NDArray[np.float64], class_axes: tuple[int, ...]
) -> NDArray[np.bool_]:
    """Return complete-slot masks; raise ValueError for invalid or partial counts.

    counts has leading scan axes and trailing class_axes (negative indices allowed).
    All-NaN slots are unacquired. Return a boolean mask over the remaining axes;
    leave counts unchanged and draw nothing.
    """
    acquired = np.asarray(np.isfinite(counts).all(axis=class_axes), dtype=np.bool_)
    missing = np.isnan(counts).all(axis=class_axes)
    if not np.all(acquired | missing) or np.any(counts < 0):
        raise ValueError("RB sums contains invalid counts or a partial round")
    return acquired


def paired_means(survival: NDArray[np.float64]) -> NDArray[np.float64]:
    """Average complete rounds inside each seed, then seeds equally, per arm.

    survival is (depth,seed,arm,round), with two arms and unavailable entries NaN.
    Use rounds finite across every depth and both arms for each seed. Return
    (2,depth) arm means or all NaN when no seed is complete. Mutate no inputs;
    do not acquire or draw.
    """
    complete = np.asarray(np.isfinite(survival).all(axis=(0, 2)), dtype=np.bool_)
    counts = complete.sum(axis=1)
    kept = counts > 0
    if not np.any(kept):
        return np.full((2, survival.shape[0]), np.nan)
    seed_means = np.where(complete[None, :, None, :], survival, 0).sum(axis=-1)
    return (seed_means[:, kept, :] / counts[None, kept, None]).mean(axis=1).T


def validate_axes(
    axes: tuple[NDArray[np.int64], ...],
    expected: tuple[Sequence[int] | NDArray[np.int64], ...],
) -> None:
    """Raise ValueError when persisted axis identities do not match saved cfg.

    axes and expected list persisted and configured identities in the same axis
    order (depth,seed,arm,round). Compare exact values; unequal sequence lengths also
    raise ValueError. Return None without modifying either sequence or drawing.
    """
    if any(
        not np.array_equal(actual, wanted)
        for actual, wanted in zip(axes, expected, strict=True)
    ):
        raise ValueError("RB count axes do not match saved cfg")


def restore_result(data: GroupedLoadData[RBCfg]) -> RBResult:
    """Restore counts; old per-shot IQ files fail because sums is missing.

    data supplies the grouped sums variable and its persisted scan axes. Return
    this experiment's Result with float64 class counts and its original axis
    ordering. Missing variables and invalid data propagate load errors. Do not
    mutate the loaded data, acquire or draw; no raw-IQ reconstruction is attempted.
    """
    sums = data.variable("sums")
    return RBResult(
        depths=sums.axes[5].astype(np.int64),
        seeds=sums.axes[4].astype(np.int64),
        rounds=sums.axes[2].astype(np.int64),
        sums=sums.z.astype(np.float64),
    )


class RBExp(Experiment[RBResult, RBCfg]):
    """Paired RB counts, native IO and complete-round joint00 analysis."""

    AXES_SPEC: ClassVar[GroupedAxesSpec[RBResult, RBCfg]] = GroupedAxesSpec(
        variables=(
            VariableSpec(
                variable="sums",
                axes=(
                    VariableAxisSpec.generated_arange(label="terminal_f2", unit=""),
                    VariableAxisSpec.generated_arange(label="terminal_f1", unit=""),
                    VariableAxisSpec(
                        label="Round", unit="", field_name="rounds", dtype=np.int64
                    ),
                    VariableAxisSpec(
                        label="Arm", unit="", values=(0, 1), dtype=np.int64
                    ),
                    VariableAxisSpec(
                        label="Seed", unit="", field_name="seeds", dtype=np.int64
                    ),
                    VariableAxisSpec(
                        label="Clifford depth",
                        unit="",
                        field_name="depths",
                        dtype=np.int64,
                    ),
                ),
                z=VariableZSpec(
                    field_name="sums",
                    label="Joint class counts",
                    unit="",
                    dtype=np.float64,
                ),
            ),
        ),
        result_type=RBResult,
        cfg_type=RBCfg,
        cfg_schema_version="5.0",
        tag="two_qubit/rb",
        result_builder=restore_result,
    )

    Options = RBOptions

    def run(self, config: RBCfg, *, context: RunContext) -> RBResult:
        """Acquire optional reset → concurrent per-qubit sequence → terminal counts.

        Each host round acquires reps shots with rounds=1, not a host average.
        Return fixed raw count slots; cfg and saved sequence identities are unchanged.

        """
        # 1. Compare reference and CZ-interleaved joint00 survival across Clifford depth.
        fig, ax = context.plots.subplots(num="measurement")
        ax.set(xlabel="Random Clifford depth", ylabel="Joint00 survival")
        viewer = live_line(ax, num_lines=2)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.grid(False)
        fig.show()

        @context.plots.on_plot_thread
        def update_view(sums: NDArray[np.float64]) -> tuple[Line2D, ...]:
            totals = np.nansum(sums, axis=(1, 3))
            return viewer.set(
                np.array(config.depth), joint_probabilities(totals)[..., 0, 0].T
            )

        # 2. Compile each paired depth/seed/arm sequence once and keep individual host-round count slots.
        buffer = SignalBuffer(
            (len(config.depth), len(config.seed), 2, config.rounds, 3, 3),
            dtype=np.float64,
            on_update=update_view,
        )
        with Schedule(buffer, scope=context.scope) as sched:
            # Host depth/seed/arm change the gate sequence. Host rounds preserve paired slots.
            for depth, depth_step in sched.scan("depth", config.depth):
                for seed, seed_step in depth_step.scan("seed", config.seed):
                    pair = make_paired_sequences(depth, seed)
                    for arm, arm_step in seed_step.scan("arm", [0, 1]):
                        sequence = (pair.reference, pair.interleaved)[arm]
                        builder = TimelineBuilder(
                            context.soccfg,
                            reps=config.reps,
                            initial_delay=config.initial_delay,
                            relax_delay=config.relax_delay,
                        )
                        readout = MuxReadout("readout", config.readout)
                        if config.reset is not None:
                            builder.emit(Reset("reset", config.reset))

                        # Both arms overlap X90s within a Clifford and align at Clifford ends.
                        emit_sequence(
                            builder,
                            sequence,
                            x90_f1=config.x90_f1,
                            x90_f2=config.x90_f2,
                            drive=config.drive,
                            phase_pair=config.phase_pair,
                        )
                        builder.emit(readout, tag="terminal")
                        prog = builder.build()
                        processor = shot_sums(
                            prog,
                            [
                                Classify("terminal", config.roles.f1_tone),
                                Classify("terminal", config.roles.f2_tone),
                            ],
                        )
                        for _, round_step in arm_step.repeat("round", config.rounds):
                            round_step.run_acquire_summed(
                                prog,
                                context.soc,
                                processor=processor,
                                rounds=1,
                                options=RunOptions(),
                            )
        buffer.trigger_update(flush=True)
        # 3. Preserve paired sequence identities and raw terminal counts; unfinished round slots stay NaN.
        return RBResult(
            np.array(config.depth, dtype=np.int64),
            np.array(config.seed, dtype=np.int64),
            np.arange(config.rounds, dtype=np.int64),
            buffer.array,
        )

    def analyze(
        self,
        source: RunRecord[RBCfg, RBResult],
        options: RBOptions,
        *,
        plots: Plots,
    ) -> RBAnalysis:
        """Fit saved joint00 count ratios from complete paired seed/round curves.

        options controls deterministic paired-seed bootstrap, not acquisition.
        Malformed counts raise ValueError. Missing cfg, AnalysisFailed or existing
        fit ValueError/RuntimeError failures return a reason with count diagnostics.
        Other exceptions propagate unchanged.

        """
        # 1. Select seed/round pairs complete across every depth and both arms before fitting joint00 decay.
        data, cfg = source.result, source.cfg
        shape = (len(data.depths), len(data.seeds), 2, len(data.rounds))
        survival = np.full(shape, np.nan)
        acquired = np.zeros(shape, dtype=np.bool_)
        means = np.full((2, len(data.depths)), np.nan)
        curves = means.copy()
        estimate = None
        quality = None
        reason = None
        if cfg is None:
            reason = "RB analysis requires saved cfg; current Form is not a substitute"
        else:
            validate_axes(
                (data.depths, data.seeds, np.arange(2, dtype=np.int64), data.rounds),
                (cfg.depth, cfg.seed, [0, 1], np.arange(cfg.rounds)),
            )
            if data.sums.shape != (*shape, 3, 3):
                raise ValueError("Expected RB sums shape (depth,seed,arm,round,3,3)")
            counts = data.sums
            acquired = count_completion(counts, (-2, -1))
            joint00 = joint_probabilities(counts)[..., 0, 0]
            paired = np.asarray(
                (acquired & np.isfinite(joint00)).all(axis=(0, 2)), dtype=np.bool_
            )
            survival = np.where(paired[None, :, None, :], joint00, np.nan)
            means = paired_means(survival)
            # 2. Fit paired decay and bootstrap whole seeds before the d=4 error conversion.
            try:
                fit = analyze_irb(
                    data.depths,
                    survival.transpose(3, 1, 2, 0).astype(np.complex128),
                    bootstrap_samples=options.bootstrap_samples,
                    bootstrap_seed=options.bootstrap_seed,
                )
            except (AnalysisFailed, ValueError, RuntimeError) as exc:
                # Isolate model failure while retaining measured count diagnostics.
                reason = str(exc)
            else:
                estimate = estimate_d4(fit.summary)
                quality = (
                    compute_fit_quality(
                        fit.mean_signals[0], fit.fit_signals[0], {}, None
                    ),
                    compute_fit_quality(
                        fit.mean_signals[1], fit.fit_signals[1], {}, None
                    ),
                )
                # Restore probability coordinates, not the decay ratio.
                sign = 1.0 if np.sum(fit.mean_signals * means) >= 0 else -1.0
                curves = sign * fit.fit_signals
        # 3. Compare paired reference/interleaved mean survival and fitted probability curves with failure reasons.
        fig, ax = plots.subplots(num="fit", figsize=(9, 6))
        for arm, label in enumerate(("Reference", "CZ interleaved")):
            (line,) = ax.plot(data.depths, means[arm], "o", label=label)
            if estimate is not None:
                ax.plot(data.depths, curves[arm], color=line.get_color())
        ax.set(xlabel="Random Clifford depth", ylabel="Same-shot joint00 survival")
        ax.legend()
        if reason is not None:
            ax.set_title(f"No RB estimate: {reason}")
        else:
            ax.set_title("d=4 paired RB; classified ensemble, statistical seed CI only")
        fig.tight_layout()
        # 4. Keep the d=4 estimate and statistical seed uncertainty, or a failure without a calibration proposal.
        return RBAnalysis(estimate, quality, reason)
