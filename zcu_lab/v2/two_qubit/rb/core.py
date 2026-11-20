"""Paired two-qubit RB with acquisition-side classification and round reduction."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Annotated, ClassVar, TypeAlias

import numpy as np
from matplotlib.ticker import MaxNLocator
from numpy.typing import NDArray
from pydantic import Field
from qick import QickConfig
from zcu_tools.analysis.failure import AnalysisFailed
from zcu_tools.analysis.fitting import (
    FitQuality,
    IRBAnalysis,
    analyze_irb,
    seed_standard_deviation,
)
from zcu_tools.analysis.fitting.shared import SharedFitResult
from zcu_tools.cfg_model import ConfigBase
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
from zcu_tools.experiment.v2.runtime import Schedule, SignalBuffer
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import (
    Classify,
    MuxHeraldResetCfg,
    Reset,
    ResetCfg,
    TimelineBuilder,
    TimelineProgramV2,
    shot_sums,
)
from zcu_tools.program.v2.modules import MuxReadout

from zcu_lab.v2.support import live_errorbar

from .._rb_scheduling import compile_depth_sweep, emit_depth_sweep
from .._rb_sequence import RBArm, make_rb_arms
from .._sums import joint_probabilities
from ._config import RBInputs


@dataclass(frozen=True)
class RBResult:
    """Float64 class counts with axes (seed, arm, depth, F1, F2).

    depths counts random Cliffords; seeds retains exact random.Random identities.
    The fixed arm axis [0,1] means reference/CZ-interleaved. sums ends in (3,3),
    ordered g/e/other on each role. Counts include only kept shots from finished
    rounds. A measured cell with no kept shots is all zero; an unacquired cell
    stays NaN.
    """

    depths: NDArray[np.int64]
    seeds: NDArray[np.int64]
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
    n_paired_seeds counts seeds with complete curves in both arms.
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
    bootstrap_successes: int
    warning: str


@dataclass(frozen=True)
class RBAnalysis:
    """Fixed-size d=4 paired RB estimate and reference/interleaved fit quality.

    estimate is None with reason when no fit is identifiable. fit_quality holds
    diagnostics including the relative error of decay_time in each arm, or None
    without a fit. Survival arrays and model curves stay inside analyze.
    """

    estimate: RBEstimate | None
    fit_quality: tuple[FitQuality, FitQuality] | None
    reason: str | None


_RBResetCfg: TypeAlias = Annotated[
    ResetCfg | MuxHeraldResetCfg, Field(discriminator="type")
]


class RBCfg(RBInputs[_RBResetCfg]):
    """Optional reset → paired random Clifford + own recovery → terminal read.

    Use None for no reset, an ordinary reset to prepare every shot, or
    reset/mux_herald with the terminal readout settings to retain matching shots.
    See RBInputs for acquisition and scan fields.
    """


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
) -> RBEstimate:
    """Convert joint arm decays and a seed-bootstrap interval to d=4 estimates.

    fit supplies reference p_0 and interleaved p_1. ci_low/ci_high bound the
    statistical gate-error interval; bootstrap_successes counts accepted fits.
    paired_seeds counts the available seed identity union. Return
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
        bootstrap_successes,
        warning,
    )


def validate_counts(counts: NDArray[np.float64], class_axes: tuple[int, ...]) -> None:
    """Reject negative counts or a cell that is neither all finite nor all NaN.

    counts ends in the given class_axes (negative indices allowed).
    All-NaN cells are unacquired. Leave counts unchanged.
    """
    finite = np.isfinite(counts).all(axis=class_axes)
    missing = np.isnan(counts).all(axis=class_axes)
    if not np.all(finite | missing) or np.any(counts < 0):
        raise ValueError("RB sums contains invalid counts or a partial cell")


def validate_axes(
    axes: tuple[NDArray[np.int64], ...],
    expected: tuple[Sequence[int] | NDArray[np.int64], ...],
) -> None:
    """Raise ValueError when persisted axis identities do not match saved cfg.

    axes and expected list persisted and configured identities in the same axis
    order (seed, depth). Compare exact values; unequal sequence lengths also
    raise ValueError. Return None without modifying either sequence or drawing.
    """
    if any(
        not np.array_equal(actual, wanted)
        for actual, wanted in zip(axes, expected, strict=True)
    ):
        raise ValueError("RB count axes do not match saved cfg")


def restore_result(data: GroupedLoadData[RBCfg]) -> RBResult:
    """Restore counts on (seed, arm, depth, F1, F2) from saved arrays.

    data supplies the grouped sums variable and its persisted scan axes. Return
    this experiment's Result with float64 class counts and its original axis
    ordering. Missing variables and invalid data propagate load errors. Do not
    mutate the loaded data, acquire or draw; no raw-IQ reconstruction is attempted.
    """
    sums = data.variable("sums")
    return RBResult(
        depths=sums.axes[2].astype(np.int64),
        seeds=sums.axes[4].astype(np.int64),
        sums=sums.z.astype(np.float64),
    )


def build_program(config: RBCfg, soccfg: QickConfig, arm: RBArm) -> TimelineProgramV2:
    """Compile one seed and arm: optional reset, every depth on board, then readout.

    Use config's pulse calibrations and shot settings on soccfg. The arm supplies
    seed and depths. Raise ValueError with the board limit if the program exceeds
    PMEM capacity; other compile errors propagate unchanged.
    """
    builder = TimelineBuilder(
        soccfg,
        reps=config.reps,
        initial_delay=config.initial_delay,
        relax_delay=config.relax_delay,
    )
    readout = MuxReadout("readout", config.readout)
    if config.reset is not None:
        builder.emit(Reset("reset", config.reset, terminal=readout))
    emit_depth_sweep(
        builder,
        arm,
        x90=(config.x90_f1, config.x90_f2),
        x180=(config.x180_f1, config.x180_f2),
        drive=config.drive,
        phase_pair=config.phase_pair,
    )
    builder.emit(readout, tag="terminal")
    return compile_depth_sweep(builder, arm)


class RBExp(Experiment[RBResult, RBCfg]):
    """Paired RB counts, native IO and complete-seed joint00 analysis."""

    AXES_SPEC: ClassVar[GroupedAxesSpec[RBResult, RBCfg]] = GroupedAxesSpec(
        variables=(
            VariableSpec(
                variable="sums",
                axes=(
                    VariableAxisSpec.generated_arange(label="terminal_f2", unit=""),
                    VariableAxisSpec.generated_arange(label="terminal_f1", unit=""),
                    VariableAxisSpec(
                        label="Clifford depth",
                        unit="",
                        field_name="depths",
                        dtype=np.int64,
                    ),
                    VariableAxisSpec(
                        label="Arm", unit="", values=(0, 1), dtype=np.int64
                    ),
                    VariableAxisSpec(
                        label="Seed", unit="", field_name="seeds", dtype=np.int64
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
        cfg_schema_version="6.0",
        tag="two_qubit/rb",
        result_builder=restore_result,
    )

    Options = RBOptions

    def run(self, config: RBCfg, *, context: RunContext) -> RBResult:
        """Acquire optional reset → concurrent per-qubit sequence → terminal counts.

        Compile one board program per seed and arm before acquiring. Each
        program acquires reps shots per depth for each acquisition round and the
        reducer adds finished rounds. A mux herald reset excludes rejected shots.
        Return counts on (seed, arm, depth, F1, F2), NaN in unacquired cells.
        """
        # 1. Compare reference and CZ-interleaved joint00 survival across Clifford depth.
        fig, ax = context.plots.subplots(num="measurement")
        ax.set(
            xlabel="Random Clifford depth", ylabel="Joint (g,g) survival probability"
        )
        viewer = live_errorbar(ax, num_lines=2)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        fig.show()

        @context.plots.on_plot_thread
        def update_view(sums: NDArray[np.float64]):
            totals = np.nansum(sums, axis=0)  # Pool seeds for the markers.
            survival = joint_probabilities(sums)[..., 0, 0]
            return viewer.set(
                np.array(config.depth),
                joint_probabilities(totals)[..., 0, 0],
                seed_standard_deviation(survival),
            )

        # 2. Compile all arms before acquiring any shots.
        programs = [
            [
                build_program(config, context.soccfg, arm)
                for arm in make_rb_arms(seed, config.depth)
            ]
            for seed in config.seed
        ]

        # 3. ADR-0083: The reducer sums completed rounds in one acquire per seed and arm.
        buffer = SignalBuffer(
            (len(config.seed), 2, len(config.depth), 3, 3),
            dtype=np.float64,
            on_update=update_view,
        )
        with Schedule(buffer, scope=context.scope) as sched:
            for seed_programs, seed_step in sched.scan("seed", programs):
                for program, arm_step in seed_step.scan("arm", seed_programs):
                    reducer = shot_sums(
                        program,
                        [
                            Classify("terminal", config.roles.f1_tone),
                            Classify("terminal", config.roles.f2_tone),
                        ],
                    )
                    arm_step.run_acquire_summed(
                        program, context.soc, reducer=reducer, rounds=config.rounds
                    )
        buffer.trigger_update(flush=True)
        return RBResult(
            np.array(config.depth, dtype=np.int64),
            np.array(config.seed, dtype=np.int64),
            buffer.array,
        )

    def analyze(
        self,
        source: RunRecord[RBCfg, RBResult],
        options: RBOptions,
        *,
        plots: Plots,
    ) -> RBAnalysis:
        """Fit saved joint00 count ratios from complete paired seed curves.

        options controls deterministic paired-seed bootstrap, not acquisition.
        Cells with zero counts have undefined survival. A fit uses only seeds
        with finite survival across both arms and every depth.
        Malformed counts raise ValueError. Missing cfg, AnalysisFailed or existing
        fit ValueError/RuntimeError failures return a reason with count diagnostics.
        Other exceptions propagate unchanged.

        """
        # 1. Normalize each cell by its own acquired shots.
        data, cfg = source.result, source.cfg
        shape = (len(data.seeds), 2, len(data.depths), 3, 3)
        if cfg is not None:
            validate_axes((data.seeds, data.depths), (cfg.seed, cfg.depth))
            if data.sums.shape != shape:
                raise ValueError("Expected RB sums shape (seed,arm,depth,3,3)")
            validate_counts(data.sums, (-2, -1))
        survival = joint_probabilities(data.sums)[..., 0, 0]
        means = np.full((2, len(data.depths)), np.nan)
        curves = means.copy()
        seed_std = means.copy()
        estimate = None
        quality = None
        reason = None
        if cfg is None:
            reason = "RB analysis requires saved cfg; current Form is not a substitute"
        else:
            complete = np.isfinite(survival).all(axis=(1, 2))
            if np.any(complete):
                means = survival[complete].mean(axis=0)
            # 2. Fit paired decay and bootstrap whole seeds before the d=4 error conversion.
            try:
                fit = analyze_irb(
                    data.depths,
                    survival.astype(np.complex128),
                    bootstrap_samples=options.bootstrap_samples,
                    bootstrap_seed=options.bootstrap_seed,
                )
            except (AnalysisFailed, ValueError, RuntimeError) as exc:
                # Isolate model failure while retaining measured count diagnostics.
                reason = str(exc)
                seed_std = seed_standard_deviation(survival[complete])
            else:
                seed_std = fit.seed_std
                estimate = estimate_d4(fit.summary)
                quality = (fit.arm_fits[0].quality, fit.arm_fits[1].quality)
                # Restore probability coordinates, not the decay ratio.
                sign = 1.0 if np.sum(fit.mean_signals * means) >= 0 else -1.0
                curves = sign * fit.fit_signals
        # 3. Compare paired reference/interleaved mean survival and fitted probability curves with failure reasons.
        # Show pooled counts as in the live plot before the equal-seed fit.
        pooled = joint_probabilities(np.nansum(data.sums, axis=0))
        pooled_std = seed_standard_deviation(survival)
        _, ax = plots.subplots(num="pooled_survival", layout="constrained")
        for arm, label in enumerate(("Reference", "CZ interleaved")):
            ax.errorbar(
                data.depths,
                pooled[arm, :, 0, 0],
                yerr=pooled_std[arm],
                fmt=".",
                label=label,
            )
        ax.set(
            xlabel="Random Clifford depth", ylabel="Joint (g,g) survival probability"
        )
        ax.legend()

        fig, ax = plots.subplots(num="fit", figsize=(9, 6))
        for arm, label in enumerate(("Reference", "CZ interleaved")):
            (line,) = ax.plot(data.depths, means[arm], ".", label=label)
            ax.errorbar(
                data.depths,
                means[arm],
                yerr=seed_std[arm],
                fmt="none",
                ecolor=line.get_color(),
            )
            if estimate is not None:
                ax.plot(data.depths, curves[arm], color=line.get_color())
        ax.set(
            xlabel="Random Clifford depth", ylabel="Joint (g,g) survival probability"
        )
        ax.legend()
        if reason is not None:
            ax.set_title(f"No RB estimate: {reason}")
        else:
            ax.set_title("d=4 paired RB; classified ensemble, statistical seed CI only")
        fig.tight_layout()
        # 4. Keep the d=4 estimate and statistical seed uncertainty, or a failure without a calibration proposal.
        return RBAnalysis(estimate, quality, reason)
