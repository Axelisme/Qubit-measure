"""Paired two-qubit RB with acquisition-side classification and round reduction."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Annotated, ClassVar, TypeAlias

import numpy as np
from matplotlib.axes import Axes
from matplotlib.ticker import MaxNLocator
from numpy.typing import NDArray
from pydantic import Field
from qick import QickConfig
from zcu_tools.analysis.failure import AnalysisFailed
from zcu_tools.analysis.fitting import (
    FitQuality,
    IRBAnalysis,
    IRBFitResult,
    analyze_irb,
    expfunc,
    seed_standard_deviation,
)
from zcu_tools.analysis.fitting.shared import SharedFitResult
from zcu_tools.analysis.shot_uncertainty import class_probability_covariance
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

from .._rb_scheduling import compile_depth_sweep, emit_depth_sweep
from .._rb_sequence import RBArm, make_rb_arms
from .._sums import joint_probabilities
from ._config import RBInputs


@dataclass(frozen=True)
class RBResult:
    """Float64 class counts with axes (seed, arm, depth, Qubit1, Qubit2).

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
    """Restore counts on (seed, arm, depth, Qubit1, Qubit2) from saved arrays.

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
        x90=(config.x90_qubit1, config.x90_qubit2),
        x180=(config.x180_qubit1, config.x180_qubit2),
        drive=config.drive,
        phase_pair=config.phase_pair,
    )
    builder.emit(readout, tag="terminal")
    return compile_depth_sweep(builder, arm)


def _draw_pooled_survival(
    ax: Axes, depths: NDArray[np.int64], sums: NDArray[np.float64]
) -> None:
    """Draw kept-shot P(g,g), shot SEM and seed spread for each arm and depth."""
    counts = np.nansum(sums, axis=0)
    successes = counts[..., 0, 0]
    binary = np.stack((successes, counts.sum(axis=(-2, -1)) - successes), axis=-1)
    sem = np.sqrt(class_probability_covariance(binary)[..., 0, 0])
    survival = joint_probabilities(counts)[..., 0, 0]
    seed_std = seed_standard_deviation(joint_probabilities(sums)[..., 0, 0])
    for arm, label in enumerate(("Reference", "CZ interleaved")):
        color = f"C{arm}"
        ax.errorbar(
            depths,
            survival[arm],
            seed_std[arm],
            fmt="none",
            color=color,
            alpha=0.3,
            lw=3,
        )
        ax.errorbar(
            depths,
            survival[arm],
            sem[arm],
            fmt=".",
            color=color,
            capsize=3,
            label=label,
        )
    ax.set(xlabel="Random Clifford depth (count)", ylabel="P(terminal g,g)")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1), ncols=2)


def _draw_seed_fit(
    ax: Axes,
    depths: NDArray[np.int64],
    means: NDArray[np.float64],
    seed_std: NDArray[np.float64],
    fit: IRBFitResult | None,
) -> None:
    """Compare the complete-pair mean and seed SD against each fitted decay."""
    xs = np.linspace(depths.min(), depths.max(), 201)
    for arm, label in enumerate(("Reference", "CZ interleaved")):
        color = f"C{arm}"
        ax.errorbar(
            depths,
            means[arm],
            seed_std[arm],
            fmt=".",
            color=color,
            capsize=3,
            label=f"{label} data",
        )
        if fit is not None:
            sign = 1.0 if np.sum(fit.mean_signals * means) >= 0 else -1.0
            ax.plot(
                xs,
                sign * expfunc(xs, *fit.arm_fits[arm].params),
                color=color,
                label=f"{label} fit",
            )
    ax.set(xlabel="Random Clifford depth (count)", ylabel="P(terminal g,g)")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1), ncols=2)


class RBExp(Experiment[RBResult, RBCfg]):
    """Paired RB counts, native IO and complete-seed joint00 analysis."""

    AXES_SPEC: ClassVar[GroupedAxesSpec[RBResult, RBCfg]] = GroupedAxesSpec(
        variables=(
            VariableSpec(
                variable="sums",
                axes=(
                    VariableAxisSpec.generated_arange(label="terminal_qubit2", unit=""),
                    VariableAxisSpec.generated_arange(label="terminal_qubit1", unit=""),
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
        cfg_schema_version="7.0",
        tag="two_qubit/rb",
        result_builder=restore_result,
    )

    Options = RBOptions

    def run(self, config: RBCfg, *, context: RunContext) -> RBResult:
        """Acquire optional reset → concurrent per-qubit sequence → terminal counts.

        Compile one board program per seed and arm before acquiring. Each
        program acquires reps shots per depth for each acquisition round and the
        reducer adds finished rounds. A mux herald reset excludes rejected shots.
        Return counts on (seed, arm, depth, Qubit1, Qubit2), NaN in unacquired cells.
        """
        # 1. Compare reference and CZ-interleaved joint00 survival across Clifford depth.
        fig, ax = context.plots.subplots(num="measurement", layout="constrained")
        depths = np.array(config.depth, dtype=np.int64)
        fig.suptitle("Two-qubit RB · Qubit1/Qubit2 · Joint ground return")
        fig.show()

        @context.plots.on_plot_thread
        def update_view(sums: NDArray[np.float64]):
            for bars in tuple(ax.containers):
                bars.remove()
            ax.relim()
            _draw_pooled_survival(ax, depths, sums)
            cells = np.isfinite(sums[..., 0, 0])
            fig.supxlabel(
                f"Measured seed/arm/depth cells: {cells.sum()}/{cells.size} count\n"
                f"Total retained shots: {np.nansum(sums):,.0f} count\n"
                "Capped: shot SEM · Pale: seed SD",
                fontsize=10,
            )
            return (fig,)

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
                            Classify("terminal", config.qubit1_tone),
                            Classify("terminal", config.qubit2_tone),
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
        seed_std = means.copy()
        fit = None
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
        # 3. Show the live pooled view first, then the complete-pair fit.
        fig, ax = plots.subplots(num="pooled_survival", layout="constrained")
        _draw_pooled_survival(ax, data.depths, data.sums)
        fig.suptitle("Two-qubit RB · Qubit1/Qubit2 · Joint ground return")
        fig.supxlabel("Pooled counts · Capped: shot SEM · Pale: seed SD", fontsize=10)

        fig, ax = plots.subplots(num="fit", figsize=(7, 5), layout="constrained")
        _draw_seed_fit(ax, data.depths, means, seed_std, fit)
        fig.suptitle("Two-qubit RB · Qubit1/Qubit2 · Paired decay fit")
        report = ["Equal-seed means · Bars: seed SD"]
        if estimate is not None and fit is not None:
            report.append(
                f"CZ error estimate = {estimate.gate_error:.3g}; 95% seed CI "
                f"[{estimate.error_ci_low:.3g}, {estimate.error_ci_high:.3g}]"
            )
            for label, arm in zip(
                ("Reference", "CZ interleaved"), fit.arm_fits, strict=True
            ):
                quality_item = arm.quality
                r2 = (
                    "unavailable"
                    if quality_item.r2 is None
                    else f"{quality_item.r2:.3g}"
                )
                rms = quality_item.normalized_residual_rms
                rms_label = "unavailable" if rms is None else f"{rms:.3g}"
                report.append(
                    f"{label}: 1/e depth = {arm.decay_time:.3g} ± {arm.decay_time_err:.2g} count; "
                    f"R² {r2}; normalized RMS {rms_label}"
                )
                # Keep the owner's bound and value; the heading translates its conclusion.
                report.extend(
                    f"{label}: decay unresolved in depth window (count)\n"
                    + issue.reason.removesuffix(": 窗口內無法解析衰減時間")
                    for issue in quality_item.invalid
                    if issue.path == "decay_time"
                )
        elif reason is not None:
            report.append(f"No RB estimate: {reason}")
        fig.supxlabel("\n".join(report), fontsize=10)
        # 4. Keep the d=4 estimate and statistical seed uncertainty, or a failure without a calibration proposal.
        return RBAnalysis(estimate, quality, reason)
