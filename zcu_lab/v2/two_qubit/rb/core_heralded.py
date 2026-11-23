"""Paired RB with gg/ee herald groups and shared return-survival decays."""

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from matplotlib.axes import Axes
from matplotlib.ticker import MaxNLocator
from numpy.typing import NDArray
from qick import QickConfig
from zcu_tools.analysis.fitting import (
    FitQuality,
    RBJointDecayFit,
    compute_fit_quality,
    decay_time_bounds,
    fit_joint_decay,
    seed_standard_deviation,
    seed_statistics,
)
from zcu_tools.analysis.shot_uncertainty import class_probability_covariance
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
    Reset,
    TimelineBuilder,
    TimelineProgramV2,
    shot_sums,
)
from zcu_tools.program.v2.modules import MuxReadout

from .._rb_scheduling import compile_depth_sweep, emit_depth_sweep
from .._rb_sequence import RBArm, make_rb_arms
from .._sums import (
    herald_probabilities,
    joint_probabilities,
    terminal_counts,
)
from ._config import RBInputs
from .core import (
    RBEstimate,
    RBOptions,
    estimate_joint_d4,
    validate_axes,
    validate_counts,
)


@dataclass(frozen=True)
class RBHeraldedResult:
    """Raw float64 sums (seed,arm,depth,3,3,3,3), NaN when unacquired.

    depths counts random Cliffords, seeds stores RNG identities, and arm 0/1
    is reference/CZ-interleaved. Trailing axes are herald Qubit1/Qubit2 then terminal
    Qubit1/Qubit2, each ordered g/e/other. All herald outcomes from finished rounds
    remain saved; counts are not normalized or filtered.
    """

    depths: NDArray[np.int64]
    seeds: NDArray[np.int64]
    sums: NDArray[np.float64]


@dataclass(frozen=True)
class RBHeraldedAnalysis:
    """Joint gg/ee d=4 estimate and diagnostics of shared arm decays.

    estimate is None with reason when unavailable. fit_quality is ordered gg
    reference, gg interleaved, ee reference, ee interleaved, or None without a
    fit. Its decay_time relative error is shared across the two herald groups
    for each arm. Per-depth survival and counts stay in analyze.
    """

    estimate: RBEstimate | None
    fit_quality: tuple[FitQuality, ...] | None
    reason: str | None


class RBHeraldedCfg(RBInputs[Reset.Cfg]):
    """Optional reset → herald → paired sequence → terminal, per shot.

    Fields and units match RBCfg. Both reads share the saved full mux cfg and
    circle radii. Its post_delay supplies the wait after herald. reset=None
    emits no reset. The reducer sums finished acquisition rounds.
    """


def restore_result(data: GroupedLoadData[RBHeraldedCfg]) -> RBHeraldedResult:
    """Restore 81-class counts from saved (seed, arm, depth, ...) arrays.

    data supplies the grouped sums variable and its persisted scan axes. Return
    this experiment's Result with float64 class counts and its original axis
    ordering. Missing variables and invalid data propagate load errors. Do not
    mutate the loaded data, acquire or draw; no raw-IQ reconstruction is attempted.
    """
    sums = data.variable("sums")
    return RBHeraldedResult(
        depths=sums.axes[4].astype(np.int64),
        seeds=sums.axes[6].astype(np.int64),
        sums=sums.z.astype(np.float64),
    )


def bootstrap_joint_error(
    depths: NDArray[np.int64],
    survival: NDArray[np.float64],
    totals: NDArray[np.float64],
    *,
    samples: int,
    seed: int,
) -> tuple[float, float, int]:
    """Bootstrap the 95% d=4 ratio CI; raise ValueError below 90% valid fits.

    Inputs are (herald,seed,arm,depth), with totals in shot counts. NaN denotes
    unavailable survival. Draw from the union of complete seed identities;
    each herald retains only its valid draws,
    including multiplicity. Both arms/depths travel together. Fewer than two
    retained draws in either group fails that resample, never fills missing data.
    depths is the shared integer Clifford-depth axis. samples is the number of
    resamples and seed initializes a local RNG. Return (low, high, valid fits)
    without changing inputs, global random state or plots.
    """
    available = np.isfinite(survival).all(axis=(2, 3))
    identities = np.flatnonzero(available.any(axis=0))
    rng = np.random.default_rng(seed)
    errors = []
    for _ in range(samples):
        draw = rng.choice(identities, size=len(identities), replace=True)
        try:
            statistics = [
                seed_statistics(
                    survival[h, draw],
                    totals[h, draw],
                )
                for h in range(2)
            ]
            means = np.array([item.means for item in statistics])
            uncertainty = np.array([item.errors for item in statistics])
            fit = fit_joint_decay(depths, means, uncertainty).fit
        except (ValueError, RuntimeError):
            continue
        errors.append(0.75 * (1 - fit.values["p_1"] / fit.values["p_0"]))
    if len(errors) < 0.9 * samples:
        raise ValueError(f"Unstable bootstrap: {len(errors)}/{samples} valid fits")
    low, high = np.quantile(errors, [0.025, 0.975])
    return float(low), float(high), len(errors)


def _draw_count_bars(
    ax: Axes,
    depths: NDArray[np.int64],
    survival: NDArray[np.float64],
    sem: NDArray[np.float64],
    seed_std: NDArray[np.float64],
    *,
    fit_overlay: bool = False,
) -> None:
    """Draw each arm's shot SEM and seed SD; omit connecting lines over a fit."""
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
            fmt="." if fit_overlay else ".-",
            color=color,
            capsize=3,
            label=f"{label} data" if fit_overlay else label,
        )
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))


def _draw_pooled_survival(
    ax: Axes,
    depths: NDArray[np.int64],
    sums: NDArray[np.float64],
    fit: RBJointDecayFit | None = None,
) -> None:
    """Draw pooled gg/ee returns, shot SEM, seed SD and optional joint fit."""
    gg = terminal_counts(sums, ("g", "g"))
    ee = terminal_counts(sums, ("e", "e"))
    gg_trials = gg.sum(axis=(-2, -1))
    ee_trials = ee.sum(axis=(-2, -1))
    returns = gg[..., 0, 0] + ee[..., 1, 1]
    trials = gg_trials + ee_trials
    seed_survival = np.divide(
        returns, trials, out=np.full_like(returns, np.nan), where=trials > 0
    )
    seed_std = seed_standard_deviation(seed_survival)

    pooled_gg = np.nansum(gg_trials, axis=0)
    pooled_trials = pooled_gg + np.nansum(ee_trials, axis=0)
    pooled_returns = np.nansum(returns, axis=0)
    survival = np.divide(
        pooled_returns,
        pooled_trials,
        out=np.full_like(pooled_returns, np.nan),
        where=pooled_trials > 0,
    )
    binary = np.stack((pooled_returns, pooled_trials - pooled_returns), axis=-1)
    sem = np.sqrt(class_probability_covariance(binary)[..., 0, 0])
    _draw_count_bars(ax, depths, survival, sem, seed_std, fit_overlay=fit is not None)

    if fit is not None:
        xs = np.union1d(depths, np.linspace(depths.min(), depths.max(), 201))
        params = fit.fit.values
        for arm, label in enumerate(("Reference", "CZ interleaved")):
            measured = pooled_trials[arm] > 0
            gg_fraction = pooled_gg[arm, measured] / pooled_trials[arm, measured]
            weight = np.interp(xs, depths[measured], gg_fraction)
            decay = params[f"p_{arm}"] ** xs
            gg_curve = params[f"A_0_{arm}"] * decay + params[f"B_0_{arm}"]
            ee_curve = params[f"A_1_{arm}"] * decay + params[f"B_1_{arm}"]
            ax.plot(
                xs,
                weight * gg_curve + (1 - weight) * ee_curve,
                color=f"C{arm}",
                label=f"{label} fit",
            )
    ax.set(
        xlabel="Random Clifford depth (count)",
        ylabel="P(return to herald state | gg or ee herald)",
    )
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1), ncols=2)


def _draw_return_survival(
    ax: Axes,
    depths: NDArray[np.int64],
    means: NDArray[np.float64],
    seed_std: NDArray[np.float64],
    fit: RBJointDecayFit | None,
    herald: int,
) -> None:
    """Draw equal-seed means with seed SD and the existing shared-decay model."""
    xs = np.linspace(depths.min(), depths.max(), 201)
    for arm, label in enumerate(("Reference", "CZ interleaved")):
        color = f"C{arm}"
        ax.errorbar(
            depths,
            means[arm],
            seed_std[arm],
            fmt="." if fit is not None else ".-",
            color=color,
            capsize=3,
            label=f"{label} data",
        )
        if fit is not None:
            params = fit.fit.values
            curve = (
                params[f"A_{herald}_{arm}"] * params[f"p_{arm}"] ** xs
                + params[f"B_{herald}_{arm}"]
            )
            ax.plot(xs, curve, color=color, label=f"{label} fit")
    state = ("gg", "ee")[herald]
    ax.set(
        xlabel="Random Clifford depth (count)",
        ylabel=f"P({state} return | {state} herald)",
    )
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1), ncols=2)


def _plot_pooled_survival(
    plots: Plots,
    data: RBHeraldedResult,
    fit: RBJointDecayFit | None,
    analysis: RBHeraldedAnalysis,
) -> None:
    """Show the pooled live return view with the joint fit and result."""
    fig, ax = plots.subplots(
        num="pooled_survival", figsize=(7, 5), layout="constrained"
    )
    _draw_pooled_survival(ax, data.depths, data.sums, fit)
    fig.suptitle("Heralded RB · Qubit1/Qubit2 · Conditional return")
    report = ["Pooled counts · Capped: shot SEM · Pale: seed SD"]
    estimate = analysis.estimate
    if estimate is not None:
        report.extend(
            (
                f"CZ error estimate = {estimate.gate_error:.3g}; 95% seed CI "
                f"[{estimate.error_ci_low:.3g}, {estimate.error_ci_high:.3g}]",
                f"Joint decay p: Reference {estimate.p_reference:.4g} · "
                f"CZ interleaved {estimate.p_interleaved:.4g} · Fit quality: see gg/ee diagnostics",
            )
        )
    elif analysis.reason is not None:
        report.append(f"No RB estimate: {analysis.reason}")
    fig.supxlabel("\n".join(report), fontsize=10)


def _plot_grouped_survival(plots: Plots, data: RBHeraldedResult) -> None:
    """Show separate shot-pooled gg/ee returns, including incomplete seeds."""
    fig, _ = plots.subplots(
        2, 1, num="grouped_survival", sharex=True, figsize=(7, 6), layout="constrained"
    )
    fig.suptitle("Heralded RB · Qubit1/Qubit2 · Conditional return by herald")
    fig.supxlabel("Pooled counts · Capped: shot SEM · Pale: seed SD", fontsize=10)
    for herald, ax in enumerate(fig.axes):
        counts = data.sums[..., herald, herald, :, :]
        pooled = np.nansum(counts, axis=0)
        successes = pooled[..., herald, herald]
        binary = np.stack((successes, pooled.sum(axis=(-2, -1)) - successes), axis=-1)
        sem = np.sqrt(class_probability_covariance(binary)[..., 0, 0])
        survival = joint_probabilities(pooled)[..., herald, herald]
        seed_std = seed_standard_deviation(
            joint_probabilities(counts)[..., herald, herald]
        )
        _draw_count_bars(ax, data.depths, survival, sem, seed_std)
        state = ("gg", "ee")[herald]
        ax.set_ylabel(f"P({state} return | {state} herald)")
    fig.axes[0].legend(loc="lower center", bbox_to_anchor=(0.5, 1), ncols=2)
    fig.axes[-1].set_xlabel("Random Clifford depth (count)")


def _plot_herald_fractions(
    plots: Plots,
    depths: NDArray[np.int64],
    pooled: NDArray[np.float64],
) -> None:
    """Group the nine herald outcomes by the first qubit, three per panel."""
    aggregate = herald_probabilities(pooled)
    counts = pooled.sum(axis=(-2, -1)).reshape(*pooled.shape[:2], 9)
    cov = class_probability_covariance(counts)
    sem = np.sqrt(np.diagonal(cov, axis1=-2, axis2=-1)).reshape(aggregate.shape)
    classes = ("g", "e", "other")
    for arm, label in enumerate(("Reference", "CZ interleaved")):
        fig, _ = plots.subplots(
            3, 1, num=f"herald_{arm}", sharex=True, figsize=(7, 8), layout="constrained"
        )
        fig.suptitle(f"Heralded RB · Qubit1/Qubit2\n{label} · Herald outcome fractions")
        for qubit1, ax in enumerate(fig.axes):
            for qubit2, state in enumerate(classes):
                ax.errorbar(
                    depths,
                    aggregate[arm, :, qubit1, qubit2],
                    sem[arm, :, qubit1, qubit2],
                    fmt=".-",
                    color=f"C{qubit2 + 2}",
                    capsize=3,
                    label=f"Qubit2 {state}",
                )
            ax.set_ylabel(f"P(Qubit1 {classes[qubit1]}, Qubit2)")
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
            ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1), ncols=3)
        fig.axes[-1].set_xlabel("Random Clifford depth (count)")
        fig.supxlabel("Bars: shot SEM", fontsize=10)


def _plot_decay_fits(
    plots: Plots,
    depths: NDArray[np.int64],
    means: NDArray[np.float64],
    seed_std: NDArray[np.float64],
    fit: RBJointDecayFit | None,
    analysis: RBHeraldedAnalysis,
) -> None:
    """Show each conditional fit, its returned quality and the shared CZ estimate."""
    estimate, quality, reason = analysis.estimate, analysis.fit_quality, analysis.reason
    for herald, label in enumerate(("gg", "ee")):
        fig, ax = plots.subplots(
            num=f"fit_{label}", figsize=(7, 5), layout="constrained"
        )
        _draw_return_survival(ax, depths, means[herald], seed_std[herald], fit, herald)
        fig.suptitle(f"Heralded RB · Qubit1/Qubit2 · {label} decay fit diagnostic")
        report = ["Equal-seed means · Bars: seed SD"]
        if estimate is not None and quality is not None and fit is not None:
            report.append(
                f"CZ error estimate = {estimate.gate_error:.3g}; 95% seed CI "
                f"[{estimate.error_ci_low:.3g}, {estimate.error_ci_high:.3g}]"
            )
            for arm, arm_label in enumerate(("Reference", "CZ interleaved")):
                tau, variance = fit.decay_time_and_variance(arm)
                item = quality[herald * 2 + arm]
                r2 = "unavailable" if item.r2 is None else f"{item.r2:.3g}"
                rms = item.normalized_residual_rms
                rms_label = "unavailable" if rms is None else f"{rms:.3g}"
                report.append(
                    f"{arm_label}: 1/e depth = {tau:.3g} ± {np.sqrt(variance):.2g} count; "
                    f"R² {r2}; normalized RMS {rms_label}"
                )
                # Keep the owner's bound and value; the heading translates its conclusion.
                report.extend(
                    f"{arm_label}: decay unresolved in depth window (count)\n"
                    + issue.reason.removesuffix(": 窗口內無法解析衰減時間")
                    for issue in item.invalid
                    if issue.path == "decay_time"
                )
        elif reason is not None:
            report.append(f"No RB estimate: {reason}")
        fig.supxlabel("\n".join(report), fontsize=10)


def build_program(
    config: RBHeraldedCfg, soccfg: QickConfig, arm: RBArm
) -> TimelineProgramV2:
    """Compile one seed and arm: optional reset, herald, board sweep, terminal read.

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
        builder.emit(Reset("reset", config.reset))
    builder.emit(readout, tag="herald")
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


class RBHeraldedExp(Experiment[RBHeraldedResult, RBHeraldedCfg]):
    """Paired RB with separate gg/ee complete seeds and shared arm decays."""

    AXES_SPEC: ClassVar[GroupedAxesSpec[RBHeraldedResult, RBHeraldedCfg]] = (
        GroupedAxesSpec(
            variables=(
                VariableSpec(
                    variable="sums",
                    axes=(
                        VariableAxisSpec.generated_arange(
                            label="terminal_qubit2", unit=""
                        ),
                        VariableAxisSpec.generated_arange(
                            label="terminal_qubit1", unit=""
                        ),
                        VariableAxisSpec.generated_arange(
                            label="herald_qubit2", unit=""
                        ),
                        VariableAxisSpec.generated_arange(
                            label="herald_qubit1", unit=""
                        ),
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
                        label="Herald and terminal class counts",
                        unit="",
                        dtype=np.float64,
                    ),
                ),
            ),
            result_type=RBHeraldedResult,
            cfg_type=RBHeraldedCfg,
            cfg_schema_version="4.0",
            tag="two_qubit/rb_heralded",
            result_builder=restore_result,
        )
    )

    Options = RBOptions

    def run(self, config: RBHeraldedCfg, *, context: RunContext) -> RBHeraldedResult:
        """Acquire optional reset → herald → concurrent per-qubit sequence → terminal.

        Compile one board program per seed and arm before acquiring. Each
        round acquires reps shots at every depth in the program. Save all 81
        counts per cell summed across finished rounds; unacquired cells stay NaN.
        """
        # 1. Compare pooled gg/ee return for reference and CZ-interleaved arms across depth.
        fig, ax = context.plots.subplots(
            num="measurement", figsize=(7, 5), layout="constrained"
        )
        depths = np.array(config.depth, dtype=np.int64)
        fig.suptitle("Heralded RB · Qubit1/Qubit2 · Conditional return")
        fig.show()

        @context.plots.on_plot_thread
        def update_view(sums: NDArray[np.float64]):
            for bars in tuple(ax.containers):
                bars.remove()
            ax.relim()
            _draw_pooled_survival(ax, depths, sums)
            cells = np.isfinite(sums[..., 0, 0, 0, 0])
            fig.supxlabel(
                f"Measured seed/arm/depth cells: {cells.sum()}/{cells.size} count\n"
                f"Total heralded shots (all outcomes): {np.nansum(sums):,.0f} count\n"
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
            (len(config.seed), 2, len(config.depth), 3, 3, 3, 3),
            dtype=np.float64,
            on_update=update_view,
        )
        with Schedule(buffer, scope=context.scope) as sched:
            for seed_programs, seed_step in sched.scan("seed", programs):
                for program, arm_step in seed_step.scan("arm", seed_programs):
                    reducer = shot_sums(
                        program,
                        [
                            Classify("herald", config.qubit1_tone),
                            Classify("herald", config.qubit2_tone),
                            Classify("terminal", config.qubit1_tone),
                            Classify("terminal", config.qubit2_tone),
                        ],
                    )
                    arm_step.run_acquire_summed(
                        program, context.soc, reducer=reducer, rounds=config.rounds
                    )
        buffer.trigger_update(flush=True)
        return RBHeraldedResult(
            np.array(config.depth, dtype=np.int64),
            np.array(config.seed, dtype=np.int64),
            buffer.array,
        )

    def analyze(
        self,
        source: RunRecord[RBHeraldedCfg, RBHeraldedResult],
        options: RBOptions,
        *,
        plots: Plots,
    ) -> RBHeraldedAnalysis:
        """Fit gg/ee return survival with a shared decay per arm.

        Select seeds with finite survival across both arms and every depth,
        independently per herald. Weight them equally and derive binomial errors
        from each cell's actual accepted shot count. Zero accepted shots mean
        unavailable survival. Bootstrap whole seeds across arms, depths and heralds.
        Malformed counts or mismatched axes raise ValueError; missing cfg or
        failed fits return a reason instead of an estimate.

        """
        # 1. Normalize each herald's cells by its own acquired shots.
        data, cfg = source.result, source.cfg
        shape = (len(data.seeds), 2, len(data.depths))
        if data.sums.shape != (*shape, 3, 3, 3, 3):
            raise ValueError("Expected heralded RB sums (seed,2,depth,3,3,3,3)")
        validate_counts(data.sums, (-4, -3, -2, -1))
        survival = np.full((2, *shape), np.nan)
        paired = np.zeros((2, len(data.seeds)), dtype=np.bool_)
        means = np.full((2, 2, len(data.depths)), np.nan)
        errors = np.full_like(means, np.nan)
        seed_std = np.full_like(means, np.nan)
        rendered_fit = None
        totals = np.full_like(survival, np.nan)
        estimate, quality = None, None
        reason = None
        if cfg is None:
            reason = "RB analysis requires saved cfg; current Form is not a substitute"
        else:
            validate_axes((data.seeds, data.depths), (cfg.seed, cfg.depth))
            for herald, counts in enumerate(
                (
                    terminal_counts(data.sums, ("g", "g")),
                    terminal_counts(data.sums, ("e", "e")),
                )
            ):
                totals[herald] = counts.sum(axis=(-2, -1))
                returns = joint_probabilities(counts)[..., herald, herald]
                survival[herald] = returns
                paired[herald] = np.isfinite(returns).all(axis=(1, 2))
                if np.any(paired[herald]):
                    means[herald] = returns[paired[herald]].mean(axis=0)
            # 2. Use equal-seed means and a shared decay per arm; bootstrap whole paired seed identities.
            try:
                if len(np.unique(data.depths)) < 4:
                    raise ValueError("Need at least four distinct depths")
                for herald in range(2):
                    statistics = seed_statistics(survival[herald], totals[herald])
                    means[herald], errors[herald] = statistics.means, statistics.errors
                    seed_std[herald] = statistics.seed_std
                joint_fit = fit_joint_decay(data.depths, means, errors)
                low, high, successes = bootstrap_joint_error(
                    data.depths,
                    survival,
                    totals,
                    samples=options.bootstrap_samples,
                    seed=options.bootstrap_seed,
                )
            except (ValueError, RuntimeError) as exc:
                reason = str(exc)
                for herald in range(2):
                    seed_std[herald] = seed_standard_deviation(
                        survival[herald, paired[herald]]
                    )
            else:
                rendered_fit = joint_fit
                estimate = estimate_joint_d4(
                    joint_fit.fit,
                    ci_low=low,
                    ci_high=high,
                    bootstrap_successes=successes,
                    paired_seeds=int(paired.any(axis=0).sum()),
                )
                decays = [joint_fit.decay_time_and_variance(arm) for arm in range(2)]
                bounds = decay_time_bounds(data.depths.astype(np.float64))
                quality = tuple(
                    compute_fit_quality(
                        means[h, arm],
                        joint_fit.curves[h, arm],
                        {"decay_time": tau},
                        np.array([[variance]]),
                        decay_bounds=bounds,
                    )
                    for h in range(2)
                    for arm, (tau, variance) in enumerate(decays)
                )
        # 3. Show the pooled return and joint-fit overlay before the group diagnostics.
        analysis = RBHeraldedAnalysis(estimate, quality, reason)
        _plot_pooled_survival(plots, data, rendered_fit, analysis)
        _plot_grouped_survival(plots, data)
        _plot_decay_fits(plots, data.depths, means, seed_std, rendered_fit, analysis)
        _plot_herald_fractions(plots, data.depths, np.nansum(data.sums, axis=0))
        # 4. Keep the shared-decay d=4 estimate and seed-bootstrap interval, or its numerical failure reason.
        return analysis
