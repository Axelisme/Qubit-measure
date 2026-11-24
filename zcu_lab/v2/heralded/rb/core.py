"""Paired RB with four herald groups and shared return-survival decays."""

from collections.abc import Mapping
from dataclasses import dataclass
from textwrap import fill
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
from zcu_tools.analysis.fitting.shared import SharedFitResult
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

from zcu_lab.v2.support import paired_rb_analysis
from zcu_lab.v2.support.paired_rb_inputs import RBInputs
from zcu_lab.v2.support.paired_rb_scheduling import (
    compile_depth_sweep,
    emit_depth_sweep,
)
from zcu_lab.v2.support.paired_rb_sequence import RBArm, make_rb_arms

from ..counts import herald_probabilities

_HERALDS = ((0, 0), (0, 1), (1, 0), (1, 1))
_HERALD_LABELS = tuple("ge"[q1] + "ge"[q2] for q1, q2 in _HERALDS)


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
    """Four-group d=4 estimate and diagnostics of shared arm decays.

    estimate and fit_quality are None with reason unless both fit and CI
    succeed. Otherwise fit_quality has eight entries in gg, ge, eg, ee order,
    each reference then interleaved. Its decay_time relative error is shared
    across herald groups for each arm. Per-depth survival and counts stay in analyze.
    """

    estimate: paired_rb_analysis.RBEstimate | None
    fit_quality: tuple[FitQuality, ...] | None
    reason: str | None


class RBHeraldedCfg(RBInputs[Reset.Cfg]):
    """Optional reset → herald → paired sequence → terminal, per shot.

    See RBInputs for acquisition and scan fields. Both reads share the saved full mux cfg and
    circle radii. Its post_delay supplies the wait after herald. reset=None
    emits no reset. The reducer sums finished acquisition rounds.
    """


def estimate_joint_d4(
    fit: SharedFitResult,
    *,
    ci_low: float,
    ci_high: float,
    bootstrap_successes: int,
    paired_seeds: int,
) -> paired_rb_analysis.RBEstimate:
    """Convert joint arm decays and a seed-bootstrap interval to d=4 estimates.

    fit supplies reference p_0 and interleaved p_1. ci_low/ci_high bound the
    statistical gate-error interval; bootstrap_successes counts accepted fits.
    paired_seeds counts the available seed identity union. Return
    paired_rb_analysis.RBEstimate with unclipped error and a statistical-only warning. Mutate no
    inputs, draw nothing and propagate missing decay keys or arithmetic errors.
    """
    pr, pi = fit.values["p_0"], fit.values["p_1"]
    error = 0.75 * (1 - pi / pr)
    warning = (
        "Statistical CI only; gate-dependent/coherent noise can bias the IRB ratio."
    )
    if error < 0:
        warning += " Interleaved decay exceeds reference: negative error estimate retained, not clipped."
    return paired_rb_analysis.RBEstimate(
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
    each herald retains only its valid draws, including multiplicity.
    Both arms/depths travel together. Fewer than two retained draws in any
    group fails that resample, never fills missing data or drops the group.
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
                for h in range(survival.shape[0])
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


def _group_decay_curve(
    params: Mapping[str, float],
    decay: NDArray[np.float64],
    *,
    herald_index: int,
    arm: int,
) -> NDArray[np.float64]:
    """Apply the group's amplitude and offset to the arm's precomputed p**depth."""
    return params[f"A_{herald_index}_{arm}"] * decay + params[f"B_{herald_index}_{arm}"]


def _return_counts(
    sums: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return (matching terminal counts, all terminal counts) in gg/ge/eg/ee order."""
    returns = np.stack([sums[..., q1, q2, q1, q2] for q1, q2 in _HERALDS])
    trials = np.stack(
        [sums[..., q1, q2, :, :].sum(axis=(-2, -1)) for q1, q2 in _HERALDS]
    )
    return returns, trials


def _survival(
    returns: NDArray[np.float64], trials: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Normalize returns by their own shot counts, leaving empty points missing."""
    return np.divide(
        returns, trials, out=np.full_like(returns, np.nan), where=trials > 0
    )


def _shot_sem(
    returns: NDArray[np.float64], trials: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Estimate binomial shot SEM from success and failure counts."""
    binary = np.stack((returns, trials - returns), axis=-1)
    return np.sqrt(class_probability_covariance(binary)[..., 0, 0])


def _draw_pooled_fit(
    ax: Axes,
    depths: NDArray[np.int64],
    group_trials: NDArray[np.float64],
    fit: RBJointDecayFit,
) -> None:
    """Draw each arm's group fits weighted by its observed herald shot fractions."""
    xs = np.asarray(
        np.union1d(depths, np.linspace(depths.min(), depths.max(), 201)),
        dtype=np.float64,
    )
    params = fit.fit.values
    pooled_trials = group_trials.sum(axis=0)
    for arm, label in enumerate(("Reference", "CZ interleaved")):
        measured = pooled_trials[arm] > 0
        decay = params[f"p_{arm}"] ** xs
        pooled = np.zeros_like(xs)
        for herald, trials in enumerate(group_trials[:, arm]):
            fractions = trials[measured] / pooled_trials[arm, measured]
            weights = np.interp(xs, depths[measured], fractions)
            pooled += weights * _group_decay_curve(
                params, decay, herald_index=herald, arm=arm
            )
        ax.plot(xs, pooled, color=f"C{arm}", label=f"{label} fit")


def _draw_pooled_survival(
    ax: Axes,
    depths: NDArray[np.int64],
    sums: NDArray[np.float64],
    fit: RBJointDecayFit | None = None,
) -> None:
    """Draw four-group pooled returns, shot SEM, seed SD and optional joint fit."""
    returns, trials = _return_counts(sums)
    seed_std = seed_standard_deviation(
        _survival(returns.sum(axis=0), trials.sum(axis=0))
    )
    pooled_returns = np.nansum(returns, axis=(0, 1))
    group_trials = np.nansum(trials, axis=1)
    pooled_trials = group_trials.sum(axis=0)
    _draw_count_bars(
        ax,
        depths,
        _survival(pooled_returns, pooled_trials),
        _shot_sem(pooled_returns, pooled_trials),
        seed_std,
        fit_overlay=fit is not None,
    )
    if fit is not None:
        _draw_pooled_fit(ax, depths, group_trials, fit)
    ax.set(
        xlabel="Random Clifford depth (count)",
        ylabel="P(return to herald state | gg/ge/eg/ee)",
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
            curve = _group_decay_curve(
                params, params[f"p_{arm}"] ** xs, herald_index=herald, arm=arm
            )
            ax.plot(xs, curve, color=color, label=f"{label} fit")
    state = _HERALD_LABELS[herald]
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
    fidelity_comparison: str | None,
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
        report.append(
            f"CZ error estimate = {estimate.gate_error:.3g}; 95% seed CI "
            f"[{estimate.error_ci_low:.3g}, {estimate.error_ci_high:.3g}]"
        )
        if fidelity_comparison is not None:
            report.append(fidelity_comparison)
        report.append(
            f"Joint decay p: Reference {estimate.p_reference:.4g} · "
            f"CZ interleaved {estimate.p_interleaved:.4g}"
        )
        if analysis.fit_quality is not None:
            rms = [
                "unavailable"
                if item.normalized_residual_rms is None
                else f"{item.normalized_residual_rms:.3g}"
                for item in analysis.fit_quality
            ]
            grouped_rms = [
                f"{herald}: {rms[2 * h]} / {rms[2 * h + 1]}"
                for h, herald in enumerate(_HERALD_LABELS)
            ]
            report.extend(
                (
                    "Grouped fit normalized residual RMS (Reference / CZ interleaved)",
                    " · ".join(grouped_rms[:2]),
                    " · ".join(grouped_rms[2:]),
                )
            )
    elif analysis.reason is not None:
        report.append(fill(f"No RB estimate: {analysis.reason}", width=82))
    fig.supxlabel("\n".join(report), fontsize=10)


def _plot_grouped_survival(plots: Plots, data: RBHeraldedResult) -> None:
    """Show separate shot-pooled returns for all heralds, including incomplete seeds."""
    fig, _ = plots.subplots(
        2,
        2,
        num="grouped_survival",
        sharex=True,
        sharey=True,
        figsize=(10, 7),
        layout="constrained",
    )
    fig.suptitle("Heralded RB · Qubit1/Qubit2 · Conditional return by herald")
    fig.supxlabel(
        "Random Clifford depth (count) · Capped: shot SEM · Pale: seed SD", fontsize=10
    )
    fig.supylabel("P(return to herald state)", fontsize=10)
    returns, trials = _return_counts(data.sums)
    for ax, label, group_returns, group_trials in zip(
        fig.axes, _HERALD_LABELS, returns, trials, strict=True
    ):
        pooled_returns, pooled_trials = (
            np.nansum(group_returns, axis=0),
            np.nansum(group_trials, axis=0),
        )
        _draw_count_bars(
            ax,
            data.depths,
            _survival(pooled_returns, pooled_trials),
            _shot_sem(pooled_returns, pooled_trials),
            seed_standard_deviation(_survival(group_returns, group_trials)),
        )
        ax.set_title(f"Herald {label}")
    fig.axes[0].legend(loc="lower center", bbox_to_anchor=(0.5, 1.10), ncols=2)


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
    for herald, label in enumerate(_HERALD_LABELS):
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
            report.append(fill(f"No RB estimate: {reason}", width=82))
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


def _complete_seed_profiles(
    survival: NDArray[np.float64],
    totals: NDArray[np.float64],
    paired: NDArray[np.bool_],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return equal-seed means, fit errors and seed SD for each herald.

    Even one complete seed can supply a plotted mean, but not a fit error or
    seed SD. An empty cohort leaves the diagnostic trace missing.
    """
    shape = (len(_HERALDS), *survival.shape[2:])
    means = np.full(shape, np.nan)
    errors, seed_std = np.full_like(means, np.nan), np.full_like(means, np.nan)
    for h, group_paired in enumerate(paired):
        count = group_paired.sum()
        if count >= 2:
            statistics = seed_statistics(survival[h], totals[h])
            means[h], errors[h], seed_std[h] = (
                statistics.means,
                statistics.errors,
                statistics.seed_std,
            )
        elif count == 1:
            means[h] = survival[h, group_paired].mean(axis=0)
    return means, errors, seed_std


def _fit_with_seed_interval(
    depths: NDArray[np.int64],
    means: NDArray[np.float64],
    errors: NDArray[np.float64],
    survival: NDArray[np.float64],
    totals: NDArray[np.float64],
    options: paired_rb_analysis.RBOptions,
) -> tuple[RBHeraldedAnalysis, RBJointDecayFit | None]:
    """Publish the shared fit only when both main fit and seed interval succeed."""
    try:
        joint_fit = fit_joint_decay(depths, means, errors)
    except (ValueError, RuntimeError) as exc:
        return RBHeraldedAnalysis(None, None, f"Joint decay fit failed: {exc}"), None
    try:
        low, high, successes = bootstrap_joint_error(
            depths,
            survival,
            totals,
            samples=options.bootstrap_samples,
            seed=options.bootstrap_seed,
        )
    except (ValueError, RuntimeError) as exc:
        return RBHeraldedAnalysis(None, None, str(exc)), None
    complete = np.isfinite(survival).all(axis=(2, 3))
    estimate = estimate_joint_d4(
        joint_fit.fit,
        ci_low=low,
        ci_high=high,
        bootstrap_successes=successes,
        paired_seeds=int(complete.any(axis=0).sum()),
    )
    decays = [joint_fit.decay_time_and_variance(arm) for arm in range(2)]
    bounds = decay_time_bounds(depths.astype(np.float64))
    quality = tuple(
        compute_fit_quality(
            means[herald, arm],
            joint_fit.curves[herald, arm],
            {"decay_time": tau},
            np.array([[variance]]),
            decay_bounds=bounds,
        )
        for herald in range(len(_HERALDS))
        for arm, (tau, variance) in enumerate(decays)
    )
    return RBHeraldedAnalysis(estimate, quality, None), joint_fit


class RBHeraldedExp(Experiment[RBHeraldedResult, RBHeraldedCfg]):
    """Paired RB with separate gg/ge/eg/ee complete seeds and shared arm decays."""

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

    Options = paired_rb_analysis.RBOptions

    def run(self, config: RBHeraldedCfg, *, context: RunContext) -> RBHeraldedResult:
        """Acquire optional reset → herald → concurrent per-qubit sequence → terminal.

        Compile one board program per seed and arm before acquiring. Each
        round acquires reps shots at every depth in the program. Save all 81
        counts per cell summed across finished rounds; unacquired cells stay NaN.
        """
        # 1. Compare pooled four-group return for both arms across depth.
        fig, ax = context.plots.subplots(
            num="measurement", figsize=(7, 5), layout="constrained"
        )
        depths = np.array(config.depth, dtype=np.int64)
        fig.suptitle("Heralded RB · Qubit1/Qubit2 · Conditional return")
        fig.supxlabel("Capped: shot SEM · Pale: seed SD", fontsize=10)
        fig.show()

        @context.plots.on_plot_thread
        def update_view(sums: NDArray[np.float64]):
            for bars in tuple(ax.containers):
                bars.remove()
            ax.relim()
            _draw_pooled_survival(ax, depths, sums)
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
        options: paired_rb_analysis.RBOptions,
        *,
        plots: Plots,
    ) -> RBHeraldedAnalysis:
        """Fit gg/ge/eg/ee return survival with one shared decay per arm.

        Select complete reference/interleaved seed pairs independently per herald.
        Weight seeds equally and derive binomial errors from actual shot counts.
        Bootstrap whole seed identities from the union of those cohorts. A usable
        fit and seed CI are both needed for an estimate and fit overlays; otherwise
        show observed data with the failure reason. Missing cfg also returns a
        reason. Malformed counts or mismatched axes raise ValueError.
        """
        # 1. Normalize each herald's returns by its own acquired shots.
        data, cfg = source.result, source.cfg
        shape = (len(data.seeds), 2, len(data.depths))
        if data.sums.shape != (*shape, 3, 3, 3, 3):
            raise ValueError("Expected heralded RB sums (seed,2,depth,3,3,3,3)")
        paired_rb_analysis.validate_counts(data.sums, (-4, -3, -2, -1))
        rendered_fit = None
        fidelity_comparison = None
        if cfg is None:
            analysis = RBHeraldedAnalysis(
                None,
                None,
                "RB analysis requires saved cfg; current Form is not a substitute",
            )
            means = np.full((len(_HERALDS), 2, len(data.depths)), np.nan)
            seed_std = np.full_like(means, np.nan)
        else:
            paired_rb_analysis.validate_axes(
                (data.seeds, data.depths), (cfg.seed, cfg.depth)
            )
            returns, totals = _return_counts(data.sums)
            survival = _survival(returns, totals)
            paired = np.asarray(np.isfinite(survival).all(axis=(2, 3)))
            means, errors, seed_std = _complete_seed_profiles(survival, totals, paired)
            # 2. Require all four cohorts, then fit and bootstrap the same model.
            insufficient = [
                herald
                for herald, count in zip(
                    _HERALD_LABELS, paired.sum(axis=1), strict=True
                )
                if count < 2
            ]
            if len(np.unique(data.depths)) < 4:
                analysis = RBHeraldedAnalysis(
                    None, None, "Need at least four distinct depths"
                )
            elif insufficient:
                analysis = RBHeraldedAnalysis(
                    None,
                    None,
                    f"Need at least two complete paired seeds for herald {', '.join(insufficient)}",
                )
            else:
                analysis, rendered_fit = _fit_with_seed_interval(
                    data.depths,
                    means,
                    errors,
                    survival,
                    totals,
                    options,
                )
                # 3. Compare the asymptote models without changing the free-B result or CI.
                if analysis.estimate is not None:
                    fidelity_comparison = paired_rb_analysis.cz_fidelity_comparison(
                        analysis.estimate, data.depths, means, errors
                    )
        # 4. Show observed returns and group diagnostics even when the joint result is absent.
        _plot_pooled_survival(plots, data, rendered_fit, analysis, fidelity_comparison)
        _plot_grouped_survival(plots, data)
        _plot_decay_fits(plots, data.depths, means, seed_std, rendered_fit, analysis)
        _plot_herald_fractions(plots, data.depths, np.nansum(data.sums, axis=0))
        return analysis
