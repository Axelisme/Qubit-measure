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
    compute_fit_quality,
    decay_time_bounds,
    fit_joint_decay,
    seed_standard_deviation,
    seed_statistics,
)
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

from zcu_lab.v2.support import live_errorbar

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


def _draw_return_survival(
    ax: Axes,
    depths: NDArray[np.int64],
    means: NDArray[np.float64],
    seed_std: NDArray[np.float64],
    curves: NDArray[np.float64],
    label: str,
) -> None:
    """Draw selected arm means, error bars and fitted curves on ax.

    depths is the integer Clifford-depth axis. means, seed_std and curves are
    (2,N) reference/interleaved means, seed spread and fitted return probabilities
    on that axis. label names the selected gg/ee herald and return class.
    Add artists, labels and a legend to caller-owned ax. Return None without
    modifying the input arrays.
    """
    for arm, arm_label in enumerate(("Reference", "CZ interleaved")):
        (line,) = ax.plot(depths, means[arm], ".", label=arm_label)
        ax.errorbar(
            depths, means[arm], yerr=seed_std[arm], fmt="none", ecolor=line.get_color()
        )
        ax.plot(depths, curves[arm], color=line.get_color())
    ax.set(xlabel="Random Clifford depth", ylabel=f"Return to {label} | herald {label}")
    ax.legend()


def _plot_herald_fractions(
    plots: Plots,
    depths: NDArray[np.int64],
    pooled: NDArray[np.float64],
) -> None:
    """Draw nine depth-resolved herald probabilities per arm from pooled counts."""
    aggregate = herald_probabilities(pooled)
    classes = ("g", "e", "other")
    for arm, label in enumerate(("Reference", "CZ interleaved")):
        fig, ax = plots.subplots(num=f"herald_{arm}", figsize=(10, 6))
        for qubit1 in range(3):
            for qubit2 in range(3):
                ax.plot(
                    depths,
                    aggregate[arm, :, qubit1, qubit2],
                    ".-",
                    label=f"{classes[qubit1]},{classes[qubit2]}",
                )
        ax.set(xlabel="Random Clifford depth", ylabel="Herald probability")
        ax.legend()
        ax.set_title(f"{label}: (g,g) is pass fraction, not purity")
        fig.tight_layout()


def _plot_pooled_survival(
    plots: Plots,
    depths: NDArray[np.int64],
    pooled: NDArray[np.float64],
) -> None:
    """Draw per-arm return probabilities from seed-pooled herald counts."""
    _, ax = plots.subplots(num="pooled_survival", layout="constrained")
    for cell, (herald, counts) in enumerate(
        (
            ("g,g", terminal_counts(pooled, ("g", "g"))),
            ("e,e", terminal_counts(pooled, ("e", "e"))),
        )
    ):
        probabilities = joint_probabilities(counts)
        for arm, label in enumerate(("Reference", "CZ interleaved")):
            ax.plot(
                depths,
                probabilities[arm, :, cell, cell],
                ".",
                label=f"{label} · {herald}",
            )
    ax.set(
        xlabel="Random Clifford depth",
        ylabel="Return probability | matching herald state",
    )
    ax.legend()


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
        # 1. Compare gg/ee return-to-herald survival for reference and CZ-interleaved arms across depth.
        fig, ax = context.plots.subplots(num="measurement")
        ax.set(
            xlabel="Random Clifford depth",
            ylabel="Return probability | matching herald state",
        )
        viewer = live_errorbar(ax, num_lines=4)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        fig.show()

        @context.plots.on_plot_thread
        def update_view(sums: NDArray[np.float64]):
            curves = []
            deviations = []
            for cell, counts in enumerate(
                (terminal_counts(sums, ("g", "g")), terminal_counts(sums, ("e", "e")))
            ):
                totals = np.nansum(counts, axis=0)  # Pool seeds for the markers.
                returns = joint_probabilities(counts)[..., cell, cell]
                curves.extend(joint_probabilities(totals)[..., cell, cell])
                deviations.extend(seed_standard_deviation(returns))
            return viewer.set(
                np.array(config.depth), np.array(curves), np.array(deviations)
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
        curves = np.full_like(means, np.nan)
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
                curves = joint_fit.curves
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
                        curves[h, arm],
                        {"decay_time": tau},
                        np.array([[variance]]),
                        decay_bounds=bounds,
                    )
                    for h in range(2)
                    for arm, (tau, variance) in enumerate(decays)
                )
        # 3. Compare gg/ee survival fits and herald fractions separately from acquisition completeness.
        pooled = np.nansum(data.sums, axis=0)
        _plot_pooled_survival(plots, data.depths, pooled)

        for herald, label in enumerate(("gg", "ee")):
            fig, ax = plots.subplots(num=f"fit_{label}", figsize=(9, 6))
            _draw_return_survival(
                ax,
                data.depths,
                means[herald],
                seed_std[herald],
                curves[herald],
                label,
            )
            ax.set_title(
                f"No RB estimate: {reason}"
                if reason
                else "d=4 joint decay; equal-seed means, statistical seed CI only"
            )
            fig.tight_layout()
        _plot_herald_fractions(plots, data.depths, pooled)
        # 4. Keep the shared-decay d=4 estimate and seed-bootstrap interval, or its numerical failure reason.
        return RBHeraldedAnalysis(estimate, quality, reason)
