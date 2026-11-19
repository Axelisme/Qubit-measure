"""Paired RB with gg/ee herald groups and shared return-survival decays."""

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from numpy.typing import NDArray
from zcu_tools.analysis.fitting import (
    FitQuality,
    compute_fit_quality,
    fit_joint_decay,
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
from zcu_tools.experiment.v2.runtime import RunOptions, Schedule, SignalBuffer
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import Classify, Reset, TimelineBuilder, shot_sums
from zcu_tools.program.v2.modules import MuxReadout

from zcu_lab.v2.support import live_line

from .._sums import (
    herald_probabilities,
    joint_probabilities,
    terminal_counts,
)
from ._config import RBInputs
from .core import (
    RBEstimate,
    RBOptions,
    count_completion,
    estimate_joint_d4,
    paired_means,
    validate_axes,
)
from .scheduling import emit_sequence
from .sequence import make_paired_sequences


@dataclass(frozen=True)
class RBHeraldedResult:
    """Raw float64 sums (depth,seed,arm,round,3,3,3,3), NaN when unacquired.

    depths, seeds, arms and rounds have the same meanings as RBResult.
    Trailing axes are herald F1/F2 then terminal F1/F2, each ordered g/e/other.
    All herald outcomes remain saved; counts are not normalized or filtered.
    """

    depths: NDArray[np.int64]
    seeds: NDArray[np.int64]

    rounds: NDArray[np.int64]
    sums: NDArray[np.float64]


@dataclass(frozen=True)
class RBHeraldedAnalysis:
    """Joint gg/ee d=4 estimate and fixed residual diagnostics.

    estimate is None with reason when unavailable. fit_quality is ordered gg
    reference, gg interleaved, ee reference, ee interleaved, or None without a
    fit. Per-depth survival, counts and herald fractions stay in analyze.
    """

    estimate: RBEstimate | None
    fit_quality: tuple[FitQuality, ...] | None
    reason: str | None


class RBHeraldedCfg(RBInputs[Reset.Cfg]):
    """Optional reset → herald → paired sequence → terminal, per host round.

    Fields and units match RBCfg. Both reads share the saved full mux cfg and
    circle radii. Its post_delay supplies the wait after herald. reset=None
    emits no reset; rounds stays a host axis rather than an acquisition average.
    """


def restore_result(data: GroupedLoadData[RBHeraldedCfg]) -> RBHeraldedResult:
    """Restore all 81 class counts; old per-shot IQ files fail without sums.

    data supplies the grouped sums variable and its persisted scan axes. Return
    this experiment's Result with float64 class counts and its original axis
    ordering. Missing variables and invalid data propagate load errors. Do not
    mutate the loaded data, acquire or draw; no raw-IQ reconstruction is attempted.
    """
    sums = data.variable("sums")
    return RBHeraldedResult(
        depths=sums.axes[7].astype(np.int64),
        seeds=sums.axes[6].astype(np.int64),
        rounds=sums.axes[4].astype(np.int64),
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

    Inputs are (herald,depth,seed,arm,round), survival masked to paired
    rounds and totals in shot counts. NaN denotes unavailable survival. Draw from
    their valid identity union; each herald retains only its valid draws,
    including multiplicity. Both arms/depths travel together. Fewer than two
    retained draws in either group fails that resample, never fills missing data.
    depths is the shared integer Clifford-depth axis. samples is the number of
    resamples and seed initializes a local RNG. Return (low, high, valid fits)
    without changing inputs, global random state or plots.
    """
    available = np.isfinite(survival).all(axis=(1, 3)).any(axis=-1)
    identities = np.flatnonzero(available.any(axis=0))
    rng = np.random.default_rng(seed)
    errors = []
    for _ in range(samples):
        draw = rng.choice(identities, size=len(identities), replace=True)
        try:
            statistics = [
                seed_statistics(
                    np.take(survival[h], draw, axis=1),
                    np.take(totals[h], draw, axis=1),
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
    errors: NDArray[np.float64],
    curves: NDArray[np.float64],
    label: str,
) -> None:
    """Draw selected arm means, error bars and fitted curves on ax.

    depths is the integer Clifford-depth axis; means, errors and curves are
    (2,N) reference/interleaved return probabilities on that axis. label names
    the selected gg/ee herald and return class. Add artists, labels and a legend
    to caller-owned ax, return None and leave input arrays unchanged.
    """
    for arm, arm_label in enumerate(("Reference", "CZ interleaved")):
        (line,) = ax.plot(depths, means[arm], "o", label=arm_label)
        ax.errorbar(
            depths, means[arm], yerr=errors[arm], fmt="none", ecolor=line.get_color()
        )
        ax.plot(depths, curves[arm], color=line.get_color())
    ax.set(xlabel="Random Clifford depth", ylabel=f"Return to {label} | herald {label}")
    ax.legend()


def _draw_herald_fractions(
    ax: Axes,
    depths: NDArray[np.int64],
    probabilities: NDArray[np.float64],
) -> None:
    """Draw the nine aggregated herald fractions and their legend on ax.

    depths is the integer Clifford-depth axis; probabilities is (N,3,3), with
    both herald axes ordered g/e/other. Add artists and axis labels without
    filtering or changing the arrays. Return None; ax remains caller-owned.
    """
    classes = ("g", "e", "other")
    for f1 in range(3):
        for f2 in range(3):
            ax.plot(
                depths,
                probabilities[:, f1, f2],
                "o-",
                label=f"{classes[f1]},{classes[f2]}",
            )
    ax.set(xlabel="Random Clifford depth", ylabel="Herald probability")
    ax.legend()


class RBHeraldedExp(Experiment[RBHeraldedResult, RBHeraldedCfg]):
    """Paired RB with separate gg/ee complete rounds and shared arm decays."""

    AXES_SPEC: ClassVar[GroupedAxesSpec[RBHeraldedResult, RBHeraldedCfg]] = (
        GroupedAxesSpec(
            variables=(
                VariableSpec(
                    variable="sums",
                    axes=(
                        VariableAxisSpec.generated_arange(label="terminal_f2", unit=""),
                        VariableAxisSpec.generated_arange(label="terminal_f1", unit=""),
                        VariableAxisSpec.generated_arange(label="herald_f2", unit=""),
                        VariableAxisSpec.generated_arange(label="herald_f1", unit=""),
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
                        label="Herald and terminal class counts",
                        unit="",
                        dtype=np.float64,
                    ),
                ),
            ),
            result_type=RBHeraldedResult,
            cfg_type=RBHeraldedCfg,
            cfg_schema_version="2.0",
            tag="two_qubit/rb_heralded",
            result_builder=restore_result,
        )
    )

    Options = RBOptions

    def run(self, config: RBHeraldedCfg, *, context: RunContext) -> RBHeraldedResult:
        """Acquire optional reset → herald → concurrent per-qubit sequence → terminal.

        Compile once per depth/seed/arm. Each host round acquires reps shots with
        rounds=1 and saves all 81 counts. Unfinished slots stay NaN; cfg remains
        unchanged. Acquisition exceptions propagate unchanged. The stop signal
        carries only stop requests and reason text.

        """
        # 1. Compare gg/ee return-to-herald survival for reference and CZ-interleaved arms across depth.
        fig, ax = context.plots.subplots(num="measurement")
        ax.set(xlabel="Random Clifford depth", ylabel="Return to herald state")
        viewer = live_line(ax, num_lines=4)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.grid(False)
        fig.show()

        @context.plots.on_plot_thread
        def update_view(sums: NDArray[np.float64]) -> tuple[Line2D, ...]:
            curves = []
            for cell, counts in enumerate(
                (terminal_counts(sums, ("g", "g")), terminal_counts(sums, ("e", "e")))
            ):
                totals = np.nansum(counts, axis=(1, 3))
                curves.extend(joint_probabilities(totals)[..., cell, cell].T)
            return viewer.set(np.array(config.depth), np.array(curves))

        # 2. Compile herald/paired-sequence/terminal programs per depth/seed/arm and reuse them across host rounds.
        buffer = SignalBuffer(
            (len(config.depth), len(config.seed), 2, config.rounds, 3, 3, 3, 3),
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
                        builder.emit(readout, tag="herald")

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
                                Classify("herald", config.roles.f1_tone),
                                Classify("herald", config.roles.f2_tone),
                                Classify("terminal", config.roles.f1_tone),
                                Classify("terminal", config.roles.f2_tone),
                            ],
                        )
                        for _, round_step in arm_step.repeat("round", config.rounds):
                            round_step.run_acquire_summed(
                                prog,
                                context.soc,
                                reducer=processor,
                                rounds=1,
                                options=RunOptions(),
                            )
        buffer.trigger_update(flush=True)
        # 3. Preserve all 81 class counts and paired round identities; unfinished slots stay NaN.
        return RBHeraldedResult(
            np.array(config.depth, dtype=np.int64),
            np.array(config.seed, dtype=np.int64),
            np.arange(config.rounds, dtype=np.int64),
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

        Select complete (seed,round) independently per herald across every depth
        and both arms. Average rounds, then seeds equally. Zero accepted shots
        mean unavailable survival, not an unacquired round. Whole seed identity
        bootstrap draws preserve correlations across arms, depths and heralds.

        """
        # 1. Select complete paired seed/round curves independently for gg and ee, not by acceptance size.
        data, cfg = source.result, source.cfg
        shape = (len(data.depths), len(data.seeds), 2, len(data.rounds))
        if data.sums.shape != (*shape, 3, 3, 3, 3):
            raise ValueError("Expected heralded RB sums (depth,seed,2,round,3,3,3,3)")
        acquired = count_completion(data.sums, (-4, -3, -2, -1))
        survival = np.full((2, *shape), np.nan)
        paired = np.zeros((2, len(data.seeds), len(data.rounds)), dtype=np.bool_)
        means = np.full((2, 2, len(data.depths)), np.nan)
        errors = np.full_like(means, np.nan)
        curves = np.full_like(means, np.nan)
        totals = np.full_like(survival, np.nan)
        estimate, fit, quality = None, None, None
        reason = None
        if cfg is None:
            reason = "RB analysis requires saved cfg; current Form is not a substitute"
        else:
            validate_axes(
                (data.depths, data.seeds, np.arange(2, dtype=np.int64), data.rounds),
                (cfg.depth, cfg.seed, [0, 1], np.arange(cfg.rounds)),
            )
            for herald, counts in enumerate(
                (
                    terminal_counts(data.sums, ("g", "g")),
                    terminal_counts(data.sums, ("e", "e")),
                )
            ):
                total = counts.sum(axis=(-2, -1))
                returns = joint_probabilities(counts)[..., herald, herald]
                paired[herald] = (acquired & np.isfinite(returns)).all(axis=(0, 2))
                survival[herald] = np.where(
                    paired[herald][None, :, None, :], returns, np.nan
                )
                means[herald] = paired_means(survival[herald])
                totals[herald] = total
            # 2. Use equal-seed means and a shared decay per arm; bootstrap whole paired seed identities.
            try:
                if len(np.unique(data.depths)) < 4:
                    raise ValueError(
                        "Need at least four distinct depths and 100 bootstrap samples"
                    )
                for herald in range(2):
                    statistics = seed_statistics(survival[herald], totals[herald])
                    means[herald], errors[herald] = statistics.means, statistics.errors
                joint_fit = fit_joint_decay(data.depths, means, errors)
                fit, fitted = joint_fit.fit, joint_fit.curves
                low, high, successes = bootstrap_joint_error(
                    data.depths,
                    survival,
                    totals,
                    samples=options.bootstrap_samples,
                    seed=options.bootstrap_seed,
                )
            except (ValueError, RuntimeError) as exc:
                fit = None
                reason = str(exc)
            else:
                curves = fitted
                # Union counts describe available paired identities, not an intersection.
                union = np.asarray(paired.any(axis=0), dtype=np.bool_)
                estimate = estimate_joint_d4(
                    fit,
                    ci_low=low,
                    ci_high=high,
                    bootstrap_successes=successes,
                    paired_seeds=int(union.any(axis=1).sum()),
                    paired_rounds=int(union.sum()),
                )
                quality = tuple(
                    compute_fit_quality(means[h, arm], curves[h, arm], {}, None)
                    for h in range(2)
                    for arm in range(2)
                )
        # 3. Compare gg/ee survival fits and herald fractions separately from acquisition completeness.
        aggregate = herald_probabilities(np.nansum(data.sums, axis=(1, 3)))
        for herald, label in enumerate(("gg", "ee")):
            fig, ax = plots.subplots(num=f"fit_{label}", figsize=(9, 6))
            _draw_return_survival(
                ax,
                data.depths,
                means[herald],
                errors[herald],
                curves[herald],
                label,
            )
            ax.set_title(
                f"No RB estimate: {reason}"
                if reason
                else "d=4 joint decay; equal-seed means, statistical seed CI only"
            )
            fig.tight_layout()
        for arm, label in enumerate(("Reference", "CZ interleaved")):
            fig, ax = plots.subplots(num=f"herald_{arm}", figsize=(10, 6))
            _draw_herald_fractions(ax, data.depths, aggregate[:, arm])
            ax.set_title(f"{label}: (g,g) is pass fraction, not purity")
            fig.tight_layout()
        # 4. Keep the shared-decay d=4 estimate and seed-bootstrap interval, or its numerical failure reason.
        return RBHeraldedAnalysis(estimate, quality, reason)
