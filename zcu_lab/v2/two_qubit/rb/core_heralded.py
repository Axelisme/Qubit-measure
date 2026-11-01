"""Paired RB with gg/ee herald groups and shared return-survival decays."""

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from matplotlib.axes import Axes
from numpy.typing import NDArray
from zcu_tools.analysis.fitting import FitQuality, compute_fit_quality
from zcu_tools.analysis.fitting.shared import SharedFitResult
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
from zcu_tools.program.v2 import Classify, Pulse, Reset, TimelineBuilder, shot_sums
from zcu_tools.program.v2.modules import MuxReadout

from zcu_lab.v2.support import live_line

from .._sums import (
    herald_probabilities,
    joint_probabilities,
    terminal_counts,
    validate_classifiers,
)
from .analysis import (
    RBEstimate,
    RBOptions,
    aggregate_seeds,
    bootstrap_joint_error,
    count_completion,
    estimate_joint_d4,
    fit_joint_decay,
    paired_means,
    seed_statistics,
    validate_axes,
)
from .core import RBCfg
from .sequence import TwoQubitGate, make_paired_sequences


class RBHeraldedCfg(RBCfg):
    """Optional reset → herald → paired sequence → terminal, per host round.

    Fields and units match RBCfg. Both reads share the saved full mux cfg and
    circle radii. Its post_delay supplies the wait after herald. reset=None
    emits no reset; rounds stays a host axis rather than an acquisition average.
    """


@dataclass(frozen=True)
class RBHeraldedResult:
    """Raw float64 sums (depth,seed,arm,round,3,3,3,3), NaN when unacquired.

    depths, seeds, arms and rounds have the same meanings as RBResult.
    Trailing axes are herald F1/F2 then terminal F1/F2, each ordered g/e/other.
    All herald outcomes remain saved; counts are not normalized or filtered.
    """

    depths: NDArray[np.int64]
    seeds: NDArray[np.int64]
    arms: NDArray[np.int64]
    rounds: NDArray[np.int64]
    sums: NDArray[np.float64]


@dataclass(frozen=True)
class RBHeraldedAnalysis:
    """Joint gg/ee return fits and separate acquired/accepted-round diagnostics.

    survival is (herald,depth,seed,arm,round), herald order gg/ee.
    acquired_rounds is (depth,seed,arm,round) for all 81 finite counts, even
    when a herald has no accepted shots. paired_rounds is (herald,seed,round).
    mean_survival/fit_survival are (herald,arm,depth). The shared fit has one
    decay per arm and independent A/B per herald. fit_quality order is gg ref,
    gg interleaved, ee ref, ee interleaved. estimate counts are identity/round
    unions; paired_rounds preserves per-herald counts. reason explains no estimate.
    herald_matrix is (depth,seed,arm,round,3,3); pass_fraction is its gg cell,
    not state purity. Empty or unmeasured rounds give NaN probabilities.
    """

    survival: NDArray[np.float64]
    acquired_rounds: NDArray[np.bool_]
    paired_rounds: NDArray[np.bool_]
    mean_survival: NDArray[np.float64]
    fit_survival: NDArray[np.float64]
    estimate: RBEstimate | None
    fit: SharedFitResult | None
    fit_quality: tuple[FitQuality, ...] | None
    reason: str | None
    herald_matrix: NDArray[np.float64]
    pass_fraction: NDArray[np.float64]


def restore_result(data: GroupedLoadData[RBHeraldedCfg]) -> RBHeraldedResult:
    """Restore all 81 class counts; old per-shot IQ files fail without sums."""
    sums = data.variable("sums")
    return RBHeraldedResult(
        depths=sums.axes[7].astype(np.int64),
        seeds=sums.axes[6].astype(np.int64),
        arms=sums.axes[5].astype(np.int64),
        rounds=sums.axes[4].astype(np.int64),
        sums=sums.z.astype(np.float64),
    )


class RBHeraldedExp(Experiment[RBHeraldedResult, RBHeraldedCfg]):
    """Paired RB with separate gg/ee complete rounds and shared arm decays."""

    Options = RBOptions
    AXES_SPEC: ClassVar[GroupedAxesSpec[RBHeraldedResult, RBHeraldedCfg]] = (
        GroupedAxesSpec(
            variables=(
                VariableSpec(
                    variable="sums",
                    axes=(
                        VariableAxisSpec.generated_arange("terminal_f2", ""),
                        VariableAxisSpec.generated_arange("terminal_f1", ""),
                        VariableAxisSpec.generated_arange("herald_f2", ""),
                        VariableAxisSpec.generated_arange("herald_f1", ""),
                        VariableAxisSpec(
                            "Round", "", field_name="rounds", dtype=np.int64
                        ),
                        VariableAxisSpec("Arm", "", field_name="arms", dtype=np.int64),
                        VariableAxisSpec(
                            "Seed", "", field_name="seeds", dtype=np.int64
                        ),
                        VariableAxisSpec(
                            "Clifford depth", "", field_name="depths", dtype=np.int64
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
            tag="two_qubit/rb_heralded",
            result_builder=restore_result,
        )
    )

    def run(self, config: RBHeraldedCfg, *, context: RunContext) -> RBHeraldedResult:
        """Acquire optional reset → herald → paired sequence → terminal.

        Compile once per depth/seed/arm. Each host round acquires reps shots with
        rounds=1 and saves all 81 counts. Unfinished slots stay NaN; cfg remains
        unchanged. Acquisition exceptions propagate unchanged. The stop signal
        carries only stop requests and reason text.
        """
        validate_classifiers(config.readout, config.roles)
        viewer = live_line(
            context.plots,
            "measurement",
            "Random Clifford depth",
            "Return to herald state",
            num_lines=4,
        )

        def update_view(sums: NDArray[np.float64]) -> None:
            curves = []
            for cell, counts in enumerate(
                (terminal_counts(sums, ("g", "g")), terminal_counts(sums, ("e", "e")))
            ):
                totals = np.nansum(counts, axis=(1, 3))
                curves.extend(joint_probabilities(totals)[..., cell, cell].T)
            viewer.update(np.array(config.depths), np.array(curves))

        buffer = SignalBuffer(
            (len(config.depths), len(config.seeds), 2, config.rounds, 3, 3, 3, 3),
            dtype=np.float64,
            on_update=update_view,
        )
        with Schedule(buffer, scope=context.scope) as sched:
            # Host depth/seed/arm change the gate sequence. Host rounds preserve paired slots.
            for depth, depth_step in sched.scan("depth", config.depths):
                for seed, seed_step in depth_step.scan("seed", config.seeds):
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

                        # The recovery makes the net logical operation identity.
                        # Positive logical Z advances a frame; X uses base minus frame.
                        frames = [0.0, 0.0]
                        pulses = (config.x90_f1, config.x90_f2)
                        for index, gate in enumerate(sequence.gates):
                            if gate in (TwoQubitGate.Z90_Q0, TwoQubitGate.Z90_Q1):
                                role = 0 if gate == TwoQubitGate.Z90_Q0 else 1
                                frames[role] = (frames[role] + 90.0) % 360.0
                            elif gate in (TwoQubitGate.X90_Q0, TwoQubitGate.X90_Q1):
                                role = 0 if gate == TwoQubitGate.X90_Q0 else 1
                                pulse = pulses[role]
                                builder.emit(
                                    Pulse(
                                        f"gate_{index}_x90_f{role + 1}",
                                        pulse,
                                        phase=(float(pulse.phase) - frames[role])
                                        % 360.0,
                                    )
                                )
                            elif gate == TwoQubitGate.CZ:
                                builder.fork(
                                    "default",
                                    into=(
                                        f"gate_{index}_cz_f1_time",
                                        f"gate_{index}_cz_f2_time",
                                    ),
                                )
                                builder.emit(
                                    Pulse(f"gate_{index}_cz_f1", config.drive.f1),
                                    timeline=f"gate_{index}_cz_f1_time",
                                )
                                builder.emit(
                                    Pulse(f"gate_{index}_cz_f2", config.drive.f2),
                                    timeline=f"gate_{index}_cz_f2_time",
                                )
                                builder.join(
                                    *(
                                        f"gate_{index}_cz_f1_time",
                                        f"gate_{index}_cz_f2_time",
                                    ),
                                    into="default",
                                )
                                frames[0] = (
                                    frames[0] + config.phase_pair.correction_f1_deg
                                ) % 360.0
                                frames[1] = (
                                    frames[1] + config.phase_pair.correction_f2_deg
                                ) % 360.0
                            else:
                                raise ValueError(f"Unknown two-qubit gate: {gate!r}")
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
                                processor=processor,
                                rounds=1,
                                options=RunOptions(),
                            )
        buffer.trigger_update(flush=True)
        return RBHeraldedResult(
            np.array(config.depths, dtype=np.int64),
            np.array(config.seeds, dtype=np.int64),
            np.arange(2, dtype=np.int64),
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
        seed_means = np.full((2, len(data.seeds), 2, len(data.depths)), np.nan)
        seed_variances = np.full_like(seed_means, np.nan)
        estimate, fit, quality = None, None, None
        reason = None
        if cfg is None:
            reason = "RB analysis requires saved cfg; current Form is not a substitute"
        else:
            validate_axes(
                (data.depths, data.seeds, data.arms, data.rounds),
                (cfg.depths, cfg.seeds, [0, 1], np.arange(cfg.rounds)),
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
                seed_means[herald], seed_variances[herald] = seed_statistics(
                    survival[herald], total
                )
            try:
                if len(np.unique(data.depths)) < 4:
                    raise ValueError(
                        "Need at least four distinct depths and 100 bootstrap samples"
                    )
                for herald in range(2):
                    means[herald], errors[herald] = aggregate_seeds(
                        seed_means[herald], seed_variances[herald]
                    )
                fit, fitted = fit_joint_decay(data.depths, means, errors)
                low, high, successes = bootstrap_joint_error(
                    data.depths,
                    seed_means,
                    seed_variances,
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
                    (low, high, successes),
                    (int(union.any(axis=1).sum()), int(union.sum())),
                )
                quality = tuple(
                    compute_fit_quality(means[h, arm], curves[h, arm], {}, None)
                    for h in range(2)
                    for arm in range(2)
                )
        matrix = herald_probabilities(data.sums)
        aggregate = herald_probabilities(np.nansum(data.sums, axis=(1, 3)))
        for herald, label in enumerate(("gg", "ee")):
            fig, ax = plots.subplots(f"fit_{label}", figsize=(9, 6))
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
            fig, ax = plots.subplots(f"herald_{arm}", figsize=(10, 6))
            _draw_herald_fractions(ax, data.depths, aggregate[:, arm])
            ax.set_title(f"{label}: (g,g) is pass fraction, not purity")
            fig.tight_layout()
        return RBHeraldedAnalysis(
            survival,
            acquired,
            paired,
            means,
            curves,
            estimate,
            fit,
            quality,
            reason,
            matrix,
            matrix[..., 0, 0],
        )


def _draw_return_survival(
    ax: Axes,
    depths: NDArray[np.int64],
    means: NDArray[np.float64],
    errors: NDArray[np.float64],
    curves: NDArray[np.float64],
    label: str,
) -> None:
    """Render already-selected arm means, error bars and curves on this axis."""
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
    """Render the caller's nine aggregated herald fractions, without filtering."""
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
