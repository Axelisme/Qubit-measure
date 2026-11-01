"""Paired two-qubit RB with acquisition-side classification and host rounds."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, ClassVar, Self

import numpy as np
from numpy.typing import NDArray
from pydantic import Field, model_validator
from zcu_tools.analysis.failure import AnalysisFailed
from zcu_tools.analysis.fitting import compute_fit_quality
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

from ...twotone.irb.analysis import analyze_irb
from .._drive import DualLineCfg
from .._readout import JointRoles
from .._sums import (
    joint_probabilities,
    validate_classifiers,
)
from .analysis import (
    RBAnalysis,
    RBOptions,
    count_completion,
    estimate_d4,
    paired_means,
    validate_axes,
)
from .program import CZPhasePair
from .sequence import TwoQubitGate, make_paired_sequences


class RBCfg(ConfigBase):
    """Optional reset → paired random Clifford + own recovery → terminal read.

    roles maps Q0 to F1 and Q1 to F2 at their original full mux tone positions.
    phase_pair is required, ordered and component-named. Delays are us, pulse
    frequency MHz and phase degrees. No DC device is driven. reset=None emits
    no reset. F1/F2 need circle classifiers with radius from GE Post before Run.
    depths counts random Cliffords, not primitive gates; unique increasing
    nonnegative integers. seeds are distinct integer random.Random identities
    shared across depths/arms/rounds. The 31-bit bound preserves exact axes.
    reps is shots per host round; rounds remains a separate saved axis.
    """

    reps: Annotated[
        int, FieldMeta(label="Reps", description="Number of repetitions per round.")
    ] = Field(default=1, gt=0)
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
    ] = 1.0
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
            description="填入 F1 calibrated x90 pulse slot。先做 amplitude／length Rabi，再確認 frequency MHz、gain a.u. within [-1, 1] 與 duration us。",
        ),
    ]
    x90_f2: Annotated[
        Pulse.Cfg,
        FieldMeta(
            label="F2 Xπ/2",
            description="填入 F2 calibrated x90 pulse slot。先做 amplitude／length Rabi，再確認 frequency MHz、gain a.u. within [-1, 1] 與 duration us。",
        ),
    ]
    readout: Annotated[
        MuxReadout.Cfg,
        FieldMeta(
            label="Mux readout",
            description="填入 full-group mux readout slot。先用 lookback 取得 timing，再做 mux onetone，確認 frequency MHz、gain a.u. within [-1, 1] 與 timing us。",
        ),
    ]
    roles: JointRoles
    drive: Annotated[
        DualLineCfg,
        FieldMeta(
            label="Dual-line drive",
            description="填入 F1/F2 charge-line pulses。先做 conditional spectroscopy／chevron，再確認 coupler 的 cz_f1／cz_f2 slots。",
        ),
    ]
    phase_pair: CZPhasePair
    depths: list[Annotated[int, Field(strict=True, ge=0)]] = Field(
        default_factory=lambda: [0, 1, 2, 3], min_length=1
    )
    seeds: list[Annotated[int, Field(strict=True, ge=0, le=2**31 - 1)]] = Field(
        default_factory=lambda: [0, 1], min_length=1
    )

    @model_validator(mode="after")
    def validate_sequence_inputs(self) -> Self:
        """Reject swapped phases, duplicate axes and mismatched role generators."""
        validate_classifiers(self.readout, self.roles)
        self.phase_pair.validate_roles(self.roles)
        if self.depths != sorted(set(self.depths)):
            raise ValueError("RB depths must be unique and increasing")
        if len(set(self.seeds)) != len(self.seeds):
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


@dataclass(frozen=True)
class RBResult:
    """Float64 class counts with axes (depth,seed,arm,round,F1,F2).

    depths counts random Cliffords; seeds retains exact random.Random identities.
    arms=[0,1] means reference/CZ-interleaved. rounds=[0..rounds-1] is a host
    acquisition axis. sums ends in (3,3), ordered g/e/other on each role.
    Counts are unnormalized; a wholly unacquired round stays NaN.
    """

    depths: NDArray[np.int64]
    seeds: NDArray[np.int64]
    arms: NDArray[np.int64]
    rounds: NDArray[np.int64]
    sums: NDArray[np.float64]


def restore_result(data: GroupedLoadData[RBCfg]) -> RBResult:
    """Restore counts; old per-shot IQ files fail because sums is missing."""
    sums = data.variable("sums")
    return RBResult(
        depths=sums.axes[5].astype(np.int64),
        seeds=sums.axes[4].astype(np.int64),
        arms=sums.axes[3].astype(np.int64),
        rounds=sums.axes[2].astype(np.int64),
        sums=sums.z.astype(np.float64),
    )


class RBExp(Experiment[RBResult, RBCfg]):
    """Paired RB counts, native IO and complete-round joint00 analysis."""

    Options = RBOptions
    AXES_SPEC: ClassVar[GroupedAxesSpec[RBResult, RBCfg]] = GroupedAxesSpec(
        variables=(
            VariableSpec(
                variable="sums",
                axes=(
                    VariableAxisSpec.generated_arange("terminal_f2", ""),
                    VariableAxisSpec.generated_arange("terminal_f1", ""),
                    VariableAxisSpec("Round", "", field_name="rounds", dtype=np.int64),
                    VariableAxisSpec("Arm", "", field_name="arms", dtype=np.int64),
                    VariableAxisSpec("Seed", "", field_name="seeds", dtype=np.int64),
                    VariableAxisSpec(
                        "Clifford depth", "", field_name="depths", dtype=np.int64
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
        cfg_schema_version="4.0",
        tag="two_qubit/rb",
        result_builder=restore_result,
    )

    def run(self, config: RBCfg, *, context: RunContext) -> RBResult:
        """Acquire optional reset → paired sequence → terminal class counts.

        Each host round acquires reps shots with rounds=1, not a host average.
        Return fixed raw count slots; cfg and saved sequence identities are unchanged.
        """
        validate_classifiers(config.readout, config.roles)
        viewer = live_line(
            context.plots,
            "measurement",
            "Random Clifford depth",
            "Joint00 survival",
            num_lines=2,
        )

        def update_view(sums: NDArray[np.float64]) -> None:
            totals = np.nansum(sums, axis=(1, 3))
            viewer.update(
                np.array(config.depths), joint_probabilities(totals)[..., 0, 0].T
            )

        buffer = SignalBuffer(
            (len(config.depths), len(config.seeds), 2, config.rounds, 3, 3),
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
                                    into=[
                                        f"gate_{index}_cz_f1_time",
                                        f"gate_{index}_cz_f2_time",
                                    ],
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
                                    
                                        f"gate_{index}_cz_f1_time",
                                        f"gate_{index}_cz_f2_time",
                                    ,
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
        return RBResult(
            np.array(config.depths, dtype=np.int64),
            np.array(config.seeds, dtype=np.int64),
            np.arange(2, dtype=np.int64),
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
                (data.depths, data.seeds, data.arms, data.rounds),
                (cfg.depths, cfg.seeds, [0, 1], np.arange(cfg.rounds)),
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
                estimate = estimate_d4(fit)
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
        answer = RBAnalysis(
            survival, acquired, means, curves, estimate, quality, reason
        )
        fig, ax = plots.subplots("fit", figsize=(9, 6))
        for arm, label in enumerate(("Reference", "CZ interleaved")):
            (line,) = ax.plot(data.depths, answer.mean_survival[arm], "o", label=label)
            if answer.estimate is not None:
                ax.plot(data.depths, answer.fit_survival[arm], color=line.get_color())
        ax.set(xlabel="Random Clifford depth", ylabel="Same-shot joint00 survival")
        ax.legend()
        if answer.reason is not None:
            ax.set_title(f"No RB estimate: {answer.reason}")
        else:
            ax.set_title("d=4 paired RB; classified ensemble, statistical seed CI only")
        fig.tight_layout()
        return answer
