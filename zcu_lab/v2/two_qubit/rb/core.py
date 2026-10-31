"""Paired two-qubit RB with acquisition-side classification and host rounds."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Annotated, ClassVar, Self

import numpy as np
from numpy.typing import NDArray
from pydantic import Field, model_validator
from qick import QickConfig
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
    Pulse,
    Reset,
    TimelineBuilder,
    TimelineProgramV2,
    shot_sums,
)
from zcu_tools.program.v2.modules import MuxReadout

from zcu_lab.v2.support import live_line

from .._drive import DualLineCfg
from .._readout import JointRoles
from .._sums import (
    joint_probabilities,
    readout_axes,
    terminal_counts,
    validate_classifiers,
)
from .._timeline import emit_readout_prefix
from .analysis import RBAnalysis, RBOptions, analyze_rb
from .program import CZPhasePair, sequence_modules
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


def build_program(
    cfg: RBCfg,
    soccfg: QickConfig,
    gates: Sequence[TwoQubitGate],
    *,
    heralded: bool,
) -> TimelineProgramV2:
    """Compile playback gates with optional reset, optional herald and terminal.

    gates comes from sequence.make_paired_sequences, including its own recovery.
    sequence_modules owns pulse translation and fresh zero-origin virtual frames.
    Every build creates fresh modules; the two tagged reads reuse one mux object.
    No acquisition occurs. Invalid role classifiers fail before compilation.
    """
    validate_classifiers(cfg.readout, cfg.roles)
    builder = TimelineBuilder(
        soccfg,
        reps=cfg.reps,
        initial_delay=cfg.initial_delay,
        relax_delay=cfg.relax_delay,
    )
    readout = emit_readout_prefix(builder, cfg.readout, cfg.reset, heralded=heralded)
    for module in sequence_modules(
        gates,
        x90_f1=cfg.x90_f1,
        x90_f2=cfg.x90_f2,
        drive=cfg.drive,
        phase_pair=cfg.phase_pair,
    ):
        builder.emit(module)
    builder.emit(readout, tag="terminal")
    return builder.build()


def acquire_counts(
    config: RBCfg, *, context: RunContext, heralded: bool
) -> NDArray[np.float64]:
    """Acquire each arm's host rounds separately, preserving raw count slots.

    Return (depth,seed,2,round,3,3), or four trailing class axes for herald.
    Compile once per depth/seed/arm and always acquire with rounds=1.
    Live curves show joint00 from accumulated counts over seeds and rounds.
    Cancellation leaves unfinished slots NaN; callers check context.scope.stop
    for acquisition errors. The input cfg and sequence identities stay unchanged.
    """
    validate_classifiers(config.readout, config.roles)
    viewer = live_line(
        context.plots,
        "measurement",
        "Random Clifford depth",
        "Joint00 survival" if not heralded else "Joint00 survival | herald (g,g)",
        num_lines=2,
    )

    def update_view(sums: NDArray[np.float64]) -> None:
        counts = terminal_counts(sums, heralded=heralded)
        # Unmeasured rounds do not contribute to the cumulative count ratio.
        totals = np.nansum(counts, axis=(1, 3))
        viewer.update(np.array(config.depths), joint_probabilities(totals)[..., 0, 0].T)

    class_shape = (3, 3, 3, 3) if heralded else (3, 3)
    buffer = SignalBuffer(
        (len(config.depths), len(config.seeds), 2, config.rounds, *class_shape),
        dtype=np.float64,
        on_update=update_view,
    )
    with Schedule(config, buffer, scope=context.scope) as sched:
        cfg = sched.cfg
        for depth, depth_step in sched.scan("depth", cfg.depths):
            for seed, seed_step in depth_step.scan("seed", cfg.seeds):
                pair = make_paired_sequences(depth, seed)
                for arm, arm_step in seed_step.scan("arm", [0, 1]):
                    sequence = (pair.reference, pair.interleaved)[arm]
                    prog = build_program(
                        cfg, context.soccfg, sequence.gates, heralded=heralded
                    )
                    processor = shot_sums(
                        prog, readout_axes(cfg.roles, heralded=heralded)
                    )
                    for _, round_step in arm_step.repeat("round", cfg.rounds):
                        round_step.run_acquire_summed(
                            prog,
                            context.soc,
                            processor=processor,
                            rounds=1,
                            options=RunOptions(),
                        )
    buffer.trigger_update(flush=True)
    return buffer.array


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
        return RBResult(
            np.array(config.depths, dtype=np.int64),
            np.array(config.seeds, dtype=np.int64),
            np.arange(2, dtype=np.int64),
            np.arange(config.rounds, dtype=np.int64),
            acquire_counts(config, context=context, heralded=False),
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
        Malformed counts raise; missing cfg or unidentifiable fits return a reason.
        """
        return analyze_rb(source, options, plots=plots)
