"""Paired two-qubit RB acquisition with explicit depth/seed/arm/round/shot/tone."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Annotated, Self

import numpy as np
from numpy.typing import NDArray
from pydantic import Field, model_validator
from zcu_tools.cfg_model import ConfigBase
from zcu_tools.experiment import IDENTITY, AxesSpec, Axis, PersistableExperiment, ZSpec
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import ModularProgramV2, ProgramV2Cfg, PulseCfg
from zcu_tools.program.v2.modules import MuxReadout, MuxReadoutCfg

from .._drive import DualLineCfg
from .._readout import JointRoles, collect_joint_shots
from ..joint_readout.core import GECenters
from .analysis import RBAnalysis, RBOptions, analyze_rb
from .program import CZPhasePair, sequence_modules
from .sequence import make_paired_sequences


def _round_collector(
    program: ModularProgramV2,
) -> Callable[[Sequence[NDArray[np.float64]]], NDArray[np.complex128]]:
    shots: NDArray[np.complex128] | None = None

    def collect(acquired: Sequence[NDArray[np.float64]]) -> NDArray[np.complex128]:
        nonlocal shots
        if shots is None:
            shots = collect_joint_shots(program, acquired)
        return shots

    return collect


class RBModules(ConfigBase):
    """Positive calibrated X90s for F1/F2 and the complete ordered mux readout.

    x90_f1/x90_f2 define each role's physical X axis in degrees; their channels
    must match that role's CZ charge line. readout retains every original tone.
    """

    x90_f1: PulseCfg
    x90_f2: PulseCfg
    readout: MuxReadoutCfg


class RBCfg(ProgramV2Cfg, ExpCfgModel):
    """Thermal B00 → paired random Clifford + own recovery → one mux read.

    modules/drive are captured calibrated inputs, never simulator fallbacks.
    roles maps Q0 to F1 and Q1 to F2, including original readout tone positions.
    phase_pair is required, ordered and component-named. Delays are us, pulse
    frequency MHz, phase degrees. No active reset/heralding/DC drive is added.
    depths counts random Cliffords, not primitive gates; unique increasing
    nonnegative integers. seeds are explicit distinct integer random.Random
    identities, shared across depths/arms/rounds and fully persisted (no child
    entropy reconstruction). The 31-bit bound preserves exact native axes.
    f1_centers/f2_centers are optional saved GE calibration; absent centers
    allow raw acquisition but analysis returns an explicit missing-source reason.
    """

    modules: RBModules
    roles: JointRoles
    drive: DualLineCfg
    phase_pair: CZPhasePair
    depths: list[Annotated[int, Field(strict=True, ge=0)]] = Field(
        default_factory=lambda: [0, 1, 2, 3], min_length=1
    )
    seeds: list[Annotated[int, Field(strict=True, ge=0, le=2**31 - 1)]] = Field(
        default_factory=lambda: [0, 1], min_length=1
    )
    f1_centers: GECenters | None = None
    f2_centers: GECenters | None = None

    @model_validator(mode="after")
    def validate_sequence_inputs(self) -> Self:
        """Reject swapped phases, duplicate axes and mismatched role generators."""
        self.roles.validate_readout(self.modules.readout)
        self.phase_pair.validate_roles(self.roles)
        if self.depths != sorted(set(self.depths)):
            raise ValueError("RB depths must be unique and increasing")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("RB seeds must be distinct complete integer identities")
        for drive, x90 in (
            (self.drive.f1, self.modules.x90_f1),
            (self.drive.f2, self.modules.x90_f2),
        ):
            if drive.ch != x90.ch:
                raise ValueError("CZ and X90 must use the same captured role generator")
            if not isinstance(x90.phase, (int, float)) or not np.isfinite(x90.phase):
                raise ValueError("X90 phase must be a finite scalar")
        return self


@dataclass(frozen=True)
class RBResult:
    """Normalized complex raw IQ (depth,seed,arm,round,shot,tone).

    depths counts random Cliffords; seeds retains exact random.Random identities.
    arms=[0,1] means reference/CZ-interleaved. rounds=[0..rounds-1] and
    shot_indices=[0..reps-1] stay separate. tone_indices is the complete original
    order. signals retains every shot without averaging/filtering/normalization
    changes. A wholly unacquired round stays complex NaN; acquired IQ is finite.
    """

    depths: NDArray[np.int64]
    seeds: NDArray[np.int64]
    arms: NDArray[np.int64]
    rounds: NDArray[np.int64]
    shot_indices: NDArray[np.int64]
    tone_indices: NDArray[np.int64]
    signals: NDArray[np.complex128]


class RBExp(PersistableExperiment[RBResult, RBCfg]):
    """Stateless paired RB definition, native IO and saved-GE joint00 analysis."""

    Options = RBOptions
    AXES_SPEC = AxesSpec(
        axes=(
            Axis("tone_indices", "Tone", "", IDENTITY, dtype=np.int64),
            Axis("shot_indices", "Shot", "", IDENTITY, dtype=np.int64),
            Axis("rounds", "Round", "", IDENTITY, dtype=np.int64),
            Axis("arms", "Arm", "", IDENTITY, dtype=np.int64),
            Axis("seeds", "Seed", "", IDENTITY, dtype=np.int64),
            Axis("depths", "Clifford depth", "", IDENTITY, dtype=np.int64),
        ),
        z=ZSpec("signals", "Raw IQ", "a.u."),
        result_type=RBResult,
        cfg_type=RBCfg,
        cfg_schema_version="1.0",
        tag="ftf/rb",
    )

    def run(self, config: RBCfg, *, context: RunContext) -> RBResult:
        """Compile each depth/seed/arm once; retain completed rounds in fixed slots.

        Each round acquires exactly reps same-shot full-tone IQ samples through
        the public MuxReadout collector. Its final conversion reuses that round's
        collected array, overwriting the same slot, never appending. Cancellation
        preserves completed slots and leaves unfinished rounds NaN. Caller owns
        handles/plots and checks cancel_signal.raise_if_error() for failures.
        The input config, aliases and library are not changed or reread.
        """
        cfg = config.model_copy(deep=True, update={"rounds": 1})
        ntones = len(cfg.modules.readout.tones)
        viewer = context.plots.liveplot_1d(
            "measurement", "Random Clifford depth", "Completed shots", num_lines=2
        )

        def update_view(raw: NDArray[np.complex128]) -> None:
            viewer.update(
                np.array(cfg.depths),
                np.all(np.isfinite(raw), axis=-1).sum(axis=(1, 3, 4)).T,
            )

        buffer = SignalBuffer(
            (len(cfg.depths), len(cfg.seeds), 2, config.rounds, cfg.reps, ntones),
            on_update=update_view,
        )
        with Schedule(cfg, buffer, scope=context.scope) as sched:
            for depth, depth_step in sched.scan("depth", cfg.depths):
                for seed, seed_step in depth_step.scan("seed", cfg.seeds):
                    pair = make_paired_sequences(depth, seed)
                    for arm, arm_step in seed_step.scan("arm", [0, 1]):
                        sequence = (pair.reference, pair.interleaved)[arm]
                        builder = arm_step.prog_builder(context.soc, context.soccfg)
                        builder.add(
                            sequence_modules(
                                sequence.gates,
                                x90_f1=cfg.modules.x90_f1,
                                x90_f2=cfg.modules.x90_f2,
                                drive=cfg.drive,
                                phase_pair=cfg.phase_pair,
                            )
                        )
                        builder.add(MuxReadout("readout", cfg.modules.readout))
                        prog = builder.build()
                        for _, round_step in arm_step.repeat("round", config.rounds):
                            _ = round_step.prog_builder(
                                context.soc, context.soccfg
                            ).run_program(prog, raw2signal_fn=_round_collector(prog))
        buffer.trigger_update(flush=True)
        return RBResult(
            np.array(cfg.depths, dtype=np.int64),
            np.array(cfg.seeds, dtype=np.int64),
            np.arange(2, dtype=np.int64),
            np.arange(config.rounds, dtype=np.int64),
            np.arange(cfg.reps, dtype=np.int64),
            np.arange(ntones, dtype=np.int64),
            buffer.array,
        )

    def analyze(
        self,
        source: RunRecord[RBCfg, RBResult],
        options: RBOptions,
        *,
        plots: Plots,
    ) -> RBAnalysis:
        """Classify saved same-shot joint00 and fit complete paired seed curves.

        Missing calibration/data or an unidentifiable model is a reason-bearing
        result, not a substituted fit. Malformed/nonfinite acquired raw raises.
        options controls only deterministic paired-seed bootstrap, not acquisition.
        """
        return analyze_rb(source, options, plots=plots)
