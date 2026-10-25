"""Thermal conditional phase acquisition with independent X/Y projections."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Annotated, ClassVar, Literal, Self

import numpy as np
from numpy.typing import NDArray
from pydantic import Field, model_validator
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.experiment import AxesSpec, Experiment
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import Pulse, SweepCfg
from zcu_tools.program.v2.modules import MuxReadout

from zcu_lab.v2.support import live_line

from .._drive import DualLineCfg, dual_line_drive
from .._readout import JointRoles, collect_joint_shots
from ..joint_readout.core import GECenters
from .analysis import ConditionalPhaseAnalysis, analyze_phase


class ConditionalPhaseCfg(ConfigBase):
    """Spectator I/Xπ → target X90 → n parallel CZ pairs → X90/Y90 → mux.

    Target selects F1/F2, never tone order. Projection Y changes target X90
    phase by +90 degrees relative to preparation. Each projection is an independent
    acquisition. Frequency is a common MHz host scan, gate_counts are positive
    integer pulse-train lengths. Full tones and round-major shots are retained.
    Optional GE centers are saved analysis inputs, not simulator truth.
    """

    reps: Annotated[
        int, FieldMeta(label="Reps", description="Number of repetitions per round.")
    ] = 1
    rounds: Annotated[
        int,
        FieldMeta(label="Rounds", description="Number of software acquisition rounds."),
    ] = 1
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
    x180_f1: Annotated[
        Pulse.Cfg,
        FieldMeta(
            label="F1 Xπ",
            description="填入 F1 calibrated x180 pulse slot。先做 amplitude／length Rabi，再確認 frequency MHz、gain a.u. within [-1, 1] 與 duration us。",
        ),
    ]
    x180_f2: Annotated[
        Pulse.Cfg,
        FieldMeta(
            label="F2 Xπ",
            description="填入 F2 calibrated x180 pulse slot。先做 amplitude／length Rabi，再確認 frequency MHz、gain a.u. within [-1, 1] 與 duration us。",
        ),
    ]
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
    target: Literal["f1", "f2"] = "f1"
    frequency: Annotated[
        SweepCfg,
        FieldMeta(
            unit="MHz",
            label="Frequency",
            description="填入 common CZ frequency 起訖值，單位 MHz。先用 conditional spectroscopy／chevron 定位雙線 CZ pulse。",
        ),
    ]
    gate_counts: list[Annotated[int, Field(strict=True, ge=1)]] = Field(
        default_factory=lambda: [1, 3, 5], min_length=1
    )
    f1_centers: Annotated[
        GECenters | None,
        FieldMeta(
            label="F1 GE centers",
            description="填入 singleshot/ge 取得的 F1 ground／excited complex IQ centers，使用 normalized raw ADC units。",
        ),
    ] = None
    f2_centers: Annotated[
        GECenters | None,
        FieldMeta(
            label="F2 GE centers",
            description="填入 singleshot/ge 取得的 F2 ground／excited complex IQ centers，使用 normalized raw ADC units。",
        ),
    ] = None

    @model_validator(mode="after")
    def validate_pair(self) -> Self:
        """Reject mismatched role generators, nonfinite frequencies and repeated n."""
        self.roles.validate_readout(self.readout)
        for drive, x180, x90 in (
            (self.drive.f1, self.x180_f1, self.x90_f1),
            (self.drive.f2, self.x180_f2, self.x90_f2),
        ):
            if drive.ch != x180.ch or drive.ch != x90.ch:
                raise ValueError(
                    "CZ, Xπ and X90 must use the same captured role generator"
                )
            if not isinstance(x90.phase, (int, float)) or not np.isfinite(x90.phase):
                raise ValueError("X90 phase must be a finite scalar")
        frequency = sweep2array(self.frequency)
        if not np.all(np.isfinite(frequency)) or np.any(np.diff(frequency) <= 0):
            raise ValueError(
                "Common frequency coordinates must be finite and strictly increasing"
            )
        if len(set(self.gate_counts)) != len(self.gate_counts):
            raise ValueError("Gate counts must be unique")
        return self


@dataclass(frozen=True)
class ConditionalPhaseResult:
    """Raw IQ (2 spectator branches,2 quadratures,Nn,Nf,rounds*reps,Ntones).

    branches=[0,1] mean nominal spectator I/Xπ on a thermal ensemble;
    quadratures=[0,1] mean X90/Y90 projection pulse axes. No heralding,
    postselection or averaging precedes persistence. Unacquired whole round
    slots remain complex NaN; acquired IQ must be finite ADC units/sample.
    """

    branches: NDArray[np.int64]
    quadratures: NDArray[np.int64]
    gate_counts: NDArray[np.int64]
    frequencies: NDArray[np.float64]
    shot_indices: NDArray[np.int64]
    tone_indices: NDArray[np.int64]
    signals: NDArray[np.complex128]


class ConditionalPhaseExp(Experiment[ConditionalPhaseResult, ConditionalPhaseCfg]):
    """Stateless conditional phase acquisition, saved-source analysis and native IO."""

    AXES_SPEC: ClassVar[AxesSpec[ConditionalPhaseResult, ConditionalPhaseCfg]] = (
        AxesSpec(
            axes=(
                AxesSpec.Axis("tone_indices", "Tone", "", dtype=np.int64),
                AxesSpec.Axis("shot_indices", "Shot", "", dtype=np.int64),
                AxesSpec.Axis("frequencies", "Common CZ frequency", "Hz", 1e6),
                AxesSpec.Axis("gate_counts", "CZ gate count", "", dtype=np.int64),
                AxesSpec.Axis("quadratures", "Projection X or Y", "", dtype=np.int64),
                AxesSpec.Axis("branches", "Spectator I or Xπ", "", dtype=np.int64),
            ),
            z=AxesSpec.Z("signals", "Raw IQ", "a.u."),
            result_type=ConditionalPhaseResult,
            cfg_type=ConditionalPhaseCfg,
            cfg_schema_version="3.0",
            tag="two_qubit/conditional_phase",
        )
    )

    def run(
        self, config: ConditionalPhaseCfg, *, context: RunContext
    ) -> ConditionalPhaseResult:
        """Acquire fixed branch/projection/n/frequency/round slots without cfg mutation.

        Caller owns transport/plots and checks cancel_signal.raise_if_error() to
        distinguish transport/nonfinite acquired IQ errors from normal cancellation.
        Final raw conversion overwrites its round slot; it never duplicates shots.
        """
        cfg = config.model_copy(deep=True, update={"rounds": 1})
        frequencies = sweep2array(cfg.frequency)
        ntones = len(cfg.readout.tones)
        viewer = live_line(
            context.plots,
            "measurement",
            "Common CZ frequency (MHz)",
            "Completed shots (all branches/projections/n)",
            num_lines=1,
        )

        def update_view(raw: NDArray[np.complex128]) -> None:
            viewer.update(
                frequencies, np.all(np.isfinite(raw), axis=-1).sum(axis=(0, 1, 2, 4, 5))
            )

        buffer = SignalBuffer(
            (
                2,
                2,
                len(cfg.gate_counts),
                len(frequencies),
                config.rounds,
                cfg.reps,
                ntones,
            ),
            on_update=update_view,
        )
        m = cfg
        target = m.x90_f1 if cfg.target == "f1" else m.x90_f2
        spectator = m.x180_f2 if cfg.target == "f1" else m.x180_f1
        with Schedule(cfg, buffer, scope=context.scope) as sched:
            for branch, branch_step in sched.scan("branch", [0, 1]):
                for quadrature, projection_step in branch_step.scan(
                    "quadrature", [0, 1]
                ):
                    for n, count_step in projection_step.scan(
                        "gate_count", cfg.gate_counts
                    ):
                        for frequency, step in count_step.scan(
                            "frequency", frequencies.tolist()
                        ):
                            builder = step.prog_builder(
                                context.soc,
                                context.soccfg,
                                reps=step.cfg.reps,
                                initial_delay=step.cfg.initial_delay,
                                relax_delay=step.cfg.relax_delay,
                            )
                            if branch:
                                builder.add_pulse("spectator", spectator)
                            builder.add_pulse("prepare_target", target)
                            for gate in range(n):
                                builder.add(
                                    dual_line_drive(
                                        cfg.drive,
                                        frequency=frequency,
                                        name=f"cz_{gate}",
                                    )
                                )
                            builder.add_pulse(
                                "projection",
                                target.with_updates(
                                    phase=target.phase + 90 * quadrature
                                ),
                            )
                            builder.add(MuxReadout("readout", m.readout))
                            prog = builder.build()
                            for _, round_step in step.repeat("round", config.rounds):
                                _ = round_step.prog_builder(
                                    context.soc,
                                    context.soccfg,
                                    reps=round_step.cfg.reps,
                                    initial_delay=round_step.cfg.initial_delay,
                                    relax_delay=round_step.cfg.relax_delay,
                                ).run_program(
                                    prog,
                                    raw2signal_fn=partial(collect_joint_shots, prog),
                                    rounds=round_step.cfg.rounds,
                                )
        buffer.trigger_update(flush=True)
        return ConditionalPhaseResult(
            np.arange(2, dtype=np.int64),
            np.arange(2, dtype=np.int64),
            np.array(cfg.gate_counts, dtype=np.int64),
            frequencies,
            np.arange(config.rounds * cfg.reps, dtype=np.int64),
            np.arange(ntones, dtype=np.int64),
            buffer.array.reshape(
                2,
                2,
                len(cfg.gate_counts),
                len(frequencies),
                config.rounds * cfg.reps,
                ntones,
            ),
        )

    def analyze(
        self,
        source: RunRecord[ConditionalPhaseCfg, ConditionalPhaseResult],
        options: None,
        *,
        plots: Plots,
    ) -> ConditionalPhaseAnalysis:
        """Analyze complete saved-source projection pairs; never consult current Form."""
        del options
        return analyze_phase(source, plots=plots)
